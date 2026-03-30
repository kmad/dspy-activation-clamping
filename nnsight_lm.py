"""
Custom DSPy LM that wraps a HuggingFace model via nnsight.
Supports both normal generation AND activation recording/injection.
"""

import torch
import time
import uuid
import datetime
from dataclasses import dataclass
from nnsight import LanguageModel
from transformers import AutoTokenizer
from dspy.clients.base_lm import BaseLM


# Minimal OpenAI-compatible response objects for DSPy
@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iter__(self):
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage
    model: str = ""
    id: str = ""
    created: int = 0


class NNsightLM:
    """
    Core model wrapper: HuggingFace model via nnsight for activation access.
    """

    def __init__(self, model_name: str, device: str = "mps", max_new_tokens: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LanguageModel(
            model_name,
            device_map=device,
            dtype=torch.float32,
            dispatch=True,
        )

        self._detect_architecture()
        print(f"Loaded. {self.num_layers} layers, hidden_dim={self.hidden_dim}")

    def _detect_architecture(self):
        """Detect transformer block path for different architectures."""
        inner = self.model._model

        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            self.layer_path = "model.layers"
            self.num_layers = len(list(inner.model.layers.children()))
            self.hidden_dim = inner.config.hidden_size
        elif hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
            self.layer_path = "transformer.h"
            self.num_layers = len(list(inner.transformer.h.children()))
            self.hidden_dim = inner.config.n_embd
        else:
            raise ValueError(f"Unknown architecture for {self.model_name}")

    def get_layer(self, idx: int):
        """Get the nnsight Envoy for a specific transformer layer."""
        return self.model.get(f"{self.layer_path}.{idx}")

    def generate_text(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        """Generate text from chat messages."""
        max_new_tokens = max_new_tokens or self.max_new_tokens

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def extract_activations(self, messages: list[dict], layers: list[int] | None = None) -> dict[int, torch.Tensor]:
        """
        Run a forward pass and record residual stream activations at specified layers.
        Returns: {layer_idx: tensor of shape [1, seq_len, hidden_dim]}
        """
        if layers is None:
            layers = list(range(self.num_layers))

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        saved = {}
        with self.model.trace(prompt):
            for layer_idx in layers:
                layer = self.get_layer(layer_idx)
                saved[layer_idx] = layer.output[0].save()

        activations = {}
        for layer_idx in layers:
            activations[layer_idx] = saved[layer_idx].detach().cpu()

        return activations

    def generate_with_steering(
        self,
        messages: list[dict],
        steering_vectors: dict[int, torch.Tensor],
        alpha: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """
        Generate text while injecting steering vectors at specified layers.
        Uses forward hooks to add the steering vector during each forward pass.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Use PyTorch forward hooks directly for steering during generation
        handles = []
        inner = self.model._model

        for layer_idx, vec in steering_vectors.items():
            vec_device = vec.to(self.device)

            # Get the actual nn.Module for this layer
            parts = self.layer_path.split(".")
            module = inner
            for part in parts:
                module = getattr(module, part)
            layer_module = module[layer_idx]

            def make_hook(v, a):
                def hook(module, input, output):
                    # output is typically (hidden_states, ...) tuple
                    if isinstance(output, tuple):
                        h = output[0]
                        h = h + a * v
                        return (h,) + output[1:]
                    else:
                        return output + a * v
                return hook

            handle = layer_module.register_forward_hook(make_hook(vec_device, alpha))
            handles.append(handle)

        try:
            with torch.no_grad():
                output_ids = self.model._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            for h in handles:
                h.remove()

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class NNsightDSPyLM(BaseLM):
    """
    DSPy-compatible LM adapter. Subclasses dspy.BaseLM and returns
    OpenAI-format response objects from forward().
    """

    def __init__(self, nnsight_lm: NNsightLM):
        super().__init__(
            model=nnsight_lm.model_name,
            model_type="chat",
            temperature=0.0,
            max_tokens=nnsight_lm.max_new_tokens,
        )
        self.nnsight_lm = nnsight_lm

    def forward(self, prompt=None, messages=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        response_text = self.nnsight_lm.generate_text(messages)

        return _Response(
            choices=[_Choice(message=_Message(content=response_text))],
            usage=_Usage(),
            model=self.model,
            id=str(uuid.uuid4()),
            created=int(time.time()),
        )
