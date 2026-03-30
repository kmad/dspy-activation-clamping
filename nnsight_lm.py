"""
Custom DSPy LM that wraps a HuggingFace model via nnsight.
Supports normal generation, activation recording, steering injection,
and continuation scoring for strict classification-style evaluation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import time
import uuid

import torch
from dspy.clients.base_lm import BaseLM
from nnsight import LanguageModel
from transformers import AutoTokenizer


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

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
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

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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

    def _build_prompt(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def _encode_prompt(self, messages: list[dict]) -> torch.Tensor:
        prompt = self._build_prompt(messages)
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    @contextmanager
    def _steering_hooks(
        self,
        steering_vectors: dict[int, torch.Tensor] | None,
        alpha: float,
        token_strategy: str,
    ):
        if not steering_vectors:
            yield
            return

        if token_strategy not in {"all", "last"}:
            raise ValueError(f"Unsupported token_strategy: {token_strategy}")

        handles = []
        inner = self.model._model

        parts = self.layer_path.split(".")
        module = inner
        for part in parts:
            module = getattr(module, part)

        for layer_idx, vec in steering_vectors.items():
            layer_module = module[layer_idx]
            vec_device = vec.to(self.device)

            def make_hook(v, a, strategy):
                def hook(_module, _input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        rest = output[1:]
                    else:
                        hidden_states = output
                        rest = ()

                    steer = (a * v).to(
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                    if strategy == "all":
                        steered = hidden_states + steer.view(1, 1, -1)
                    else:
                        steered = hidden_states.clone()
                        steered[:, -1, :] = steered[:, -1, :] + steer.view(1, -1)

                    if isinstance(output, tuple):
                        return (steered,) + rest
                    return steered

                return hook

            handles.append(
                layer_module.register_forward_hook(
                    make_hook(vec_device, alpha, token_strategy)
                )
            )

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate_text(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        """Generate text from chat messages."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        input_ids = self._encode_prompt(messages)

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

        prompt = self._build_prompt(messages)

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
        token_strategy: str = "all",
    ) -> str:
        """
        Generate text while injecting steering vectors at specified layers.
        Uses forward hooks to add the steering vector during each forward pass.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        input_ids = self._encode_prompt(messages)

        with self._steering_hooks(steering_vectors, alpha, token_strategy):
            with torch.no_grad():
                output_ids = self.model._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def score_continuations(
        self,
        messages: list[dict],
        continuations: list[str],
        steering_vectors: dict[int, torch.Tensor] | None = None,
        alpha: float = 1.0,
        token_strategy: str = "all",
        normalize_by_length: bool = True,
    ) -> dict[str, float]:
        """
        Score candidate continuations with summed token log-probabilities.

        This is the primary evaluation path for rigorous classification tasks because
        it isolates semantic preference from output-format failures.
        """
        prompt_ids = self._encode_prompt(messages)
        prompt_len = prompt_ids.shape[1]
        scores = {}

        with self._steering_hooks(steering_vectors, alpha, token_strategy):
            for continuation in continuations:
                continuation_ids = self.tokenizer.encode(
                    continuation,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(self.device)
                input_ids = torch.cat([prompt_ids, continuation_ids], dim=1)

                with torch.no_grad():
                    logits = self.model._model(input_ids).logits[0]
                    log_probs = torch.log_softmax(logits, dim=-1)

                continuation_tokens = continuation_ids[0]
                token_log_probs = []
                for offset, token_id in enumerate(continuation_tokens):
                    position = prompt_len - 1 + offset
                    token_log_probs.append(log_probs[position, token_id].item())

                score = sum(token_log_probs)
                if normalize_by_length and token_log_probs:
                    score /= len(token_log_probs)
                scores[continuation] = score

        return scores


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
