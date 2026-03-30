# DSPy Activation Clamping

**Can we replace a system prompt with a steering vector?**

This repo explores whether the effect of a DSPy-optimized (or hand-crafted) system prompt can be captured as a steering vector in a transformer's activation space, then replayed at inference time *without* the prompt — saving tokens and potentially TTFB.

## Key Results

Using **Qwen2.5-0.5B-Instruct** on a 20-example sentiment classification task:

| Condition | Accuracy | Prompt Tokens | Extra |
|---|---|---|---|
| A. Full prompt (100 tokens of instructions) | **90%** | 100 | - |
| B. No task prompt (baseline) | **0%** | 6 | - |
| C. Steering vector only (no task prompt) | **65%** | 6 | + 3.5KB vector |
| D. Hybrid: 4-token prompt + steering vector | **85%** | 4 | + 3.5KB vector |
| E. 4-token prompt alone | **80%** | 4 | - |

The steering vector at layer 18 takes the model from **0% to 65% accuracy** with zero task-specific tokens — recovering **72% of the full prompt's performance**.

The hybrid approach (minimal semantic prompt + format steering vector) achieves **85%** — within 5 points of the 100-token full prompt, using **96% fewer tokens**.

### What the steering vector captures

The system prompt's effect on activations is remarkably consistent across different inputs:

```
Layer  Norm   Consistency
    0  0.46      0.9806  █████████████████████████████░
    5  3.61      0.9493  ████████████████████████████░░
   10  6.52      0.9241  ███████████████████████████░░░
   15 15.27      0.8775  ██████████████████████████░░░░
   18 22.99      0.8505  █████████████████████████░░░░░
   23 51.68      0.8497  █████████████████████████░░░░░
```

Consistency (cosine similarity) > 0.85 at every layer means the prompt pushes ALL inputs in nearly the same activation-space direction — regardless of input content.

### Per-class breakdown

| Condition | Positive (8) | Negative (7) | Neutral (5) |
|---|---|---|---|
| Full Prompt | 8/8 | 7/7 | 3/5 |
| Steered L18 | 8/8 | 5/7 | 0/5 |
| Hybrid | 8/8 | 7/7 | 2/5 |

The steering vector captures positive/negative classification well but struggles with neutral — the averaged vector has a slight positive bias since positive + negative examples don't cancel out to neutral.

### Concrete example: what happens at each layer

With input *"This restaurant has the best pasta I have ever tasted"*:

| Layer | Alpha | Output |
|---|---|---|
| 0 | 1.0 | "I'm sorry, but as an AI language model..." (unchanged) |
| 8 | 1.0 | "That's amazing! It sounds like you had an exceptional..." (slightly shifted) |
| 12 | 1.0 | "this restaurant has the best pasta I have..." (echoing) |
| 16 | 1.0 | "best pasta." (format shifting) |
| 18 | 1.0 | **"positive"** (correct label) |
| 22 | 0.5 | **"positive"** (correct label) |
| 22 | 1.5 | "neutral" (over-steered) |

The transition from verbose prose to single-word labels happens sharply at layers 16-18 (67-75% depth in the 24-layer model).

## The Core Idea

1. **DSPy signatures guide the model** to specific regions in activation space
2. We can **measure** this by recording activations with and without the optimized prompt
3. The difference (steering vector) captures the prompt's effect as a fixed-size artifact
4. **Injecting** this vector at inference eliminates the need for the prompt tokens

This is related to:
- [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023) — steering vectors for behaviors
- [Prefix Tuning](https://arxiv.org/abs/2101.00190) (Li & Liang, 2021) — continuous prompt optimization
- [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023) — behavioral steering via vector addition

## Setup

```bash
# Requires Python 3.11+, uv
cd dspy-activation-clamping
uv sync

# Run the experiment (downloads Qwen2.5-0.5B-Instruct, ~1GB)
uv run python experiment_final.py

# With a different model
uv run python experiment_final.py --model HuggingFaceTB/SmolLM2-135M-Instruct
```

## Architecture

```
nnsight_lm.py       — NNsight model wrapper with activation extraction and forward-hook steering
steering.py         — Steering vector computation, evaluation, and analysis
experiment_final.py — Main experiment with concrete inputs/outputs at every stage
```

### How it works

**Extraction**: For each training input, run the model twice — once with the optimized prompt, once with a minimal prompt. Record the residual stream activation at each transformer layer. The steering vector is the mean difference.

**Injection**: During generation, register PyTorch forward hooks on the target layer(s). Each hook adds `alpha * steering_vector` to the hidden states. The model generates as if the prompt were present — but no prompt tokens are consumed.

```python
from nnsight_lm import NNsightLM
from steering import extract_steering_vectors, build_messages

lm = NNsightLM("Qwen/Qwen2.5-0.5B-Instruct")

# Extract
steering = extract_steering_vectors(
    lm,
    inputs=["Great movie!", "Terrible food", "It was okay"],
    optimized_system_prompt="Classify sentiment as positive/negative/neutral. One word only.",
    baseline_system_prompt="You are a helpful assistant.",
)

# Inject
msgs = build_messages("You are a helpful assistant.", "Best pasta ever!")
output = lm.generate_with_steering(
    msgs,
    steering_vectors={18: steering.vectors[18]},
    alpha=1.0,
)
# -> "positive" (without any classification instructions in the prompt)
```

## Limitations

1. **Neutral class weakness**: The averaged steering vector biases toward positive/negative, making neutral classification unreliable
2. **Alpha sensitivity**: The optimal alpha has a narrow window (~0.5 units) between "not enough effect" and "corrupted output"
3. **Static vs dynamic**: A fixed vector can't capture context-dependent prompt reinterpretation — this is why 65% < 90%
4. **Small model**: Results on 0.5B params; larger models may show different patterns

## Future Directions

- **Per-class steering vectors**: Separate vectors for positive/negative/neutral contexts
- **Conditional steering**: Lightweight input classifier selects which vector to apply
- **DSPy integration**: Automatic steering vector extraction as a DSPy compilation target
- **Larger models**: Test on 7B+ to see if the gap closes with model scale
- **Cross-task transfer**: Do format-steering vectors generalize across different classification tasks?
