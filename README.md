# DSPy Activation Clamping

**Can we replace a system prompt with a steering vector?**

This repo explores whether the effect of a DSPy-optimized (or hand-crafted) system prompt can be captured as a steering vector in a transformer's activation space, then replayed at inference time *without* the prompt — saving tokens and potentially TTFB.

Idea originally posed by **Noah Vandal** in the [DSPy Discord](https://discord.gg/XCGy2WDCQB).

## Does it work?

**Short answer: not as a prompt replacement.**

The rigorous experiment in `results_rigorous/results.json` gives the canonical answer for this repo:

- **Binary sentiment**: not supported
- **Ternary sentiment**: not supported

The distinction that matters is:

- **Prompt-induced activation shift** is real.
- **Prompt replacement via a single steering vector** is not supported here.
- See `RESULTS_RIGOROUS.md` for the concise repo-level summary.

`experiment_rigorous.py` is the source of truth because it uses:

- train/validation/test splits
- validation-only layer/alpha tuning
- strict exact-label generation scoring
- separate log-prob classification scoring to separate semantics from formatting
- repeated seeded runs

## Rigorous Results

### Binary sentiment

| Condition | LogProb Accuracy | Generation Accuracy | Format Compliance | Tokens |
|---|---:|---:|---:|---:|
| Full prompt | **100.0%** | **100.0%** | **100.0%** | 78 |
| Minimal prompt | 90.0% | 0.0% | 0.0% | 6 |
| Hybrid prompt only | **100.0%** | **100.0%** | **100.0%** | 8 |
| Steering only | 90.0% | 0.0% | 0.0% | 6 |
| Hybrid + steering | **100.0%** | **100.0%** | **100.0%** | 8 |

Interpretation:

- The steering vector does **not** beat the minimal prompt semantically.
- It does **not** produce valid one-word labels under strict generation scoring.
- The short hybrid prompt already solves the task perfectly, so the vector adds nothing useful.

### Ternary sentiment

| Condition | LogProb Accuracy | Generation Accuracy | Format Compliance | Tokens |
|---|---:|---:|---:|---:|
| Full prompt | **91.7%** | **91.7%** | **100.0%** | 100 |
| Minimal prompt | 58.3% | 0.0% | 0.0% | 6 |
| Hybrid prompt only | 66.7% | 58.3% | 77.8% | 4 |
| Steering only | 58.3% | 0.0% | 0.0% | 6 |
| Hybrid + steering | 66.7% | 66.7% | 86.1% | 4 |

Interpretation:

- Steering alone again does **not** improve semantics over the minimal prompt.
- With a short prompt, steering appears to help **formatting / output mode** somewhat.
- It still does **not** recover the full prompt's semantic effect.

### Updated conclusion

- **Does prompt replacement work in this repo?** No.
- **Does steering do anything at all?** Some evidence says it can help with formatting when paired with a short prompt.
- **Does it replace the prompt's semantic function?** No, not under the rigorous evaluation.

The earlier scripts remain in the repo as exploratory artifacts, but they should not be treated as the main conclusion.

## The Core Idea

1. **DSPy signatures guide the model** to specific regions in activation space
2. We can **measure** this by recording activations with and without the optimized prompt
3. The difference (steering vector) captures the prompt's effect as a fixed-size artifact
4. **Injecting** this vector at inference eliminates the need for the prompt tokens

Related work:
- [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023) — steering vectors for behaviors
- [Prefix Tuning](https://arxiv.org/abs/2101.00190) (Li & Liang, 2021) — continuous prompt optimization
- [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023) — behavioral steering via vector addition

## Setup

```bash
# Requires Python 3.11+, uv
cd dspy-activation-clamping
uv sync

# Run the rigorous experiment (recommended)
uv run python experiment_rigorous.py --task all

# Legacy exploratory script
uv run python experiment_final.py

# With a different model
uv run python experiment_final.py --model HuggingFaceTB/SmolLM2-135M-Instruct
```

## Architecture

```
nnsight_lm.py        — NNsight model wrapper with activation extraction and forward-hook steering
steering.py          — Steering vector computation, evaluation, and analysis
rigorous_protocol.py — Strict scoring and stratified-split utilities for defensible evaluation
experiment_rigorous.py — Validation-tuned, held-out evaluation protocol
RESULTS_RIGOROUS.md — Canonical interpretation of the rigorous result
experiment_final.py  — Three-class experiment with concrete inputs/outputs at every stage
experiment_binary.py — Binary classification experiment (50 train / 50 test)
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

## What would close the gap

- **Per-class or conditional vectors** — separate vectors for different output contexts, selected by a lightweight input classifier. But this defeats the simplicity of the approach.
- **Soft prompts / prefix tuning** — these already exist, work better, and solve the same problem with continuous embeddings. They require gradient access (open-weight models only).
- **Larger models** — representations may be more linearly separable at scale, making the vector approximation more accurate.
- **DSPy integration** — automatic steering vector extraction as a compilation target, where DSPy optimizes the prompt AND extracts the corresponding vector in one pass.

## Credits

- Idea by **Noah Vandal** ([DSPy Discord](https://discord.gg/XCGy2WDCQB))
- Built with [nnsight](https://github.com/ndif-team/nnsight) for activation access and [DSPy](https://github.com/stanfordnlp/dspy) for prompt optimization
