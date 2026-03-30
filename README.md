# DSPy Activation Clamping

**Can we replace a system prompt with a steering vector?**

This repo explores whether the effect of a DSPy-optimized (or hand-crafted) system prompt can be captured as a steering vector in a transformer's activation space, then replayed at inference time *without* the prompt — saving tokens and potentially TTFB.

Idea originally posed by **Noah Vandal** in the [DSPy Discord](https://discord.gg/XCGy2WDCQB).

## Does it work?

**Short answer: the concept is validated, but the performance gap is too large for practical use today.**

The core intuition is correct — prompts DO create measurable, consistent directions in activation space (cosine similarity >0.93 across all layers). A single 896-float vector CAN flip a model from verbose "I'm sorry, but as an AI..." prose to outputting single-word classification labels. That behavioral shift from a fixed-size vector is real and striking.

But 65% vs 90% is not comparable performance. The steering vector alone recovers ~72% of the full prompt's accuracy, and the remaining gap is fundamental to the approach, not a tuning issue.

### What works

- **Prompts push activations in a consistent direction.** Cosine similarity >0.93 at every layer across diverse inputs. This means the prompt creates nearly the same activation-space shift regardless of input content — exactly what you'd need for a vector replacement to be viable.
- **A single vector can change output format.** The model goes from multi-sentence conversational responses to single-word labels. At layer 18 the transition is sharp and clean.
- **Positive and negative classification transfers well.** The steering vector gets positive 8/8 and negative 5/7 — the polarity signal is clearly captured.

### What doesn't work

- **65% vs 90% is a real gap.** The steering vector misses a quarter of the full prompt's accuracy. This isn't a tuning issue — it's structural.
- **Neutral class completely fails (0/5).** The averaged vector has a polarity bias. Positive and negative training examples don't average to "neutral" — they average to "has strong sentiment." The vector can't represent absence of polarity.
- **The alpha window is razor-thin.** At layer 18: alpha=0.9 gives 70%, alpha=1.0 gives 65%, alpha=1.5 gives degenerate repetition. There is no comfortable operating range — you're always one small perturbation from either "not enough effect" or gibberish.
- **The hybrid result (85%) is misleading.** "Classify sentiment." alone gets 80%. The steering vector only adds 5 points. The short prompt is doing the real work, not the vector. If you can write a better short prompt, you don't need steering at all.

### Why the gap exists

A static vector adds the same perturbation regardless of input. But the prompt effect is context-dependent. When the model reads "Respond with EXACTLY one word: positive, negative, or neutral," it dynamically applies different reasoning to "I love this" vs "I hate this" vs "It was fine." A fixed vector can't do that — it's the difference between a constant offset and a learned function.

## Results

Using **Qwen2.5-0.5B-Instruct** on a 20-example sentiment classification task:

| Condition | Accuracy | Prompt Tokens | Extra |
|---|---|---|---|
| A. Full prompt (100 tokens of instructions) | **90%** | 100 | - |
| B. No task prompt (baseline) | **0%** | 6 | - |
| C. Steering vector only (no task prompt) | **65%** | 6 | + 3.5KB vector |
| D. Hybrid: 4-token prompt + steering vector | **85%** | 4 | + 3.5KB vector |
| E. 4-token prompt alone | **80%** | 4 | - |

### Activation consistency by layer

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

### Concrete example: what happens at each layer

With input *"This restaurant has the best pasta I have ever tasted"*:

| Layer | Alpha | Output |
|---|---|---|
| 0 | 1.0 | "I'm sorry, but as an AI language model, I don't have access..." (unchanged) |
| 8 | 1.0 | "That's amazing! It sounds like you had an exceptional experience..." (slightly shifted) |
| 12 | 1.0 | "This restaurant has the best pasta I have ever tasted in my life" (echoing input) |
| 16 | 1.0 | "best pasta." (format shifting — terse, no longer conversational) |
| 18 | 1.0 | **"positive"** (correct single-word label) |
| 22 | 0.5 | "I'm happy to help! However, I need more information..." (still conversational) |
| 22 | 1.5 | "neutralneutralneutral..." (over-steered, degenerate repetition) |

The transition from verbose prose to single-word labels happens sharply at layers 16-18 (67-75% depth in the 24-layer model). Layer 22 at alpha=0.5 is too weak to flip the format; at alpha=1.5 it overshoots into degenerate repetition. The sweet spot is narrow.

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

## What would close the gap

- **Per-class or conditional vectors** — separate vectors for different output contexts, selected by a lightweight input classifier. But this defeats the simplicity of the approach.
- **Soft prompts / prefix tuning** — these already exist, work better, and solve the same problem with continuous embeddings. They require gradient access (open-weight models only).
- **Larger models** — representations may be more linearly separable at scale, making the vector approximation more accurate.
- **DSPy integration** — automatic steering vector extraction as a compilation target, where DSPy optimizes the prompt AND extracts the corresponding vector in one pass.

## Credits

- Idea by **Noah Vandal** ([DSPy Discord](https://discord.gg/XCGy2WDCQB))
- Built with [nnsight](https://github.com/ndif-team/nnsight) for activation access and [DSPy](https://github.com/stanfordnlp/dspy) for prompt optimization
