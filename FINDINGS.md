# DSPy Activation Clamping: Experimental Findings

> Note
> These findings come from the earlier exploratory scripts in this repo.
> For a defensible answer, use `experiment_rigorous.py`, which adds held-out
> validation/test splits, strict scoring, and seeded repeats.

## Rigorous Update

The current repo-level conclusion is **not supported**:

- steering alone does not match the full prompt on held-out evaluation
- steering alone does not outperform the short hybrid prompt baseline
- the strongest remaining signal is modest help with **formatting**, not semantics

See `RESULTS_RIGOROUS.md` and `results_rigorous/results.json` for the canonical result.

## Core Hypothesis
Can we extract the effect of a DSPy-optimized prompt as a steering vector in activation space, then reproduce that effect by injecting the vector WITHOUT the prompt — saving tokens and potentially TTFB?

## Setup
- **Model**: Qwen2.5-0.5B-Instruct (24 layers, hidden_dim=896)
- **Task**: Sentiment classification (positive/negative/neutral)
- **Optimized prompt**: 90-token detailed system prompt with rules
- **Baseline prompt**: 5-token minimal prompt ("Respond to the user.")
- **Tool**: nnsight for activation extraction and injection via PyTorch forward hooks

## Key Results

### The prompt effect is real and measurable
| Condition | Accuracy |
|-----------|----------|
| Baseline (no system prompt) | 0% |
| Optimized (90-token system prompt) | 93% |

### Steering vectors capture consistent directions
- **Cosine similarity 0.91-0.98** across all layers when comparing activation differences between optimized and baseline prompts across different inputs
- Mean aggregation shows even higher consistency (0.99+), though norms are inflated
- The prompt pushes activations in an **almost perfectly consistent direction** regardless of input content

### Steering partially reproduces prompt behavior
At **layer 16, alpha=1.5-2.0**, steering vectors:
- Successfully flip the model from conversational outputs to single-word label format
- Correctly classify positive and neutral examples
- Struggle with negative examples (produce corrupted tokens like "passenest", "passual")
- Achieve ~33% accuracy vs 93% for the full prompt

### Best results by layer
| Layer | Behavior |
|-------|----------|
| 0-8 | Minimal effect at reasonable alphas; model still conversational |
| 12 | Starts shifting toward labels but inconsistent |
| 16 | Best results — flips output format, gets sentiment partially right |
| 20 | Labels appear but accuracy drops |
| 24+ | Degenerates at even small alphas (norms too large) |

## Analysis

### What works
1. **Format steering**: A single vector CAN change the model's output format from verbose conversation to structured single-word labels
2. **Direction consistency**: The DSPy prompt creates a remarkably consistent activation shift across diverse inputs — cosine similarity >0.93 at every layer
3. **The middle-to-late layers** (66-80% depth) are the sweet spot for behavioral steering

### What doesn't work yet
1. **Semantic precision**: The steering vector is an average across positive, negative, and neutral examples. This creates a "label-ness" direction but slightly biases toward the most common label in the training set
2. **Format fragility**: The line between "correct single-word label" and "corrupted token" is very thin in alpha space (~0.3 units)
3. **Single vector limitation**: A 90-token prompt encodes multiple independent behaviors (format constraint, classification rules, tone). A single steering vector collapses these into one direction

### Why the gap exists (93% prompt vs 33% steered)
The system prompt does at least three things:
1. **Format**: "respond with EXACTLY one word" — steering captures this
2. **Semantic mapping**: positive=happiness, negative=disappointment — partially captured
3. **Dynamic re-interpretation**: the LLM reads the prompt and applies it to each input differently. A static vector can't do this.

This third point is the fundamental limitation. Steering vectors are **context-independent** — they add the same perturbation regardless of input. But the prompt effect is **context-dependent** — the same prompt words interact differently with "I love this" vs "I hate this".

## Implications for the original question

### Partially validated
- DSPy signatures DO guide the model to specific regions in activation space
- These regions CAN be identified and the direction CAN be extracted
- Simple format/behavior steering (e.g., "respond concisely", "use JSON format") is likely achievable

### Not yet achievable
- Full semantic task performance replacement via activation clamping
- The 94% token reduction doesn't come close to matching prompt performance yet

### What would close the gap
1. **Per-class steering vectors**: Instead of one average vector, compute separate vectors for different output classes or contexts
2. **Conditional steering**: Use a lightweight classifier on the input to select which steering vector to apply
3. **Multi-layer injection with learned alphas**: Use a small MLP to predict optimal alpha per layer
4. **Soft prompts**: The existing prefix-tuning literature already solves this with continuous embeddings — the activation clamping approach is essentially prefix tuning at deeper layers

### Most promising path forward
The hybrid approach: use steering vectors for **format/behavioral constraints** (which are consistent across inputs) and a minimal prompt for **task semantics** (which vary per input). This could reduce a 90-token prompt to ~15 tokens + a steering vector, getting most of the benefit with significant token savings.

## Files
- `nnsight_lm.py` — NNsight wrapper with activation extraction and forward-hook steering
- `steering.py` — Steering vector computation and evaluation
- `experiment.py` — Full DSPy optimization → steering pipeline
- `experiment_direct.py` — Direct prompt-vs-steering comparison (bypasses DSPy)
