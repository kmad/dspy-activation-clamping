# Rigorous Result

This file summarizes the canonical result from `results_rigorous/results.json`.

## Verdict

**Not supported.**

Under the held-out, validation-tuned protocol implemented in `experiment_rigorous.py`, a single steering vector does **not** replace the semantic effect of the task prompt for the tested model/task setup.

## Binary sentiment

| Condition | LogProb Accuracy | Generation Accuracy | Format Compliance |
|---|---:|---:|---:|
| Full prompt | **100.0%** | **100.0%** | **100.0%** |
| Minimal prompt | 90.0% | 0.0% | 0.0% |
| Hybrid prompt only | **100.0%** | **100.0%** | **100.0%** |
| Steering only | 90.0% | 0.0% | 0.0% |
| Hybrid + steering | **100.0%** | **100.0%** | **100.0%** |

Takeaway: the vector adds nothing over the short prompt, and steering-only does not produce valid labels under strict generation scoring.

## Ternary sentiment

| Condition | LogProb Accuracy | Generation Accuracy | Format Compliance |
|---|---:|---:|---:|
| Full prompt | **91.7%** | **91.7%** | **100.0%** |
| Minimal prompt | 58.3% | 0.0% | 0.0% |
| Hybrid prompt only | 66.7% | 58.3% | 77.8% |
| Steering only | 58.3% | 0.0% | 0.0% |
| Hybrid + steering | 66.7% | 66.7% | 86.1% |

Takeaway: steering helps formatting somewhat when paired with a short prompt, but it does not recover the full prompt's semantic effect.

## Interpretation

- The repo **does** show that prompts induce measurable activation-space shifts.
- The repo **does not** show that those shifts can replace the prompt as a semantic task specification.
- The strongest surviving claim is narrower: steering may be useful as a **formatting / response-mode assist**, not as a full prompt substitute.
