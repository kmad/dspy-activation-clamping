"""
Steering vector extraction and evaluation.

Core idea:
  1. Run inputs through model WITH the optimized DSPy prompt → record activations
  2. Run inputs through model with a BASELINE (minimal) prompt → record activations
  3. Steering vector = mean(optimized_activations - baseline_activations)
  4. At inference: inject steering vector into baseline prompt → should recover optimized behavior
"""

import torch
import numpy as np
from dataclasses import dataclass
from nnsight_lm import NNsightLM


@dataclass
class SteeringResult:
    """Result of steering vector extraction."""
    vectors: dict[int, torch.Tensor]  # layer_idx -> [hidden_dim]
    layers: list[int]
    num_samples: int
    # Per-layer statistics
    norms: dict[int, float]  # L2 norm of each steering vector
    cosine_similarities: dict[int, float]  # avg cosine sim between individual diffs


def build_messages(system_prompt: str, user_content: str) -> list[dict]:
    """Build chat messages from system prompt and user content."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def extract_steering_vectors(
    lm: NNsightLM,
    inputs: list[str],
    optimized_system_prompt: str,
    baseline_system_prompt: str = "",
    layers: list[int] | None = None,
    token_aggregation: str = "last",  # "last", "mean", or "all"
) -> SteeringResult:
    """
    Extract steering vectors by comparing activations with optimized vs baseline prompts.

    Args:
        lm: The NNsight-wrapped language model
        inputs: List of user-message strings to run through the model
        optimized_system_prompt: The DSPy-optimized system prompt
        baseline_system_prompt: Minimal/empty system prompt for comparison
        layers: Which layers to extract from (default: all)
        token_aggregation: How to aggregate across token positions
            - "last": use only the last token's activation (most relevant for generation)
            - "mean": average across all token positions
            - "all": keep full sequence (for per-position analysis)
    """
    if layers is None:
        # Sample key layers: early, middle, late
        n = lm.num_layers
        layers = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    # Collect per-sample activation differences
    diffs_per_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for i, user_input in enumerate(inputs):
        print(f"  Processing input {i+1}/{len(inputs)}...")

        # Get activations with optimized prompt
        opt_messages = build_messages(optimized_system_prompt, user_input)
        opt_acts = lm.extract_activations(opt_messages, layers)

        # Get activations with baseline prompt
        base_messages = build_messages(baseline_system_prompt, user_input)
        base_acts = lm.extract_activations(base_messages, layers)

        for layer_idx in layers:
            opt_h = opt_acts[layer_idx]   # [seq_len, hidden] or [1, seq_len, hidden]
            base_h = base_acts[layer_idx]

            # Normalize to 2D [seq_len, hidden]
            if opt_h.dim() == 3:
                opt_h = opt_h[0]
            if base_h.dim() == 3:
                base_h = base_h[0]

            if token_aggregation == "last":
                opt_vec = opt_h[-1, :]    # [hidden]
                base_vec = base_h[-1, :]  # [hidden]
            elif token_aggregation == "mean":
                opt_vec = opt_h.mean(dim=0)   # [hidden]
                base_vec = base_h.mean(dim=0)  # [hidden]
            else:
                raise ValueError(f"Unsupported aggregation: {token_aggregation}")

            diff = opt_vec - base_vec  # [hidden]
            diffs_per_layer[layer_idx].append(diff)

    # Compute mean steering vector per layer
    vectors = {}
    norms = {}
    cosine_sims = {}

    for layer_idx in layers:
        diffs = torch.stack(diffs_per_layer[layer_idx])  # [num_samples, hidden]
        mean_diff = diffs.mean(dim=0)  # [hidden]
        vectors[layer_idx] = mean_diff
        norms[layer_idx] = mean_diff.norm().item()

        # Compute average pairwise cosine similarity of individual diffs
        # (measures consistency: are all samples shifting in the same direction?)
        if len(diffs) > 1:
            normed = torch.nn.functional.normalize(diffs, dim=1)
            sim_matrix = normed @ normed.T
            # Average off-diagonal elements
            n = len(diffs)
            mask = ~torch.eye(n, dtype=torch.bool)
            avg_sim = sim_matrix[mask].mean().item()
            cosine_sims[layer_idx] = avg_sim
        else:
            cosine_sims[layer_idx] = 1.0

    return SteeringResult(
        vectors=vectors,
        layers=layers,
        num_samples=len(inputs),
        norms=norms,
        cosine_similarities=cosine_sims,
    )


def evaluate_steering(
    lm: NNsightLM,
    test_inputs: list[str],
    expected_outputs: list[str],
    steering_result: SteeringResult,
    optimized_system_prompt: str,
    baseline_system_prompt: str = "",
    alpha_values: list[float] = [0.5, 1.0, 1.5, 2.0],
    target_layers: list[int] | None = None,
    score_fn=None,
) -> dict:
    """
    Evaluate whether steering vectors reproduce optimized behavior.

    Compares three conditions:
      1. Baseline (no optimization, no steering)
      2. Optimized (with full DSPy prompt)
      3. Steered (baseline prompt + steering vectors at various alphas)

    score_fn: callable(predicted, expected) -> float (0-1)
              Default: exact match after strip/lower
    """
    if score_fn is None:
        def score_fn(pred, exp):
            return 1.0 if pred.strip().lower() == exp.strip().lower() else 0.0

    if target_layers is None:
        target_layers = steering_result.layers

    results = {
        "baseline": {"outputs": [], "scores": []},
        "optimized": {"outputs": [], "scores": []},
    }
    for alpha in alpha_values:
        results[f"steered_alpha_{alpha}"] = {"outputs": [], "scores": []}

    for i, (user_input, expected) in enumerate(zip(test_inputs, expected_outputs)):
        print(f"  Evaluating {i+1}/{len(test_inputs)}...")

        # Condition 1: Baseline
        base_msgs = build_messages(baseline_system_prompt, user_input)
        base_out = lm.generate_text(base_msgs)
        results["baseline"]["outputs"].append(base_out)
        results["baseline"]["scores"].append(score_fn(base_out, expected))

        # Condition 2: Optimized prompt
        opt_msgs = build_messages(optimized_system_prompt, user_input)
        opt_out = lm.generate_text(opt_msgs)
        results["optimized"]["outputs"].append(opt_out)
        results["optimized"]["scores"].append(score_fn(opt_out, expected))

        # Condition 3: Steered at various alphas
        for alpha in alpha_values:
            steering_vecs = {
                l: steering_result.vectors[l]
                for l in target_layers
                if l in steering_result.vectors
            }
            steered_out = lm.generate_with_steering(
                base_msgs, steering_vecs, alpha=alpha
            )
            key = f"steered_alpha_{alpha}"
            results[key]["outputs"].append(steered_out)
            results[key]["scores"].append(score_fn(steered_out, expected))

    # Compute aggregate scores
    summary = {}
    for condition, data in results.items():
        scores = data["scores"]
        summary[condition] = {
            "mean_score": np.mean(scores),
            "num_correct": sum(1 for s in scores if s > 0.5),
            "total": len(scores),
        }

    return {"details": results, "summary": summary}


def find_best_layer(steering_result: SteeringResult) -> int:
    """
    Heuristic: the best layer for steering has high cosine similarity
    (consistent direction) and moderate norm (not too large = won't
    destroy generation, not too small = has signal).

    We prefer layers in the upper-middle range (25-75th percentile of norms)
    with the highest cosine similarity.
    """
    norms = steering_result.norms
    sorted_norms = sorted(norms.values())
    p25 = sorted_norms[len(sorted_norms) // 4]
    p75 = sorted_norms[3 * len(sorted_norms) // 4]

    best_layer = None
    best_sim = -1

    for layer_idx in steering_result.layers:
        norm = norms[layer_idx]
        sim = steering_result.cosine_similarities[layer_idx]
        # Prefer layers with moderate norms
        if p25 <= norm <= p75 and sim > best_sim:
            best_sim = sim
            best_layer = layer_idx

    # Fallback: highest cosine sim overall
    if best_layer is None:
        best_layer = max(
            steering_result.layers,
            key=lambda l: steering_result.cosine_similarities[l],
        )

    return best_layer
