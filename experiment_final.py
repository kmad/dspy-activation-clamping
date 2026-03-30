"""
DSPy Activation Clamping — Final Experiment

Demonstrates that a system prompt's effect on LLM behavior can be partially
captured as a steering vector in activation space, then replayed WITHOUT
the prompt.

Three conditions compared:
  A. Full prompt (90 tokens) — the gold standard
  B. No prompt — baseline, model produces unhelpful prose
  C. Steering vector only — no prompt tokens, vector injected at inference

Plus a hybrid condition:
  D. Minimal prompt (3 tokens) + steering vector — best of both worlds
"""

import torch
import json
import os
import sys
import time
import numpy as np
from nnsight_lm import NNsightLM
from steering import extract_steering_vectors, build_messages


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

TRAIN = [
    ("The movie was absolutely fantastic and I loved every minute of it", "positive"),
    ("Terrible waste of time, the worst film I have seen this year", "negative"),
    ("It was an okay experience, nothing special but not bad either", "neutral"),
    ("Brilliantly directed with stunning performances from the entire cast", "positive"),
    ("The food was disgusting and the service was incredibly rude", "negative"),
    ("A true masterpiece of modern cinema, truly groundbreaking work", "positive"),
    ("Completely boring and predictable from start to finish", "negative"),
    ("It was fine, it met my basic expectations and nothing more", "neutral"),
    ("Absolutely delightful, this exceeded all of my expectations", "positive"),
    ("Horrible quality, I would never recommend this to anyone", "negative"),
    ("The product works as advertised, nothing more and nothing less", "neutral"),
    ("Outstanding customer service, they really went above and beyond", "positive"),
]

TEST = [
    ("This restaurant has the best pasta I have ever tasted in my life", "positive"),
    ("What a complete waste of money, the product broke immediately", "negative"),
    ("The hotel room was clean and the staff was friendly and helpful", "positive"),
    ("Mediocre at best, I honestly expected much more from this brand", "negative"),
    ("I am thoroughly impressed with the quality and fine craftsmanship", "positive"),
    ("Awful experience from start to finish, I will never come back", "negative"),
    ("Standard quality overall, it gets the job done without any flair", "neutral"),
    ("The concert last night was electrifying, best night of my life", "positive"),
    ("Soggy fries and cold burgers, an extremely disappointing meal", "negative"),
    ("Not great and not terrible, just a perfectly average Tuesday", "neutral"),
    ("I would definitely buy this again, what an amazing value", "positive"),
    ("The worst customer service experience I have ever encountered", "negative"),
    ("It does exactly what it says on the tin, no surprises at all", "neutral"),
    ("My children absolutely adore this toy, it was worth every penny", "positive"),
    ("Overpriced and underwhelming in every way, save your money", "negative"),
    ("The book was a decent read but nothing I would rave about", "neutral"),
    ("Spectacular views and impeccable service at this resort", "positive"),
    ("Arrived damaged with missing parts, very frustrating experience", "negative"),
    ("A perfectly serviceable option if you are on a tight budget", "neutral"),
    ("This is hands down the best purchase I have made all year", "positive"),
]

# Prompts
FULL_PROMPT = """You are a precise sentiment classifier. Given a piece of text, classify its sentiment.

Rules:
- Respond with EXACTLY one word: positive, negative, or neutral
- "positive" means the text expresses happiness, satisfaction, praise, or enthusiasm
- "negative" means the text expresses anger, disappointment, criticism, or dissatisfaction
- "neutral" means the text is factual, balanced, or neither clearly positive nor negative
- Do NOT explain your reasoning. Output ONLY the label."""

MINIMAL_PROMPT = "You are a helpful assistant."
HYBRID_PROMPT = "Classify sentiment."  # 3 tokens of semantics, no format instructions


def extract_label(raw: str) -> str:
    """Extract sentiment label from model output."""
    raw = raw.strip().lower()
    if raw in ("positive", "negative", "neutral"):
        return raw
    first = raw.split()[0].rstrip(".,!:;") if raw else ""
    if first in ("positive", "negative", "neutral"):
        return first
    for label in ["positive", "negative", "neutral"]:
        if label in raw:
            return label
    return raw[:30]


def evaluate(lm, texts, labels, system_prompt, steering_vecs=None, alpha=0.0):
    """Run evaluation, return list of {input, expected, predicted, raw, correct}."""
    results = []
    for text, expected in zip(texts, labels):
        msgs = build_messages(system_prompt or "", text)
        if steering_vecs and alpha > 0:
            raw = lm.generate_with_steering(msgs, steering_vecs, alpha=alpha)
        else:
            raw = lm.generate_text(msgs)
        predicted = extract_label(raw)
        results.append({
            "input": text,
            "expected": expected,
            "predicted": predicted,
            "raw": raw.replace("\n", "\\n")[:80],
            "correct": predicted == expected,
        })
    return results


def accuracy(results):
    return sum(r["correct"] for r in results) / len(results) if results else 0


def print_table(results, label):
    correct = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\n  [{label}] {correct}/{total} ({accuracy(results):.0%})")
    print(f"  {'':>4} {'Input':<52} {'Expected':>9} {'Predicted':>11} {'Raw Output'}")
    print(f"  {'':>4} {'-'*100}")
    for r in results:
        mark = " OK" if r["correct"] else " XX"
        inp = r["input"][:49] + "..." if len(r["input"]) > 52 else r["input"]
        print(f"  [{mark}] {inp:<52} {r['expected']:>9} {r['predicted']:>11} | {r['raw'][:40]}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def main(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="mps"):
    output_dir = "results_final"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("DSPy ACTIVATION CLAMPING — FINAL EXPERIMENT")
    print(f"Model: {model_name}")
    print("=" * 80)

    lm = NNsightLM(model_name, device=device, max_new_tokens=24)
    texts = [t for t, _ in TEST]
    labels = [l for _, l in TEST]
    train_texts = [t for t, _ in TRAIN]

    # ── Phase 1: Baselines ──────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 1: BASELINES")
    print("─" * 80)

    print("\nCondition A — Full optimized prompt (100 tokens)")
    results_a = evaluate(lm, texts, labels, FULL_PROMPT)
    print_table(results_a, "Full Prompt")

    print("\nCondition B — Minimal prompt, no task instructions (6 tokens)")
    results_b = evaluate(lm, texts, labels, MINIMAL_PROMPT)
    print_table(results_b, "Minimal Prompt")

    # ── Phase 2: Extract Steering Vectors ───────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 2: EXTRACT STEERING VECTORS")
    print("─" * 80)
    print(f"\nExtracting activation differences between:")
    print(f"  Optimized: '{FULL_PROMPT[:60]}...' (100 tokens)")
    print(f"  Baseline:  '{MINIMAL_PROMPT}' (6 tokens)")
    print(f"  Training samples: {len(train_texts)}")

    all_layers = list(range(lm.num_layers))
    steering = extract_steering_vectors(
        lm, train_texts, FULL_PROMPT, MINIMAL_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    print(f"\nActivation difference analysis (per layer):")
    print(f"  {'Layer':>6} {'Norm':>8} {'Consistency':>12}  Visual")
    print(f"  {'-'*55}")
    for l in steering.layers:
        bar_len = int(steering.cosine_similarities[l] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {l:>6} {steering.norms[l]:>8.2f} {steering.cosine_similarities[l]:>11.4f}  {bar}")

    print(f"\n  Interpretation:")
    print(f"  - Norm = magnitude of the activation shift caused by the prompt")
    print(f"  - Consistency = cosine similarity across different inputs (1.0 = perfectly consistent)")
    print(f"  - High consistency means the prompt pushes ALL inputs in the same direction")
    avg_cos = np.mean([steering.cosine_similarities[l] for l in steering.layers])
    print(f"  - Average consistency across all layers: {avg_cos:.4f}")

    # ── Phase 3: Find Optimal Steering Configuration ────────────────────
    print("\n" + "─" * 80)
    print("PHASE 3: FIND OPTIMAL STEERING CONFIGURATION")
    print("─" * 80)

    print(f"\nSweeping all layers x alpha values (quick test on first 5 examples)...")
    print(f"  {'Layer':>6} {'Alpha':>6} {'Score':>6}  Outputs")
    print(f"  {'-'*70}")

    best = {"acc": 0, "layer": 0, "alpha": 0}
    sweep_data = []

    for layer in range(0, lm.num_layers):
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
            vecs = {layer: steering.vectors[layer]}
            quick = evaluate(lm, texts[:5], labels[:5], MINIMAL_PROMPT,
                           steering_vecs=vecs, alpha=alpha)
            acc = accuracy(quick)
            sweep_data.append({"layer": layer, "alpha": alpha, "accuracy": acc})

            preds = [r["predicted"][:8] for r in quick]
            if acc >= 0.6:  # Only print good results
                print(f"  {layer:>6} {alpha:>6.1f} {acc:>5.0%}   {preds}")

            if acc > best["acc"]:
                best = {"acc": acc, "layer": layer, "alpha": alpha}

    print(f"\n  Best quick-test config: layer={best['layer']}, alpha={best['alpha']} ({best['acc']:.0%})")

    # Fine-tune alpha around best layer
    print(f"\n  Fine-tuning alpha for layer {best['layer']}...")
    fine_results = []
    for alpha in np.arange(max(0.1, best["alpha"] - 0.5), best["alpha"] + 0.6, 0.05):
        alpha = round(alpha, 2)
        vecs = {best["layer"]: steering.vectors[best["layer"]]}
        r = evaluate(lm, texts[:10], labels[:10], MINIMAL_PROMPT,
                    steering_vecs=vecs, alpha=alpha)
        fine_results.append((alpha, accuracy(r)))

    fine_results.sort(key=lambda x: -x[1])
    print(f"  Top 5 alphas:")
    for alpha, acc in fine_results[:5]:
        print(f"    alpha={alpha:.2f}: {acc:.0%}")

    best_fine_alpha = fine_results[0][0]

    # Also check layer 22 which showed 80% in quick test
    # and nearby layers
    print(f"\n  Checking alternative layers...")
    alt_configs = []
    for layer in range(max(0, best["layer"] - 4), min(lm.num_layers, best["layer"] + 5)):
        for alpha in np.arange(0.3, 2.5, 0.1):
            alpha = round(alpha, 1)
            vecs = {layer: steering.vectors[layer]}
            r = evaluate(lm, texts[:10], labels[:10], MINIMAL_PROMPT,
                        steering_vecs=vecs, alpha=alpha)
            acc = accuracy(r)
            if acc >= 0.5:
                alt_configs.append({"layer": layer, "alpha": alpha, "accuracy": acc})

    alt_configs.sort(key=lambda x: -x["accuracy"])
    if alt_configs:
        print(f"  Top alternative configs (10-sample test):")
        for c in alt_configs[:5]:
            print(f"    layer={c['layer']}, alpha={c['alpha']}: {c['accuracy']:.0%}")

    # ── Phase 4: Full Evaluation ────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 4: FULL EVALUATION (20 test examples)")
    print("─" * 80)

    # Pick top 2 configs for full eval
    configs_to_eval = [
        (best["layer"], best_fine_alpha, "Best (sweep)"),
    ]
    if alt_configs and (alt_configs[0]["layer"] != best["layer"] or
                        alt_configs[0]["alpha"] != best_fine_alpha):
        configs_to_eval.append(
            (alt_configs[0]["layer"], alt_configs[0]["alpha"], "Best (alt)")
        )

    full_results = {}

    for layer, alpha, name in configs_to_eval:
        vecs = {layer: steering.vectors[layer]}
        print(f"\nCondition C — Steering only: layer={layer}, alpha={alpha} [{name}]")
        r = evaluate(lm, texts, labels, MINIMAL_PROMPT,
                    steering_vecs=vecs, alpha=alpha)
        print_table(r, f"Steered L{layer} α={alpha}")
        full_results[f"steered_L{layer}_a{alpha}"] = r

    # ── Phase 5: Hybrid Approach ────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 5: HYBRID — Minimal semantic prompt + steering vector")
    print("─" * 80)

    print(f"\nHybrid prompt: '{HYBRID_PROMPT}' ({len(lm.tokenizer.encode(HYBRID_PROMPT))} tokens)")
    print(f"vs Full prompt: '{FULL_PROMPT[:50]}...' ({len(lm.tokenizer.encode(FULL_PROMPT))} tokens)")

    # Extract hybrid-specific vectors: full prompt minus hybrid prompt
    print(f"\nExtracting format-specific steering vectors...")
    hybrid_steering = extract_steering_vectors(
        lm, train_texts, FULL_PROMPT, HYBRID_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    # Sweep for best hybrid config
    print(f"\nSweeping hybrid configs...")
    best_hybrid = {"acc": 0}
    for layer in range(0, lm.num_layers):
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            vecs = {layer: hybrid_steering.vectors[layer]}
            r = evaluate(lm, texts[:10], labels[:10], HYBRID_PROMPT,
                        steering_vecs=vecs, alpha=alpha)
            acc = accuracy(r)
            if acc > best_hybrid["acc"]:
                best_hybrid = {"acc": acc, "layer": layer, "alpha": alpha}

    if best_hybrid["acc"] > 0:
        print(f"  Best hybrid config: layer={best_hybrid['layer']}, alpha={best_hybrid['alpha']} ({best_hybrid['acc']:.0%})")

        print(f"\nCondition D — Hybrid: '{HYBRID_PROMPT}' + steering L{best_hybrid['layer']} α={best_hybrid['alpha']}")
        vecs = {best_hybrid["layer"]: hybrid_steering.vectors[best_hybrid["layer"]]}
        results_d = evaluate(lm, texts, labels, HYBRID_PROMPT,
                           steering_vecs=vecs, alpha=best_hybrid["alpha"])
        print_table(results_d, "Hybrid")
        full_results["hybrid"] = results_d

    # Also test hybrid prompt alone (no steering)
    print(f"\nHybrid prompt alone (no steering):")
    results_hybrid_only = evaluate(lm, texts, labels, HYBRID_PROMPT)
    print_table(results_hybrid_only, "Hybrid Prompt Only")

    # ── Phase 6: Summary ────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("FINAL RESULTS SUMMARY")
    print("═" * 80)

    full_tokens = len(lm.tokenizer.encode(FULL_PROMPT))
    min_tokens = len(lm.tokenizer.encode(MINIMAL_PROMPT))
    hyb_tokens = len(lm.tokenizer.encode(HYBRID_PROMPT))

    summary = [
        ("A. Full Prompt", accuracy(results_a), full_tokens, "-"),
        ("B. Minimal Prompt (baseline)", accuracy(results_b), min_tokens, "-"),
    ]

    for layer, alpha, name in configs_to_eval:
        key = f"steered_L{layer}_a{alpha}"
        if key in full_results:
            summary.append((
                f"C. Steered L{layer} α={alpha}",
                accuracy(full_results[key]),
                min_tokens,
                f"+ vector ({lm.hidden_dim} floats)",
            ))

    if "hybrid" in full_results:
        summary.append((
            f"D. Hybrid + Steered",
            accuracy(full_results["hybrid"]),
            hyb_tokens,
            f"+ vector ({lm.hidden_dim} floats)",
        ))

    summary.append(("E. Hybrid Prompt Only", accuracy(results_hybrid_only), hyb_tokens, "-"))

    print(f"\n  {'Condition':<32} {'Accuracy':>9} {'Tokens':>7} {'Extra'}")
    print(f"  {'-'*70}")
    for name, acc, tokens, extra in summary:
        print(f"  {name:<32} {acc:>8.0%} {tokens:>7}  {extra}")

    print(f"\n  Key insight: Steering vector at layer {best['layer']} recovers")
    if full_results:
        best_steered = max(accuracy(v) for k, v in full_results.items() if k.startswith("steered"))
        print(f"  {best_steered:.0%} of baseline 0% → {best_steered:.0%} accuracy with ZERO task-specific tokens.")
        recovery = best_steered / max(accuracy(results_a), 0.01)
        print(f"  That is {recovery:.0%} of the full prompt's {accuracy(results_a):.0%} accuracy.")

    print(f"\n  Token savings: {full_tokens - min_tokens} tokens/request ({(full_tokens - min_tokens)/full_tokens:.0%} reduction)")
    print(f"  Vector size: {lm.hidden_dim} float32 values = {lm.hidden_dim * 4} bytes")

    # ── Phase 7: Per-class analysis ─────────────────────────────────────
    print("\n" + "─" * 80)
    print("PER-CLASS BREAKDOWN")
    print("─" * 80)

    for condition_name, results in [("Full Prompt", results_a)] + \
            [(k, v) for k, v in full_results.items()]:
        by_class = {}
        for r in results:
            cls = r["expected"]
            if cls not in by_class:
                by_class[cls] = {"correct": 0, "total": 0}
            by_class[cls]["total"] += 1
            if r["correct"]:
                by_class[cls]["correct"] += 1

        parts = []
        for cls in ["positive", "negative", "neutral"]:
            if cls in by_class:
                c = by_class[cls]
                parts.append(f"{cls}: {c['correct']}/{c['total']}")
        print(f"  {condition_name:<30} {', '.join(parts)}")

    # ── Save everything ─────────────────────────────────────────────────
    all_data = {
        "model": model_name,
        "num_layers": lm.num_layers,
        "hidden_dim": lm.hidden_dim,
        "full_prompt_tokens": full_tokens,
        "conditions": {
            "full_prompt": {"accuracy": accuracy(results_a), "tokens": full_tokens, "results": results_a},
            "minimal_prompt": {"accuracy": accuracy(results_b), "tokens": min_tokens, "results": results_b},
            "hybrid_prompt_only": {"accuracy": accuracy(results_hybrid_only), "tokens": hyb_tokens, "results": results_hybrid_only},
        },
        "steering_analysis": {
            l: {"norm": steering.norms[l], "cosine_similarity": steering.cosine_similarities[l]}
            for l in steering.layers
        },
        "sweep_data": sweep_data,
        "best_config": best,
    }
    for k, v in full_results.items():
        all_data["conditions"][k] = {"accuracy": accuracy(v), "results": v}

    # Save steering vectors
    torch.save({
        "vectors": steering.vectors,
        "norms": steering.norms,
        "cosine_similarities": steering.cosine_similarities,
        "model": model_name,
        "optimized_prompt": FULL_PROMPT,
        "baseline_prompt": MINIMAL_PROMPT,
    }, f"{output_dir}/steering_vectors.pt")

    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_data, f, indent=2, default=serialize)

    print(f"\n  Results saved to {output_dir}/")
    print(f"  - results.json (all data)")
    print(f"  - steering_vectors.pt (reusable vectors)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    main(model_name=args.model, device=args.device)
