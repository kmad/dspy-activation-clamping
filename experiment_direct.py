"""
Direct steering experiment — bypasses DSPy optimization to test the core hypothesis:

    Can we extract the effect of a system prompt as a steering vector,
    then reproduce that effect by injecting the vector WITHOUT the prompt?

This uses a hand-crafted "good" system prompt vs a minimal baseline,
so we can clearly measure whether steering vectors capture prompt effects.
"""

import torch
import json
import os
from nnsight_lm import NNsightLM
from steering import (
    extract_steering_vectors,
    evaluate_steering,
    find_best_layer,
    build_messages,
)


# ── Task: Sentiment Classification with a strong system prompt ──────────

GOOD_SYSTEM_PROMPT = """You are a sentiment classifier. Given a text, respond with EXACTLY one word: positive, negative, or neutral.

Rules:
- "positive" = the text expresses happiness, satisfaction, praise, or enthusiasm
- "negative" = the text expresses anger, disappointment, criticism, or dissatisfaction
- "neutral" = the text is factual, balanced, or neither clearly positive nor negative

Respond with ONLY the sentiment label, nothing else."""

BASELINE_SYSTEM_PROMPT = "Respond to the user."

TRAIN_TEXTS = [
    "The movie was absolutely fantastic and I loved every minute",
    "Terrible waste of time, worst film I've seen this year",
    "An okay experience, nothing special but not bad either",
    "Brilliantly directed with stunning performances throughout",
    "The food was disgusting and the service was rude",
    "A masterpiece of modern cinema, truly groundbreaking work",
    "Completely boring and predictable from start to finish",
    "It was fine, met my basic expectations",
    "Absolutely delightful, exceeded all my expectations",
    "Horrible quality, would never recommend to anyone",
    "The product works as advertised, nothing more nothing less",
    "Outstanding customer service, they went above and beyond",
]

TEST_DATA = [
    ("This restaurant has the best pasta I have ever tasted", "positive"),
    ("What a waste of money, the product broke immediately", "negative"),
    ("The hotel was clean and the staff was friendly and helpful", "positive"),
    ("Mediocre at best, I expected much more from this brand", "negative"),
    ("I am thoroughly impressed with the quality and craftsmanship", "positive"),
    ("Awful experience, I will never come back again", "negative"),
    ("Standard quality, gets the job done without any flair", "neutral"),
    ("The concert was electrifying, best night of my life", "positive"),
    ("Soggy fries and cold burgers, extremely disappointing meal", "negative"),
    ("Not great not terrible, a perfectly average Tuesday", "neutral"),
    ("I would definitely buy this again, amazing value", "positive"),
    ("The worst customer service I have encountered", "negative"),
    ("It does what it says on the tin", "neutral"),
    ("My kids absolutely adore this toy", "positive"),
    ("Overpriced and underwhelming, save your money", "negative"),
]


def score_fn(pred, expected):
    pred_lower = pred.strip().lower()
    # Check for exact match first
    if pred_lower == expected.lower():
        return 1.0
    # Check if the expected label appears in the response
    for label in ["positive", "negative", "neutral"]:
        if pred_lower.startswith(label):
            return 1.0 if label == expected.lower() else 0.0
    # Fuzzy: label appears anywhere
    for label in ["positive", "negative", "neutral"]:
        if label in pred_lower:
            return 1.0 if label == expected.lower() else 0.0
    return 0.0


def run_experiment(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    device: str = "mps",
    output_dir: str = "results_direct",
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("\n═══ Step 1: Loading model ═══")
    lm = NNsightLM(model_name, device=device, max_new_tokens=32)

    # ── Step 2: Verify the system prompt actually helps ─────────────────
    print("\n═══ Step 2: Verifying system prompt effect ═══")
    test_inputs = [t for t, _ in TEST_DATA]
    test_labels = [l for _, l in TEST_DATA]

    print("\nBaseline (minimal prompt):")
    base_scores = []
    for text, label in TEST_DATA[:5]:
        msgs = build_messages(BASELINE_SYSTEM_PROMPT, f"Text: {text}\nSentiment:")
        out = lm.generate_text(msgs)
        s = score_fn(out, label)
        base_scores.append(s)
        print(f"  [{label:>8}] score={s:.0f} | {out[:60]}")

    print("\nOptimized (strong prompt):")
    opt_scores = []
    for text, label in TEST_DATA[:5]:
        msgs = build_messages(GOOD_SYSTEM_PROMPT, text)
        out = lm.generate_text(msgs)
        s = score_fn(out, label)
        opt_scores.append(s)
        print(f"  [{label:>8}] score={s:.0f} | {out[:60]}")

    print(f"\nBaseline: {sum(base_scores)/len(base_scores):.0%}")
    print(f"Optimized: {sum(opt_scores)/len(opt_scores):.0%}")

    # ── Step 3: Extract Steering Vectors ────────────────────────────────
    print("\n═══ Step 3: Extracting Steering Vectors ═══")
    all_layers = list(range(lm.num_layers))

    # Use "mean" aggregation for more robust vectors
    for agg in ["last", "mean"]:
        print(f"\n--- Token aggregation: {agg} ---")
        steering = extract_steering_vectors(
            lm,
            inputs=TRAIN_TEXTS,
            optimized_system_prompt=GOOD_SYSTEM_PROMPT,
            baseline_system_prompt=BASELINE_SYSTEM_PROMPT,
            layers=all_layers,
            token_aggregation=agg,
        )

        print(f"\n{'Layer':>6} {'Norm':>10} {'CosSim':>8}")
        print("-" * 28)
        for l in steering.layers:
            print(f"{l:>6} {steering.norms[l]:>10.2f} {steering.cosine_similarities[l]:>8.4f}")

        best = find_best_layer(steering)
        print(f"\nBest layer: {best} (norm={steering.norms[best]:.2f}, cos_sim={steering.cosine_similarities[best]:.4f})")

        # Save
        torch.save({
            "vectors": steering.vectors,
            "norms": steering.norms,
            "cosine_similarities": steering.cosine_similarities,
            "best_layer": best,
            "aggregation": agg,
        }, f"{output_dir}/steering_{agg}.pt")

    # ── Step 4: Evaluate Steering ───────────────────────────────────────
    print("\n═══ Step 4: Evaluating Steering ═══")

    for agg in ["last", "mean"]:
        data = torch.load(f"{output_dir}/steering_{agg}.pt", weights_only=False)
        steering = extract_steering_vectors(
            lm,
            inputs=TRAIN_TEXTS[:4],  # smaller set for speed
            optimized_system_prompt=GOOD_SYSTEM_PROMPT,
            baseline_system_prompt=BASELINE_SYSTEM_PROMPT,
            layers=all_layers,
            token_aggregation=agg,
        )
        best = data["best_layer"]

        # Test single-layer steering
        print(f"\n--- Evaluating {agg} aggregation, layer {best} ---")
        results = evaluate_steering(
            lm,
            test_inputs=test_inputs,
            expected_outputs=test_labels,
            steering_result=steering,
            optimized_system_prompt=GOOD_SYSTEM_PROMPT,
            baseline_system_prompt=BASELINE_SYSTEM_PROMPT,
            alpha_values=[0.1, 0.3, 0.5, 0.7, 1.0],
            target_layers=[best],
            score_fn=score_fn,
        )

        print(f"\n{'Condition':<25} {'Accuracy':>10}")
        print("-" * 40)
        for cond, stats in results["summary"].items():
            print(f"{cond:<25} {stats['mean_score']:>9.1%}")

        # Test multi-layer (top 3 by cosine sim, early-to-mid range)
        mid_layers = [l for l in steering.layers if l < lm.num_layers * 0.6]
        top3 = sorted(mid_layers, key=lambda l: steering.cosine_similarities[l], reverse=True)[:3]
        print(f"\n--- Multi-layer steering: layers {top3} ---")

        multi_results = evaluate_steering(
            lm,
            test_inputs=test_inputs,
            expected_outputs=test_labels,
            steering_result=steering,
            optimized_system_prompt=GOOD_SYSTEM_PROMPT,
            baseline_system_prompt=BASELINE_SYSTEM_PROMPT,
            alpha_values=[0.05, 0.1, 0.2, 0.3],
            target_layers=top3,
            score_fn=score_fn,
        )

        print(f"\n{'Condition':<25} {'Accuracy':>10}")
        print("-" * 40)
        for cond, stats in multi_results["summary"].items():
            print(f"{cond:<25} {stats['mean_score']:>9.1%}")

    # ── Step 5: Token savings analysis ──────────────────────────────────
    print("\n═══ Step 5: Token Savings Analysis ═══")
    opt_tokens = len(lm.tokenizer.encode(GOOD_SYSTEM_PROMPT))
    base_tokens = len(lm.tokenizer.encode(BASELINE_SYSTEM_PROMPT))
    print(f"Optimized prompt: {opt_tokens} tokens")
    print(f"Baseline prompt:  {base_tokens} tokens")
    print(f"Tokens saved per request: {opt_tokens - base_tokens}")
    print(f"Reduction: {(opt_tokens - base_tokens) / opt_tokens:.0%}")

    print("\n═══ Done ═══")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="results_direct")
    args = parser.parse_args()
    run_experiment(model_name=args.model, device=args.device, output_dir=args.output)
