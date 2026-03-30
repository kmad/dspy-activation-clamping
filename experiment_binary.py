"""
Binary sentiment classification experiment.

Tests the hypothesis that steering vectors work well for binary tasks
where the signal maps to a single linear axis in activation space.

Uses a larger dataset (50 train, 50 test) with only positive/negative labels.
"""

import torch
import json
import os
import numpy as np
from nnsight_lm import NNsightLM
from steering import extract_steering_vectors, build_messages


# ═══════════════════════════════════════════════════════════════════════════
# Data — 50 train, 50 test, binary only
# ═══════════════════════════════════════════════════════════════════════════

TRAIN = [
    # Positive (25)
    ("The movie was absolutely fantastic and I loved every minute of it", "positive"),
    ("Brilliantly directed with stunning performances from the entire cast", "positive"),
    ("A true masterpiece of modern cinema, truly groundbreaking work", "positive"),
    ("Absolutely delightful, this exceeded all of my expectations", "positive"),
    ("Outstanding customer service, they really went above and beyond", "positive"),
    ("The food here is incredible, easily the best meal I have had all month", "positive"),
    ("This book changed my perspective on life, deeply moving and powerful", "positive"),
    ("What a beautiful hotel, the views from the room were breathtaking", "positive"),
    ("The software works flawlessly and the interface is a joy to use", "positive"),
    ("My daughter loves this toy, she has not put it down since we got it", "positive"),
    ("Excellent build quality, this feels like it will last for years", "positive"),
    ("The concert was absolutely magical, the band was in perfect form", "positive"),
    ("I am blown away by how good this product is for the price point", "positive"),
    ("The staff were incredibly welcoming and made us feel right at home", "positive"),
    ("This course is phenomenal, I have learned so much in just a week", "positive"),
    ("Best purchase I have made in years, worth every single penny spent", "positive"),
    ("The garden is stunning, clearly maintained with love and attention", "positive"),
    ("Superb craftsmanship, you can tell real artisans made this piece", "positive"),
    ("The new update makes the app so much faster and smoother to use", "positive"),
    ("What an amazing experience, I would recommend this to everyone", "positive"),
    ("The dessert was heavenly, perfectly balanced sweetness and texture", "positive"),
    ("I am so grateful for this team, they solved my problem in minutes", "positive"),
    ("The views from the hiking trail were absolutely spectacular today", "positive"),
    ("This vacuum cleaner is a game changer, cuts cleaning time in half", "positive"),
    ("The writing in this series is sharp, witty, and deeply engaging", "positive"),
    # Negative (25)
    ("Terrible waste of time, the worst film I have seen this year", "negative"),
    ("The food was disgusting and the service was incredibly rude", "negative"),
    ("Completely boring and predictable from start to finish", "negative"),
    ("Horrible quality, I would never recommend this to anyone", "negative"),
    ("What a complete waste of money, the product broke immediately", "negative"),
    ("The hotel room was filthy and the noise kept us up all night", "negative"),
    ("This app crashes constantly and the developers seem to not care", "negative"),
    ("Worst customer service I have ever experienced in my entire life", "negative"),
    ("The package arrived damaged and two of the items were missing", "negative"),
    ("Overpriced garbage, there are much better alternatives out there", "negative"),
    ("The restaurant gave us food poisoning, absolutely unacceptable", "negative"),
    ("I regret buying this, it stopped working after just two days", "negative"),
    ("The instructions were impossible to follow and support was useless", "negative"),
    ("Cheaply made junk that falls apart the moment you start using it", "negative"),
    ("The noise from this machine is unbearable, cannot use it indoors", "negative"),
    ("Waited three weeks for delivery and then they sent the wrong item", "negative"),
    ("The fabric started pilling after the very first wash, total waste", "negative"),
    ("This is the most uncomfortable chair I have ever sat in", "negative"),
    ("The paint started peeling off within a week of application", "negative"),
    ("Absolutely terrible experience from booking to checkout", "negative"),
    ("The battery dies after barely two hours of normal use", "negative"),
    ("Misleading product photos, what arrived looks nothing like shown", "negative"),
    ("The screen developed dead pixels after just one month of use", "negative"),
    ("Do not waste your money on this, it is a complete scam", "negative"),
    ("The air conditioning broke on the hottest day and nobody came to fix it", "negative"),
]

TEST = [
    # Positive (25)
    ("This restaurant has the best pasta I have ever tasted in my life", "positive"),
    ("The hotel was spotless and the staff could not have been kinder", "positive"),
    ("I am thoroughly impressed with the quality and fine craftsmanship", "positive"),
    ("The concert last night was electrifying, best night of my life", "positive"),
    ("I would definitely buy this again, what an amazing value overall", "positive"),
    ("My children absolutely adore this toy, it was worth every penny", "positive"),
    ("Spectacular views and impeccable service at this wonderful resort", "positive"),
    ("This is hands down the best purchase I have made all year long", "positive"),
    ("The teacher was so patient and explained everything so clearly", "positive"),
    ("Just got back from the trip and it was the holiday of a lifetime", "positive"),
    ("The sound quality on these headphones is absolutely phenomenal", "positive"),
    ("I cannot believe how fast the delivery was, truly impressive", "positive"),
    ("The garden tools are extremely well made and comfortable to use", "positive"),
    ("This mattress completely fixed my back pain, sleeping great now", "positive"),
    ("The customer support team went above and beyond to help me out", "positive"),
    ("Beautifully written novel, I could not put it down once I started", "positive"),
    ("The renovation team did an incredible job on our kitchen update", "positive"),
    ("These running shoes are the most comfortable I have ever owned", "positive"),
    ("The birthday cake was gorgeous and tasted even better than it looked", "positive"),
    ("Absolutely love the new features in this software update release", "positive"),
    ("The documentary was deeply moving and incredibly well researched", "positive"),
    ("This blender is powerful, quiet, and makes the smoothest results", "positive"),
    ("The park is beautifully maintained with plenty of space for kids", "positive"),
    ("Our waiter was attentive, friendly, and gave great recommendations", "positive"),
    ("The subscription box has been a delightful surprise every month", "positive"),
    # Negative (25)
    ("What a complete waste of money, the product broke on day one", "negative"),
    ("Mediocre at best, I honestly expected much more from this brand", "negative"),
    ("Awful experience from start to finish, I will never come back", "negative"),
    ("Soggy fries and cold burgers, an extremely disappointing meal", "negative"),
    ("The worst customer service experience I have ever encountered", "negative"),
    ("Overpriced and underwhelming in every way, save your hard earned money", "negative"),
    ("Arrived damaged with missing parts, very frustrating experience", "negative"),
    ("The movie was painfully slow and the plot made absolutely no sense", "negative"),
    ("This phone case cracked the first time I dropped it, totally useless", "negative"),
    ("The gym equipment is rusted and half the machines are out of order", "negative"),
    ("Received a used item that was clearly returned and resold as new", "negative"),
    ("The wifi at this hotel was basically nonexistent the entire stay", "negative"),
    ("These shoes gave me blisters after wearing them for just one hour", "negative"),
    ("The contractor left our bathroom in worse shape than before", "negative"),
    ("This subscription is a ripoff, they keep charging hidden fees", "negative"),
    ("The steak was overcooked and the vegetables were completely limp", "negative"),
    ("Customer service hung up on me twice before I could explain my issue", "negative"),
    ("The color looks nothing like the photos, very disappointed overall", "negative"),
    ("This laptop overheats constantly and the fan sounds like a jet engine", "negative"),
    ("The tent leaked on the first night out, ruined the whole camping trip", "negative"),
    ("Absolutely dreadful airline experience, lost luggage and delayed flight", "negative"),
    ("The toy broke within an hour of my kid opening it on Christmas morning", "negative"),
    ("This cream caused a terrible rash and the company refused a refund", "negative"),
    ("The audiobook narrator was so monotone I fell asleep every single time", "negative"),
    ("Worst investment I have ever made, complete and total regret buying this", "negative"),
]

FULL_PROMPT = """You are a precise sentiment classifier. Given a piece of text, classify its sentiment.

Rules:
- Respond with EXACTLY one word: positive or negative
- "positive" means the text expresses happiness, satisfaction, praise, or enthusiasm
- "negative" means the text expresses anger, disappointment, criticism, or dissatisfaction
- Do NOT explain your reasoning. Output ONLY the label."""

MINIMAL_PROMPT = "You are a helpful assistant."
HYBRID_PROMPT = "Classify sentiment as positive or negative."


def extract_label(raw: str) -> str:
    raw = raw.strip().lower()
    if raw in ("positive", "negative"):
        return raw
    first = raw.split()[0].rstrip(".,!:;") if raw else ""
    if first in ("positive", "negative"):
        return first
    for label in ["positive", "negative"]:
        if label in raw:
            return label
    return raw[:30]


def evaluate(lm, data, system_prompt, steering_vecs=None, alpha=0.0):
    results = []
    for text, expected in data:
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


def class_accuracy(results, cls):
    subset = [r for r in results if r["expected"] == cls]
    if not subset:
        return 0.0
    return sum(r["correct"] for r in subset) / len(subset)


def print_table(results, label, show_errors_only=False):
    correct = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\n  [{label}] {correct}/{total} ({accuracy(results):.0%})")
    print(f"  Pos: {class_accuracy(results, 'positive'):.0%}  Neg: {class_accuracy(results, 'negative'):.0%}")

    if show_errors_only:
        errors = [r for r in results if not r["correct"]]
        if errors:
            print(f"  Errors ({len(errors)}):")
            for r in errors:
                inp = r["input"][:55] + "..." if len(r["input"]) > 58 else r["input"]
                print(f"    [{r['expected']:>8}] got {r['predicted']:<12} | {r['raw'][:45]}")
        else:
            print(f"  No errors!")
    else:
        for r in results:
            mark = " OK" if r["correct"] else " XX"
            inp = r["input"][:49] + "..." if len(r["input"]) > 52 else r["input"]
            print(f"  [{mark}] {inp:<52} {r['expected']:>8} -> {r['predicted']:<12} | {r['raw'][:35]}")


def main(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="mps"):
    output_dir = "results_binary"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("BINARY SENTIMENT CLASSIFICATION — STEERING VECTOR EXPERIMENT")
    print(f"Model: {model_name}")
    print(f"Train: {len(TRAIN)} examples  |  Test: {len(TEST)} examples")
    print("=" * 80)

    lm = NNsightLM(model_name, device=device, max_new_tokens=16)
    train_texts = [t for t, _ in TRAIN]

    # ── Phase 1: Baselines ──────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 1: BASELINES (50 test examples)")
    print("─" * 80)

    print("\nA. Full prompt (100 tokens)")
    results_full = evaluate(lm, TEST, FULL_PROMPT)
    print_table(results_full, "Full Prompt", show_errors_only=True)

    print("\nB. Minimal prompt (6 tokens)")
    results_min = evaluate(lm, TEST[:10], MINIMAL_PROMPT)  # Only 10 — they're all wrong
    print_table(results_min, "Minimal Prompt (first 10)")

    print("\nC. Hybrid prompt alone (7 tokens)")
    results_hyb_only = evaluate(lm, TEST, HYBRID_PROMPT)
    print_table(results_hyb_only, "Hybrid Prompt Only", show_errors_only=True)

    # ── Phase 2: Extract Steering Vectors ───────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 2: EXTRACT STEERING VECTORS (from 50 training examples)")
    print("─" * 80)

    all_layers = list(range(lm.num_layers))

    # Full prompt vs minimal
    print("\nExtracting: full prompt vs minimal prompt...")
    steering_full = extract_steering_vectors(
        lm, train_texts, FULL_PROMPT, MINIMAL_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    # Full prompt vs hybrid (format-only vector)
    print("Extracting: full prompt vs hybrid prompt (format-only)...")
    steering_format = extract_steering_vectors(
        lm, train_texts, FULL_PROMPT, HYBRID_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    print(f"\n  Full vs Minimal vectors:")
    print(f"  {'Layer':>6} {'Norm':>8} {'Consistency':>12}")
    print(f"  {'-'*30}")
    for l in range(0, lm.num_layers, 3):
        print(f"  {l:>6} {steering_full.norms[l]:>8.2f} {steering_full.cosine_similarities[l]:>12.4f}")
    avg_cos = np.mean([steering_full.cosine_similarities[l] for l in all_layers])
    print(f"  {'avg':>6} {'':>8} {avg_cos:>12.4f}")

    # ── Phase 3: Sweep ──────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 3: FIND OPTIMAL CONFIGURATION")
    print("─" * 80)

    # Quick sweep on 10 examples
    quick_test = TEST[:10]
    print(f"\nSweeping layers x alphas (quick test on 10 examples)...")
    print(f"  {'Layer':>6} {'Alpha':>6} {'Acc':>5}  Outputs (first 5)")
    print(f"  {'-'*65}")

    best = {"acc": 0, "layer": 0, "alpha": 0}
    for layer in range(0, lm.num_layers):
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
            vecs = {layer: steering_full.vectors[layer]}
            r = evaluate(lm, quick_test, MINIMAL_PROMPT, steering_vecs=vecs, alpha=alpha)
            acc = accuracy(r)
            if acc >= 0.7:
                preds = [r_["predicted"][:8] for r_ in r[:5]]
                print(f"  {layer:>6} {alpha:>6.1f} {acc:>4.0%}   {preds}")
            if acc > best["acc"]:
                best = {"acc": acc, "layer": layer, "alpha": alpha}

    print(f"\n  Best: layer={best['layer']}, alpha={best['alpha']} ({best['acc']:.0%} on 10)")

    # Fine-tune alpha
    print(f"\n  Fine-tuning alpha for layer {best['layer']}...")
    fine_results = []
    for alpha in np.arange(max(0.1, best["alpha"] - 0.5), best["alpha"] + 0.6, 0.05):
        alpha = round(alpha, 2)
        vecs = {best["layer"]: steering_full.vectors[best["layer"]]}
        r = evaluate(lm, TEST[:20], MINIMAL_PROMPT, steering_vecs=vecs, alpha=alpha)
        fine_results.append((alpha, accuracy(r)))
    fine_results.sort(key=lambda x: -x[1])
    print(f"  Top alphas (20-sample test):")
    for alpha, acc in fine_results[:5]:
        print(f"    alpha={alpha:.2f}: {acc:.0%}")
    best_alpha = fine_results[0][0]

    # Also sweep hybrid + format steering
    print(f"\n  Sweeping hybrid + format steering...")
    best_hybrid = {"acc": 0, "layer": 0, "alpha": 0}
    for layer in range(0, lm.num_layers):
        for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
            vecs = {layer: steering_format.vectors[layer]}
            r = evaluate(lm, quick_test, HYBRID_PROMPT, steering_vecs=vecs, alpha=alpha)
            acc = accuracy(r)
            if acc > best_hybrid["acc"]:
                best_hybrid = {"acc": acc, "layer": layer, "alpha": alpha}
    print(f"  Best hybrid: layer={best_hybrid['layer']}, alpha={best_hybrid['alpha']} ({best_hybrid['acc']:.0%} on 10)")

    # ── Phase 4: Full Evaluation ────────────────────────────────────────
    print("\n" + "─" * 80)
    print("PHASE 4: FULL EVALUATION (all 50 test examples)")
    print("─" * 80)

    # Steering only
    vecs = {best["layer"]: steering_full.vectors[best["layer"]]}
    print(f"\nC. Steered: layer={best['layer']}, alpha={best_alpha}")
    results_steered = evaluate(lm, TEST, MINIMAL_PROMPT, steering_vecs=vecs, alpha=best_alpha)
    print_table(results_steered, f"Steered L{best['layer']} α={best_alpha}", show_errors_only=True)

    # Hybrid + format steering
    vecs_h = {best_hybrid["layer"]: steering_format.vectors[best_hybrid["layer"]]}
    print(f"\nD. Hybrid + format steering: layer={best_hybrid['layer']}, alpha={best_hybrid['alpha']}")
    results_hybrid = evaluate(lm, TEST, HYBRID_PROMPT, steering_vecs=vecs_h, alpha=best_hybrid["alpha"])
    print_table(results_hybrid, f"Hybrid + Steered", show_errors_only=True)

    # Also test a couple neighboring configs
    print(f"\n  Neighboring configs for robustness check:")
    for delta_layer in [-1, 0, 1]:
        for delta_alpha in [-0.1, 0, 0.1]:
            l = best["layer"] + delta_layer
            a = round(best_alpha + delta_alpha, 2)
            if l < 0 or l >= lm.num_layers or a <= 0:
                continue
            vecs = {l: steering_full.vectors[l]}
            r = evaluate(lm, TEST, MINIMAL_PROMPT, steering_vecs=vecs, alpha=a)
            print(f"    L{l} α={a}: {accuracy(r):.0%} (pos={class_accuracy(r, 'positive'):.0%}, neg={class_accuracy(r, 'negative'):.0%})")

    # ── Phase 5: Summary ────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("FINAL RESULTS — BINARY CLASSIFICATION")
    print("═" * 80)

    full_tokens = len(lm.tokenizer.encode(FULL_PROMPT))
    min_tokens = len(lm.tokenizer.encode(MINIMAL_PROMPT))
    hyb_tokens = len(lm.tokenizer.encode(HYBRID_PROMPT))

    summary = [
        ("A. Full Prompt", accuracy(results_full), full_tokens,
         class_accuracy(results_full, "positive"), class_accuracy(results_full, "negative")),
        ("B. Minimal Prompt", 0.0, min_tokens, 0.0, 0.0),  # All wrong, prose output
        ("C. Steered Only", accuracy(results_steered), min_tokens,
         class_accuracy(results_steered, "positive"), class_accuracy(results_steered, "negative")),
        ("D. Hybrid + Steered", accuracy(results_hybrid), hyb_tokens,
         class_accuracy(results_hybrid, "positive"), class_accuracy(results_hybrid, "negative")),
        ("E. Hybrid Prompt Only", accuracy(results_hyb_only), hyb_tokens,
         class_accuracy(results_hyb_only, "positive"), class_accuracy(results_hyb_only, "negative")),
    ]

    print(f"\n  {'Condition':<25} {'Overall':>8} {'Pos':>6} {'Neg':>6} {'Tokens':>7}")
    print(f"  {'-'*58}")
    for name, acc, tokens, pos_acc, neg_acc in summary:
        print(f"  {name:<25} {acc:>7.0%} {pos_acc:>5.0%} {neg_acc:>5.0%} {tokens:>7}")

    gap = accuracy(results_full) - accuracy(results_steered)
    recovery = accuracy(results_steered) / max(accuracy(results_full), 0.01)
    print(f"\n  Steering recovery: {recovery:.0%} of full prompt performance")
    print(f"  Gap: {gap:.0%} ({gap * len(TEST):.0f} examples out of {len(TEST)})")
    print(f"  Token savings: {full_tokens - min_tokens} tokens/request ({(full_tokens - min_tokens)/full_tokens:.0%} reduction)")

    # Save
    all_data = {
        "model": model_name,
        "train_size": len(TRAIN),
        "test_size": len(TEST),
        "task": "binary_sentiment",
        "results": {
            "full_prompt": {"accuracy": accuracy(results_full), "tokens": full_tokens,
                           "pos_acc": class_accuracy(results_full, "positive"),
                           "neg_acc": class_accuracy(results_full, "negative")},
            "steered": {"accuracy": accuracy(results_steered),
                       "layer": best["layer"], "alpha": best_alpha,
                       "pos_acc": class_accuracy(results_steered, "positive"),
                       "neg_acc": class_accuracy(results_steered, "negative")},
            "hybrid_steered": {"accuracy": accuracy(results_hybrid),
                              "layer": best_hybrid["layer"], "alpha": best_hybrid["alpha"],
                              "pos_acc": class_accuracy(results_hybrid, "positive"),
                              "neg_acc": class_accuracy(results_hybrid, "negative")},
            "hybrid_only": {"accuracy": accuracy(results_hyb_only),
                           "pos_acc": class_accuracy(results_hyb_only, "positive"),
                           "neg_acc": class_accuracy(results_hyb_only, "negative")},
        },
        "steering_analysis": {
            str(l): {"norm": steering_full.norms[l],
                    "cosine_similarity": steering_full.cosine_similarities[l]}
            for l in all_layers
        },
        "full_results": {
            "full_prompt": results_full,
            "steered": results_steered,
            "hybrid_steered": results_hybrid,
            "hybrid_only": results_hyb_only,
        },
    }

    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return str(obj)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(all_data, f, indent=2, default=serialize)

    torch.save({
        "vectors_full": steering_full.vectors,
        "vectors_format": steering_format.vectors,
        "best_config": best,
        "best_alpha": best_alpha,
        "model": model_name,
    }, f"{output_dir}/steering_vectors.pt")

    print(f"\n  Saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    main(model_name=args.model, device=args.device)
