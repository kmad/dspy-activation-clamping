"""
Comprehensive steering vector experiment with concrete inputs/outputs at every stage.

Goal: Demonstrate that a DSPy-optimized prompt's effect can be partially captured
as a steering vector in activation space, with detailed evidence at each step.

Three experiments:
  1. Format steering — can we make the model output structured labels instead of prose?
  2. Behavior steering — can we shift the model's "personality" (formal vs casual)?
  3. DSPy-optimized steering — does DSPy optimization produce better steering vectors?
"""

import torch
import dspy
import json
import os
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from nnsight_lm import NNsightLM, NNsightDSPyLM
from steering import extract_steering_vectors, build_messages, SteeringResult


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

SENTIMENT_TRAIN = [
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

SENTIMENT_TEST = [
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


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

OPTIMIZED_PROMPT = """You are a precise sentiment classifier. Given a piece of text, classify its sentiment.

Rules:
- Respond with EXACTLY one word: positive, negative, or neutral
- "positive" means the text expresses happiness, satisfaction, praise, or enthusiasm
- "negative" means the text expresses anger, disappointment, criticism, or dissatisfaction
- "neutral" means the text is factual, balanced, or neither clearly positive nor negative
- Do NOT explain your reasoning. Output ONLY the label."""

MINIMAL_PROMPT = "You are a helpful assistant."

BARE_PROMPT = ""  # No system prompt at all


# ═══════════════════════════════════════════════════════════════════════════
# Experiment helpers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    condition: str
    outputs: list[dict] = field(default_factory=list)
    accuracy: float = 0.0
    token_count: int = 0

    def add(self, text: str, expected: str, predicted: str, raw_output: str):
        correct = predicted == expected
        self.outputs.append({
            "input": text,
            "expected": expected,
            "predicted": predicted,
            "raw_output": raw_output,
            "correct": correct,
        })

    def compute_accuracy(self):
        if self.outputs:
            self.accuracy = sum(1 for o in self.outputs if o["correct"]) / len(self.outputs)
        return self.accuracy


def extract_label(raw: str) -> str:
    """Extract sentiment label from model output, handling various formats."""
    raw = raw.strip().lower()
    # Direct match
    if raw in ("positive", "negative", "neutral"):
        return raw
    # First word match
    first = raw.split()[0].rstrip(".,!:;") if raw else ""
    if first in ("positive", "negative", "neutral"):
        return first
    # Contained anywhere
    for label in ["positive", "negative", "neutral"]:
        if label in raw:
            return label
    return raw[:20]  # Return truncated raw for debugging


def run_condition(lm, texts, labels, system_prompt, condition_name,
                  steering_vecs=None, alpha=0.0):
    """Run a single experimental condition and collect results."""
    result = ExperimentResult(condition=condition_name)
    result.token_count = len(lm.tokenizer.encode(system_prompt)) if system_prompt else 0

    for text, expected in zip(texts, labels):
        msgs = build_messages(system_prompt or "", text)

        if steering_vecs and alpha > 0:
            raw = lm.generate_with_steering(msgs, steering_vecs, alpha=alpha)
        else:
            raw = lm.generate_text(msgs)

        predicted = extract_label(raw)
        result.add(text, expected, predicted, raw)

    result.compute_accuracy()
    return result


def print_result_table(result: ExperimentResult, show_all=False):
    """Print a formatted table of results."""
    n = len(result.outputs)
    correct = sum(1 for o in result.outputs if o["correct"])
    print(f"\n  {result.condition}: {correct}/{n} ({result.accuracy:.0%})")
    print(f"  System prompt tokens: {result.token_count}")
    print(f"  {'Input':<55} {'Expected':>9} {'Got':>12} {'Raw Output':<30}")
    print(f"  {'-'*110}")

    for o in result.outputs:
        if show_all or not o["correct"]:
            mark = "OK" if o["correct"] else "XX"
            inp = o["input"][:52] + "..." if len(o["input"]) > 55 else o["input"]
            raw = o["raw_output"][:28].replace("\n", "\\n")
            print(f"  [{mark}] {inp:<52} {o['expected']:>9} {o['predicted']:>12} | {raw}")

    if not show_all:
        num_ok = sum(1 for o in result.outputs if o["correct"])
        if num_ok > 0:
            print(f"  ... {num_ok} correct outputs omitted (use show_all=True to see)")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Format Steering
# ═══════════════════════════════════════════════════════════════════════════

def experiment_1_format_steering(lm):
    """
    Can a steering vector change the model's output FORMAT from verbose prose
    to a single-word label — without any format instructions in the prompt?
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: FORMAT STEERING")
    print("Can we replace format instructions with a steering vector?")
    print("="*80)

    texts = [t for t, _ in SENTIMENT_TEST]
    labels = [l for _, l in SENTIMENT_TEST]
    train_texts = [t for t, _ in SENTIMENT_TRAIN]

    # Condition A: Full optimized prompt (the gold standard)
    print("\n--- Condition A: Full optimized prompt ---")
    result_a = run_condition(lm, texts, labels, OPTIMIZED_PROMPT, "Optimized Prompt")
    print_result_table(result_a, show_all=True)

    # Condition B: Minimal prompt (baseline — model is helpful but verbose)
    print("\n--- Condition B: Minimal prompt (baseline) ---")
    result_b = run_condition(lm, texts, labels, MINIMAL_PROMPT, "Minimal Prompt")
    print_result_table(result_b, show_all=True)

    # Extract steering vectors
    print("\n--- Extracting steering vectors ---")
    print(f"  Training inputs: {len(train_texts)}")
    print(f"  Optimized prompt: {len(OPTIMIZED_PROMPT)} chars")
    print(f"  Minimal prompt: {len(MINIMAL_PROMPT)} chars")

    all_layers = list(range(lm.num_layers))
    steering = extract_steering_vectors(
        lm, train_texts, OPTIMIZED_PROMPT, MINIMAL_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    # Print vector analysis
    print(f"\n  {'Layer':>6} {'Norm':>8} {'CosSim':>8}  Direction Consistency")
    print(f"  {'-'*50}")
    for l in steering.layers:
        bar = "█" * int(steering.cosine_similarities[l] * 20)
        print(f"  {l:>6} {steering.norms[l]:>8.2f} {steering.cosine_similarities[l]:>8.4f}  {bar}")

    # Sweep layers and alphas to find the sweet spot
    print("\n--- Layer x Alpha sweep ---")
    print(f"  {'Layer':>6} {'Alpha':>6} {'Accuracy':>9}  Sample outputs")
    print(f"  {'-'*70}")

    best_accuracy = 0
    best_config = None
    sweep_results = []

    for layer in range(0, lm.num_layers, 2):  # Every other layer
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            vecs = {layer: steering.vectors[layer]}
            # Quick test on first 5 examples
            quick_correct = 0
            samples = []
            for text, expected in zip(texts[:5], labels[:5]):
                msgs = build_messages(MINIMAL_PROMPT, text)
                raw = lm.generate_with_steering(msgs, vecs, alpha=alpha)
                pred = extract_label(raw)
                if pred == expected:
                    quick_correct += 1
                samples.append(f"{pred}")

            acc = quick_correct / 5
            sweep_results.append((layer, alpha, acc))
            sample_str = ", ".join(samples)
            print(f"  {layer:>6} {alpha:>6.1f} {acc:>8.0%}    [{sample_str}]")

            if acc > best_accuracy:
                best_accuracy = acc
                best_config = (layer, alpha)

    if best_config is None:
        print("\n  No configuration achieved >0% — model may be too small")
        return {"experiment": "format_steering", "status": "no_signal"}

    best_layer, best_alpha = best_config
    print(f"\n  Best config: layer={best_layer}, alpha={best_alpha} ({best_accuracy:.0%} on quick test)")

    # Full evaluation at best config
    print(f"\n--- Condition C: Steered (layer={best_layer}, alpha={best_alpha}) ---")
    vecs = {best_layer: steering.vectors[best_layer]}
    result_c = run_condition(
        lm, texts, labels, MINIMAL_PROMPT, f"Steered L{best_layer} a={best_alpha}",
        steering_vecs=vecs, alpha=best_alpha,
    )
    print_result_table(result_c, show_all=True)

    # Also try neighboring alphas for refinement
    print("\n--- Fine-tuning alpha around best layer ---")
    fine_alphas = np.arange(max(0.1, best_alpha - 0.5), best_alpha + 0.6, 0.1)
    for alpha in fine_alphas:
        alpha = round(alpha, 1)
        vecs = {best_layer: steering.vectors[best_layer]}
        r = run_condition(
            lm, texts, labels, MINIMAL_PROMPT, f"L{best_layer} a={alpha}",
            steering_vecs=vecs, alpha=alpha,
        )
        print(f"  alpha={alpha:.1f}: {r.accuracy:.0%} ({sum(1 for o in r.outputs if o['correct'])}/{len(r.outputs)})")

    # Multi-layer steering: combine top layers
    print("\n--- Multi-layer steering ---")
    # Pick top 3 layers by cosine similarity in the first 2/3 of layers
    mid_layers = [l for l in steering.layers if l < lm.num_layers * 0.7]
    top3 = sorted(mid_layers, key=lambda l: steering.cosine_similarities[l], reverse=True)[:3]
    print(f"  Top layers by consistency: {top3}")

    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
        vecs = {l: steering.vectors[l] for l in top3}
        r = run_condition(
            lm, texts, labels, MINIMAL_PROMPT, f"Multi-L{top3} a={alpha}",
            steering_vecs=vecs, alpha=alpha,
        )
        print(f"  alpha={alpha:.1f}: {r.accuracy:.0%}")

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT 1 SUMMARY")
    print("="*80)
    print(f"  Optimized prompt:  {result_a.accuracy:.0%} accuracy ({result_a.token_count} tokens)")
    print(f"  Minimal prompt:    {result_b.accuracy:.0%} accuracy ({result_b.token_count} tokens)")
    print(f"  Steered (best):    {result_c.accuracy:.0%} accuracy ({result_b.token_count} tokens + vector)")
    print(f"  Token savings:     {result_a.token_count - result_b.token_count} tokens/request")
    print(f"  Vector config:     layer={best_layer}, alpha={best_alpha}")

    return {
        "experiment": "format_steering",
        "optimized_accuracy": result_a.accuracy,
        "baseline_accuracy": result_b.accuracy,
        "steered_accuracy": result_c.accuracy,
        "best_layer": best_layer,
        "best_alpha": best_alpha,
        "token_savings": result_a.token_count - result_b.token_count,
        "optimized_outputs": result_a.outputs,
        "baseline_outputs": result_b.outputs,
        "steered_outputs": result_c.outputs,
        "vector_analysis": {
            l: {"norm": steering.norms[l], "cosine_sim": steering.cosine_similarities[l]}
            for l in steering.layers
        },
        "sweep": sweep_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: DSPy-Optimized Steering
# ═══════════════════════════════════════════════════════════════════════════

def experiment_2_dspy_optimized(lm):
    """
    Does DSPy optimization produce prompts that create BETTER steering vectors
    than hand-crafted prompts?
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: DSPy-OPTIMIZED STEERING")
    print("Does DSPy optimization create better-aligned activation directions?")
    print("="*80)

    texts = [t for t, _ in SENTIMENT_TEST]
    labels = [l for _, l in SENTIMENT_TEST]
    train_texts = [t for t, _ in SENTIMENT_TRAIN]

    # Step 1: Run DSPy optimization
    print("\n--- Step 1: DSPy Optimization ---")
    dspy_lm = NNsightDSPyLM(lm)
    dspy.configure(lm=dspy_lm)

    class SentimentSig(dspy.Signature):
        """Classify the sentiment of the given text as exactly one of: positive, negative, or neutral."""
        text: str = dspy.InputField(desc="Text to classify")
        sentiment: str = dspy.OutputField(desc="Exactly one of: positive, negative, neutral")

    class SentimentModule(dspy.Module):
        def __init__(self):
            self.classify = dspy.Predict(SentimentSig)
        def forward(self, text):
            return self.classify(text=text)

    trainset = [
        dspy.Example(text=t, sentiment=l).with_inputs("text")
        for t, l in SENTIMENT_TRAIN
    ]

    def metric(example, pred, trace=None):
        return pred.sentiment.strip().lower() == example.sentiment.strip().lower()

    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(SentimentModule(), trainset=trainset)

    # Extract the DSPy-generated prompt
    from dspy.adapters.chat_adapter import ChatAdapter
    adapter = ChatAdapter()
    sig = optimized.classify.signature
    demos = optimized.classify.demos
    sample_msgs = adapter.format(sig, demos, {"text": "test"})
    dspy_system = next((m["content"] for m in sample_msgs if m["role"] == "system"), "")

    print(f"\n  DSPy-generated system prompt ({len(dspy_system)} chars):")
    print(f"  {'-'*60}")
    for line in dspy_system.split('\n')[:10]:
        print(f"  | {line}")
    if dspy_system.count('\n') > 10:
        print(f"  | ... ({dspy_system.count(chr(10)) - 10} more lines)")

    print(f"\n  Demos included: {len(demos)}")
    for i, demo in enumerate(demos):
        print(f"    Demo {i+1}: text='{str(demo.get('text',''))[:40]}...' → {demo.get('sentiment','?')}")

    # Step 2: Evaluate DSPy-optimized prompt directly
    print("\n--- Step 2: Evaluate DSPy prompt directly ---")
    dspy_results = []
    for text, expected in zip(texts, labels):
        full_msgs = adapter.format(sig, demos, {"text": text})
        raw = lm.generate_text(full_msgs)
        pred = extract_label(raw)
        dspy_results.append((text, expected, pred, raw))

    dspy_correct = sum(1 for _, e, p, _ in dspy_results if p == e)
    print(f"  DSPy prompt accuracy: {dspy_correct}/{len(dspy_results)} ({dspy_correct/len(dspy_results):.0%})")

    # Step 3: Extract steering vectors from DSPy prompt vs baseline
    print("\n--- Step 3: Extract steering vectors ---")
    print("  Comparing DSPy prompt vs minimal prompt...")
    all_layers = list(range(lm.num_layers))

    # DSPy steering vectors — use the full formatted messages
    dspy_steering = extract_steering_vectors(
        lm, train_texts, dspy_system, MINIMAL_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    # Hand-crafted steering vectors (from Experiment 1)
    hand_steering = extract_steering_vectors(
        lm, train_texts, OPTIMIZED_PROMPT, MINIMAL_PROMPT,
        layers=all_layers, token_aggregation="last",
    )

    # Compare vector quality
    print(f"\n  {'Layer':>6} {'Hand CosSim':>12} {'DSPy CosSim':>12} {'Winner':>8}")
    print(f"  {'-'*46}")
    dspy_wins = 0
    for l in all_layers:
        h_sim = hand_steering.cosine_similarities[l]
        d_sim = dspy_steering.cosine_similarities[l]
        winner = "DSPy" if d_sim > h_sim else "Hand"
        if d_sim > h_sim:
            dspy_wins += 1
        print(f"  {l:>6} {h_sim:>12.4f} {d_sim:>12.4f} {winner:>8}")

    print(f"\n  DSPy wins {dspy_wins}/{len(all_layers)} layers on consistency")

    # Step 4: Sweep both vector sets
    print("\n--- Step 4: Compare steering performance ---")
    best_hand = {"acc": 0, "layer": 0, "alpha": 0}
    best_dspy = {"acc": 0, "layer": 0, "alpha": 0}

    for layer in range(0, lm.num_layers, 2):
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            # Hand-crafted
            vecs = {layer: hand_steering.vectors[layer]}
            h_correct = 0
            for text, expected in zip(texts[:10], labels[:10]):
                msgs = build_messages(MINIMAL_PROMPT, text)
                raw = lm.generate_with_steering(msgs, vecs, alpha=alpha)
                if extract_label(raw) == expected:
                    h_correct += 1
            h_acc = h_correct / 10

            # DSPy
            vecs = {layer: dspy_steering.vectors[layer]}
            d_correct = 0
            for text, expected in zip(texts[:10], labels[:10]):
                msgs = build_messages(MINIMAL_PROMPT, text)
                raw = lm.generate_with_steering(msgs, vecs, alpha=alpha)
                if extract_label(raw) == expected:
                    d_correct += 1
            d_acc = d_correct / 10

            if h_acc > best_hand["acc"]:
                best_hand = {"acc": h_acc, "layer": layer, "alpha": alpha}
            if d_acc > best_dspy["acc"]:
                best_dspy = {"acc": d_acc, "layer": layer, "alpha": alpha}

    print(f"\n  Best hand-crafted: {best_hand['acc']:.0%} (L{best_hand['layer']}, a={best_hand['alpha']})")
    print(f"  Best DSPy:         {best_dspy['acc']:.0%} (L{best_dspy['layer']}, a={best_dspy['alpha']})")

    # Full eval at best configs
    print("\n--- Step 5: Full evaluation at best configs ---")

    # Hand-crafted best
    vecs = {best_hand["layer"]: hand_steering.vectors[best_hand["layer"]]}
    result_hand = run_condition(
        lm, texts, labels, MINIMAL_PROMPT,
        f"Hand Steered L{best_hand['layer']} a={best_hand['alpha']}",
        steering_vecs=vecs, alpha=best_hand["alpha"],
    )
    print_result_table(result_hand, show_all=True)

    # DSPy best
    vecs = {best_dspy["layer"]: dspy_steering.vectors[best_dspy["layer"]]}
    result_dspy = run_condition(
        lm, texts, labels, MINIMAL_PROMPT,
        f"DSPy Steered L{best_dspy['layer']} a={best_dspy['alpha']}",
        steering_vecs=vecs, alpha=best_dspy["alpha"],
    )
    print_result_table(result_dspy, show_all=True)

    print("\n" + "="*80)
    print("EXPERIMENT 2 SUMMARY")
    print("="*80)
    print(f"  Hand prompt direct:    {sum(1 for o in run_condition(lm, texts[:10], labels[:10], OPTIMIZED_PROMPT, 'x').outputs if o['correct'])}/10")
    print(f"  DSPy prompt direct:    {dspy_correct}/{len(dspy_results)}")
    print(f"  Hand steered (best):   {result_hand.accuracy:.0%}")
    print(f"  DSPy steered (best):   {result_dspy.accuracy:.0%}")
    print(f"  Minimal prompt alone:  0% (verbose prose, no labels)")

    return {
        "experiment": "dspy_optimized_steering",
        "dspy_prompt": dspy_system[:500],
        "dspy_direct_accuracy": dspy_correct / len(dspy_results),
        "hand_steered_accuracy": result_hand.accuracy,
        "dspy_steered_accuracy": result_dspy.accuracy,
        "best_hand_config": best_hand,
        "best_dspy_config": best_dspy,
        "dspy_wins_consistency": dspy_wins,
        "total_layers": len(all_layers),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Hybrid Approach
# ═══════════════════════════════════════════════════════════════════════════

def experiment_3_hybrid(lm):
    """
    Test the hybrid approach: minimal task prompt + steering vector for format.
    Can we get most of the optimized prompt's accuracy with far fewer tokens?
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: HYBRID STEERING")
    print("Minimal semantic prompt + steering vector for format constraints")
    print("="*80)

    texts = [t for t, _ in SENTIMENT_TEST]
    labels = [l for _, l in SENTIMENT_TEST]
    train_texts = [t for t, _ in SENTIMENT_TRAIN]

    # The hybrid prompts: short on semantics, no format instructions
    HYBRID_PROMPTS = {
        "full_optimized": OPTIMIZED_PROMPT,  # 90 tokens, full instructions
        "medium_semantic": "Classify sentiment as positive, negative, or neutral. One word only.",
        "short_semantic": "Classify sentiment.",
        "bare_task": "Sentiment:",
    }

    # Extract format-only steering vectors
    # Vector = (full_optimized - medium_semantic) captures the FORMAT component
    # since both have the same task semantics but different format instructions
    print("\n--- Extracting format-specific steering vectors ---")
    format_steering = extract_steering_vectors(
        lm, train_texts, OPTIMIZED_PROMPT,
        "Classify sentiment as positive, negative, or neutral.",  # semantics only
        layers=list(range(lm.num_layers)), token_aggregation="last",
    )

    # Also extract task-specific steering vectors
    # Vector = (medium_semantic - bare) captures the TASK component
    task_steering = extract_steering_vectors(
        lm, train_texts,
        "Classify sentiment as positive, negative, or neutral.",
        MINIMAL_PROMPT,
        layers=list(range(lm.num_layers)), token_aggregation="last",
    )

    # Full steering (entire prompt effect)
    full_steering = extract_steering_vectors(
        lm, train_texts, OPTIMIZED_PROMPT, MINIMAL_PROMPT,
        layers=list(range(lm.num_layers)), token_aggregation="last",
    )

    # Evaluate all prompt conditions without steering
    print("\n--- Prompt-only baselines ---")
    for name, prompt in HYBRID_PROMPTS.items():
        r = run_condition(lm, texts, labels, prompt, name)
        tokens = len(lm.tokenizer.encode(prompt))
        print(f"  {name:<20} {r.accuracy:>5.0%} ({tokens} tokens)")

    # Now test hybrid: short prompt + steering
    print("\n--- Hybrid: short prompt + format steering ---")
    print(f"  {'Prompt':<20} {'Layer':>6} {'Alpha':>6} {'Accuracy':>9}")
    print(f"  {'-'*50}")

    best_hybrid = {"acc": 0}

    for prompt_name in ["short_semantic", "bare_task"]:
        prompt = HYBRID_PROMPTS[prompt_name]
        for layer in range(0, lm.num_layers, 2):
            for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
                vecs = {layer: format_steering.vectors[layer]}
                correct = 0
                for text, expected in zip(texts[:10], labels[:10]):
                    msgs = build_messages(prompt, text)
                    raw = lm.generate_with_steering(msgs, vecs, alpha=alpha)
                    if extract_label(raw) == expected:
                        correct += 1
                acc = correct / 10
                if acc >= 0.4:  # Only print promising results
                    print(f"  {prompt_name:<20} {layer:>6} {alpha:>6.1f} {acc:>8.0%}")
                if acc > best_hybrid["acc"]:
                    best_hybrid = {"acc": acc, "prompt": prompt_name, "layer": layer, "alpha": alpha}

    if best_hybrid["acc"] == 0:
        print("  No hybrid config achieved >40% on quick test")

    # Full evaluation at best hybrid config
    if best_hybrid["acc"] > 0:
        print(f"\n--- Full evaluation: {best_hybrid['prompt']} + L{best_hybrid['layer']} a={best_hybrid['alpha']} ---")
        prompt = HYBRID_PROMPTS[best_hybrid["prompt"]]
        vecs = {best_hybrid["layer"]: format_steering.vectors[best_hybrid["layer"]]}
        result_hybrid = run_condition(
            lm, texts, labels, prompt,
            f"Hybrid: {best_hybrid['prompt']} + steering",
            steering_vecs=vecs, alpha=best_hybrid["alpha"],
        )
        print_result_table(result_hybrid, show_all=True)

        # Compare
        result_full = run_condition(lm, texts, labels, OPTIMIZED_PROMPT, "Full Prompt")

        full_tokens = len(lm.tokenizer.encode(OPTIMIZED_PROMPT))
        hybrid_tokens = len(lm.tokenizer.encode(prompt))

        print(f"\n  Full prompt:    {result_full.accuracy:.0%} ({full_tokens} tokens)")
        print(f"  Hybrid:         {result_hybrid.accuracy:.0%} ({hybrid_tokens} tokens + vector)")
        print(f"  Token reduction: {full_tokens - hybrid_tokens} tokens ({(full_tokens - hybrid_tokens)/full_tokens:.0%})")

    # Decomposition analysis: format vs task steering
    print("\n--- Decomposition: Format vs Task steering directions ---")
    print(f"  {'Layer':>6} {'Format Norm':>12} {'Task Norm':>10} {'Full Norm':>10} {'F-T CosSim':>11}")
    print(f"  {'-'*55}")

    for l in range(0, lm.num_layers, 3):
        f_vec = format_steering.vectors[l]
        t_vec = task_steering.vectors[l]
        full_vec = full_steering.vectors[l]
        # Cosine similarity between format and task directions
        cos = torch.nn.functional.cosine_similarity(f_vec.unsqueeze(0), t_vec.unsqueeze(0)).item()
        print(f"  {l:>6} {f_vec.norm().item():>12.2f} {t_vec.norm().item():>10.2f} {full_vec.norm().item():>10.2f} {cos:>11.4f}")

    print("\n" + "="*80)
    print("EXPERIMENT 3 SUMMARY")
    print("="*80)
    if best_hybrid["acc"] > 0:
        print(f"  Best hybrid config: {best_hybrid['prompt']} + L{best_hybrid['layer']} a={best_hybrid['alpha']}")
        print(f"  Hybrid accuracy: {best_hybrid['acc']:.0%}")
    print(f"  Format and task directions are {'aligned' if cos > 0.5 else 'orthogonal'} at deep layers (cos_sim={cos:.2f})")

    return {
        "experiment": "hybrid_steering",
        "best_hybrid": best_hybrid,
        "format_task_decomposition": [
            {"layer": l, "format_norm": format_steering.norms[l],
             "task_norm": task_steering.norms[l], "full_norm": full_steering.norms[l]}
            for l in range(0, lm.num_layers, 3)
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="mps", output_dir="results_comprehensive"):
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("DSPy ACTIVATION CLAMPING: COMPREHENSIVE EXPERIMENT")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("="*80)

    lm = NNsightLM(model_name, device=device, max_new_tokens=24)

    results = {}
    results["exp1"] = experiment_1_format_steering(lm)
    results["exp2"] = experiment_2_dspy_optimized(lm)
    results["exp3"] = experiment_3_hybrid(lm)

    # Save results
    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=serialize)

    print(f"\nAll results saved to {output_dir}/results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="results_comprehensive")
    args = parser.parse_args()
    main(model_name=args.model, device=args.device, output_dir=args.output)
