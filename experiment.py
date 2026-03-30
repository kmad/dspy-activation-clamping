"""
End-to-end experiment: DSPy optimize → extract steering vectors → evaluate clamping.

Uses a simple sentiment classification task to demonstrate the concept.
Small model (Qwen2.5-0.5B-Instruct) for fast iteration on local hardware.
"""

import torch
import dspy
import json
import os
from pathlib import Path
from nnsight_lm import NNsightLM, NNsightDSPyLM
from steering import (
    extract_steering_vectors,
    extract_steering_vectors_from_message_pairs,
    evaluate_steering,
    find_best_layer,
    build_messages,
)


# ── Task: Sentiment Classification ──────────────────────────────────────────
# Simple enough to optimize quickly, clear enough to measure steering success.

TRAIN_DATA = [
    ("The movie was absolutely fantastic and I loved every minute", "positive"),
    ("Terrible waste of time, worst film I've seen this year", "negative"),
    ("An okay experience, nothing special but not bad either", "neutral"),
    ("Brilliantly directed with stunning performances throughout", "positive"),
    ("The food was disgusting and the service was rude", "negative"),
    ("A masterpiece of modern cinema, truly groundbreaking work", "positive"),
    ("Completely boring and predictable from start to finish", "negative"),
    ("It was fine, met my basic expectations", "neutral"),
    ("Absolutely delightful, exceeded all my expectations", "positive"),
    ("Horrible quality, would never recommend to anyone", "negative"),
    ("The product works as advertised, nothing more nothing less", "neutral"),
    ("Outstanding customer service, they went above and beyond", "positive"),
    ("Broken on arrival, total disappointment", "negative"),
    ("Average performance, does the job adequately", "neutral"),
    ("Pure joy from beginning to end, a triumph", "positive"),
]

TEST_DATA = [
    ("This restaurant has the best pasta I have ever tasted", "positive"),
    ("What a waste of money, the product broke immediately", "negative"),
    ("The hotel was clean and the staff was friendly and helpful", "positive"),
    ("Mediocre at best, I expected much more from this brand", "neutral"),
    ("I am thoroughly impressed with the quality and craftsmanship", "positive"),
    ("Awful experience, I will never come back again", "negative"),
    ("Standard quality, gets the job done without any flair", "neutral"),
    ("The concert was electrifying, best night of my life", "positive"),
    ("Soggy fries and cold burgers, extremely disappointing meal", "negative"),
    ("Not great not terrible, a perfectly average Tuesday", "neutral"),
]


class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of the given text as positive, negative, or neutral."""
    text: str = dspy.InputField(desc="The text to classify")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")


class SentimentModule(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(SentimentClassifier)

    def forward(self, text):
        return self.classify(text=text)


def make_dspy_examples(data):
    return [
        dspy.Example(text=text, sentiment=label).with_inputs("text")
        for text, label in data
    ]


def sentiment_metric(example, prediction, trace=None):
    pred = prediction.sentiment.strip().lower()
    gold = example.sentiment.strip().lower()
    return pred == gold


# ── Prompt Extraction ───────────────────────────────────────────────────────

def extract_dspy_prompt(program, sample_input: str) -> str:
    """
    Extract the full system prompt that DSPy generates for a compiled program.
    We do this by formatting the adapter manually.
    """
    predictor = program.classify
    signature = predictor.signature
    demos = predictor.demos

    # Use DSPy's ChatAdapter to format
    from dspy.adapters.chat_adapter import ChatAdapter
    adapter = ChatAdapter()

    inputs = {"text": sample_input}
    messages = adapter.format(signature, demos, inputs)

    # The system message is the "optimized prompt" we want to steer with
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    return system_msg, messages


# ── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    device: str = "mps",
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load model ──────────────────────────────────────────────
    print("\n═══ Step 1: Loading model ═══")
    lm = NNsightLM(model_name, device=device, max_new_tokens=32)
    dspy_lm = NNsightDSPyLM(lm)

    # ── Step 2: DSPy Optimization ───────────────────────────────────────
    print("\n═══ Step 2: DSPy Optimization ═══")
    dspy.configure(lm=dspy_lm)

    trainset = make_dspy_examples(TRAIN_DATA)
    testset = make_dspy_examples(TEST_DATA)

    # Baseline: unoptimized
    baseline_program = SentimentModule()
    print("Evaluating baseline (unoptimized)...")
    baseline_eval = dspy.Evaluate(devset=testset, metric=sentiment_metric, num_threads=1)
    baseline_result = baseline_eval(baseline_program)
    baseline_score = float(baseline_result.score) if hasattr(baseline_result, 'score') else float(baseline_result)
    print(f"Baseline accuracy: {baseline_score:.1f}%")

    # Optimize with BootstrapFewShot
    print("Optimizing with BootstrapFewShot...")
    optimizer = dspy.BootstrapFewShot(
        metric=sentiment_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )
    optimized_program = optimizer.compile(baseline_program, trainset=trainset)

    print("Evaluating optimized program...")
    opt_result = baseline_eval(optimized_program)
    opt_score = float(opt_result.score) if hasattr(opt_result, 'score') else float(opt_result)
    print(f"Optimized accuracy: {opt_score:.1f}%")

    # ── Step 3: Extract the prompts ─────────────────────────────────────
    print("\n═══ Step 3: Extracting prompts ═══")
    sample_input = TEST_DATA[0][0]

    # Get the optimized system prompt
    opt_system_prompt, opt_messages = extract_dspy_prompt(optimized_program, sample_input)
    print(f"Optimized system prompt length: {len(opt_system_prompt)} chars")

    # Baseline system prompt (minimal)
    base_system_prompt = "Classify the sentiment as positive, negative, or neutral."

    # Also get the full message format for optimized
    # The "user" messages contain the DSPy field formatting
    opt_user_template = next(
        (m["content"] for m in opt_messages if m["role"] == "user"), ""
    )
    print(f"Optimized user message template length: {len(opt_user_template)} chars")

    # Save prompts
    with open(f"{output_dir}/prompts.json", "w") as f:
        json.dump({
            "optimized_system": opt_system_prompt,
            "baseline_system": base_system_prompt,
            "optimized_user_template": opt_user_template,
            "optimized_messages": opt_messages,
        }, f, indent=2)

    # ── Step 4: Extract Steering Vectors ────────────────────────────────
    print("\n═══ Step 4: Extracting Steering Vectors ═══")

    # Use training inputs for extraction
    extraction_inputs = [text for text, _ in TRAIN_DATA[:8]]

    # Build the full DSPy-formatted messages for each input
    # This captures the FULL prompt difference (system + user formatting + demos)
    print("Recording activations with optimized prompt...")
    opt_activations_inputs = []
    base_activations_inputs = []
    for text in extraction_inputs:
        _, full_msgs = extract_dspy_prompt(optimized_program, text)
        opt_activations_inputs.append(full_msgs)

        # Baseline: simple format
        base_activations_inputs.append([
            {"role": "system", "content": base_system_prompt},
            {"role": "user", "content": f"Text: {text}\nSentiment:"},
        ])

    # Extract at all layers for analysis
    all_layers = list(range(lm.num_layers))

    steering_result = extract_steering_vectors_from_message_pairs(
        lm,
        optimized_messages=opt_activations_inputs,
        baseline_messages=base_activations_inputs,
        layers=all_layers,
        token_aggregation="last",
    )

    # Report on steering vectors
    print("\nSteering vector analysis:")
    print(f"{'Layer':>6} {'Norm':>10} {'Cosine Sim':>12}")
    print("-" * 32)
    for layer_idx in steering_result.layers:
        print(
            f"{layer_idx:>6} "
            f"{steering_result.norms[layer_idx]:>10.4f} "
            f"{steering_result.cosine_similarities[layer_idx]:>12.4f}"
        )

    best_layer = find_best_layer(steering_result)
    print(f"\nBest layer for steering: {best_layer}")
    print(f"  Norm: {steering_result.norms[best_layer]:.4f}")
    print(f"  Cosine sim: {steering_result.cosine_similarities[best_layer]:.4f}")

    # Save steering vectors
    torch.save({
        "vectors": steering_result.vectors,
        "norms": steering_result.norms,
        "cosine_similarities": steering_result.cosine_similarities,
        "layers": steering_result.layers,
        "best_layer": best_layer,
        "model_name": model_name,
    }, f"{output_dir}/steering_vectors.pt")

    # ── Step 5: Evaluate Steering ───────────────────────────────────────
    print("\n═══ Step 5: Evaluating Steering Vectors ═══")

    test_inputs = [text for text, _ in TEST_DATA]
    test_labels = [label for _, label in TEST_DATA]

    def score_fn(pred, expected):
        # Extract just the sentiment word from potentially verbose output
        pred_lower = pred.strip().lower()
        for label in ["positive", "negative", "neutral"]:
            if label in pred_lower:
                return 1.0 if label == expected.lower() else 0.0
        return 0.0

    # Test with just the best layer
    print(f"\nTesting with best layer ({best_layer}) at various alpha values...")
    eval_results = evaluate_steering(
        lm,
        test_inputs=test_inputs,
        expected_outputs=test_labels,
        steering_result=steering_result,
        optimized_system_prompt=opt_system_prompt,
        baseline_system_prompt=base_system_prompt,
        alpha_values=[0.5, 1.0, 1.5, 2.0, 3.0],
        target_layers=[best_layer],
        score_fn=score_fn,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Condition':<25} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 50)
    for condition, stats in eval_results["summary"].items():
        print(
            f"{condition:<25} "
            f"{stats['mean_score']:>9.1%} "
            f"{stats['num_correct']:>5}/{stats['total']}"
        )

    # Token savings analysis
    opt_tokens = len(lm.tokenizer.encode(opt_system_prompt))
    base_tokens = len(lm.tokenizer.encode(base_system_prompt))
    saved_tokens = opt_tokens - base_tokens
    print(f"\nToken analysis:")
    print(f"  Optimized prompt: {opt_tokens} tokens")
    print(f"  Baseline prompt:  {base_tokens} tokens")
    print(f"  Tokens saved by steering: {saved_tokens} ({saved_tokens/opt_tokens:.0%} reduction)")

    # Save full results
    # Convert tensors to lists for JSON serialization
    serializable_results = {
        "summary": eval_results["summary"],
        "details": {
            k: {
                "outputs": v["outputs"],
                "scores": v["scores"],
            }
            for k, v in eval_results["details"].items()
        },
        "token_analysis": {
            "optimized_tokens": opt_tokens,
            "baseline_tokens": base_tokens,
            "saved_tokens": saved_tokens,
        },
        "model": model_name,
        "best_layer": best_layer,
    }
    with open(f"{output_dir}/eval_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    # ── Step 6: Multi-layer steering test ───────────────────────────────
    print("\n═══ Step 6: Multi-layer steering ═══")

    # Test with top-3 most consistent layers
    sorted_layers = sorted(
        steering_result.layers,
        key=lambda l: steering_result.cosine_similarities[l],
        reverse=True,
    )
    top_layers = sorted_layers[:3]
    print(f"Testing with top-3 layers: {top_layers}")

    multi_results = evaluate_steering(
        lm,
        test_inputs=test_inputs,
        expected_outputs=test_labels,
        steering_result=steering_result,
        optimized_system_prompt=opt_system_prompt,
        baseline_system_prompt=base_system_prompt,
        alpha_values=[0.5, 1.0, 1.5],
        target_layers=top_layers,
        score_fn=score_fn,
    )

    print(f"\n{'Condition':<25} {'Accuracy':>10}")
    print("-" * 40)
    for condition, stats in multi_results["summary"].items():
        print(f"{condition:<25} {stats['mean_score']:>9.1%}")

    return eval_results, steering_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DSPy Activation Clamping Experiment")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
    )
