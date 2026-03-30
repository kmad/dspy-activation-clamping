"""
Rigorous steering-vector experiment.

This script is designed to answer the core question defensibly:

Can a steering vector replace a task-specific system prompt, or does it only
recover part of the prompt's effect?

Protocol:
  1. Use stratified train/val/test splits.
  2. Extract steering vectors on train only.
  3. Tune layer/alpha on val only.
  4. Report prompt-only and steered baselines on held-out test.
  5. Measure both semantic accuracy (label log-prob scoring) and strict
     free-generation accuracy / format compliance.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch

from experiment_binary import (
    FULL_PROMPT as BINARY_FULL_PROMPT,
    HYBRID_PROMPT as BINARY_HYBRID_PROMPT,
    MINIMAL_PROMPT as SHARED_MINIMAL_PROMPT,
    TEST as BINARY_TEST,
    TRAIN as BINARY_TRAIN,
)
from experiment_final import (
    FULL_PROMPT as TERNARY_FULL_PROMPT,
    HYBRID_PROMPT as TERNARY_HYBRID_PROMPT,
    TEST as TERNARY_TEST,
    TRAIN as TERNARY_TRAIN,
)
from nnsight_lm import NNsightLM
from rigorous_protocol import (
    Example,
    TaskSpec,
    accuracy_from_predictions,
    mean_and_stderr,
    strict_label_from_output,
    stratified_split,
)
from steering import build_messages, extract_steering_vectors_from_message_pairs


EXTRA_TERNARY_EXAMPLES = (
    Example("The new cafe has a cheerful atmosphere and surprisingly great coffee", "positive"),
    Example("The software update fixed every issue I had and made the app faster", "positive"),
    Example("The replacement order arrived late and missing half the parts", "negative"),
    Example("The speaker sounds tinny and the battery drains absurdly fast", "negative"),
    Example("The claims in the article were overstated and the reporting felt sloppy", "negative"),
    Example("Support closed my ticket without solving the problem", "negative"),
    Example("The package includes the charger, manual, and carrying case", "neutral"),
    Example("The meeting starts at noon and should last about thirty minutes", "neutral"),
    Example("This version has the same feature set as last year's model", "neutral"),
    Example("The apartment is two blocks from the station and faces the courtyard", "neutral"),
    Example("The report summarizes sales by region and quarter", "neutral"),
    Example("The seat is firm, the fabric is gray, and assembly takes about ten minutes", "neutral"),
    Example("The recipe uses olive oil, garlic, and canned tomatoes", "neutral"),
)


def build_tasks() -> dict[str, TaskSpec]:
    binary_examples = tuple(
        Example(text=text, label=label)
        for text, label in (BINARY_TRAIN + BINARY_TEST)
    )

    ternary_pool = [
        Example(text=text, label=label)
        for text, label in (TERNARY_TRAIN + TERNARY_TEST)
    ] + list(EXTRA_TERNARY_EXAMPLES)

    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in ternary_pool:
        by_label[example.label].append(example)

    ternary_examples = tuple(
        by_label["positive"][:15]
        + by_label["negative"][:15]
        + by_label["neutral"][:15]
    )

    return {
        "binary": TaskSpec(
            name="binary_sentiment",
            labels=("positive", "negative"),
            examples=binary_examples,
            full_prompt=BINARY_FULL_PROMPT,
            minimal_prompt=SHARED_MINIMAL_PROMPT,
            hybrid_prompt=BINARY_HYBRID_PROMPT,
        ),
        "ternary": TaskSpec(
            name="ternary_sentiment",
            labels=("positive", "negative", "neutral"),
            examples=ternary_examples,
            full_prompt=TERNARY_FULL_PROMPT,
            minimal_prompt=SHARED_MINIMAL_PROMPT,
            hybrid_prompt=TERNARY_HYBRID_PROMPT,
        ),
    }


def continuation_variants(label: str) -> list[str]:
    # Keep this conservative to avoid tokenizer-specific evaluation tricks.
    return [label]


def choose_candidate_layers(
    norms: dict[int, float],
    cosine_similarities: dict[int, float],
    top_k: int,
) -> list[int]:
    sorted_norms = sorted(norms.values())
    lower = sorted_norms[max(0, len(sorted_norms) // 5 - 1)]
    upper = sorted_norms[min(len(sorted_norms) - 1, (4 * len(sorted_norms)) // 5)]

    preferred = [
        layer
        for layer, norm in norms.items()
        if lower <= norm <= upper
    ]
    if not preferred:
        preferred = list(norms.keys())

    preferred.sort(
        key=lambda layer: cosine_similarities[layer],
        reverse=True,
    )
    return preferred[:top_k]


def predict_by_logprob(
    lm: NNsightLM,
    system_prompt: str,
    text: str,
    labels: tuple[str, ...],
    steering_vectors: dict[int, torch.Tensor] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
) -> tuple[str, dict[str, float]]:
    continuations = []
    continuation_to_label = {}
    for label in labels:
        for variant in continuation_variants(label):
            continuations.append(variant)
            continuation_to_label[variant] = label

    scores = lm.score_continuations(
        build_messages(system_prompt, text),
        continuations=continuations,
        steering_vectors=steering_vectors,
        alpha=alpha,
        token_strategy=token_strategy,
    )

    per_label_scores = {}
    for label in labels:
        per_label_scores[label] = max(
            scores[variant] for variant in continuation_variants(label)
        )

    prediction = max(per_label_scores, key=per_label_scores.get)
    return prediction, per_label_scores


def evaluate_logprob_condition(
    lm: NNsightLM,
    examples: list[Example],
    system_prompt: str,
    labels: tuple[str, ...],
    steering_vectors: dict[int, torch.Tensor] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
) -> dict:
    predictions = []
    details = []

    for example in examples:
        predicted, scores = predict_by_logprob(
            lm=lm,
            system_prompt=system_prompt,
            text=example.text,
            labels=labels,
            steering_vectors=steering_vectors,
            alpha=alpha,
            token_strategy=token_strategy,
        )
        predictions.append(predicted)
        details.append(
            {
                "input": example.text,
                "expected": example.label,
                "predicted": predicted,
                "scores": scores,
                "correct": predicted == example.label,
            }
        )

    expected = [example.label for example in examples]
    return {
        "accuracy": accuracy_from_predictions(predictions, expected),
        "details": details,
    }


def evaluate_generation_condition(
    lm: NNsightLM,
    examples: list[Example],
    system_prompt: str,
    labels: tuple[str, ...],
    steering_vectors: dict[int, torch.Tensor] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
    max_new_tokens: int = 6,
) -> dict:
    parsed_predictions = []
    details = []

    for example in examples:
        messages = build_messages(system_prompt, example.text)
        if steering_vectors:
            raw = lm.generate_with_steering(
                messages,
                steering_vectors=steering_vectors,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                token_strategy=token_strategy,
            )
        else:
            raw = lm.generate_text(messages, max_new_tokens=max_new_tokens)

        parsed = strict_label_from_output(raw, labels)
        parsed_predictions.append(parsed)
        details.append(
            {
                "input": example.text,
                "expected": example.label,
                "predicted": parsed,
                "raw": raw,
                "format_ok": parsed is not None,
                "correct": parsed == example.label,
            }
        )

    expected = [example.label for example in examples]
    format_ok = sum(item["format_ok"] for item in details) / len(details)
    return {
        "accuracy": accuracy_from_predictions(parsed_predictions, expected),
        "format_compliance": format_ok,
        "details": details,
    }


def tune_steering_configuration(
    lm: NNsightLM,
    validation_examples: list[Example],
    labels: tuple[str, ...],
    system_prompt: str,
    steering_result,
    alphas: list[float],
    token_strategies: list[str],
    top_k_layers: int,
) -> dict:
    candidate_layers = choose_candidate_layers(
        steering_result.norms,
        steering_result.cosine_similarities,
        top_k=top_k_layers,
    )

    best = None
    leaderboard = []

    for token_strategy in token_strategies:
        for layer in candidate_layers:
            steering_vectors = {layer: steering_result.vectors[layer]}
            for alpha in alphas:
                evaluation = evaluate_logprob_condition(
                    lm=lm,
                    examples=validation_examples,
                    system_prompt=system_prompt,
                    labels=labels,
                    steering_vectors=steering_vectors,
                    alpha=alpha,
                    token_strategy=token_strategy,
                )
                score = evaluation["accuracy"]
                candidate = {
                    "layer": layer,
                    "alpha": alpha,
                    "token_strategy": token_strategy,
                    "validation_accuracy": score,
                    "cosine_similarity": steering_result.cosine_similarities[layer],
                    "norm": steering_result.norms[layer],
                }
                leaderboard.append(candidate)
                if best is None or (
                    score,
                    steering_result.cosine_similarities[layer],
                    -alpha,
                    token_strategy == "last",
                ) > (
                    best["validation_accuracy"],
                    best["cosine_similarity"],
                    -best["alpha"],
                    best["token_strategy"] == "last",
                ):
                    best = candidate

    leaderboard.sort(
        key=lambda item: (
            item["validation_accuracy"],
            item["cosine_similarity"],
            -item["alpha"],
            item["token_strategy"] == "last",
        ),
        reverse=True,
    )
    result = dict(best)
    result["leaderboard"] = [dict(item) for item in leaderboard[:10]]
    return result


def summarize_condition(
    logprob_result: dict,
    generation_result: dict,
    system_prompt_tokens: int,
    extra: str = "-",
) -> dict:
    return {
        "logprob_accuracy": logprob_result["accuracy"],
        "generation_accuracy": generation_result["accuracy"],
        "format_compliance": generation_result["format_compliance"],
        "prompt_tokens": system_prompt_tokens,
        "extra": extra,
        "logprob_details": logprob_result["details"],
        "generation_details": generation_result["details"],
    }


def evaluate_seed(
    lm: NNsightLM,
    task: TaskSpec,
    seed: int,
    train_frac: float,
    val_frac: float,
    alphas: list[float],
    token_strategies: list[str],
    top_k_layers: int,
) -> dict:
    train_examples, val_examples, test_examples = stratified_split(
        task.examples,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    train_full_messages = [build_messages(task.full_prompt, example.text) for example in train_examples]
    train_minimal_messages = [build_messages(task.minimal_prompt, example.text) for example in train_examples]
    train_hybrid_messages = [build_messages(task.hybrid_prompt, example.text) for example in train_examples]

    steering_full = extract_steering_vectors_from_message_pairs(
        lm,
        optimized_messages=train_full_messages,
        baseline_messages=train_minimal_messages,
        layers=list(range(lm.num_layers)),
        token_aggregation="last",
    )
    steering_format = extract_steering_vectors_from_message_pairs(
        lm,
        optimized_messages=train_full_messages,
        baseline_messages=train_hybrid_messages,
        layers=list(range(lm.num_layers)),
        token_aggregation="last",
    )

    steering_only_config = tune_steering_configuration(
        lm=lm,
        validation_examples=val_examples,
        labels=task.labels,
        system_prompt=task.minimal_prompt,
        steering_result=steering_full,
        alphas=alphas,
        token_strategies=token_strategies,
        top_k_layers=top_k_layers,
    )
    hybrid_config = tune_steering_configuration(
        lm=lm,
        validation_examples=val_examples,
        labels=task.labels,
        system_prompt=task.hybrid_prompt,
        steering_result=steering_format,
        alphas=alphas,
        token_strategies=token_strategies,
        top_k_layers=top_k_layers,
    )

    full_tokens = len(lm.tokenizer.encode(task.full_prompt))
    minimal_tokens = len(lm.tokenizer.encode(task.minimal_prompt))
    hybrid_tokens = len(lm.tokenizer.encode(task.hybrid_prompt))

    conditions = {}

    for name, system_prompt, prompt_tokens in [
        ("full_prompt", task.full_prompt, full_tokens),
        ("minimal_prompt", task.minimal_prompt, minimal_tokens),
        ("hybrid_prompt_only", task.hybrid_prompt, hybrid_tokens),
    ]:
        logprob_result = evaluate_logprob_condition(
            lm=lm,
            examples=test_examples,
            system_prompt=system_prompt,
            labels=task.labels,
        )
        generation_result = evaluate_generation_condition(
            lm=lm,
            examples=test_examples,
            system_prompt=system_prompt,
            labels=task.labels,
        )
        conditions[name] = summarize_condition(
            logprob_result,
            generation_result,
            system_prompt_tokens=prompt_tokens,
        )

    steering_vectors = {steering_only_config["layer"]: steering_full.vectors[steering_only_config["layer"]]}
    steering_logprob = evaluate_logprob_condition(
        lm=lm,
        examples=test_examples,
        system_prompt=task.minimal_prompt,
        labels=task.labels,
        steering_vectors=steering_vectors,
        alpha=steering_only_config["alpha"],
        token_strategy=steering_only_config["token_strategy"],
    )
    steering_generation = evaluate_generation_condition(
        lm=lm,
        examples=test_examples,
        system_prompt=task.minimal_prompt,
        labels=task.labels,
        steering_vectors=steering_vectors,
        alpha=steering_only_config["alpha"],
        token_strategy=steering_only_config["token_strategy"],
    )
    conditions["steering_only"] = summarize_condition(
        steering_logprob,
        steering_generation,
        system_prompt_tokens=minimal_tokens,
        extra=f"+ vector @L{steering_only_config['layer']} a={steering_only_config['alpha']} {steering_only_config['token_strategy']}",
    )

    hybrid_vectors = {hybrid_config["layer"]: steering_format.vectors[hybrid_config["layer"]]}
    hybrid_logprob = evaluate_logprob_condition(
        lm=lm,
        examples=test_examples,
        system_prompt=task.hybrid_prompt,
        labels=task.labels,
        steering_vectors=hybrid_vectors,
        alpha=hybrid_config["alpha"],
        token_strategy=hybrid_config["token_strategy"],
    )
    hybrid_generation = evaluate_generation_condition(
        lm=lm,
        examples=test_examples,
        system_prompt=task.hybrid_prompt,
        labels=task.labels,
        steering_vectors=hybrid_vectors,
        alpha=hybrid_config["alpha"],
        token_strategy=hybrid_config["token_strategy"],
    )
    conditions["hybrid_steered"] = summarize_condition(
        hybrid_logprob,
        hybrid_generation,
        system_prompt_tokens=hybrid_tokens,
        extra=f"+ vector @L{hybrid_config['layer']} a={hybrid_config['alpha']} {hybrid_config['token_strategy']}",
    )

    return {
        "seed": seed,
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "conditions": conditions,
        "configs": {
            "steering_only": steering_only_config,
            "hybrid_steered": hybrid_config,
        },
        "steering_analysis": {
            "full_vs_minimal": {
                str(layer): {
                    "norm": steering_full.norms[layer],
                    "cosine_similarity": steering_full.cosine_similarities[layer],
                }
                for layer in steering_full.layers
            },
            "full_vs_hybrid": {
                str(layer): {
                    "norm": steering_format.norms[layer],
                    "cosine_similarity": steering_format.cosine_similarities[layer],
                }
                for layer in steering_format.layers
            },
        },
    }


def aggregate_seed_runs(seed_runs: list[dict]) -> dict:
    aggregated = {}
    for condition_name in seed_runs[0]["conditions"]:
        aggregated[condition_name] = {}
        for metric in ["logprob_accuracy", "generation_accuracy", "format_compliance", "prompt_tokens"]:
            values = [run["conditions"][condition_name][metric] for run in seed_runs]
            mean, stderr = mean_and_stderr(values)
            aggregated[condition_name][metric] = {
                "mean": mean,
                "stderr": stderr,
                "values": values,
            }
    return aggregated


def derive_verdict(aggregate: dict) -> dict:
    full_prompt = aggregate["full_prompt"]["logprob_accuracy"]["mean"]
    steering_only = aggregate["steering_only"]["logprob_accuracy"]["mean"]
    hybrid_prompt = aggregate["hybrid_prompt_only"]["logprob_accuracy"]["mean"]
    steering_generation = aggregate["steering_only"]["generation_accuracy"]["mean"]

    if steering_only >= full_prompt - 0.05 and steering_only > hybrid_prompt and steering_generation >= full_prompt - 0.10:
        return {
            "label": "supported",
            "reason": "Steering recovered nearly all full-prompt accuracy on held-out test and outperformed the short-prompt baseline.",
        }
    if steering_only <= hybrid_prompt or steering_only < full_prompt - 0.15:
        return {
            "label": "not_supported",
            "reason": "Steering did not match the full prompt on held-out test and was not clearly better than the short-prompt baseline.",
        }
    return {
        "label": "mixed",
        "reason": "Steering improved over the minimal baseline but did not cleanly replace the full prompt.",
    }


def print_task_summary(task_name: str, aggregate: dict, verdict: dict):
    print("\n" + "=" * 88)
    print(f"RIGOROUS RESULTS — {task_name}")
    print("=" * 88)
    print(f"{'Condition':<20} {'LogProb':>9} {'Gen':>9} {'Format':>9} {'Tokens':>8}")
    print("-" * 88)
    ordered = [
        "full_prompt",
        "minimal_prompt",
        "hybrid_prompt_only",
        "steering_only",
        "hybrid_steered",
    ]
    for condition_name in ordered:
        metrics = aggregate[condition_name]
        print(
            f"{condition_name:<20} "
            f"{metrics['logprob_accuracy']['mean']:>8.1%} "
            f"{metrics['generation_accuracy']['mean']:>8.1%} "
            f"{metrics['format_compliance']['mean']:>8.1%} "
            f"{metrics['prompt_tokens']['mean']:>8.1f}"
        )
    print(f"\nVerdict: {verdict['label']} — {verdict['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Rigorous steering-vector experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", choices=["binary", "ternary", "all"], default="all")
    parser.add_argument("--output-dir", default="results_rigorous")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6])
    parser.add_argument("--token-strategies", nargs="+", choices=["last", "all"], default=["last"])
    parser.add_argument("--top-k-layers", type=int, default=6)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lm = NNsightLM(
        args.model,
        device=args.device,
        max_new_tokens=6,
        local_files_only=args.local_files_only,
    )

    tasks = build_tasks()
    selected_tasks = tasks.values() if args.task == "all" else [tasks[args.task]]
    all_results = {}

    for task in selected_tasks:
        print(f"\nRunning task={task.name} with seeds={args.seeds}")
        seed_runs = []
        for seed in args.seeds:
            print(f"\n  Seed {seed}")
            seed_runs.append(
                evaluate_seed(
                    lm=lm,
                    task=task,
                    seed=seed,
                    train_frac=args.train_frac,
                    val_frac=args.val_frac,
                    alphas=args.alphas,
                    token_strategies=args.token_strategies,
                    top_k_layers=args.top_k_layers,
                )
            )

        aggregate = aggregate_seed_runs(seed_runs)
        verdict = derive_verdict(aggregate)
        print_task_summary(task.name, aggregate, verdict)

        all_results[task.name] = {
            "task": asdict(task),
            "seeds": seed_runs,
            "aggregate": aggregate,
            "verdict": verdict,
        }

    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)

    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as handle:
        json.dump(all_results, handle, indent=2, default=serialize)

    print(f"\nSaved rigorous results to {output_path}")


if __name__ == "__main__":
    main()
