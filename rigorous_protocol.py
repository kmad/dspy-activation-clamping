"""
Utilities for running a defensible steering-vector experiment.

This module is intentionally pure-Python so its core logic can be tested without
loading models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re


STRICT_LABEL_RE = re.compile(r"^\s*([A-Za-z_ -]+?)\s*([.!?])?\s*$")


@dataclass(frozen=True)
class Example:
    text: str
    label: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    labels: tuple[str, ...]
    examples: tuple[Example, ...]
    full_prompt: str
    minimal_prompt: str
    hybrid_prompt: str


def strict_label_from_output(raw: str, labels: tuple[str, ...]) -> str | None:
    """
    Parse a generated response only if it is exactly one label, optionally with
    trailing punctuation. Anything else counts as format failure.
    """
    match = STRICT_LABEL_RE.match(raw)
    if not match:
        return None

    candidate = match.group(1).strip().lower()
    if candidate in labels:
        return candidate
    return None


def stratified_split(
    examples: tuple[Example, ...] | list[Example],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[Example], list[Example], list[Example]]:
    """
    Split examples into train/val/test sets while preserving label balance.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1")

    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    train, val, test = [], [], []
    for label_examples in by_label.values():
        items = list(label_examples)
        rng.shuffle(items)

        n = len(items)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            raise ValueError("Not enough examples per class for requested split")

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return mean, stderr


def accuracy_from_predictions(predictions: list[str | None], expected: list[str]) -> float:
    if not expected:
        return 0.0
    correct = sum(pred == gold for pred, gold in zip(predictions, expected))
    return correct / len(expected)
