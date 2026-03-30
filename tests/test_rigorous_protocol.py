import unittest
import json

from rigorous_protocol import (
    Example,
    mean_and_stderr,
    strict_label_from_output,
    stratified_split,
)
from experiment_rigorous import tune_steering_configuration


class RigorousProtocolTests(unittest.TestCase):
    def test_strict_label_accepts_exact_label(self):
        labels = ("positive", "negative", "neutral")
        self.assertEqual(strict_label_from_output("positive", labels), "positive")
        self.assertEqual(strict_label_from_output(" Negative. ", labels), "negative")

    def test_strict_label_rejects_extra_text(self):
        labels = ("positive", "negative")
        self.assertIsNone(strict_label_from_output("positive sentiment", labels))
        self.assertIsNone(strict_label_from_output("the answer is positive", labels))

    def test_stratified_split_preserves_each_class(self):
        examples = tuple(
            Example(text=f"{label}-{idx}", label=label)
            for label in ("positive", "negative", "neutral")
            for idx in range(5)
        )
        train, val, test = stratified_split(examples, train_frac=0.4, val_frac=0.2, seed=7)

        for split in (train, val, test):
            labels = {example.label for example in split}
            self.assertEqual(labels, {"positive", "negative", "neutral"})

        all_texts = {example.text for example in train + val + test}
        self.assertEqual(len(all_texts), len(examples))

    def test_mean_and_stderr(self):
        mean, stderr = mean_and_stderr([1.0, 1.0, 1.0])
        self.assertEqual(mean, 1.0)
        self.assertEqual(stderr, 0.0)

    def test_tuning_result_is_json_serializable(self):
        class DummyLM:
            pass

        class DummySteering:
            norms = {1: 1.0, 2: 2.0}
            cosine_similarities = {1: 0.9, 2: 0.8}
            vectors = {1: object(), 2: object()}

        original_eval = __import__("experiment_rigorous").evaluate_logprob_condition

        def fake_eval(**kwargs):
            layer = next(iter(kwargs["steering_vectors"]))
            return {"accuracy": 0.7 if layer == 1 else 0.6, "details": []}

        module = __import__("experiment_rigorous")
        module.evaluate_logprob_condition = fake_eval
        try:
            result = tune_steering_configuration(
                lm=DummyLM(),
                validation_examples=[],
                labels=("positive", "negative"),
                system_prompt="x",
                steering_result=DummySteering(),
                alphas=[0.3, 0.5],
                token_strategies=["last"],
                top_k_layers=2,
            )
            json.dumps(result)
        finally:
            module.evaluate_logprob_condition = original_eval


if __name__ == "__main__":
    unittest.main()
