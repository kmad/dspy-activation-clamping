import unittest

import torch

from steering import extract_steering_vectors_from_message_pairs


class DummyLM:
    num_layers = 2

    def extract_activations(self, messages, layers):
        signal = sum(len(message["content"]) for message in messages)
        activations = {}
        for layer in layers:
            hidden = torch.tensor(
                [[[signal + layer, signal + layer + 1, signal + layer + 2]]],
                dtype=torch.float32,
            )
            activations[layer] = hidden
        return activations


class SteeringExtractionTests(unittest.TestCase):
    def test_message_pair_extraction_uses_full_messages(self):
        lm = DummyLM()
        optimized_messages = [
            [{"role": "system", "content": "optimized"}, {"role": "user", "content": "abc"}],
            [{"role": "system", "content": "optimized"}, {"role": "user", "content": "abcdef"}],
        ]
        baseline_messages = [
            [{"role": "system", "content": "base"}, {"role": "user", "content": "abc"}],
            [{"role": "system", "content": "base"}, {"role": "user", "content": "abcdef"}],
        ]

        result = extract_steering_vectors_from_message_pairs(
            lm,
            optimized_messages=optimized_messages,
            baseline_messages=baseline_messages,
            layers=[0, 1],
            token_aggregation="last",
        )

        expected_layer0 = torch.tensor([5.0, 5.0, 5.0])
        expected_layer1 = torch.tensor([5.0, 5.0, 5.0])
        self.assertTrue(torch.equal(result.vectors[0], expected_layer0))
        self.assertTrue(torch.equal(result.vectors[1], expected_layer1))
        self.assertEqual(result.num_samples, 2)

    def test_message_pair_length_must_match(self):
        lm = DummyLM()
        with self.assertRaises(ValueError):
            extract_steering_vectors_from_message_pairs(
                lm,
                optimized_messages=[[{"role": "system", "content": "a"}]],
                baseline_messages=[],
                layers=[0],
            )


if __name__ == "__main__":
    unittest.main()
