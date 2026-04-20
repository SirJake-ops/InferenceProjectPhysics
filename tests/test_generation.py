from __future__ import annotations

import unittest
from unittest.mock import patch

from physics_class_inference.generation import GenerationResult
from physics_class_inference.generation import generate
from physics_class_inference.generation import generate_with_model
from physics_class_inference.generation import get_next_token


def _make_logits(best_token: int, vocab_size: int = 50257) -> list[float]:
    logits = [-1.0] * vocab_size
    logits[best_token] = 10.0
    return logits


class _FakeModel:
    def __init__(self) -> None:
        self.reset_count = 0
        self.calls: list[tuple[list[int], int]] = []
        self.required_cache_layer_count = 2
        self.cached_sequence_length = 4
        self.responses = [
            _make_logits(2),
            _make_logits(1),
            _make_logits(3),
        ]

    def get_required_cache_layer_count(self) -> int:
        return self.required_cache_layer_count

    def reset_cache(self) -> None:
        self.reset_count += 1

    def run_inference(self, input_ids: list[int], number_of_layers: int) -> list[float]:
        self.calls.append((input_ids, number_of_layers))
        return self.responses[len(self.calls) - 1]

    def get_cached_sequence_length(self) -> int:
        return self.cached_sequence_length


class GenerationTests(unittest.TestCase):
    def test_get_next_token_uses_last_vocabulary_window(self) -> None:
        logits = [9.0, 8.0, 7.0, 0.2, 0.9, 0.1]
        self.assertEqual(get_next_token(logits, 3), 1)

    def test_get_next_token_rejects_short_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "full vocabulary window"):
            get_next_token([0.1, 0.2, 0.3], 4)

    def test_generate_with_model_rejects_empty_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prompt body must not be empty"):
            generate_with_model("", 2)

    def test_generate_with_model_rejects_zero_max_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens must be greater than 0"):
            generate_with_model("Hi", 0)

    def test_generate_with_model_runs_prefill_then_single_token_steps(self) -> None:
        fake_model = _FakeModel()

        with patch("physics_class_inference.generation._get_model", return_value=fake_model):
            result = generate_with_model("Hi", 3, tokenizer_backend="byte")

        self.assertEqual(fake_model.reset_count, 1)
        self.assertEqual(fake_model.calls, [([72, 105], 2), ([2], 2), ([1], 2)])
        self.assertEqual(result.prompt, "Hi")
        self.assertEqual(result.generated_token_ids, [2, 1, 3])
        self.assertEqual(result.generated_text, "\x02\x01\x03")
        self.assertEqual(result.prompt_token_ids, [72, 105])
        self.assertEqual(result.cache_layers, 2)
        self.assertEqual(result.cache_sequence_length, 4)
        self.assertEqual(result.tokenizer_backend, "byte")
        self.assertEqual(result.response_text, "Hi\x02\x01\x03")
        self.assertEqual(result.prompt_token_count, 2)
        self.assertEqual(result.generated_token_count, 3)

    def test_generate_uses_default_model_loader(self) -> None:
        fake_model = _FakeModel()

        with patch("physics_class_inference.generation._get_model", return_value=fake_model):
            result = generate("A", 1, tokenizer_backend="byte")

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.generated_token_ids, [2])


if __name__ == "__main__":
    unittest.main()
