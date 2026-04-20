from __future__ import annotations

import unittest

from physics_class_inference.config import DEFAULT_MODEL_PATH
from physics_class_inference.generation import generate_with_model
from physics_class_inference.inference import ModelInference
from physics_class_inference.tokenizer import Tokenizer


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), f"Missing model asset: {DEFAULT_MODEL_PATH}")
@unittest.skipUnless(_has_onnxruntime(), "onnxruntime is not installed")
class RealModelRuntimeTests(unittest.TestCase):
    def test_model_session_loads_and_reports_cache_layers(self) -> None:
        model = ModelInference(DEFAULT_MODEL_PATH)
        self.assertGreater(model.get_required_cache_layer_count(), 0)
        self.assertEqual(model.get_cached_sequence_length(), 0)

    def test_real_model_inference_returns_logits_for_prompt_tokens(self) -> None:
        model = ModelInference(DEFAULT_MODEL_PATH)
        tokenizer = Tokenizer()
        prompt_token_ids = tokenizer.encode("Hello")

        logits = model.run_inference(prompt_token_ids, model.get_required_cache_layer_count())

        self.assertTrue(logits)
        self.assertEqual(len(logits) % 50257, 0)
        self.assertEqual(model.get_cached_sequence_length(), len(prompt_token_ids))

    def test_real_model_generation_returns_expected_metadata(self) -> None:
        result = generate_with_model("Hello", max_new_tokens=1, model_path=DEFAULT_MODEL_PATH)
        tokenizer = Tokenizer()
        prompt_token_ids = tokenizer.encode("Hello")

        self.assertEqual(result.prompt, "Hello")
        self.assertEqual(result.prompt_token_ids, prompt_token_ids)
        self.assertEqual(result.prompt_token_count, len(prompt_token_ids))
        self.assertEqual(result.generated_token_count, 1)
        self.assertEqual(len(result.generated_token_ids), 1)
        self.assertEqual(result.cache_layers, 24)
        self.assertEqual(result.cache_sequence_length, len(prompt_token_ids))
        self.assertEqual(result.tokenizer_backend, tokenizer.backend_name)
        self.assertEqual(result.response_text, result.prompt + result.generated_text)


if __name__ == "__main__":
    unittest.main()
