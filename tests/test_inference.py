from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from physics_class_inference.inference import ModelInference


class _FakeIo:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads: int | None = None
        self.graph_optimization_level = None


class _FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "ORT_ENABLE_ALL"


class _FakeOrt:
    SessionOptions = _FakeSessionOptions
    GraphOptimizationLevel = _FakeGraphOptimizationLevel


class _FakeSession:
    def __init__(self, *_args, **_kwargs) -> None:
        self.inputs = [
            _FakeIo("input_ids"),
            _FakeIo("attention_mask"),
            _FakeIo("position_ids"),
            _FakeIo("past_key_values.0.key"),
            _FakeIo("past_key_values.0.value"),
            _FakeIo("past_key_values.1.key"),
            _FakeIo("past_key_values.1.value"),
        ]
        self.outputs = [
            _FakeIo("logits"),
            _FakeIo("present.0.key"),
            _FakeIo("present.0.value"),
            _FakeIo("present.1.key"),
            _FakeIo("present.1.value"),
        ]
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self) -> list[_FakeIo]:
        return self.inputs

    def get_outputs(self) -> list[_FakeIo]:
        return self.outputs

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append((output_names, input_feed))
        step_index = len(self.calls)
        logits = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32)
        key_0 = np.zeros((1, 16, step_index, 64), dtype=np.float32)
        value_0 = np.zeros((1, 16, step_index, 64), dtype=np.float32)
        key_1 = np.ones((1, 16, step_index, 64), dtype=np.float32)
        value_1 = np.ones((1, 16, step_index, 64), dtype=np.float32)
        return [logits, key_0, value_0, key_1, value_1]


class ModelInferenceTests(unittest.TestCase):
    def _make_model_file(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temp_dir = tempfile.TemporaryDirectory()
        model_path = Path(temp_dir.name) / "model.onnx"
        model_path.write_text("fake")
        return temp_dir, model_path

    def test_inspects_required_cache_layer_count_from_session_inputs(self) -> None:
        temp_dir, model_path = self._make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch("physics_class_inference.inference._import_onnxruntime", return_value=_FakeOrt):
            with patch.object(_FakeOrt, "InferenceSession", _FakeSession, create=True):
                model = ModelInference(model_path)

        self.assertEqual(model.get_required_cache_layer_count(), 2)

    def test_empty_input_resizes_cache_and_returns_empty_logits(self) -> None:
        temp_dir, model_path = self._make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch("physics_class_inference.inference._import_onnxruntime", return_value=_FakeOrt):
            with patch.object(_FakeOrt, "InferenceSession", _FakeSession, create=True):
                model = ModelInference(model_path)
                logits = model.run_inference([], 3)

        self.assertEqual(logits, [])
        self.assertEqual(len(model._layer_cache), 3)

    def test_run_inference_builds_expected_inputs_and_flattens_logits(self) -> None:
        temp_dir, model_path = self._make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch("physics_class_inference.inference._import_onnxruntime", return_value=_FakeOrt):
            with patch.object(_FakeOrt, "InferenceSession", _FakeSession, create=True):
                model = ModelInference(model_path)
                logits = model.run_inference([72, 105], 1)
                session = model._session

        np.testing.assert_allclose(logits, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(len(session.calls), 1)
        _, input_feed = session.calls[0]
        np.testing.assert_array_equal(input_feed["input_ids"], np.asarray([[72, 105]], dtype=np.int64))
        np.testing.assert_array_equal(input_feed["attention_mask"], np.asarray([[1, 1]], dtype=np.int64))
        np.testing.assert_array_equal(input_feed["position_ids"], np.asarray([[0, 1]], dtype=np.int64))
        self.assertEqual(input_feed["past_key_values.0.key"].shape, (1, 16, 0, 64))
        self.assertEqual(input_feed["past_key_values.1.value"].shape, (1, 16, 0, 64))

    def test_reuses_cache_between_calls_and_tracks_sequence_length(self) -> None:
        temp_dir, model_path = self._make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch("physics_class_inference.inference._import_onnxruntime", return_value=_FakeOrt):
            with patch.object(_FakeOrt, "InferenceSession", _FakeSession, create=True):
                model = ModelInference(model_path)
                model.run_inference([1, 2], 2)
                first_length = model.get_cached_sequence_length()
                model.run_inference([3], 2)
                session = model._session

        self.assertEqual(first_length, 1)
        self.assertEqual(model.get_cached_sequence_length(), 2)
        _, second_input_feed = session.calls[1]
        self.assertEqual(second_input_feed["attention_mask"].shape, (1, 2))
        self.assertEqual(second_input_feed["position_ids"][0, 0], 1)
        self.assertEqual(second_input_feed["past_key_values.0.key"].shape, (1, 16, 1, 64))

    def test_reset_cache_clears_sequence_length(self) -> None:
        temp_dir, model_path = self._make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch("physics_class_inference.inference._import_onnxruntime", return_value=_FakeOrt):
            with patch.object(_FakeOrt, "InferenceSession", _FakeSession, create=True):
                model = ModelInference(model_path)
                model.run_inference([1], 2)
                model.reset_cache()

        self.assertEqual(model.get_cached_sequence_length(), 0)

    def test_missing_onnxruntime_raises_clear_error(self) -> None:
        temp_dir, model_path = self._make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch(
            "physics_class_inference.inference._import_onnxruntime",
            side_effect=RuntimeError("onnxruntime is required for model inference. Install the project dependencies first."),
        ):
            with self.assertRaisesRegex(RuntimeError, "onnxruntime is required"):
                ModelInference(model_path)


if __name__ == "__main__":
    unittest.main()
