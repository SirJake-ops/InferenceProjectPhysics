from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_model_path

DEFAULT_ATTENTION_HEAD_COUNT = 16
DEFAULT_ATTENTION_HEAD_SIZE = 64


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for model inference. Install the project dependencies first."
        ) from exc
    return ort


def _create_session_options(ort: Any) -> Any:
    session_options = ort.SessionOptions()
    if hasattr(session_options, "intra_op_num_threads"):
        session_options.intra_op_num_threads = 4
    if hasattr(ort, "GraphOptimizationLevel"):
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return session_options


def _get_io_name(entry: Any) -> str:
    return getattr(entry, "name")


class ModelInference:

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = resolve_model_path(model_path)
        ort = _import_onnxruntime()
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=_create_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        self._layer_cache: list[dict[str, np.ndarray | None]] = []
        self._input_names = [_get_io_name(entry) for entry in self._session.get_inputs()]
        self._output_names = [_get_io_name(entry) for entry in self._session.get_outputs()]
        self._required_cache_layer_count = sum(
            1
            for name in self._input_names
            if name.startswith("past_key_values.") and name.endswith(".key")
        )

    def _resize_layer_cache(self, number_of_layers: int) -> None:
        if len(self._layer_cache) == number_of_layers:
            return
        self._layer_cache = [{"key": None, "value": None} for _ in range(number_of_layers)]

    def _get_past_sequence_length(self) -> int:
        if not self._layer_cache:
            return 0
        first_key_cache = self._layer_cache[0]["key"]
        if first_key_cache is None or len(first_key_cache.shape) < 3:
            return 0
        return max(int(first_key_cache.shape[2]), 0)

    def _empty_cache_tensor(self, past_sequence_length: int) -> np.ndarray:
        return np.zeros(
            (1, DEFAULT_ATTENTION_HEAD_COUNT, past_sequence_length, DEFAULT_ATTENTION_HEAD_SIZE),
            dtype=np.float32,
        )

    def _build_input_feed(self, input_ids: list[int], active_cache_layers: int) -> dict[str, np.ndarray]:
        past_sequence_length = self._get_past_sequence_length()
        input_array = np.asarray([input_ids], dtype=np.int64)
        attention_mask = np.ones((1, past_sequence_length + len(input_ids)), dtype=np.int64)
        position_ids = np.asarray(
            [[past_sequence_length + index for index in range(len(input_ids))]],
            dtype=np.int64,
        )

        input_feed: dict[str, np.ndarray] = {
            "input_ids": input_array,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        empty_cache = self._empty_cache_tensor(past_sequence_length)
        for layer_index in range(active_cache_layers):
            key_name = f"past_key_values.{layer_index}.key"
            value_name = f"past_key_values.{layer_index}.value"

            key_cache = empty_cache
            value_cache = empty_cache
            if layer_index < len(self._layer_cache):
                cached_key = self._layer_cache[layer_index]["key"]
                cached_value = self._layer_cache[layer_index]["value"]
                if cached_key is not None:
                    key_cache = cached_key
                if cached_value is not None:
                    value_cache = cached_value

            input_feed[key_name] = key_cache
            input_feed[value_name] = value_cache

        return {name: value for name, value in input_feed.items() if name in self._input_names}

    def run_inference(self, input_ids: list[int], number_of_layers: int) -> list[float]:
        self._resize_layer_cache(number_of_layers)

        if not input_ids:
            return []

        active_cache_layers = max(self._required_cache_layer_count, number_of_layers)
        input_feed = self._build_input_feed(input_ids, active_cache_layers)
        output_values = self._session.run(self._output_names, input_feed)
        outputs_by_name = dict(zip(self._output_names, output_values, strict=False))

        cache_layers_to_store = min(number_of_layers, self._required_cache_layer_count)
        for layer_index in range(cache_layers_to_store):
            key_name = f"present.{layer_index}.key"
            value_name = f"present.{layer_index}.value"
            if key_name in outputs_by_name and value_name in outputs_by_name:
                self._layer_cache[layer_index]["key"] = outputs_by_name[key_name]
                self._layer_cache[layer_index]["value"] = outputs_by_name[value_name]

        logits = output_values[0]
        return np.asarray(logits, dtype=np.float32).reshape(-1).tolist()

    def reset_cache(self) -> None:
        for layer_cache in self._layer_cache:
            layer_cache["key"] = None
            layer_cache["value"] = None

    def get_required_cache_layer_count(self) -> int:
        return self._required_cache_layer_count

    def get_cached_sequence_length(self) -> int:
        return self._get_past_sequence_length()
