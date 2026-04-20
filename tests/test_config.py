from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physics_class_inference import config


class ResolveModelPathTests(unittest.TestCase):
    def test_returns_existing_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "absolute_model.onnx"
            model_path.write_text("test")

            resolved = config.resolve_model_path(model_path)

            self.assertEqual(resolved, model_path)

    def test_resolves_relative_path_from_current_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            model_path = temp_root / "cwd-model.onnx"
            model_path.write_text("test")

            with patch("pathlib.Path.cwd", return_value=temp_root):
                with patch.object(config, "PROJECT_ROOT", temp_root / "project"):
                    with patch.object(config, "DEFAULT_MODELS_DIR", temp_root / "project" / "models"):
                        resolved = config.resolve_model_path("cwd-model.onnx")

            self.assertEqual(resolved, model_path)

    def test_resolves_relative_path_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            project_root = temp_root / "project"
            project_root.mkdir()
            model_path = project_root / "models" / "nested-model.onnx"
            model_path.parent.mkdir()
            model_path.write_text("test")

            with patch.object(config, "PROJECT_ROOT", project_root):
                with patch.object(config, "DEFAULT_MODELS_DIR", project_root / "models"):
                    resolved = config.resolve_model_path(Path("models") / "nested-model.onnx")

            self.assertEqual(resolved, model_path)

    def test_resolves_filename_from_models_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            models_dir = temp_root / "models"
            models_dir.mkdir()
            model_path = models_dir / "model.onnx"
            model_path.write_text("test")

            with patch.object(config, "PROJECT_ROOT", temp_root):
                with patch.object(config, "DEFAULT_MODELS_DIR", models_dir):
                    resolved = config.resolve_model_path("model.onnx")

            self.assertEqual(resolved, model_path)

    def test_falls_back_to_first_onnx_model_when_no_path_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            models_dir = temp_root / "models"
            models_dir.mkdir()
            ignored_file = models_dir / "notes.txt"
            ignored_file.write_text("ignore me")
            model_path = models_dir / "fallback.onnx"
            model_path.write_text("test")

            with patch.object(config, "PROJECT_ROOT", temp_root):
                with patch.object(config, "DEFAULT_MODELS_DIR", models_dir):
                    resolved = config.resolve_model_path()

            self.assertEqual(resolved, model_path)

    def test_missing_explicit_path_raises_helpful_error(self) -> None:
        missing = Path("models") / "definitely_missing_decoder_model.onnx"

        with self.assertRaisesRegex(RuntimeError, str(missing)):
            config.resolve_model_path(missing)

    def test_missing_default_model_directory_raises_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            missing_models_dir = temp_root / "models"

            with patch.object(config, "PROJECT_ROOT", temp_root):
                with patch.object(config, "DEFAULT_MODELS_DIR", missing_models_dir):
                    with self.assertRaisesRegex(RuntimeError, str(missing_models_dir)):
                        config.resolve_model_path()


if __name__ == "__main__":
    unittest.main()
