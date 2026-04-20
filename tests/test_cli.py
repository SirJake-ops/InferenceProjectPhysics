from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from physics_class_inference.cli import GenerationResult
from physics_class_inference.cli import main


class CliTests(unittest.TestCase):
    def test_plain_output_uses_generated_text(self) -> None:
        result = GenerationResult(prompt="Hi", generated_text="OK")
        stdout = io.StringIO()

        with patch("physics_class_inference.cli.generate_with_model", return_value=result):
            with patch("sys.argv", ["physics-class-inference", "--prompt", "Hi"]):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "OK")

    def test_json_output_includes_derived_fields(self) -> None:
        result = GenerationResult(
            prompt="Hi",
            generated_text="OK",
            prompt_token_ids=[72, 105],
            generated_token_ids=[79, 75],
            cache_layers=24,
            cache_sequence_length=4,
            tokenizer_backend="gpt2",
        )
        stdout = io.StringIO()

        with patch("physics_class_inference.cli.generate_with_model", return_value=result):
            with patch("sys.argv", ["physics-class-inference", "--prompt", "Hi", "--json"]):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["response_text"], "HiOK")
        self.assertEqual(payload["prompt_token_count"], 2)
        self.assertEqual(payload["generated_token_count"], 2)
        self.assertEqual(payload["tokenizer_backend"], "gpt2")

    def test_verbose_output_includes_metadata(self) -> None:
        result = GenerationResult(
            prompt="Hi",
            generated_text="OK",
            prompt_token_ids=[72, 105],
            generated_token_ids=[79, 75],
            cache_layers=24,
            cache_sequence_length=4,
            tokenizer_backend="gpt2",
        )
        stdout = io.StringIO()

        with patch("physics_class_inference.cli.generate_with_model", return_value=result):
            with patch("sys.argv", ["physics-class-inference", "--prompt", "Hi", "--verbose"]):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertIn("Prompt: Hi", rendered)
        self.assertIn("Generated: OK", rendered)
        self.assertIn("Cache layers: 24", rendered)
        self.assertIn("Tokenizer backend: gpt2", rendered)

    def test_reads_prompt_from_file(self) -> None:
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_file = Path(temp_dir) / "prompt.txt"
            prompt_file.write_text("Hello from file", encoding="utf-8")

            with patch(
                "physics_class_inference.cli.generate_with_model",
                return_value=GenerationResult(prompt="Hello from file", generated_text="x"),
            ) as generate_mock:
                with patch(
                    "sys.argv",
                    ["physics-class-inference", "--prompt-file", str(prompt_file)],
                ):
                    with redirect_stdout(stdout):
                        exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_args.kwargs["prompt"], "Hello from file")

    def test_reads_prompt_from_stdin(self) -> None:
        stdout = io.StringIO()

        with patch(
            "physics_class_inference.cli.generate_with_model",
            return_value=GenerationResult(prompt="stdin prompt", generated_text="x"),
        ) as generate_mock:
            with patch("sys.argv", ["physics-class-inference", "--stdin"]):
                with patch("sys.stdin", io.StringIO("stdin prompt")):
                    with redirect_stdout(stdout):
                        exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_args.kwargs["prompt"], "stdin prompt")

    def test_passes_model_path_to_generation(self) -> None:
        stdout = io.StringIO()
        model_path = Path("models/custom.onnx")

        with patch(
            "physics_class_inference.cli.generate_with_model",
            return_value=GenerationResult(prompt="Hi", generated_text="x"),
        ) as generate_mock:
            with patch(
                "sys.argv",
                ["physics-class-inference", "--prompt", "Hi", "--model-path", str(model_path)],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_args.kwargs["model_path"], model_path)

    def test_passes_tokenizer_backend_to_generation(self) -> None:
        stdout = io.StringIO()

        with patch(
            "physics_class_inference.cli.generate_with_model",
            return_value=GenerationResult(prompt="Hi", generated_text="x"),
        ) as generate_mock:
            with patch(
                "sys.argv",
                ["physics-class-inference", "--prompt", "Hi", "--tokenizer", "byte"],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate_mock.call_args.kwargs["tokenizer_backend"], "byte")

    def test_full_format_prints_prompt_plus_generated_text(self) -> None:
        stdout = io.StringIO()

        with patch(
            "physics_class_inference.cli.generate_with_model",
            return_value=GenerationResult(prompt="Hi", generated_text=" there"),
        ):
            with patch(
                "sys.argv",
                ["physics-class-inference", "--prompt", "Hi", "--format", "full"],
            ):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "Hi there")

    def test_json_timing_adds_elapsed_ms(self) -> None:
        stdout = io.StringIO()

        with patch(
            "physics_class_inference.cli.generate_with_model",
            return_value=GenerationResult(prompt="Hi", generated_text="OK"),
        ):
            with patch("sys.argv", ["physics-class-inference", "--prompt", "Hi", "--json", "--timing"]):
                with redirect_stdout(stdout):
                    exit_code = main()

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("elapsed_ms", payload)

    def test_returns_error_code_and_message_for_generation_errors(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch(
            "physics_class_inference.cli.generate_with_model",
            side_effect=RuntimeError("model failed"),
        ):
            with patch("sys.argv", ["physics-class-inference", "--prompt", "Hi"]):
                with redirect_stdout(stdout):
                    with redirect_stderr(stderr):
                        exit_code = main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Error: model failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
