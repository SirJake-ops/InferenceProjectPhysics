from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from physics_class_inference.batch import BatchCase
from physics_class_inference.batch import BatchSummary
from physics_class_inference.batch_cli import main


class BatchCliTests(unittest.TestCase):
    def test_runs_batch_from_input_file(self) -> None:
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "cases.jsonl"
            output_dir = Path(temp_dir) / "outputs"
            input_file.write_text('{"prompt": "Hi", "expected_text": "OK"}\n', encoding="utf-8")
            summary = BatchSummary(
                total_cases=1,
                cases_with_expectations=1,
                exact_token_matches=1,
                exact_text_matches=1,
                total_token_mismatches=0,
                results_path=str(output_dir / "results.jsonl"),
                summary_path=str(output_dir / "summary.json"),
                csv_path=str(output_dir / "summary.csv"),
                graphs_dir=str(output_dir / "graphs"),
            )

            with patch("physics_class_inference.batch_cli.run_batch", return_value=summary) as run_mock:
                with patch(
                    "sys.argv",
                    [
                        "physics-class-inference-batch",
                        "--input-file",
                        str(input_file),
                        "--output-dir",
                        str(output_dir),
                        "--max-tokens",
                        "3",
                        "--tokenizer",
                        "byte",
                    ],
                ):
                    with redirect_stdout(stdout):
                        exit_code = main()

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["total_cases"], 1)
        self.assertEqual(run_mock.call_args.kwargs["max_new_tokens"], 3)
        self.assertEqual(run_mock.call_args.kwargs["tokenizer_backend"], "byte")

    def test_interactive_uses_collected_cases(self) -> None:
        stdout = io.StringIO()
        case = BatchCase(prompt="Hi", expected_text="OK")
        summary = BatchSummary(
            total_cases=1,
            cases_with_expectations=1,
            exact_token_matches=1,
            exact_text_matches=1,
            total_token_mismatches=0,
            results_path="results.jsonl",
            summary_path="summary.json",
            csv_path="summary.csv",
            graphs_dir="graphs",
        )

        with patch("physics_class_inference.batch_cli.collect_interactive_cases", return_value=[case]) as collect_mock:
            with patch("physics_class_inference.batch_cli.run_batch", return_value=summary):
                with patch("sys.argv", ["physics-class-inference-batch", "--interactive"]):
                    with redirect_stdout(stdout):
                        exit_code = main()

        self.assertEqual(exit_code, 0)
        collect_mock.assert_called_once_with()

    def test_returns_error_code_for_batch_errors(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch("physics_class_inference.batch_cli.load_cases", side_effect=ValueError("bad input")):
            with patch("sys.argv", ["physics-class-inference-batch", "--input-file", "missing.jsonl"]):
                with redirect_stdout(stdout):
                    with redirect_stderr(stderr):
                        exit_code = main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Error: bad input", stderr.getvalue())

    def test_missing_input_file_exits_with_parser_error(self) -> None:
        stderr = io.StringIO()

        with patch("sys.argv", ["physics-class-inference-batch", "--input-file", "missing-prompts.jsonl"]):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_context:
                    main()

        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn("could not read input file missing-prompts.jsonl", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
