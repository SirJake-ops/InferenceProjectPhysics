from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physics_class_inference.batch import BatchCase
from physics_class_inference.batch import load_cases
from physics_class_inference.batch import run_batch
from physics_class_inference.generation import GenerationResult


class BatchTests(unittest.TestCase):
    def test_load_cases_reads_jsonl_expectations_and_plain_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "cases.jsonl"
            input_file.write_text(
                "\n".join(
                    [
                        '{"id": "case-a", "prompt": "Hi", "expected_token_ids": [79, 75]}',
                        '{"id": "case-b", "prompt": "Hello", "expected_text": "OK"}',
                        "Plain prompt",
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_cases(input_file)

        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0].case_id, "case-a")
        self.assertEqual(cases[0].expected_token_ids, [79, 75])
        self.assertEqual(cases[1].expected_text, "OK")
        self.assertEqual(cases[2].prompt, "Plain prompt")

    def test_load_cases_reads_text_comparison_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "cases.txt"
            input_file.write_text(
                "Answer with only the final number: 2 + 2 ||| 4\n",
                encoding="utf-8",
            )

            cases = load_cases(input_file)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].prompt, "Answer with only the final number: 2 + 2")
        self.assertEqual(cases[0].expected_text, "4")

    def test_run_batch_writes_results_summary_csv_and_graphs(self) -> None:
        cases = [
            BatchCase(prompt="Hi", case_id="match", expected_token_ids=[79, 75]),
            BatchCase(prompt="Yo", case_id="miss", expected_token_ids=[1, 2, 3]),
        ]
        generated = [
            GenerationResult(
                prompt="Hi",
                generated_text="OK",
                prompt_token_ids=[72, 105],
                generated_token_ids=[79, 75],
                tokenizer_backend="byte",
            ),
            GenerationResult(
                prompt="Yo",
                generated_text="AB",
                prompt_token_ids=[89, 111],
                generated_token_ids=[65, 66],
                tokenizer_backend="byte",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch_outputs"
            with patch("physics_class_inference.batch.generate_with_model", side_effect=generated):
                summary = run_batch(cases, output_dir, max_new_tokens=2, tokenizer_backend="byte")

            results_path = output_dir / "results.jsonl"
            rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary.total_cases, 2)
            self.assertEqual(summary.exact_token_matches, 1)
            self.assertEqual(summary.total_token_mismatches, 3)
            self.assertEqual(rows[0]["mismatch_count"], 0)
            self.assertEqual(rows[1]["mismatch_count"], 3)
            self.assertEqual(rows[0]["generated_token_ids"], [79, 75])
            self.assertEqual(rows[0]["generated_token_texts"], ["O", "K"])
            self.assertEqual(rows[0]["expected_token_texts"], ["O", "K"])
            self.assertEqual(rows[0]["prompt_token_texts"], ["H", "i"])
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "graphs" / "token_mismatches.png").is_file())
            self.assertTrue((output_dir / "graphs" / "latency_ms.png").is_file())

    def test_expected_text_is_tokenized_for_comparison(self) -> None:
        case = BatchCase(prompt="Hi", case_id="text", expected_text="OK")
        result = GenerationResult(
            prompt="Hi",
            generated_text="OK",
            prompt_token_ids=[72, 105],
            generated_token_ids=[79, 75],
            tokenizer_backend="byte",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("physics_class_inference.batch.generate_with_model", return_value=result):
                summary = run_batch([case], Path(temp_dir), max_new_tokens=2, tokenizer_backend="byte")

        self.assertEqual(summary.exact_token_matches, 1)
        self.assertEqual(summary.exact_text_matches, 1)


if __name__ == "__main__":
    unittest.main()
