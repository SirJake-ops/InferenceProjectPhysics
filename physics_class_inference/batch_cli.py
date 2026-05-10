from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    from .batch import collect_interactive_cases
    from .batch import load_cases
    from .batch import run_batch
except ImportError:
    from physics_class_inference.batch import collect_interactive_cases
    from physics_class_inference.batch import load_cases
    from physics_class_inference.batch import run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-class-inference-batch",
        description="Run prompt batches and compare generated tokens with expected outputs.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input-file",
        type=Path,
        help="Read batch cases from a text or JSONL file.",
    )
    source_group.add_argument(
        "--interactive",
        action="store_true",
        help="Enter prompts and optional expectations from the command line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_outputs"),
        help="Directory for JSONL, CSV, summary, and generated graph files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate per case.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the default model path.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["auto", "gpt2", "byte"],
        default="auto",
        help="Tokenizer backend to use for generation and expected_text tokenization.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.interactive:
            cases = collect_interactive_cases()
        else:
            try:
                cases = load_cases(args.input_file)
            except OSError as exc:
                parser.error(f"could not read input file {args.input_file}: {exc}")

        summary = run_batch(
            cases=cases,
            output_dir=args.output_dir,
            max_new_tokens=args.max_tokens,
            model_path=args.model_path,
            tokenizer_backend=args.tokenizer,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
