from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

try:
    from .generation import GenerationResult
    from .generation import generate_with_model
except ImportError:
    from physics_class_inference.generation import GenerationResult
    from physics_class_inference.generation import generate_with_model


def _add_prompt_source_arguments(parser: argparse.ArgumentParser) -> None:
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompt", help="Prompt text to generate from.")
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the prompt text from a file.",
    )
    prompt_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read the prompt text from standard input.",
    )


def _read_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        try:
            return args.prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(f"could not read prompt file {args.prompt_file}: {exc}")
    if args.stdin:
        return sys.stdin.read()
    parser.error("one of --prompt, --prompt-file, or --stdin is required")
    raise AssertionError("argparse parser.error() should have exited")


def _format_verbose_output(result: GenerationResult) -> str:
    return "\n".join(
        [
            f"Prompt: {result.prompt}",
            f"Generated: {result.generated_text}",
            f"Response: {result.response_text}",
            f"Tokenizer backend: {result.tokenizer_backend}",
            f"Prompt token ids: {result.prompt_token_ids}",
            f"Generated token ids: {result.generated_token_ids}",
            f"Prompt token count: {result.prompt_token_count}",
            f"Generated token count: {result.generated_token_count}",
            f"Cache layers: {result.cache_layers}",
            f"Cache sequence length: {result.cache_sequence_length}",
        ]
    )


def _result_to_payload(result: GenerationResult, elapsed_ms: float | None = None) -> dict[str, object]:
    payload = asdict(result)
    payload["response_text"] = result.response_text
    payload["prompt_token_count"] = result.prompt_token_count
    payload["generated_token_count"] = result.generated_token_count
    if elapsed_ms is not None:
        payload["elapsed_ms"] = round(elapsed_ms, 3)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physics-class-inference",
        description="Run local text generation without an HTTP server.",
    )
    _add_prompt_source_arguments(parser)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate.",
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
        help="Tokenizer backend to use. 'auto' prefers GPT-2 BPE and falls back to byte mode.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "full", "json", "verbose"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Deprecated alias for --format json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Deprecated alias for --format verbose.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Include elapsed generation time in verbose or JSON output.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    prompt = _read_prompt(args, parser)
    output_format = args.format
    if args.json:
        output_format = "json"
    elif args.verbose:
        output_format = "verbose"

    started_at = time.perf_counter()
    try:
        result = generate_with_model(
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            model_path=args.model_path,
            tokenizer_backend=args.tokenizer,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    if output_format == "json":
        print(json.dumps(_result_to_payload(result, elapsed_ms if args.timing else None), indent=2))
    elif output_format == "verbose":
        rendered = _format_verbose_output(result)
        if args.timing:
            rendered = f"{rendered}\nElapsed ms: {elapsed_ms:.3f}"
        print(rendered)
    elif output_format == "full":
        print(result.response_text)
    else:
        print(result.generated_text)

    return 0
