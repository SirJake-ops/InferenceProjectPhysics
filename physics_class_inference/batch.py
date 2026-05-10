from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .generation import GenerationResult
from .generation import generate_with_model
from .tokenizer import Tokenizer


@dataclass(slots=True)
class BatchCase:
    prompt: str
    case_id: str = ""
    expected_text: str | None = None
    expected_token_ids: list[int] | None = None


@dataclass(slots=True)
class TokenMismatch:
    position: int
    expected: int | None
    generated: int | None


@dataclass(slots=True)
class BatchResult:
    case_id: str
    prompt: str
    expected_text: str | None
    expected_token_ids: list[int] | None
    expected_token_texts: list[str] | None
    generated_text: str
    generated_token_ids: list[int]
    generated_token_texts: list[str]
    prompt_token_ids: list[int]
    prompt_token_texts: list[str]
    tokenizer_backend: str
    elapsed_ms: float
    mismatches: list[TokenMismatch] = field(default_factory=list)
    exact_token_match: bool | None = None
    exact_text_match: bool | None = None

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)


@dataclass(slots=True)
class BatchSummary:
    total_cases: int
    cases_with_expectations: int
    exact_token_matches: int
    exact_text_matches: int
    total_token_mismatches: int
    results_path: str
    summary_path: str
    csv_path: str
    graphs_dir: str


def load_cases(path: Path) -> list[BatchCase]:
    cases: list[BatchCase] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("{"):
            payload = json.loads(line)
            prompt = str(payload.get("prompt", ""))
            if not prompt:
                raise ValueError(
                    f"{path}:{line_number}: JSONL case is missing a non-empty prompt"
                )

            expected_token_ids = payload.get("expected_token_ids")
            if expected_token_ids is not None:
                if not isinstance(expected_token_ids, list) or not all(
                    isinstance(item, int) for item in expected_token_ids
                ):
                    raise ValueError(
                        f"{path}:{line_number}: expected_token_ids must be a list of integers"
                    )

            expected_text = payload.get("expected_text")
            cases.append(
                BatchCase(
                    prompt=prompt,
                    case_id=str(
                        payload.get("id")
                        or payload.get("case_id")
                        or f"case-{line_number}"
                    ),
                    expected_text=str(expected_text)
                    if expected_text is not None
                    else None,
                    expected_token_ids=expected_token_ids,
                )
            )
        else:
            prompt, expected_text = _parse_text_case(line)
            cases.append(
                BatchCase(
                    prompt=prompt,
                    case_id=f"case-{line_number}",
                    expected_text=expected_text,
                )
            )

    if not cases:
        raise ValueError(f"No batch cases found in {path}")

    return cases


def collect_interactive_cases() -> list[BatchCase]:
    cases: list[BatchCase] = []
    index = 1

    while True:
        prompt = input("Prompt (blank to run): ").strip()
        if not prompt:
            break

        expected_text = input("Expected text (optional): ").strip() or None
        expected_token_ids = _parse_expected_token_ids(
            input("Expected token ids, comma-separated (optional): ").strip()
        )
        cases.append(
            BatchCase(
                prompt=prompt,
                case_id=f"interactive-{index}",
                expected_text=expected_text,
                expected_token_ids=expected_token_ids,
            )
        )
        index += 1

    if not cases:
        raise ValueError("No interactive cases entered")

    return cases


def _parse_text_case(line: str) -> tuple[str, str | None]:
    if "|||" not in line:
        return line, None

    prompt, expected_text = line.split("|||", maxsplit=1)
    prompt = prompt.strip()
    expected_text = expected_text.strip()
    if not prompt:
        raise ValueError("Text batch comparison line is missing a prompt before |||")

    return prompt, expected_text or None


def run_batch(
    cases: Iterable[BatchCase],
    output_dir: Path,
    max_new_tokens: int,
    model_path: str | Path | None = None,
    tokenizer_backend: str = "auto",
) -> BatchSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_backend)
    results: list[BatchResult] = []

    for index, case in enumerate(cases, start=1):
        case_id = case.case_id or f"case-{index}"
        started_at = time.perf_counter()
        generation = generate_with_model(
            prompt=case.prompt,
            max_new_tokens=max_new_tokens,
            model_path=model_path,
            tokenizer_backend=tokenizer_backend,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        expected_token_ids = _expected_token_ids(case, tokenizer)
        mismatches = _compare_tokens(expected_token_ids, generation.generated_token_ids)

        results.append(
            BatchResult(
                case_id=case_id,
                prompt=case.prompt,
                expected_text=case.expected_text,
                expected_token_ids=expected_token_ids,
                expected_token_texts=_decode_token_texts(tokenizer, expected_token_ids),
                generated_text=generation.generated_text,
                generated_token_ids=generation.generated_token_ids,
                generated_token_texts=_decode_token_texts(
                    tokenizer, generation.generated_token_ids
                )
                or [],
                prompt_token_ids=generation.prompt_token_ids,
                prompt_token_texts=_decode_token_texts(
                    tokenizer, generation.prompt_token_ids
                )
                or [],
                tokenizer_backend=generation.tokenizer_backend,
                elapsed_ms=round(elapsed_ms, 3),
                mismatches=mismatches,
                exact_token_match=None
                if expected_token_ids is None
                else not mismatches,
                exact_text_match=None
                if case.expected_text is None
                else case.expected_text == generation.generated_text,
            )
        )

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    _write_results_jsonl(results_path, results)
    _write_summary_csv(csv_path, results)
    _write_graphs(graphs_dir, results)

    summary = _summarize(results, results_path, summary_path, csv_path, graphs_dir)
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_expected_token_ids(raw: str) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _expected_token_ids(case: BatchCase, tokenizer: Tokenizer) -> list[int] | None:
    if case.expected_token_ids is not None:
        return case.expected_token_ids
    if case.expected_text is not None:
        return tokenizer.encode(case.expected_text)
    return None


def _decode_token_texts(
    tokenizer: Tokenizer, token_ids: list[int] | None
) -> list[str] | None:
    if token_ids is None:
        return None
    return [tokenizer.decode([token_id]) for token_id in token_ids]


def _compare_tokens(
    expected: list[int] | None, generated: list[int]
) -> list[TokenMismatch]:
    if expected is None:
        return []

    mismatches: list[TokenMismatch] = []
    max_length = max(len(expected), len(generated))
    for position in range(max_length):
        expected_token = expected[position] if position < len(expected) else None
        generated_token = generated[position] if position < len(generated) else None
        if expected_token != generated_token:
            mismatches.append(
                TokenMismatch(
                    position=position,
                    expected=expected_token,
                    generated=generated_token,
                )
            )
    return mismatches


def _write_results_jsonl(path: Path, results: list[BatchResult]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for result in results:
            payload = asdict(result)
            payload["mismatch_count"] = result.mismatch_count
            output.write(json.dumps(payload) + "\n")


def _write_summary_csv(path: Path, results: list[BatchResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "case_id",
                "prompt",
                "expected_text",
                "generated_text",
                "expected_token_ids",
                "expected_token_texts",
                "generated_token_ids",
                "generated_token_texts",
                "mismatch_count",
                "exact_token_match",
                "exact_text_match",
                "elapsed_ms",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "prompt": result.prompt,
                    "expected_text": result.expected_text or "",
                    "generated_text": result.generated_text,
                    "expected_token_ids": json.dumps(result.expected_token_ids),
                    "expected_token_texts": json.dumps(
                        result.expected_token_texts
                    ),
                    "generated_token_ids": json.dumps(result.generated_token_ids),
                    "generated_token_texts": json.dumps(
                        result.generated_token_texts
                    ),
                    "mismatch_count": result.mismatch_count,
                    "exact_token_match": result.exact_token_match,
                    "exact_text_match": result.exact_text_match,
                    "elapsed_ms": result.elapsed_ms,
                }
            )


def _summarize(
    results: list[BatchResult],
    results_path: Path,
    summary_path: Path,
    csv_path: Path,
    graphs_dir: Path,
) -> BatchSummary:
    expected_results = [
        result
        for result in results
        if result.expected_token_ids is not None or result.expected_text is not None
    ]
    return BatchSummary(
        total_cases=len(results),
        cases_with_expectations=len(expected_results),
        exact_token_matches=sum(
            1 for result in results if result.exact_token_match is True
        ),
        exact_text_matches=sum(
            1 for result in results if result.exact_text_match is True
        ),
        total_token_mismatches=sum(result.mismatch_count for result in results),
        results_path=str(results_path),
        summary_path=str(summary_path),
        csv_path=str(csv_path),
        graphs_dir=str(graphs_dir),
    )


def _write_graphs(graphs_dir: Path, results: list[BatchResult]) -> None:
    _write_bar_chart(
        graphs_dir / "token_mismatches.png",
        "Token mismatches by case",
        "Mismatched tokens",
        [(result.case_id, result.mismatch_count) for result in results],
    )
    _write_bar_chart(
        graphs_dir / "latency_ms.png",
        "Generation latency by case",
        "Latency (ms)",
        [(result.case_id, result.elapsed_ms) for result in results],
    )


def _write_bar_chart(
    path: Path,
    title: str,
    y_label: str,
    values: list[tuple[str, float]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate batch graphs. Install project dependencies first."
        ) from exc

    labels = [label for label, _ in values]
    measurements = [value for _, value in values]
    figure_width = max(8.0, 0.55 * max(len(values), 1))

    fig, ax = plt.subplots(figsize=(figure_width, 4.8), dpi=160)
    bars = ax.bar(labels, measurements, color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Batch case")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.3g", padding=3)
    ax.tick_params(axis="x", labelrotation=35)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
