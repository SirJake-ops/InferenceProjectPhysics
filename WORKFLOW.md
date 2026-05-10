# PhysicsClassInference Workflow

This project is a local inference and comparison tool. The main goal is to run prompts through an ONNX Runtime model, capture the generated token output, compare it against expected output when available, and produce files that are easy to inspect or use in a paper.

The important distinction is that the program keeps both views of the output:

- the exact numeric token IDs, which are useful for reproducibility and debugging
- the decoded text, which is easier to read and explain

## High-Level Flow

```text
prompt source
  -> prompt text
  -> tokenizer
  -> prompt token IDs
  -> ONNX Runtime model
  -> logits
  -> next generated token ID
  -> repeat until max token count
  -> generated token IDs
  -> decoded generated text
  -> optional expected-output comparison
  -> JSONL, CSV, summary, and graphs
```

## Runtime Pieces

| Area | File | Role |
| --- | --- | --- |
| Single prompt CLI | `physics_class_inference/cli.py` | Runs one prompt and prints the result. |
| Batch CLI | `physics_class_inference/batch_cli.py` | Runs many prompts and writes analysis outputs. |
| Batch analysis | `physics_class_inference/batch.py` | Loads cases, runs generation, compares outputs, writes files and graphs. |
| Generation | `physics_class_inference/generation.py` | Handles prompt encoding, model calls, greedy token selection, and decoding. |
| ONNX wrapper | `physics_class_inference/inference.py` | Owns the ONNX Runtime session and cache state. |
| Tokenizer | `physics_class_inference/tokenizer.py` | Converts text to token IDs and token IDs back to text. |
| Config | `physics_class_inference/config.py` | Resolves model paths. |

## Single Prompt Path

Use the single prompt path when I only want to check one input:

```bash
python -m physics_class_inference --prompt-file prompt.txt --max-tokens 5
```

The sequence is:

1. `cli.py` reads the prompt from `--prompt`, `--prompt-file`, or `--stdin`.
2. It calls `generate_with_model(...)`.
3. `generation.py` builds a tokenizer and encodes the prompt into `prompt_token_ids`.
4. The model wrapper loads or reuses the ONNX Runtime session.
5. The prompt token IDs are sent through the model.
6. The model returns logits.
7. `get_next_token(...)` selects the highest-scoring token from the final vocabulary window.
8. That token is fed back into the model until the requested max token count is reached.
9. The generated token IDs are decoded into readable text.
10. The CLI prints the selected output format.

This path is mostly for quick checks and debugging.

## Batch Path

Use the batch path when I want data for comparison:

```bash
python -m physics_class_inference.batch_cli --input-file batch_inputs/common_pg_llm_prompts.txt --max-tokens 16 --output-dir batch_outputs
```

The sequence is:

1. `batch_cli.py` reads the command-line options.
2. `batch.py` loads the input cases from a text or JSONL file.
3. Each prompt is passed into the same `generate_with_model(...)` path used by the single prompt CLI.
4. The generated output is saved as both `generated_token_ids` and `generated_text`.
5. If the case includes an expected answer, the expected answer is tokenized.
6. The expected token IDs are compared against the generated token IDs by position.
7. Each mismatch records the position, expected token ID, and generated token ID.
8. The token IDs are also decoded one token at a time into readable token text arrays.
9. The batch runner writes JSONL, CSV, summary JSON, and graph files.

This is the path I expect to use for paper data because it keeps the raw token-level results and the higher-level summary in the same output folder.

## Input Formats

Plain text prompt-only lines are supported:

```text
Explain conservation of energy in one sentence.
```

Those runs are useful for latency and smoke testing, but they do not produce token mismatch counts because there is no expected output.

For text-file comparisons, use `|||` between the prompt and expected generated output:

```text
Answer with only the final number: What is 17 multiplied by 24? ||| 408
```

For JSONL comparisons with expected text:

```json
{"id": "speed-1", "prompt": "If a car travels 60 miles in 2 hours, what is its average speed?", "expected_text": "30 miles per hour"}
```

For JSONL comparisons with expected token IDs:

```json
{"id": "tokens-1", "prompt": "Answer with only the word OK.", "expected_token_ids": [79, 75]}
```

## Output Files

Batch output is written under the selected output directory.

| File | Purpose |
| --- | --- |
| `results.jsonl` | Detailed per-case output. This is the main artifact for debugging. |
| `summary.json` | Aggregate counts and paths to generated files. |
| `summary.csv` | Spreadsheet-friendly summary. |
| `graphs/token_mismatches.png` | Matplotlib bar chart of mismatch counts. |
| `graphs/latency_ms.png` | Matplotlib bar chart of generation time. |

## Important Result Fields

| Field | Meaning |
| --- | --- |
| `prompt` | Prompt sent to the model. |
| `expected_text` | Expected answer, if provided. |
| `generated_text` | Decoded model output. |
| `prompt_token_ids` | Numeric token IDs for the prompt. |
| `prompt_token_texts` | Per-token decoded prompt text. |
| `expected_token_ids` | Numeric token IDs for the expected output. |
| `expected_token_texts` | Per-token decoded expected output. |
| `generated_token_ids` | Numeric token IDs produced by the model. |
| `generated_token_texts` | Per-token decoded generated output. |
| `mismatches` | Token-level differences. |
| `mismatch_count` | Number of mismatched token positions. |
| `elapsed_ms` | Runtime for that case. |

## Comparison Rules

Comparison is exact and position-based. If the expected output is:

```text
[79, 75]
```

and the model generates:

```text
[79, 33]
```

then the result has one mismatch:

```json
{"position": 1, "expected": 75, "generated": 33}
```

If one side has more tokens than the other, the missing side is recorded as `null`.

## Why Both Token IDs and Token Texts Are Stored

The token IDs are the precise data. They are what I need for exact comparison.

The token text fields are there because raw token IDs are hard to read. For example:

```json
"generated_token_ids": [79, 75],
"generated_token_texts": ["O", "K"]
```

This makes the output easier to inspect without losing the original numeric representation.

## Graph Generation

Graphs are generated with matplotlib using the non-interactive `Agg` backend. That lets the batch job write `.png` files from PyCharm, a terminal, or a headless run without opening a plotting window.

The graph step uses the batch results already written in memory. It does not run the model a second time.
