# PhysicsClassInference

PhysicsClassInference is a local ONNX Runtime inference project for comparing generated token IDs against expected outputs. It supports one-off prompt generation and batch comparison runs that write JSONL, CSV, and matplotlib graph outputs.

## Quick Start

Run every command from the project root, the directory that contains this `README.md`.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Confirm the model exists:

```bash
ls models/model.onnx
```

4. Run a single prompt:

```bash
python -m physics_class_inference --prompt-file prompt.txt --max-tokens 5
```

5. Run the batch comparison:

```bash
python -m physics_class_inference.batch_cli --input-file batch_inputs/common_pg_llm_prompts.txt --max-tokens 16 --output-dir batch_outputs
```

6. Inspect the batch outputs:

```text
batch_outputs/results.jsonl
batch_outputs/summary.json
batch_outputs/summary.csv
batch_outputs/graphs/token_mismatches.png
batch_outputs/graphs/latency_ms.png
```

## Requirements

- Python `3.11+`
- Linux or compatible environment
- `models/model.onnx`

The submission zip may include `models/model.onnx`. If the model is submitted separately, place it at exactly:

```text
models/model.onnx
```

The `onnxruntime` Python package installed from `requirements.txt` provides the runtime library. Do not manually copy `libonnxruntime.so` files unless you are doing low-level runtime experiments outside this project.

## Dependencies

Runtime dependencies:

- `matplotlib>=3.8`
- `numpy>=1.26`
- `onnxruntime>=1.18`
- `tiktoken>=0.9`

Dev dependencies:

- `pytest>=8.0`
- `ruff>=0.5`

Install runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
```

The project also works with `python -m ...` commands directly from the project root without editable install.

## One-Prompt Runs

Prompt from a command-line string:

```bash
python -m physics_class_inference --prompt "Hello" --max-tokens 5
```

Prompt from the included sample file:

```bash
python -m physics_class_inference --prompt-file prompt.txt --max-tokens 5
```

Prompt from stdin:

```bash
printf "Hello from stdin" | python -m physics_class_inference --stdin --max-tokens 5
```

Print the full prompt plus generated text:

```bash
python -m physics_class_inference --prompt "Hello" --max-tokens 5 --format full
```

Print verbose metadata:

```bash
python -m physics_class_inference --prompt "Hello" --max-tokens 5 --format verbose --timing
```

Print JSON:

```bash
python -m physics_class_inference --prompt "Hello" --max-tokens 5 --format json --timing
```

Override the model path:

```bash
python -m physics_class_inference --prompt "Hello" --model-path models/model.onnx
```

Choose tokenizer backend:

```bash
python -m physics_class_inference --prompt "Hello" --tokenizer gpt2
python -m physics_class_inference --prompt "Hello" --tokenizer byte
```

## Batch Runs

Run the included text comparison batch:

```bash
python -m physics_class_inference.batch_cli --input-file batch_inputs/common_pg_llm_prompts.txt --max-tokens 16 --output-dir batch_outputs
```

Run the included JSONL comparison batch:

```bash
python -m physics_class_inference.batch_cli --input-file prompts.jsonl --max-tokens 5 --output-dir batch_outputs
```

Run an interactive batch from the command line:

```bash
python -m physics_class_inference.batch_cli --interactive --max-tokens 5 --output-dir batch_outputs
```

Batch output is written to the selected output directory. The default examples use `batch_outputs`.

## Batch Input Formats

Plain prompt-only text lines are accepted, but they do not produce token mismatch comparisons:

```text
Explain conservation of energy in one sentence.
```

For text-file comparisons, separate the prompt from the expected generated text with `|||`:

```text
Answer with only the final number: What is 17 multiplied by 24? ||| 408
Answer yes or no: If all squares are rectangles, and this shape is a square, is it also a rectangle? ||| yes
```

For JSONL comparisons, use one object per line with `expected_text`:

```json
{"id": "energy-1", "prompt": "Explain conservation of energy in one sentence.", "expected_text": "Energy is conserved."}
```

Or use `expected_token_ids` directly:

```json
{"id": "tokens-1", "prompt": "Answer with only the word OK.", "expected_token_ids": [79, 75]}
```

## Batch Output Fields

The batch runner writes:

- `results.jsonl`
- `summary.json`
- `summary.csv`
- `graphs/token_mismatches.png`
- `graphs/latency_ms.png`

`results.jsonl` and `summary.csv` keep numeric token IDs and readable token text arrays:

- `generated_token_ids`
- `generated_token_texts`
- `expected_token_ids`
- `expected_token_texts`
- `prompt_token_ids`
- `prompt_token_texts`

The readable token fields make numeric arrays easier to interpret. For example:

```json
"generated_token_ids": [79, 75],
"generated_token_texts": ["O", "K"]
```

## Tests

Unit tests do not require the real ONNX model:

```bash
python -m unittest tests.test_cli tests.test_generation tests.test_inference tests.test_config tests.test_tokenizer tests.test_smoke tests.test_batch tests.test_batch_cli -v
```

Integration tests require `models/model.onnx` and `onnxruntime`:

```bash
python -m unittest tests.integration.test_model_runtime -v
```

## Documentation

- [WORKFLOW.md](WORKFLOW.md): system-design view of prompt collection, tokenization, ONNX Runtime inference, decoding, comparison, and graph generation.
- [SUBMISSION.md](SUBMISSION.md): instructions for building a clean zip submission.

## Troubleshooting

`Error: onnxruntime is required for model inference. Install the project dependencies first.`

Run:

```bash
python -m pip install -r requirements.txt
```

`Model file does not exist...`

Make sure the model is present at:

```text
models/model.onnx
```

`could not read prompt file prompt.txt`

Run the command from the project root, or pass the correct path to the prompt file.

`could not read input file prompts.jsonl`

Run the command from the project root, or pass the correct path to the batch input file.

Batch graph files are missing.

Make sure `matplotlib` installed successfully and that the batch command completed without an error. The graph files are written under:

```text
batch_outputs/graphs/
```
