# PhysicsClassInference

## Requirements

- Python `3.11+`
- Linux or compatible environment
- Local model assets under `models/`

Expected local files:

- `models/model.onnx`
- `models/libonnxruntime.so`
- `models/libonnxruntime.so.1`

## Dependencies

Runtime dependencies:

- `numpy>=1.26`
- `onnxruntime>=1.18`
- `tiktoken>=0.9`

Dev dependencies:

- `pytest>=8.0`
- `ruff>=0.5`

You can install from either `pyproject.toml` or `requirements.txt`.

## Build / Install

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the project in editable mode:

```bash
python -m pip install -e .
```

Or install the runtime dependencies directly:

```bash
python -m pip install -r requirements.txt
```

## Run

Basic text generation:

```bash
python -m physics_class_inference --prompt "Hello" --max-tokens 5
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

Read the prompt from stdin:

```bash
printf "Hello from stdin" | python -m physics_class_inference --stdin --max-tokens 5
```

Read the prompt from a file:

```bash
python -m physics_class_inference --prompt-file prompt.txt --max-tokens 5
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

## CLI Options

- `--prompt`
- `--prompt-file`
- `--stdin`
- `--max-tokens`
- `--model-path`
- `--tokenizer auto|gpt2|byte`
- `--format text|full|json|verbose`
- `--timing`

Legacy aliases still accepted:

- `--json`
- `--verbose`

## Test

Unit tests:

```bash
python -m unittest tests.test_cli tests.test_generation tests.test_inference tests.test_config tests.test_tokenizer tests.test_smoke -v
```

Integration tests:

```bash
python -m unittest tests.integration.test_model_runtime -v
```

Full suite:

```bash
python -m unittest tests.test_cli tests.test_generation tests.test_inference tests.test_config tests.test_tokenizer tests.test_smoke tests.integration.test_model_runtime -v
```
