# Submission Packaging

This project can be submitted in two ways:

1. Full out-of-the-box zip with `models/model.onnx` included.
2. Smaller code-only zip where `models/model.onnx` is provided separately.

The full zip is the easiest for grading because the instructor can unzip, install requirements, and run the commands without hunting for model assets. The tradeoff is size: `models/model.onnx` is about 1.4 GB.

## Recommended Full Submission

From the project root:

```bash
python scripts/make_submission_zip.py --include-model
```

This creates:

```text
dist/PhysicsClassInference-submission.zip
```

The zip includes:

- source code
- tests
- `README.md`
- `WORKFLOW.md`
- `SUBMISSION.md`
- `requirements.txt`
- `pyproject.toml`
- `prompt.txt`
- `prompts.jsonl`
- `batch_inputs/common_pg_llm_prompts.txt`
- `models/model.onnx`

The zip excludes:

- virtual environments
- PyCharm `.idea`
- Git metadata
- test caches
- generated `batch_outputs`
- copied ONNX Runtime shared libraries

The copied `models/libonnxruntime.so` files are intentionally excluded because `onnxruntime` is installed from `requirements.txt`.

## Instructor Setup After Unzipping

Run these commands from the unzipped project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Confirm the model exists:

```bash
ls models/model.onnx
```

## Smoke Test

Run one prompt:

```bash
python -m physics_class_inference --prompt-file prompt.txt --max-tokens 5
```

Run the batch comparison:

```bash
python -m physics_class_inference.batch_cli --input-file batch_inputs/common_pg_llm_prompts.txt --max-tokens 16 --output-dir batch_outputs
```

The batch run writes:

```text
batch_outputs/results.jsonl
batch_outputs/summary.json
batch_outputs/summary.csv
batch_outputs/graphs/token_mismatches.png
batch_outputs/graphs/latency_ms.png
```

## Smaller Code-Only Submission

If the class upload limit cannot accept a 1.4 GB model file:

```bash
python scripts/make_submission_zip.py
```

Then submit the zip plus `models/model.onnx` separately, or include a note telling the instructor to place the model here:

```text
models/model.onnx
```

The project resolves that path by default.
