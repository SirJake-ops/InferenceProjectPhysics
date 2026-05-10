from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "PhysicsClassInference-submission.zip"

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "batch_outputs",
    "build",
    "dist",
    "env",
    "htmlcov",
    "inference_env",
    "venv",
}

EXCLUDED_SUFFIXES = {
    ".egg-info",
    ".pyc",
    ".pyo",
}


def should_include(path: Path, include_model: bool) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    parts = set(relative.parts)

    if parts & EXCLUDED_DIRS:
        return False
    if any(str(part).endswith(".egg-info") for part in relative.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False

    if relative.parts and relative.parts[0] == "models":
        if not include_model:
            return False
        return path.name == "model.onnx"

    return True


def build_zip(output_path: Path, include_model: bool) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == output_path:
                continue
            if not should_include(path, include_model):
                continue

            archive.write(path, path.relative_to(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean project submission zip.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output zip path.",
    )
    parser.add_argument(
        "--include-model",
        action="store_true",
        help="Bundle models/model.onnx for out-of-the-box inference. This makes the zip large.",
    )
    args = parser.parse_args()

    build_zip(args.output, args.include_model)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
