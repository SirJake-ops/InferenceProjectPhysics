from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "model.onnx"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    candidate = Path(model_path) if model_path is not None else None

    if candidate is not None and candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate is not None:
        from_cwd = (Path.cwd() / candidate).resolve()
        if from_cwd.exists():
            return from_cwd

        from_project_root = PROJECT_ROOT / candidate
        if from_project_root.exists():
            return from_project_root

        from_models_dir = DEFAULT_MODELS_DIR / candidate.name
        if from_models_dir.exists():
            return from_models_dir

        raise RuntimeError(f"Model file does not exist: {candidate}")

    if DEFAULT_MODELS_DIR.exists():
        for entry in DEFAULT_MODELS_DIR.iterdir():
            if entry.is_file() and entry.suffix == ".onnx":
                return entry

    raise RuntimeError(
        f"Model file does not exist and no .onnx file was found in {DEFAULT_MODELS_DIR}"
    )
