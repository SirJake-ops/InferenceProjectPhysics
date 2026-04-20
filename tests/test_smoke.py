from __future__ import annotations

import physics_class_inference


def test_package_exports_are_importable() -> None:
    assert physics_class_inference.Tokenizer is not None
    assert physics_class_inference.ModelInference is not None
    assert physics_class_inference.GenerationResult is not None
