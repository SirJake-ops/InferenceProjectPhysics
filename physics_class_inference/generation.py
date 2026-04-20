from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODEL_PATH, resolve_model_path
from .inference import ModelInference
from .tokenizer import Tokenizer

DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_VOCAB_SIZE = 50257
_MODEL_CACHE: dict[Path, ModelInference] = {}


@dataclass(slots=True)
class GenerationResult:
    prompt: str = ""
    generated_text: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    cache_layers: int = 0
    cache_sequence_length: int = 0
    tokenizer_backend: str = ""

    @property
    def response_text(self) -> str:
        return self.prompt + self.generated_text

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_token_ids)


def _get_model(model_path: str | Path | None = None) -> ModelInference:
    resolved_model_path = resolve_model_path(model_path or DEFAULT_MODEL_PATH)
    if resolved_model_path not in _MODEL_CACHE:
        _MODEL_CACHE[resolved_model_path] = ModelInference(resolved_model_path)
    return _MODEL_CACHE[resolved_model_path]


def get_next_token(logits: list[float], vocab_size: int = DEFAULT_VOCAB_SIZE) -> int:
    if vocab_size == 0 or len(logits) < vocab_size:
        raise ValueError("Logits must contain at least one full vocabulary window")

    last_token_start = len(logits) - vocab_size
    best_token = 0
    best_score = logits[last_token_start]

    for token_index in range(1, vocab_size):
        score = logits[last_token_start + token_index]
        if score > best_score:
            best_score = score
            best_token = token_index

    return best_token


def generate_with_model(
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    model_path: str | Path | None = None,
    tokenizer_backend: str = "auto",
) -> GenerationResult:
    if not prompt:
        raise ValueError("Prompt body must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    model = _get_model(model_path)
    tokenizer = Tokenizer(tokenizer_backend)
    prompt_token_ids = tokenizer.encode(prompt)
    cache_layer_count = model.get_required_cache_layer_count()

    model.reset_cache()

    logits = model.run_inference(prompt_token_ids, cache_layer_count)
    generated_token_ids: list[int] = []

    for token_index in range(max_new_tokens):
        next_token = int(get_next_token(logits))
        generated_token_ids.append(next_token)

        if token_index + 1 == max_new_tokens:
            break

        logits = model.run_inference([next_token], cache_layer_count)

    return GenerationResult(
        prompt=prompt,
        generated_text=tokenizer.decode(generated_token_ids),
        prompt_token_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        cache_layers=model.get_required_cache_layer_count(),
        cache_sequence_length=model.get_cached_sequence_length(),
        tokenizer_backend=tokenizer.backend_name,
    )


def generate(
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tokenizer_backend: str = "auto",
) -> GenerationResult:
    return generate_with_model(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        tokenizer_backend=tokenizer_backend,
    )
