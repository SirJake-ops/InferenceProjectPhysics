from __future__ import annotations

from typing import Any


def _import_tiktoken() -> Any:
    try:
        import tiktoken
    except ImportError:
        return None
    return tiktoken


class Tokenizer:
    """Tokenizer with GPT-2 BPE by default and byte fallback."""

    def __init__(self, backend: str = "auto") -> None:
        if backend not in {"auto", "gpt2", "byte"}:
            raise ValueError("Tokenizer backend must be one of: auto, gpt2, byte")

        self.requested_backend = backend
        self._encoding = None
        self.backend_name = "byte"

        if backend in {"auto", "gpt2"}:
            tiktoken = _import_tiktoken()
            if tiktoken is None:
                if backend == "gpt2":
                    raise RuntimeError(
                        "tiktoken is required for the gpt2 tokenizer backend. Install project dependencies first."
                    )
            else:
                self._encoding = tiktoken.get_encoding("gpt2")
                self.backend_name = "gpt2"
                return

        self.backend_name = "byte"

    def encode(self, input_text: str) -> list[int]:
        if self._encoding is not None:
            return list(self._encoding.encode(input_text))
        return list(input_text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        if self._encoding is not None:
            output: list[str] = []
            for token_id in token_ids:
                try:
                    output.append(self._encoding.decode([token_id]))
                except Exception:
                    output.append(f"<tok:{token_id}>")
            return "".join(output)

        output: list[str] = []
        for token_id in token_ids:
            if 0 <= token_id <= 255:
                output.append(chr(token_id))
            else:
                output.append(f"<tok:{token_id}>")
        return "".join(output)
