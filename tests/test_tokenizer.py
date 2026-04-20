from __future__ import annotations

import unittest
from unittest.mock import patch

from physics_class_inference.tokenizer import Tokenizer
from physics_class_inference.tokenizer import _import_tiktoken


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer("byte")

    def test_encode_returns_byte_values_for_ascii(self) -> None:
        self.assertEqual(self.tokenizer.encode("Hi"), [72, 105])

    def test_encode_uses_utf8_bytes_for_non_ascii_text(self) -> None:
        self.assertEqual(self.tokenizer.encode("é"), [195, 169])

    def test_decode_returns_text_for_byte_range_values(self) -> None:
        self.assertEqual(self.tokenizer.decode([79, 75]), "OK")

    def test_decode_formats_out_of_range_tokens(self) -> None:
        self.assertEqual(self.tokenizer.decode([65, 256, -1]), "A<tok:256><tok:-1>")

    @unittest.skipUnless(_import_tiktoken() is not None, "tiktoken is not installed")
    def test_gpt2_backend_can_encode_and_decode_when_available(self) -> None:
        tokenizer = Tokenizer("gpt2")
        token_ids = tokenizer.encode("Hello")

        self.assertTrue(token_ids)
        self.assertEqual(tokenizer.backend_name, "gpt2")
        self.assertEqual(tokenizer.decode(token_ids), "Hello")

    def test_auto_backend_falls_back_to_byte_without_tiktoken(self) -> None:
        with patch("physics_class_inference.tokenizer._import_tiktoken", return_value=None):
            tokenizer = Tokenizer()

        self.assertEqual(tokenizer.backend_name, "byte")
        self.assertEqual(tokenizer.encode("Hi"), [72, 105])

    @unittest.skipUnless(_import_tiktoken() is not None, "tiktoken is not installed")
    def test_auto_backend_prefers_gpt2_when_available(self) -> None:
        tokenizer = Tokenizer()

        self.assertEqual(tokenizer.backend_name, "gpt2")


if __name__ == "__main__":
    unittest.main()
