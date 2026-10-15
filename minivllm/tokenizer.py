"""Tokenizer abstraction: HuggingFace tokenizers when available, a byte-level
fallback for tests and tiny random models."""
from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    eos_token_id: int
    bos_token_id: int

    def encode(self, text: str, add_bos: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """UTF-8 bytes shifted by 2 (0 = EOS, 1 = BOS); vocab size 258."""

    vocab_size = 258
    eos_token_id = 0
    bos_token_id = 1

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [b + 2 for b in text.encode("utf-8")]
        return [self.bos_token_id] + ids if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        return bytes(i - 2 for i in ids if i >= 2).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, path: str):
        from transformers import AutoTokenizer  # optional dependency

        self.tok = AutoTokenizer.from_pretrained(path)
        self.eos_token_id = self.tok.eos_token_id
        self.bos_token_id = self.tok.bos_token_id

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_bos)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
