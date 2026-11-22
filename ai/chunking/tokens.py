"""Token counting.

:class:`HFTokenCounter` uses the real BGE WordPiece tokenizer (``tokenizer.json`` via the
``tokenizers`` library; no torch import). :class:`ApproxTokenCounter` is a conservative fallback
used when the tokenizer files are not available offline.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Protocol, runtime_checkable

from ai.chunking.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_WORDS = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


@runtime_checkable
class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        """Number of tokens in ``text`` (special tokens excluded)."""
        ...


class ApproxTokenCounter:
    """WordPiece-like estimate: digits and symbols 1 token each, words ~1 token per 6 letters.

    Deliberately pessimistic (it over-counts) so a chunk that fits here fits the real tokenizer.
    """

    name = "approx"

    def count(self, text: str) -> int:
        total = 0
        for piece in _WORDS.findall(text):
            if piece[0].isalpha():
                total += 1 + (len(piece) - 1) // 6 if piece.isascii() else len(piece)
            else:
                total += 1
        return total


class HFTokenCounter:
    """Exact counts with a Hugging Face ``tokenizers`` tokenizer."""

    def __init__(self, tokenizer: object, name: str) -> None:
        self._tokenizer = tokenizer
        self.name = name
        self._cache: dict[str, int] = {}

    @classmethod
    def from_pretrained(cls, model: str = EMBEDDING_MODEL, local_files_only: bool = True) -> "HFTokenCounter":
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(model, "tokenizer.json", local_files_only=local_files_only)
        tokenizer = Tokenizer.from_file(path)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return cls(tokenizer, model)

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        n = len(self._tokenizer.encode(text, add_special_tokens=False).ids)  # type: ignore[attr-defined]
        if len(self._cache) < 200_000:
            self._cache[text] = n
        return n


@lru_cache(maxsize=4)
def default_token_counter(model: str = EMBEDDING_MODEL) -> TokenCounter:
    """The real tokenizer when it is cached locally (or downloadable), else :class:`ApproxTokenCounter`."""
    for local_only in (True, False):
        try:
            return HFTokenCounter.from_pretrained(model, local_files_only=local_only)
        except Exception as exc:  # offline, missing package, ...
            logger.debug("Tokenizer for %s unavailable (local_only=%s): %s", model, local_only, exc)
    logger.warning("Using approximate token counts (tokenizer for %s not available)", model)
    return ApproxTokenCounter()


def char_budget(tokens: int) -> int:
    """A safe character length for ``tokens`` tokens (used when splitting unbreakable strings)."""
    return max(1, math.floor(tokens * 2.5))
