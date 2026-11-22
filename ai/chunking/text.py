"""Token-aware text splitting: sentences -> words -> characters, with tail overlap."""

from __future__ import annotations

import hashlib
import re

from ai.chunking.tokens import TokenCounter, char_budget

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=\S)|\n+")


def sha_hex(text: str, length: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:length]


def split_sentences(text: str) -> list[str]:
    """Sentences (and lines) of ``text``; never returns empty strings."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _split_chars(text: str, budget: int, counter: TokenCounter) -> list[str]:
    """Split an unbreakable string (URL, base64, long formula) into pieces of <= budget tokens."""
    pieces: list[str] = []
    rest = text
    while rest:
        size = min(len(rest), char_budget(budget))
        while size > 1 and counter.count(rest[:size]) > budget:
            size = max(1, int(size * 0.8))
        pieces.append(rest[:size])
        rest = rest[size:]
    return pieces


def _pack(parts: list[str], budget: int, counter: TokenCounter, sep: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for part in parts:
        candidate = sep.join([*current, part])
        if current and counter.count(candidate) > budget:
            out.append(sep.join(current))
            current = [part]
        else:
            current.append(part)
    if current:
        out.append(sep.join(current))
    return out


def split_to_fit(text: str, budget: int, counter: TokenCounter) -> list[str]:
    """Split ``text`` into pieces of at most ``budget`` tokens, preferring sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if counter.count(text) <= budget:
        return [text]
    units: list[str] = []
    for sentence in split_sentences(text):
        if counter.count(sentence) <= budget:
            units.append(sentence)
            continue
        for word in sentence.split():
            if counter.count(word) <= budget:
                units.append(word)
            else:
                units.extend(_split_chars(word, budget, counter))
    # Pack sentences with a space, which also re-joins words of an over-long sentence.
    return [p for p in _pack(units, budget, counter, " ") if p.strip()]


def tail_text(text: str, max_tokens: int, counter: TokenCounter) -> str:
    """The last whole sentences of ``text`` totalling at most ``max_tokens`` (words if needed)."""
    if max_tokens <= 0 or not text.strip():
        return ""
    sentences = split_sentences(text)
    picked: list[str] = []
    for sentence in reversed(sentences):
        candidate = " ".join([sentence, *picked])
        if counter.count(candidate) > max_tokens:
            break
        picked.insert(0, sentence)
    if picked:
        if len(picked) == len(sentences):
            return ""  # the whole text would be repeated: no useful overlap
        return " ".join(picked)
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        if counter.count(" ".join([word, *tail])) > max_tokens:
            break
        tail.insert(0, word)
    return " ".join(tail) if len(tail) < len(words) else ""


def window_split(text: str, budget: int, overlap: int, counter: TokenCounter) -> list[tuple[str, str, int]]:
    """Split long text into windows of <= ``budget`` tokens; each window after the first starts
    with up to ``overlap`` tokens from the end of the previous one.

    Returns ``(overlap_text, new_text, overlap_tokens)`` triples; the window is
    ``overlap_text + " " + new_text``.
    """
    base = split_to_fit(text, max(budget - overlap, budget // 2), counter)
    windows: list[tuple[str, str, int]] = []
    previous = ""
    for piece in base:
        tail = tail_text(previous, overlap, counter) if previous else ""
        if tail and counter.count(f"{tail}\n{piece}") > budget:
            tail = ""
        windows.append((tail, piece, counter.count(tail) if tail else 0))
        previous = piece
    return windows
