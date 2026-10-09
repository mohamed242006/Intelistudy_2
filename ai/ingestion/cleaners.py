"""Text cleaning helpers."""

from __future__ import annotations

import re
import unicodedata

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalize extracted text without destroying paragraph/heading structure."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def is_heading(line: str) -> bool:
    """Heuristic heading detector (numbered, ALL CAPS, or short title-like)."""
    s = line.strip()
    if not s or len(s) > 100 or s.endswith((".", ",", ";")):
        return False
    if re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.|Chapter\s+\d+)\s+\S+", s, re.I):
        return True
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and len(s.split()) <= 10 and (s.isupper() or s.istitle())


def mark_headings(text: str) -> str:
    """Prefix detected headings with ``## `` so downstream chunkers can split on them."""
    out: list[str] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i == len(lines) - 1 or not lines[i + 1].strip()
        if prev_blank and next_blank and is_heading(line):
            out.append(f"## {line.strip()}")
        else:
            out.append(line)
    return "\n".join(out)


def cell_clean(cell: object) -> str:
    """Normalize a single table cell."""
    if cell is None:
        return ""
    return _MULTI_SPACE.sub(" ", str(cell).replace("\n", " ")).strip()
