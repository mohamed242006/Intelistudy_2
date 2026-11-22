"""Text normalisation shared by every reader.

* Selective NFKC: compatibility characters are folded (ligatures, full-width forms, no-break
  spaces) but superscripts, subscripts, vulgar fractions and font variants (double-struck sets such
  as ℝ, mathematical letters) are kept, so ``x²``, ``H₂O`` and ``½`` survive.
* Invisible characters (zero-width, soft hyphen, BOM, bidi controls) and lone surrogates are removed.
* Hyphen repair re-joins words broken at line ends but keeps real compounds (``Aspect-Oriented``).
"""

from __future__ import annotations

import re
import unicodedata

_KEEP_TAGS = ("<super>", "<sub>", "<fraction>", "<font>")
_INVISIBLE = re.compile(
    "[­᠎​-‏‪-‮⁠-⁤⁦-⁩﻿￹-￻]"
)
_SURROGATES = re.compile("[\ud800-\udfff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t  -   　]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CJK_OR_LETTER = re.compile(r"[^\W\d_]")

# Left halves that usually form real hyphenated compounds, so "self-\ncontained" keeps its hyphen.
_COMPOUND_PREFIXES = frozenset(
    {
        "self", "non", "well", "high", "low", "long", "short", "cross", "real", "user", "client",
        "server", "object", "aspect", "service", "event", "time", "end", "run", "built", "state",
        "domain", "data", "rule", "model", "test", "open", "closed", "first", "second", "third",
        "one", "two", "three", "x", "e", "t", "peer", "case", "fine", "large",
        "small", "full", "half", "top", "bottom", "left", "right", "pre", "post", "anti", "co",
    }
)


def normalize_unicode(text: str) -> str:
    """Selective NFKC (see module docstring)."""
    if text.isascii():
        return text
    out: list[str] = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
            continue
        decomposition = unicodedata.decomposition(ch)
        if decomposition.startswith(_KEEP_TAGS):
            out.append(ch)
        else:
            out.append(unicodedata.normalize("NFKC", ch))
    return unicodedata.normalize("NFC", "".join(out))


def strip_invisible(text: str) -> str:
    """Remove zero-width/bidi/soft-hyphen characters, control characters and lone surrogates."""
    text = _SURROGATES.sub("", text)
    text = _INVISIBLE.sub("", text)
    return _CONTROL.sub("", text)


def _repair_hyphen(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if right.isupper() or right.isdigit() or left.lower() in _COMPOUND_PREFIXES or left.isdigit():
        return f"{left}-{right}"
    return f"{left}{right}"


def clean_text(text: str) -> str:
    """Normalise extracted text without destroying paragraph/heading structure."""
    if not text:
        return ""
    text = strip_invisible(text)
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n\n")
    text = _HYPHEN_BREAK.sub(_repair_hyphen, text)
    lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_NEWLINE.sub("\n\n", text).strip()


def clean_inline(text: str) -> str:
    """Clean text that must stay on one line (headings, list items, cells)."""
    return _MULTI_SPACE.sub(" ", clean_text(text).replace("\n", " ")).strip()


def has_content(text: str) -> bool:
    """True when the text contains at least one letter or digit after cleaning."""
    return any(c.isalnum() for c in strip_invisible(text))


def is_heading(line: str) -> bool:
    """Heuristic heading detector for plain text (numbered, ALL CAPS, or short title-like)."""
    s = line.strip()
    if not s or len(s) > 100 or s.endswith((".", ",", ";", ":")):
        return False
    if re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.|Chapter\s+\d+)\s+[A-Za-z]", s, re.I):
        return True
    letters = _CJK_OR_LETTER.findall(s)
    return len(letters) >= 3 and len(s.split()) <= 10 and (s.isupper() or s.istitle())


def mark_headings(text: str) -> str:
    """Prefix detected headings with ``## `` (used for OCR text, which has no font information)."""
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
    """Normalise a single table cell."""
    if cell is None:
        return ""
    return clean_inline(str(cell))
