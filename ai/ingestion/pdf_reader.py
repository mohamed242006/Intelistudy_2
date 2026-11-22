"""Layer 1 - PyMuPDF parser: text layer -> canonical blocks.

Per page the reader produces heading / paragraph / list-item blocks in reading order:

* visual-row merging (label/value layouts such as ``Course Work ... 35``) - only when one side is a
  single-line block, so two-column papers are not interleaved;
* bullet detection (glyph spans, private-use Wingdings/Symbol glyphs, ``o Item``, ``- Item``,
  numbered markers) with wrapped-bullet re-joining and nesting depth from indentation;
* headings: font size >= 1.15 x body size, or a bold line alone on its row; <= 100 characters, not
  ending in ``.,;``, never a bullet, never inside a block of 4+ lines; wrapped headings merged;
  levels from a document-wide :class:`LayoutProfile` (a size must recur on >= 2 pages, except in
  1-2 page documents); PDF bookmarks override levels and promote missed titles;
* running header/footer removal (top/bottom 18% of the page, repeated on >= max(3, 50%) of pages),
  table-of-contents rows dropped, rotated margin text (arXiv stamps) ignored.
"""

from __future__ import annotations

import io
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pymupdf as fitz
from PIL import Image

from ai.ingestion.cleaners import clean_inline, clean_text, has_content
from ai.ingestion.models import (
    BBox,
    Block,
    BlockType,
    ImageData,
    ImageExtractionError,
    PageContent,
    PDFParseError,
)
from ai.ingestion.structure import render_blocks

logger = logging.getLogger(__name__)

BULLET_GLYPHS = frozenset("•●○◦▪▫■□►▶▸▹➢➣➤➔→✓✔✗❖◆◇⦿⁃∙·")
SYMBOL_FONTS = ("wingdings", "symbol", "webdings", "courier", "zapf", "dingbat")
SYMBOL_LETTER_BULLETS = frozenset("oqnvlüØ§Ÿ")
_TEXT_BULLET = re.compile(r"^([•●○◦▪▫■□►▶▸▹➢➣➤➔✓✔❖◆◇⦿⁃∙-])\s*(\S.*)$")
_DASH_BULLET = re.compile(r"^([-–—*])\s+(\S.*)$")
_O_BULLET = re.compile(r"^o\s+([A-Z(\"'].*)$")
_NUMBERED = re.compile(r"^(\(?\d{1,2}[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,4}[.)])\s+(\S.*)$")
_TOC_ROW = re.compile(r"((\.\s?){4,}|…{2,}|_{4,})\s*\d+\s*$")
_INLINE_BULLET = re.compile(r"\S\s*[•●▪■►➢]\s*\S")
_RUN_IN_BULLETS = re.compile(r"\s+[•●▪■►➢]\s+")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")

MARGIN_RATIO = 0.18


def _is_private_use(ch: str) -> bool:
    return len(ch) == 1 and 0xE000 <= ord(ch) <= 0xF8FF


def _norm_key(text: str) -> str:
    return _SPACES.sub(" ", _DIGITS.sub("#", text.lower())).strip()


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ---------------------------------------------------------------- raw lines
@dataclass(slots=True)
class Line:
    """One visual text line of a PDF page."""

    text: str
    bbox: BBox
    size: float
    bold: bool
    bullet: str
    block_no: int
    block_lines: int
    alone_on_row: bool = True
    numbered: bool = False

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)


def _span_text(spans: list[dict[str, Any]]) -> str:
    return "".join(s.get("text", "") for s in spans)


def _detect_bullet(spans: list[dict[str, Any]]) -> tuple[str, str, bool]:
    """Return ``(bullet, text_without_bullet, numbered)``; bullet is ``""`` when none."""
    first = spans[0]
    first_text = first.get("text", "").strip()
    font = str(first.get("font", "")).lower()
    rest = _span_text(spans[1:]).strip()
    if not rest and len(first_text) == 1 and (
        first_text in BULLET_GLYPHS or _is_private_use(first_text)
        or (first_text in SYMBOL_LETTER_BULLETS and any(f in font for f in SYMBOL_FONTS))
    ):
        return "•", "", False  # a bullet glyph on its own line; attached to its text by _attach_bullets
    if len(first_text) == 1 and rest and (
        first_text in BULLET_GLYPHS
        or _is_private_use(first_text)
        or (first_text in SYMBOL_LETTER_BULLETS and any(f in font for f in SYMBOL_FONTS))
    ):
        return "•", rest, False
    text = _span_text(spans).strip()
    for pattern in (_TEXT_BULLET, _DASH_BULLET):
        m = pattern.match(text)
        if m:
            return "•", m.group(2).strip(), False
    m = _O_BULLET.match(text)
    if m:
        return "•", m.group(1).strip(), False
    m = _NUMBERED.match(text)
    if m:
        return m.group(1), m.group(2).strip(), True
    return "", text, False


def _raw_lines(page: fitz.Page, exclude: Iterable[BBox] = ()) -> list[Line]:
    """Extract horizontal text lines (rotated text and lines inside ``exclude`` are skipped)."""
    excluded = list(exclude)
    data = page.get_text("dict", sort=True)
    lines: list[Line] = []
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        x0, y0, x1, y1 = block["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3] for ex in excluded):
            continue
        block_lines = [ln for ln in block.get("lines", []) if _span_text(ln.get("spans", [])).strip()]
        for ln in block_lines:
            direction = ln.get("dir", (1.0, 0.0))
            if abs(direction[1]) > 0.1 or direction[0] < 0:
                continue  # rotated margin stamps, vertical labels
            spans = [s for s in ln.get("spans", []) if s.get("text", "")]
            visible = [s for s in spans if s["text"].strip()]
            if not visible:
                continue
            # Leading whitespace-only spans would hide a bullet glyph.
            while spans and not spans[0]["text"].strip():
                spans.pop(0)
            bullet, text, numbered = _detect_bullet(spans)
            text = clean_inline(text)
            if not text and not bullet:
                continue
            lines.append(
                Line(
                    text=text,
                    bbox=tuple(ln["bbox"]),  # type: ignore[arg-type]
                    size=round(max(s.get("size", 0.0) for s in visible), 1),
                    bold=all((s.get("flags", 0) & 16) or "bold" in str(s.get("font", "")).lower() for s in visible),
                    bullet=bullet,
                    block_no=block_no,
                    block_lines=len(block_lines),
                    numbered=numbered,
                )
            )
    lines = _attach_bullets(lines)
    _mark_rows(lines)
    return lines


def _attach_bullets(lines: list[Line]) -> list[Line]:
    """Give a lone bullet glyph to the nearest text line on its right in the same row."""
    dropped: set[int] = set()
    for i, glyph in enumerate(lines):
        if not glyph.bullet or glyph.text:
            continue
        candidates = [
            (other.bbox[0], j) for j, other in enumerate(lines)
            if j != i and j not in dropped and other.text and not other.bullet
            and other.bbox[0] >= glyph.bbox[0] and _same_row(glyph, other)
        ]
        if candidates:
            _, j = min(candidates)
            target = lines[j]
            target.bullet = glyph.bullet
            target.bbox = (glyph.bbox[0], min(glyph.bbox[1], target.bbox[1]), target.bbox[2],
                           max(glyph.bbox[3], target.bbox[3]))
        dropped.add(i)
    return [ln for i, ln in enumerate(lines) if i not in dropped]


def _same_row(a: Line, b: Line) -> bool:
    overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    return overlap >= 0.5 * min(a.height, b.height) and (a.bbox[2] <= b.bbox[0] + 2 or b.bbox[2] <= a.bbox[0] + 2)


def _mark_rows(lines: list[Line]) -> None:
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            if _same_row(a, b):
                a.alone_on_row = b.alone_on_row = False


def _merge_rows(lines: list[Line], page_width: float = 0.0) -> list[Line]:
    """Merge label/value fragments that share a visual row.

    Allowed when one side is a single-line block, or when the fragments are far apart (>= 10% of
    the page width) and one is short - never for two multi-line blocks close together, which is
    what two-column body text looks like.
    """
    consumed: set[int] = set()
    out: list[Line] = []
    for i, line in enumerate(lines):
        if i in consumed:
            continue
        group = [i]
        for j in range(i + 1, len(lines)):
            other = lines[j]
            if j in consumed or other.bullet:
                continue
            if not any(_same_row(lines[g], other) for g in group):
                continue
            if line.block_lines > 1 and other.block_lines > 1:
                gap = max(other.bbox[0] - line.bbox[2], line.bbox[0] - other.bbox[2])
                short = min(len(line.text), len(other.text)) <= 40
                if not (page_width and gap >= 0.10 * page_width and short):
                    continue
            if abs(other.size - line.size) > 0.35 * max(line.size, other.size):
                continue
            group.append(j)
        if len(group) == 1:
            out.append(line)
            continue
        consumed.update(group)
        members = sorted((lines[g] for g in group), key=lambda ln: ln.bbox[0])
        lead = members[0]
        bullet = lead.bullet if lead is line or not line.bullet else line.bullet
        text = " ".join(m.text for m in members if m.text)
        bbox = (
            min(m.bbox[0] for m in members), min(m.bbox[1] for m in members),
            max(m.bbox[2] for m in members), max(m.bbox[3] for m in members),
        )
        out.append(Line(text=text, bbox=bbox, size=max(m.size for m in members), bold=all(m.bold for m in members),
                        bullet=bullet, block_no=lead.block_no, block_lines=max(m.block_lines for m in members),
                        alone_on_row=False, numbered=lead.numbered))
    return out


# ------------------------------------------------------------------ profile
@dataclass(slots=True)
class LayoutProfile:
    """Document-wide layout facts collected in a first pass."""

    body_size: float = 0.0
    heading_levels: dict[float, int] = field(default_factory=dict)
    repeated_keys: set[str] = field(default_factory=set)
    repeated_xrefs: set[int] = field(default_factory=set)
    toc: dict[str, int] = field(default_factory=dict)
    page_count: int = 0

    @property
    def max_level(self) -> int:
        return max(self.heading_levels.values(), default=0)

    def level_for(self, size: float) -> int:
        """Level of a heading-sized font: the smallest recurring heading size >= ``size``."""
        if not self.heading_levels:
            return 1
        bigger = [s for s in self.heading_levels if s >= size - 0.5]
        if not bigger:
            return self.max_level + 1 if size < min(self.heading_levels) else 1
        return self.heading_levels[min(bigger)]

    def is_running_text(self, line: Line, page_height: float, heading_ratio: float = 1.15) -> bool:
        if not self.repeated_keys:
            return False
        if self.body_size and line.size >= self.body_size * heading_ratio:
            return False  # repeated titles ("Chapter 3 ...") are content, not running headers
        in_margin = line.bbox[3] <= page_height * MARGIN_RATIO or line.bbox[1] >= page_height * (1 - MARGIN_RATIO)
        full = f"{line.bullet} {line.text}" if line.numbered else line.text
        return in_margin and _norm_key(full) in self.repeated_keys


class PDFReader:
    """Opens PDFs and extracts canonical blocks, images and metadata with PyMuPDF.

    Args:
        min_text_chars: Pages with fewer extractable characters are flagged as needing OCR.
        heading_size_ratio: A line is heading-sized at this multiple of the body font size.
    """

    def __init__(self, min_text_chars: int = 50, heading_size_ratio: float = 1.15) -> None:
        self.min_text_chars = min_text_chars
        self.heading_size_ratio = heading_size_ratio
        self.profile = LayoutProfile()

    # ------------------------------------------------------------------ open
    def open(self, path: str | Path) -> fitz.Document:
        """Open a PDF, raising :class:`PDFParseError` on failure."""
        p = Path(path)
        if not p.is_file():
            raise PDFParseError(f"File not found: {p}")
        try:
            doc = fitz.open(p)
        except Exception as exc:  # fitz raises several error types
            raise PDFParseError(f"Cannot open PDF '{p}': {exc}") from exc
        if doc.needs_pass:
            doc.close()
            raise PDFParseError(f"PDF is password protected: {p}")
        if not doc.is_pdf:
            doc.close()
            raise PDFParseError(f"Not a PDF file: {p}")
        logger.info("Opened %s (%d pages)", p.name, doc.page_count)
        return doc

    @staticmethod
    def extract_metadata(doc: fitz.Document) -> dict[str, object]:
        """Return document-level metadata."""
        meta: dict[str, object] = {k: v for k, v in (doc.metadata or {}).items() if v}
        meta["page_count"] = doc.page_count
        return meta

    # --------------------------------------------------------------- profile
    def build_profile(self, doc: fitz.Document) -> LayoutProfile:
        """First pass over the document: body size, heading levels, running text, logos, bookmarks."""
        n = doc.page_count
        size_chars: Counter[float] = Counter()
        page_lines: list[tuple[float, list[Line]]] = []
        xref_pages: defaultdict[int, set[int]] = defaultdict(set)
        for page in doc:
            lines = _raw_lines(page)
            page_lines.append((page.rect.height, lines))
            for ln in lines:
                size_chars[round(ln.size * 2) / 2] += len(ln.text)
            try:
                for img in page.get_images(full=True):
                    xref_pages[img[0]].add(page.number)
            except Exception:  # pragma: no cover - damaged image tables
                pass

        profile = LayoutProfile(page_count=n)
        profile.body_size = size_chars.most_common(1)[0][0] if size_chars else 0.0

        threshold = max(3, math.ceil(0.5 * n))
        if n >= 3:
            key_pages: defaultdict[str, set[int]] = defaultdict(set)
            for page_no, (height, lines) in enumerate(page_lines):
                for ln in lines:
                    if profile.body_size and ln.size >= profile.body_size * self.heading_size_ratio:
                        continue
                    if ln.bbox[3] <= height * MARGIN_RATIO or ln.bbox[1] >= height * (1 - MARGIN_RATIO):
                        full = f"{ln.bullet} {ln.text}" if ln.numbered else ln.text
                        key_pages[_norm_key(full)].add(page_no)
            profile.repeated_keys = {k for k, pages in key_pages.items() if k and len(pages) >= threshold}
            profile.repeated_xrefs = {x for x, pages in xref_pages.items() if len(pages) >= threshold}

        size_pages: defaultdict[float, set[int]] = defaultdict(set)
        for page_no, (height, lines) in enumerate(page_lines):
            for ln in lines:
                if self._heading_sized(ln, profile.body_size) and self._heading_shape(ln):
                    in_margin = ln.bbox[1] >= height * (1 - MARGIN_RATIO)
                    if not in_margin:
                        size_pages[round(ln.size)].add(page_no)
        min_pages = 1 if n <= 2 else 2
        recurring = sorted((s for s, pages in size_pages.items() if len(pages) >= min_pages), reverse=True)
        level = 0
        previous: float | None = None
        for size in recurring:
            if previous is None or previous - size > 1.0:
                level += 1
            profile.heading_levels[size] = min(level, 6)
            previous = size

        try:
            toc = doc.get_toc(simple=True)
        except Exception:  # pragma: no cover
            toc = []
        for lvl, title, _page in toc:
            key = _norm_title(str(title))
            if key and key not in profile.toc:
                profile.toc[key] = max(int(lvl), 1)
        self.profile = profile
        logger.debug("Layout profile: body=%.1f levels=%s running=%d toc=%d", profile.body_size,
                     profile.heading_levels, len(profile.repeated_keys), len(profile.toc))
        return profile

    # -------------------------------------------------------------- headings
    def _heading_sized(self, line: Line, body: float) -> bool:
        return body > 0 and line.size >= body * self.heading_size_ratio

    @staticmethod
    def _heading_shape(line: Line) -> bool:
        text = line.text
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        return (
            not (line.bullet and not line.numbered)
            and 2 <= len(text) <= 100
            and letters >= 2
            and letters >= digits
            and not text.endswith((".", ",", ";"))
            and not _INLINE_BULLET.search(text)
            and line.block_lines <= 3
        )

    def _heading_level(self, line: Line) -> int | None:
        """Heading level for a line, or ``None`` when it is body text."""
        profile = self.profile
        text = f"{line.bullet} {line.text}" if line.numbered else line.text
        toc_level = profile.toc.get(_norm_title(text)) if profile.toc else None
        if toc_level is not None and len(text) <= 120:
            return toc_level
        if not self._heading_shape(line):
            return None
        if self._heading_sized(line, profile.body_size):
            return profile.level_for(round(line.size))
        letters = sum(c.isalpha() for c in line.text)
        if line.bold and line.alone_on_row and line.block_lines == 1 and len(text) <= 80 and letters >= 3:
            return profile.max_level + 1
        return None

    # ---------------------------------------------------------------- blocks
    def extract_blocks(self, page: fitz.Page, exclude_bboxes: Iterable[BBox] = ()) -> list[Block]:
        """Heading / paragraph / list-item blocks of one page, in reading order."""
        number = page.number + 1
        height = page.rect.height
        lines = [
            ln for ln in _merge_rows(_raw_lines(page, exclude_bboxes), page.rect.width)
            if not self.profile.is_running_text(ln, height, self.heading_size_ratio) and not _TOC_ROW.search(ln.text)
        ]
        bullet_xs = sorted({round(ln.bbox[0] / 8) for ln in lines if ln.bullet})

        blocks: list[Block] = []
        current: dict[str, Any] | None = None

        def flush() -> None:
            nonlocal current
            if current is None:
                return
            text = clean_text("\n".join(current["texts"])).replace("\n", " ")
            bbox = current["bbox"]
            parts = _RUN_IN_BULLETS.split(text) if current["type"] is not BlockType.HEADING else [text]
            if len(parts) >= 3:  # run-in bullets: "intro • item • item"
                head, items = parts[0], parts[1:]
                if has_content(head):
                    blocks.append(Block(type=current["type"], text=head, page_number=number,
                                        level=current["level"], list_marker=current.get("marker", ""), bbox=bbox))
                depth = current["level"] + 1 if current["type"] is BlockType.LIST_ITEM else 0
                for item in items:
                    if has_content(item):
                        blocks.append(Block(BlockType.LIST_ITEM, item, number, level=depth, list_marker="•",
                                            bbox=bbox))
            elif has_content(text):
                blocks.append(Block(type=current["type"], text=text, page_number=number,
                                    level=current["level"], list_marker=current.get("marker", ""),
                                    bbox=bbox))
            current = None

        for ln in lines:
            prev_bottom = current["bbox"][3] if current else 0.0
            gap = ln.bbox[1] - prev_bottom
            level = self._heading_level(ln)
            glyph_bullet = bool(ln.bullet) and not ln.numbered

            if glyph_bullet or (ln.numbered and level is None):
                flush()
                depth = bullet_xs.index(round(ln.bbox[0] / 8)) if ln.bullet and round(ln.bbox[0] / 8) in bullet_xs else 0
                current = {"type": BlockType.LIST_ITEM, "texts": [ln.text], "level": depth,
                           "marker": ln.bullet if ln.numbered else "•", "bbox": ln.bbox,
                           "x0": ln.bbox[0], "size": ln.size, "block": ln.block_no}
                continue

            if (current and current["type"] is BlockType.LIST_ITEM and level is None
                    and ln.bbox[0] >= current["x0"] + 2 and gap <= 0.9 * ln.height
                    and abs(ln.size - current["size"]) <= 2.5):
                self._extend(current, ln)
                continue

            if level is not None:
                if (current and current["type"] is BlockType.HEADING and abs(ln.size - current["size"]) <= 0.6
                        and gap <= 0.8 * ln.height and len(current["texts"]) < 3 and current["level"] == level):
                    self._extend(current, ln)
                    continue
                flush()
                text = f"{ln.bullet} {ln.text}" if ln.numbered else ln.text
                current = {"type": BlockType.HEADING, "texts": [text], "level": level, "bbox": ln.bbox,
                           "x0": ln.bbox[0], "size": ln.size, "block": ln.block_no}
                continue

            if (current and current["type"] is BlockType.PARAGRAPH
                    and (ln.block_no == current["block"]
                         or (gap <= 0.9 * ln.height and abs(ln.bbox[0] - current["x0"]) <= 30
                             and abs(ln.size - current["size"]) <= 1.5))):
                self._extend(current, ln)
                continue

            flush()
            current = {"type": BlockType.PARAGRAPH, "texts": [ln.text], "level": 0, "bbox": ln.bbox,
                       "x0": ln.bbox[0], "size": ln.size, "block": ln.block_no}
        flush()
        return blocks

    @staticmethod
    def _extend(current: dict[str, Any], ln: Line) -> None:
        current["texts"].append(ln.text)
        b = current["bbox"]
        current["bbox"] = (min(b[0], ln.bbox[0]), min(b[1], ln.bbox[1]), max(b[2], ln.bbox[2]), max(b[3], ln.bbox[3]))
        current["block"] = ln.block_no

    # --------------------------------------------------------------- analyze
    def analyze_page(self, page: fitz.Page) -> PageContent:
        """Extract blocks/images for one page and compute detection metadata."""
        number = page.number + 1
        blocks = self.extract_blocks(page)
        images = self.extract_images(page)
        content = PageContent(page_number=number, blocks=blocks, images=images,
                              size=(page.rect.width, page.rect.height))
        content.text = render_blocks(blocks)
        content.image_count = len(images)
        content.text_length = len(content.text)
        content.has_text = content.text_length >= self.min_text_chars
        content.needs_ocr = not content.has_text
        return content

    def extract_text(self, page: fitz.Page, exclude_bboxes: Iterable[BBox] = ()) -> str:
        """Readable text of the page (headings marked with ``#``), kept for compatibility."""
        return render_blocks(self.extract_blocks(page, exclude_bboxes))

    # ---------------------------------------------------------------- images
    @staticmethod
    def extract_images(page: fitz.Page) -> list[ImageData]:
        """List images embedded on the page (metadata only, no pixel data)."""
        result: list[ImageData] = []
        for idx, img in enumerate(page.get_images(full=True)):
            xref, smask, width, height = img[0], img[1], img[2], img[3]
            bbox: BBox | None = None
            try:
                r = page.get_image_bbox(img)
                if r.is_valid and not r.is_empty and not r.is_infinite:
                    bbox = (r.x0, r.y0, r.x1, r.y1)
            except Exception:  # bbox not available for some image types
                logger.debug("No bbox for image xref=%s on page %d", xref, page.number + 1)
            result.append(ImageData(page_number=page.number + 1, index=idx, xref=xref,
                                    width=width, height=height, bbox=bbox, smask=smask))
        return result

    @staticmethod
    def load_image(doc: fitz.Document, image: ImageData) -> Image.Image:
        """Decode one embedded image into an RGB PIL image (transparency composited onto white).

        Raises:
            ImageExtractionError: if the image cannot be decoded.
        """
        try:
            pix = fitz.Pixmap(doc, image.xref)
            if pix.colorspace is None:  # stencil/mask-only image, nothing to read
                raise ImageExtractionError(f"Image xref={image.xref} has no colour data")
            if pix.colorspace.n != 3 or pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix) if pix.colorspace.n != 3 else pix
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img.load()
            if img.mode == "RGBA":
                base = Image.new("RGB", img.size, "white")
                base.paste(img.convert("RGB"), mask=img.getchannel("A"))
                img = base
            else:
                img = img.convert("RGB")
            if image.smask:
                mask_pix = fitz.Pixmap(doc, image.smask)
                mask = Image.open(io.BytesIO(mask_pix.tobytes("png"))).convert("L")
                if mask.size != img.size:
                    mask = mask.resize(img.size)
                base = Image.new("RGB", img.size, "white")
                base.paste(img, mask=mask)
                img = base
            return img
        except ImageExtractionError:
            raise
        except Exception as exc:
            raise ImageExtractionError(
                f"Cannot decode image xref={image.xref} on page {image.page_number}: {exc}"
            ) from exc
