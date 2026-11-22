"""DOCX reader -> canonical blocks (one logical page, cited by heading path).

* The body is walked once in document order, so tables and image text appear where they are.
* ``Title`` style = heading level 0, ``Heading N`` = level N (style inheritance and paragraph
  outline levels are honoured).
* Numbering definitions are resolved: bullet vs decimal / letter / roman, ``lvlText`` templates,
  start values, and counters that restart when a shallower level advances.
* Tables: horizontally merged cells collapsed, empty tables skipped.
* Images are de-duplicated (sha1) and OCR'd at the paragraph that references them.

Not extracted: headers/footers, footnotes, comments, text boxes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from docx import Document as load_docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from ai.ingestion.cleaners import cell_clean, clean_inline, clean_text, has_content
from ai.ingestion.image_ocr import ImageOCRHelper, mean_score
from ai.ingestion.models import (
    Block,
    BlockType,
    DocxParseError,
    ImageData,
    PageContent,
    SourceKind,
    TableData,
)
from ai.ingestion.structure import render_blocks

logger = logging.getLogger(__name__)

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.I)
_BULLET_FORMATS = {"bullet", "none"}
_R_EMBED = qn("r:embed")
_R_ID = qn("r:id")


def _style_depth(paragraph: Paragraph) -> int:
    """Nesting depth implied by numbered list styles: ``List Bullet 2`` -> 1."""
    name = (paragraph.style.name or "") if paragraph.style is not None else ""
    m = re.match(r"^list(?: bullet| number| continue)?\s+(\d)$", name.strip(), re.I)
    return int(m.group(1)) - 1 if m else 0


def _roman(n: int) -> str:
    values = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
              (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for value, sym in values:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def _letters(n: int) -> str:
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(97 + rem) + out
    return out


def format_number(n: int, fmt: str) -> str:
    """Render counter ``n`` in a Word ``numFmt``."""
    if fmt == "lowerLetter":
        return _letters(n)
    if fmt == "upperLetter":
        return _letters(n).upper()
    if fmt == "lowerRoman":
        return _roman(n)
    if fmt == "upperRoman":
        return _roman(n).upper()
    return str(n)


@dataclass(slots=True)
class _Level:
    fmt: str = "decimal"
    text: str = "%1."
    start: int = 1


@dataclass(slots=True)
class NumberingResolver:
    """Resolves ``(numId, ilvl)`` to list markers, keeping Word's counter semantics."""

    levels: dict[str, dict[int, _Level]] = field(default_factory=dict)  # numId -> ilvl -> level
    counters: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Any) -> "NumberingResolver":
        resolver = cls()
        try:
            numbering = document.part.numbering_part.element
        except Exception:  # no numbering part
            return resolver
        abstract: dict[str, dict[int, _Level]] = {}
        for absn in numbering.findall(qn("w:abstractNum")):
            levels: dict[int, _Level] = {}
            for lvl in absn.findall(qn("w:lvl")):
                ilvl = int(lvl.get(qn("w:ilvl"), "0"))
                level = _Level()
                fmt = lvl.find(qn("w:numFmt"))
                if fmt is not None:
                    level.fmt = fmt.get(qn("w:val"), "decimal")
                txt = lvl.find(qn("w:lvlText"))
                if txt is not None:
                    level.text = txt.get(qn("w:val"), "")
                start = lvl.find(qn("w:start"))
                if start is not None:
                    level.start = int(start.get(qn("w:val"), "1"))
                levels[ilvl] = level
            abstract[absn.get(qn("w:abstractNumId"), "")] = levels
        for num in numbering.findall(qn("w:num")):
            ref = num.find(qn("w:abstractNumId"))
            if ref is None:
                continue
            levels = dict(abstract.get(ref.get(qn("w:val"), ""), {}))
            for override in num.findall(qn("w:lvlOverride")):
                start = override.find(qn("w:startOverride"))
                ilvl = int(override.get(qn("w:ilvl"), "0"))
                if start is not None and ilvl in levels:
                    base = levels[ilvl]
                    levels[ilvl] = _Level(base.fmt, base.text, int(start.get(qn("w:val"), "1")))
            resolver.levels[num.get(qn("w:numId"), "")] = levels
        return resolver

    def is_bullet(self, num_id: str, ilvl: int) -> bool:
        level = self.levels.get(num_id, {}).get(ilvl)
        return level is None or level.fmt in _BULLET_FORMATS

    def marker(self, num_id: str, ilvl: int) -> str:
        levels = self.levels.get(num_id, {})
        level = levels.get(ilvl)
        if level is None or level.fmt in _BULLET_FORMATS:
            return "•"
        counters = self.counters.setdefault(num_id, [0] * 9)
        ilvl = min(max(ilvl, 0), 8)
        if counters[ilvl] == 0:
            counters[ilvl] = level.start
        else:
            counters[ilvl] += 1
        for deeper in range(ilvl + 1, 9):
            counters[deeper] = 0

        def repl(m: re.Match[str]) -> str:
            k = int(m.group(1)) - 1
            lv = levels.get(k, _Level())
            value = counters[k] if counters[k] else lv.start
            return format_number(value, lv.fmt)

        text = re.sub(r"%(\d)", repl, level.text) if level.text else f"{counters[ilvl]}."
        return text.strip() or f"{counters[ilvl]}."


class DocxReader:
    """Reads a DOCX file into one :class:`PageContent` of canonical blocks.

    Args:
        ocr: Image OCR helper (``None`` disables image OCR).
    """

    parser_name = "python-docx"

    def __init__(self, ocr: ImageOCRHelper | None = None) -> None:
        self.ocr = ocr

    def read(self, path: str | Path) -> tuple[list[PageContent], dict[str, Any]]:
        p = Path(path)
        if not p.is_file():
            raise DocxParseError(f"File not found: {p}")
        try:
            document = load_docx(str(p))
        except Exception as exc:
            raise DocxParseError(f"Cannot open DOCX '{p}': {exc}") from exc

        metadata = self._metadata(document)
        if self.ocr:
            self.ocr.reset()
        page = PageContent(page_number=1)
        numbering = NumberingResolver.from_document(document)
        style_cache: dict[str, tuple[int | None, tuple[str, int] | None]] = {}
        scores: list[float] = []
        seen_images: set[str] = set()

        for item in self._iter_body(document):
            try:
                if isinstance(item, Paragraph):
                    self._paragraph(item, document, page, numbering, style_cache, scores, seen_images)
                else:
                    self._table(item, page)
            except Exception as exc:  # one bad element must not kill the document
                logger.warning("DOCX element skipped: %s", exc)
                page.warnings.append(f"Element skipped: {exc}")

        page.text = render_blocks(page.blocks)
        page.final_text = clean_text(page.text)
        page.text_length = len(page.text)
        page.has_text = bool(page.text)
        page.image_count = len(page.images)
        page.confidence = mean_score(scores)
        metadata["page_count"] = 1
        return [page], metadata

    # --------------------------------------------------------------- walking
    @staticmethod
    def _iter_body(document: Any) -> Iterator[Paragraph | Table]:
        def walk(parent_element: Any) -> Iterator[Paragraph | Table]:
            for child in parent_element.iterchildren():
                if child.tag == qn("w:p"):
                    yield Paragraph(child, document)
                elif child.tag == qn("w:tbl"):
                    yield Table(child, document)
                elif child.tag == qn("w:sdt"):
                    content = child.find(qn("w:sdtContent"))
                    if content is not None:
                        yield from walk(content)
        yield from walk(document.element.body)

    @staticmethod
    def _metadata(document: Any) -> dict[str, Any]:
        props = document.core_properties
        meta: dict[str, Any] = {}
        for key in ("title", "author", "subject", "keywords", "language"):
            value = getattr(props, key, None)
            if value:
                meta[key] = str(value)
        for key in ("created", "modified"):
            value = getattr(props, key, None)
            if value:
                meta[key] = value.isoformat()
        return meta

    # ------------------------------------------------------------ paragraphs
    def _style_info(self, paragraph: Paragraph, cache: dict[str, tuple[int | None, tuple[str, int] | None]]
                    ) -> tuple[int | None, tuple[str, int] | None]:
        """(heading level, (numId, ilvl) from the style) following the style inheritance chain."""
        style = paragraph.style
        key = getattr(style, "style_id", None) or ""
        if key in cache:
            return cache[key]
        level: int | None = None
        num: tuple[str, int] | None = None
        current = style
        depth = 0
        while current is not None and depth < 10:
            name = (current.name or "").strip()
            if level is None:
                if name.lower() == "title":
                    level = 0
                else:
                    m = _HEADING_STYLE.match(name)
                    if m:
                        level = int(m.group(1))
                    else:
                        ppr = current.element.pPr
                        outline = ppr.find(qn("w:outlineLvl")) if ppr is not None else None
                        if outline is not None and int(outline.get(qn("w:val"), "9")) < 9:
                            level = int(outline.get(qn("w:val"))) + 1
            if num is None:
                ppr = current.element.pPr
                numpr = ppr.find(qn("w:numPr")) if ppr is not None else None
                if numpr is not None:
                    num_id = numpr.find(qn("w:numId"))
                    ilvl = numpr.find(qn("w:ilvl"))
                    if num_id is not None:
                        num = (num_id.get(qn("w:val"), ""), int(ilvl.get(qn("w:val"), "0")) if ilvl is not None else 0)
            current = current.base_style
            depth += 1
        cache[key] = (level, num)
        return level, num

    def _paragraph(self, paragraph: Paragraph, document: Any, page: PageContent, numbering: NumberingResolver,
                   style_cache: dict[str, tuple[int | None, tuple[str, int] | None]],
                   scores: list[float], seen_images: set[str]) -> None:
        text = clean_inline(paragraph.text.replace("\t", " "))
        level, style_num = self._style_info(paragraph, style_cache)

        ppr = paragraph._p.pPr
        if level is None and ppr is not None:
            outline = ppr.find(qn("w:outlineLvl"))
            if outline is not None and int(outline.get(qn("w:val"), "9")) < 9:
                level = int(outline.get(qn("w:val"))) + 1

        num: tuple[str, int] | None = style_num
        numpr = ppr.find(qn("w:numPr")) if ppr is not None else None
        if numpr is not None:
            num_id = numpr.find(qn("w:numId"))
            ilvl = numpr.find(qn("w:ilvl"))
            if num_id is not None:
                num = (num_id.get(qn("w:val"), ""), int(ilvl.get(qn("w:val"), "0")) if ilvl is not None else 0)
        if num is not None and num[0] == "0":
            num = None  # numId 0 = numbering explicitly removed

        if has_content(text):
            if level is not None and len(text) <= 200:
                page.blocks.append(Block(BlockType.HEADING, text, 1, level=level))
            elif num is not None:
                marker = numbering.marker(*num)
                page.blocks.append(Block(BlockType.LIST_ITEM, text, 1, level=max(num[1], _style_depth(paragraph)),
                                         list_marker=marker))
            else:
                style_name = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
                if style_name.startswith("list bullet"):
                    page.blocks.append(Block(BlockType.LIST_ITEM, text, 1, level=_style_depth(paragraph),
                                             list_marker="•"))
                else:
                    page.blocks.append(Block(BlockType.PARAGRAPH, text, 1))

        self._images(paragraph, document, page, scores, seen_images)

    def _images(self, paragraph: Paragraph, document: Any, page: PageContent,
                scores: list[float], seen_images: set[str]) -> None:
        element = paragraph._p
        rel_ids = [b.get(_R_EMBED) for b in element.iter(qn("a:blip"))]
        rel_ids += [b.get(_R_ID) for b in element.iter("{urn:schemas-microsoft-com:vml}imagedata")]
        for rel_id in filter(None, rel_ids):
            try:
                part = document.part.related_parts[rel_id]
                blob: bytes = part.blob
            except Exception:
                continue
            name = str(getattr(part, "partname", ""))
            extension = Path(name).suffix
            image = ImageData(page_number=1, index=len(page.images), name=name)
            if self.ocr is None:
                page.images.append(image)
                continue
            outcome = self.ocr.ocr(blob, extension)
            if outcome.content_hash in seen_images:
                continue  # the same picture again: indexed once
            seen_images.add(outcome.content_hash)
            image.content_hash = outcome.content_hash
            image.width, image.height = outcome.width, outcome.height
            image.ocr_applied = outcome.applied
            image.ocr_text = outcome.text
            image.ocr_confidence = outcome.engine_confidence
            image.ocr_skipped_reason = outcome.skipped_reason
            image.score = outcome.score
            page.images.append(image)
            scores.append(outcome.score)
            if outcome.warning:
                page.warnings.append(outcome.warning)
            if outcome.text and has_content(outcome.text):
                page.blocks.append(Block(BlockType.IMAGE_OCR, outcome.text, 1, ref_index=image.index,
                                         confidence=outcome.block_confidence, source=SourceKind.OCR_IMAGE,
                                         metadata={"image": name}))

    # ---------------------------------------------------------------- tables
    @staticmethod
    def _table(table: Table, page: PageContent) -> None:
        rows: list[list[str]] = []
        for row in table.rows:
            cells: list[str] = []
            previous = None
            try:
                row_cells = row.cells
            except Exception:  # malformed grid
                continue
            first_row = not rows
            for cell in row_cells:
                if previous is not None and cell._tc is previous:
                    # Horizontally merged cell (python-docx repeats it): keep the grid aligned; a
                    # merged header labels every column it spans, a merged data cell is stated once.
                    cells.append(cells[-1] if first_row else "")
                    continue
                previous = cell._tc
                cells.append(cell_clean("\n".join(p.text for p in cell.paragraphs)))
            if any(cells):
                rows.append(cells)
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        if len(rows) == 1 and width == 1:
            text = rows[0][0]  # a 1x1 "table" is a framed paragraph
            if has_content(text):
                page.blocks.append(Block(BlockType.PARAGRAPH, text, 1))
            return
        data = TableData(page_number=1, index=len(page.tables), rows=rows)
        page.tables.append(data)
        page.blocks.append(Block(BlockType.TABLE, data.to_pipe_text(), 1, ref_index=data.index, table=data))
