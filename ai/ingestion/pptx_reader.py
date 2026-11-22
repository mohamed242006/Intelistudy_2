"""PPTX reader -> canonical blocks, one page per slide.

* Reading order: title first, then shapes by (vertical bucket, left); groups expanded with their
  child coordinates mapped to slide coordinates.
* The title placeholder (also inside groups) is emitted once as a level-1 heading. A slide without
  a title placeholder takes its title from the text box with the clearly largest explicit font size
  (>= 1.2 x every other explicit size; ties and ambiguous cases stay untitled).
* Bullets / numbering rendered (``buChar``/``buAutoNum``/``buNone``; body placeholders bullet by
  default); tables with merged cells collapsed; empty tables skipped; charts rendered with type,
  title, axis titles, categories and plotted values; pictures OCR'd once per distinct image (EMF/WMF
  are a mild skip, not a failure).
* Slide-number, footer and date placeholders are skipped; hidden slides are kept and flagged;
  speaker notes become a ``notes`` block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.oxml.ns import qn

from ai.ingestion.cleaners import cell_clean, clean_inline, clean_text, has_content
from ai.ingestion.docs_reader import format_number
from ai.ingestion.image_ocr import ImageOCRHelper, mean_score
from ai.ingestion.models import (
    Block,
    BlockType,
    ImageData,
    PageContent,
    PptxParseError,
    SourceKind,
    TableData,
)
from ai.ingestion.structure import render_blocks

logger = logging.getLogger(__name__)

_TITLE_TYPES = {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.VERTICAL_TITLE}
_SKIP_TYPES = {PP_PLACEHOLDER.SLIDE_NUMBER, PP_PLACEHOLDER.FOOTER, PP_PLACEHOLDER.DATE, PP_PLACEHOLDER.HEADER}
_BODY_TYPES = {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT, PP_PLACEHOLDER.VERTICAL_BODY,
               PP_PLACEHOLDER.VERTICAL_OBJECT}
_AUTONUM = {
    "arabicPeriod": ("decimal", "{}."), "arabicParenR": ("decimal", "{})"), "arabicParenBoth": ("decimal", "({})"),
    "arabicPlain": ("decimal", "{}"), "alphaLcPeriod": ("lowerLetter", "{}."), "alphaUcPeriod": ("upperLetter", "{}."),
    "alphaLcParenR": ("lowerLetter", "{})"), "alphaUcParenR": ("upperLetter", "{})"),
    "alphaLcParenBoth": ("lowerLetter", "({})"), "romanLcPeriod": ("lowerRoman", "{}."),
    "romanUcPeriod": ("upperRoman", "{}."), "romanLcParenR": ("lowerRoman", "{})"),
}


@dataclass(slots=True)
class _Placed:
    shape: Any
    top: int
    left: int
    width: int
    height: int


def _placeholder_type(shape: Any) -> Any:
    if not getattr(shape, "is_placeholder", False):
        return None
    try:
        return shape.placeholder_format.type
    except Exception:  # pragma: no cover - broken placeholder
        return None


class PptxReader:
    """Reads a PPTX into one :class:`PageContent` per slide.

    Args:
        ocr: Image OCR helper (``None`` disables image OCR).
        title_font_ratio: E2E-07 title fallback threshold.
    """

    parser_name = "python-pptx"

    def __init__(self, ocr: ImageOCRHelper | None = None, title_font_ratio: float = 1.2) -> None:
        self.ocr = ocr
        self.title_font_ratio = title_font_ratio

    def read(self, path: str | Path) -> tuple[list[PageContent], dict[str, Any]]:
        p = Path(path)
        if not p.is_file():
            raise PptxParseError(f"File not found: {p}")
        try:
            prs = Presentation(str(p))
        except Exception as exc:
            raise PptxParseError(f"Cannot open PPTX '{p}': {exc}") from exc
        if self.ocr:
            self.ocr.reset()
        metadata = self._metadata(prs)
        slide_height = int(prs.slide_height or 6858000)
        pages: list[PageContent] = []
        for number, slide in enumerate(prs.slides, start=1):
            try:
                pages.append(self._slide(slide, number, slide_height, int(prs.slide_width or 9144000)))
            except Exception as exc:  # one bad slide must not kill the deck
                logger.exception("Slide %d failed: %s", number, exc)
                failed = PageContent(page_number=number, confidence=0.0)
                failed.warnings.append(f"Slide processing failed: {exc}")
                pages.append(failed)
        metadata["page_count"] = len(pages)
        return pages, metadata

    @staticmethod
    def _metadata(prs: Any) -> dict[str, Any]:
        props = prs.core_properties
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

    # ----------------------------------------------------------------- shapes
    def _flatten(self, shapes: Any, transform: tuple[float, float, float, float] | None = None) -> Iterator[_Placed]:
        """Yield leaf shapes with slide coordinates (group children mapped through the group xfrm)."""
        for shape in shapes:
            left, top = int(shape.left or 0), int(shape.top or 0)
            width, height = int(shape.width or 0), int(shape.height or 0)
            if transform:
                ox, oy, sx, sy = transform
                left, top = int(ox + left * sx), int(oy + top * sy)
                width, height = int(width * sx), int(height * sy)
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from self._flatten(shape.shapes, self._group_transform(shape, transform))
            else:
                yield _Placed(shape, top, left, width, height)

    @staticmethod
    def _group_transform(group: Any, parent: tuple[float, float, float, float] | None
                         ) -> tuple[float, float, float, float] | None:
        try:
            xfrm = group._element.grpSpPr.find(qn("a:xfrm"))
            off, ext = xfrm.find(qn("a:off")), xfrm.find(qn("a:ext"))
            ch_off, ch_ext = xfrm.find(qn("a:chOff")), xfrm.find(qn("a:chExt"))
            sx = int(ext.get("cx")) / max(int(ch_ext.get("cx")), 1)
            sy = int(ext.get("cy")) / max(int(ch_ext.get("cy")), 1)
            ox = int(off.get("x")) - int(ch_off.get("x")) * sx
            oy = int(off.get("y")) - int(ch_off.get("y")) * sy
        except Exception:
            return parent
        if parent:
            pox, poy, psx, psy = parent
            return pox + ox * psx, poy + oy * psy, sx * psx, sy * psy
        return ox, oy, sx, sy

    # ------------------------------------------------------------------ slide
    def _slide(self, slide: Any, number: int, slide_height: int, slide_width: int) -> PageContent:
        page = PageContent(page_number=number, size=(float(slide_width), float(slide_height)))
        page.hidden = slide._element.get("show") == "0"
        placed = [p for p in self._flatten(slide.shapes) if _placeholder_type(p.shape) not in _SKIP_TYPES]
        bucket = max(slide_height // 25, 1)
        placed.sort(key=lambda p: (p.top // bucket, p.left))

        title_shapes = [p for p in placed if _placeholder_type(p.shape) in _TITLE_TYPES]
        title_text = ""
        for p in title_shapes:
            if p.shape.has_text_frame:
                title_text = clean_inline(" ".join(par.text for par in p.shape.text_frame.paragraphs))
                if title_text:
                    break
        fallback = None
        if not title_text:
            fallback = self._fallback_title(placed)
            if fallback is not None:
                title_text = clean_inline(fallback.shape.text_frame.text)
        if title_text:
            page.blocks.append(Block(BlockType.HEADING, title_text, number, level=1,
                                     metadata={"title_source": "fallback" if fallback else "placeholder"}))
            page.title = title_text

        scores: list[float] = []
        for p in placed:
            if p in title_shapes or p is fallback:
                continue
            try:
                self._shape(p, page, scores)
            except Exception as exc:
                logger.warning("Slide %d: shape skipped: %s", number, exc)
                page.warnings.append(f"Shape skipped: {exc}")

        notes = self._notes(slide)
        if notes:
            page.blocks.append(Block(BlockType.NOTES, notes, number))

        page.text = render_blocks(page.blocks)
        page.final_text = clean_text(page.text)
        page.text_length = len(page.text)
        page.has_text = bool(page.text)
        page.image_count = len(page.images)
        page.confidence = mean_score(scores)
        return page

    def _fallback_title(self, placed: list[_Placed]) -> _Placed | None:
        """E2E-07: the text box with the clearly largest explicit font size, if unambiguous."""
        sizes: list[tuple[float, _Placed]] = []
        for p in placed:
            shape = p.shape
            if not getattr(shape, "has_text_frame", False) or not shape.text_frame.text.strip():
                continue
            explicit = [r.font.size.pt for par in shape.text_frame.paragraphs for r in par.runs
                        if r.font.size is not None and r.text.strip()]
            if explicit:
                sizes.append((max(explicit), p))
        if len(sizes) < 2:
            return None
        sizes.sort(key=lambda s: s[0], reverse=True)
        (best, candidate), (second, _) = sizes[0], sizes[1]
        text = clean_inline(candidate.shape.text_frame.text)
        paragraphs = [par for par in candidate.shape.text_frame.paragraphs if par.text.strip()]
        if best < second * self.title_font_ratio or len(text) > 100 or len(paragraphs) > 2 or not has_content(text):
            return None
        return candidate

    def _shape(self, p: _Placed, page: PageContent, scores: list[float]) -> None:
        shape = p.shape
        bbox = (float(p.left), float(p.top), float(p.left + p.width), float(p.top + p.height))
        if getattr(shape, "has_table", False) and shape.has_table:
            self._table(shape, page, bbox)
            return
        if getattr(shape, "has_chart", False) and shape.has_chart:
            self._chart(shape, page, bbox)
            return
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or hasattr(shape, "image"):
            try:
                image = shape.image
            except Exception:
                image = None
            if image is not None:
                self._picture(image, page, bbox, scores)
                return
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            self._text_frame(shape, page, bbox)

    def _text_frame(self, shape: Any, page: PageContent, bbox: tuple[float, float, float, float]) -> None:
        default_bullets = _placeholder_type(shape) in _BODY_TYPES
        counters: dict[int, int] = {}
        for par in shape.text_frame.paragraphs:
            text = clean_inline(par.text.replace("\v", " "))
            if not has_content(text):
                continue
            level = int(par.level or 0)
            ppr = par._p.pPr
            bullet: str | None = None
            if ppr is not None and ppr.find(qn("a:buNone")) is not None:
                bullet = None
            elif ppr is not None and ppr.find(qn("a:buAutoNum")) is not None:
                auto = ppr.find(qn("a:buAutoNum"))
                fmt, template = _AUTONUM.get(auto.get("type", "arabicPeriod"), ("decimal", "{}."))
                start = int(auto.get("startAt", "1"))
                counters[level] = counters.get(level, start - 1) + 1
                for deeper in [k for k in counters if k > level]:
                    del counters[deeper]
                bullet = template.format(format_number(counters[level], fmt))
            elif ppr is not None and ppr.find(qn("a:buChar")) is not None:
                bullet = "•"
            elif default_bullets:
                bullet = "•"
            if bullet:
                page.blocks.append(Block(BlockType.LIST_ITEM, text, page.page_number, level=level,
                                         list_marker=bullet, bbox=bbox))
            else:
                page.blocks.append(Block(BlockType.PARAGRAPH, text, page.page_number, bbox=bbox))

    @staticmethod
    def _table(shape: Any, page: PageContent, bbox: tuple[float, float, float, float]) -> None:
        rows: list[list[str]] = []
        for row in shape.table.rows:
            cells = []
            for cell in row.cells:
                cells.append("" if cell.is_spanned else cell_clean(cell.text))
            rows.append(cells)
        if rows:  # drop columns that are empty in every row (merge spacers)
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            keep = [i for i in range(width) if any(r[i] for r in rows)]
            rows = [[r[i] for i in keep] for r in rows]
        rows = [r for r in rows if any(r)]
        if not rows:
            return  # empty table
        data = TableData(page_number=page.page_number, index=len(page.tables), rows=rows, bbox=bbox)
        page.tables.append(data)
        page.blocks.append(Block(BlockType.TABLE, data.to_pipe_text(), page.page_number,
                                 ref_index=data.index, bbox=bbox, table=data))

    @staticmethod
    def _chart(shape: Any, page: PageContent, bbox: tuple[float, float, float, float]) -> None:
        chart = shape.chart
        lines: list[str] = []
        try:
            chart_type = str(chart.chart_type).split(".")[-1].split(" ")[0]
        except Exception:
            chart_type = "chart"
        title = ""
        try:
            if chart.has_title and chart.chart_title.has_text_frame:
                title = clean_inline(chart.chart_title.text_frame.text)
        except Exception:
            pass
        lines.append(f"Chart type: {chart_type}" + (f"; title: {title}" if title else ""))
        for axis_name in ("category_axis", "value_axis"):
            try:
                axis = getattr(chart, axis_name)
                if axis.has_title and axis.axis_title.has_text_frame:
                    lines.append(f"{axis_name.replace('_', ' ').capitalize()}: "
                                 f"{clean_inline(axis.axis_title.text_frame.text)}")
            except Exception:
                continue
        try:
            plot = chart.plots[0]
            categories = [clean_inline(str(c)) for c in plot.categories]
        except Exception:
            categories = []
        if categories:
            lines.append("Categories: " + ", ".join(categories))
        try:
            for series in chart.series:
                values = list(series.values)
                if categories and len(categories) == len(values):
                    pairs = ", ".join(f"{c} = {_fmt_value(v)}" for c, v in zip(categories, values))
                else:
                    pairs = ", ".join(_fmt_value(v) for v in values)
                lines.append(f"Series {clean_inline(str(series.name or ''))}: {pairs}".replace("Series :", "Series:"))
        except Exception as exc:
            page.warnings.append(f"Chart series unreadable: {exc}")
        try:
            if chart.has_legend:
                lines.append("Legend: shown")
        except Exception:
            pass
        page.blocks.append(Block(BlockType.CHART, "\n".join(lines), page.page_number, bbox=bbox,
                                 ref_index=sum(b.type is BlockType.CHART for b in page.blocks),
                                 metadata={"chart_type": chart_type, "chart_title": title}))

    def _picture(self, image: Any, page: PageContent, bbox: tuple[float, float, float, float],
                 scores: list[float]) -> None:
        data = ImageData(page_number=page.page_number, index=len(page.images), bbox=bbox,
                         name=str(getattr(image, "filename", "")))
        page.images.append(data)
        if self.ocr is None:
            return
        outcome = self.ocr.ocr(image.blob, str(image.ext or ""))
        data.content_hash = outcome.content_hash
        data.width, data.height = outcome.width, outcome.height
        data.ocr_applied = outcome.applied
        data.ocr_text = outcome.text
        data.ocr_confidence = outcome.engine_confidence
        data.ocr_skipped_reason = outcome.skipped_reason
        data.score = outcome.score
        scores.append(outcome.score)
        if outcome.warning:
            page.warnings.append(outcome.warning)
        if outcome.skipped_reason == "unsupported_format":
            page.warnings.append(f"Image format not OCR'd: {image.ext}")
        if outcome.text and has_content(outcome.text):
            page.blocks.append(Block(BlockType.IMAGE_OCR, outcome.text, page.page_number, ref_index=data.index,
                                     bbox=bbox, confidence=outcome.block_confidence, source=SourceKind.OCR_IMAGE,
                                     metadata={"image": data.name}))

    @staticmethod
    def _notes(slide: Any) -> str:
        try:
            if not slide.has_notes_slide:
                return ""
            frame = slide.notes_slide.notes_text_frame
        except Exception:
            return ""
        if frame is None:
            return ""
        text = clean_text(frame.text)
        return text if has_content(text) else ""


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
