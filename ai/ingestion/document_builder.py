"""Layer 4 - PDF page finalisation.

Pure logic, no file or engine access: takes the outputs of the earlier layers (text blocks, OCR
text, tables, image OCR) and produces finished pages: canonical blocks in reading order,
``final_text``, page type and confidence.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from ai.ingestion.cleaners import clean_inline, clean_text, has_content, is_heading
from ai.ingestion.models import Block, BlockType, PageContent, PageType, SourceKind, TableData
from ai.ingestion.structure import render_blocks

logger = logging.getLogger(__name__)

TableFormat = Literal["pipe", "key_value"]
_OCR_BULLET = re.compile(r"^([•●▪■◦o\-–*»>]|\d{1,2}[.)])\s+(\S.*)$")


class PageClassifier:
    """Assigns a :class:`PageType` to a processed page."""

    def __init__(self, min_text_chars: int = 50) -> None:
        self.min_text_chars = min_text_chars

    def classify(self, page: PageContent) -> PageType:
        body_chars = len(page.text)
        has_tables = bool(page.tables)
        if page.ocr_applied:
            if has_tables or body_chars >= self.min_text_chars:
                return PageType.MIXED_PAGE
            return PageType.SCANNED_PAGE
        if page.images_with_text:
            if body_chars < self.min_text_chars and not has_tables:
                return PageType.SCANNED_PAGE
            return PageType.MIXED_PAGE
        if has_tables:
            return PageType.TABLE_PAGE if body_chars < 4 * self.min_text_chars else PageType.MIXED_PAGE
        return PageType.TEXT_PAGE


class ConfidenceScorer:
    """Heuristic 0-1 extraction-quality score per page."""

    @staticmethod
    def score(page: PageContent) -> float:
        if page.ocr_applied:
            if page.ocr_confidence is None or not page.ocr_text:
                return 0.0
            return round(min(page.ocr_confidence / 100.0, 1.0), 3)

        native = ConfidenceScorer._native_score(page)
        images = page.images_with_text
        if not images:
            return native

        image_chars = sum(len(i.ocr_text) for i in images)
        image_conf = sum((i.ocr_confidence or 0.0) * len(i.ocr_text) for i in images) / image_chars / 100.0
        native_chars = len(page.text) + sum(len(t.to_pipe_text()) for t in page.tables)
        if native_chars == 0:
            return round(min(image_conf, 1.0), 3)
        total = native_chars + image_chars
        return round((native * native_chars + min(image_conf, 1.0) * image_chars) / total, 3)

    @staticmethod
    def _native_score(page: PageContent) -> float:
        text = page.text
        if not text and not page.tables:
            return 1.0 if not page.images else 0.0  # an empty page is not an extraction failure
        if not text:
            return 0.9  # table-only page
        bad = text.count("�") + sum(1 for c in text if ord(c) < 32 and c not in "\n\t")
        alnum = sum(c.isalnum() for c in text) / max(len(text), 1)
        penalty = min(bad / max(len(text), 1) * 10, 0.5)
        base = 0.6 + 0.4 * min(alnum / 0.7, 1.0)
        return round(max(base - penalty, 0.0), 3)


def ocr_text_to_blocks(text: str, page_number: int, confidence: float,
                       source: SourceKind = SourceKind.OCR_PAGE) -> list[Block]:
    """Turn OCR text (no font information) into heading / list / paragraph blocks."""
    blocks: list[Block] = []
    for para in re.split(r"\n\s*\n", clean_text(text)):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines) == 1 and is_heading(lines[0]):
            blocks.append(Block(BlockType.HEADING, clean_inline(lines[0]), page_number, level=1,
                                confidence=confidence, source=source))
            continue
        buffer: list[str] = []
        for ln in lines:
            m = _OCR_BULLET.match(ln)
            if m:
                if buffer:
                    blocks.append(Block(BlockType.PARAGRAPH, " ".join(buffer), page_number,
                                        confidence=confidence, source=source))
                    buffer = []
                marker = m.group(1) if m.group(1)[0].isdigit() else "•"
                blocks.append(Block(BlockType.LIST_ITEM, clean_inline(m.group(2)), page_number,
                                    list_marker=marker, confidence=confidence, source=source))
            elif blocks and blocks[-1].type is BlockType.LIST_ITEM and not buffer and ln[:1].islower():
                blocks[-1].text = f"{blocks[-1].text} {clean_inline(ln)}"
            else:
                buffer.append(clean_inline(ln))
        if buffer:
            blocks.append(Block(BlockType.PARAGRAPH, " ".join(buffer), page_number,
                                confidence=confidence, source=source))
    return [b for b in blocks if has_content(b.text)]


class DocumentBuilder:
    """Finalises PDF pages.

    Args:
        classifier: Page classifier (injectable for testing).
        scorer: Confidence scorer (injectable for testing).
        table_format: Display rendering of tables: ``"pipe"`` or ``"key_value"``.
        min_text_chars: Used by the default classifier.
    """

    def __init__(
        self,
        classifier: PageClassifier | None = None,
        scorer: ConfidenceScorer | None = None,
        table_format: TableFormat = "pipe",
        min_text_chars: int = 50,
    ) -> None:
        self.classifier = classifier or PageClassifier(min_text_chars)
        self.scorer = scorer or ConfidenceScorer()
        self.table_format = table_format

    def finalize_page(self, page: PageContent) -> PageContent:
        """Classify, score, merge all blocks in reading order and render ``final_text`` (in place)."""
        page.page_type = self.classifier.classify(page)
        page.confidence = self.scorer.score(page)
        page.blocks = self._merge_blocks(page)
        page.final_text = clean_text(render_blocks(page.blocks))
        return page

    def _merge_blocks(self, page: PageContent) -> list[Block]:
        text_blocks = list(page.blocks)
        if page.ocr_applied and page.ocr_text:
            conf = round(min((page.ocr_confidence or 0.0) / 100.0, 1.0), 3)
            text_blocks = ocr_text_to_blocks(page.ocr_text, page.page_number, conf) + text_blocks

        extras: list[Block] = []
        for table in page.tables:
            extras.append(Block(BlockType.TABLE, self._render_table(table), page.page_number,
                                ref_index=table.index, bbox=table.bbox, table=table,
                                confidence=0.9))
        for image in page.images_with_text:
            extras.append(Block(BlockType.IMAGE_OCR, image.ocr_text, page.page_number,
                                ref_index=image.index, bbox=image.bbox, source=SourceKind.OCR_IMAGE,
                                confidence=round(min((image.ocr_confidence or 0.0) / 100.0, 1.0), 3),
                                metadata={"xref": image.xref, "width": image.width, "height": image.height}))
        if not extras:
            return text_blocks

        # Insert each extra before the first text block that starts below it.
        merged = list(text_blocks)
        positioned = sorted((e for e in extras if e.bbox), key=lambda b: b.bbox[1])  # type: ignore[index]
        for extra in positioned:
            top = extra.bbox[1]  # type: ignore[index]
            index = next((i for i, b in enumerate(merged)
                          if b.bbox is not None and b.type not in (BlockType.TABLE, BlockType.IMAGE_OCR)
                          and b.bbox[1] > top), len(merged))
            merged.insert(index, extra)
        merged.extend(e for e in extras if not e.bbox)
        return merged

    def _render_table(self, table: TableData) -> str:
        return table.to_key_value_text() if self.table_format == "key_value" else table.to_pipe_text()
