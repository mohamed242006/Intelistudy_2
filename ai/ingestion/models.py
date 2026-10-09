"""Data models for the IntelliStudy document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IngestionError(Exception):
    """Base class for ingestion failures."""


class PDFParseError(IngestionError):
    """The PDF could not be opened or parsed."""


class OCRError(IngestionError):
    """OCR failed for a page."""


class TableExtractionError(IngestionError):
    """Table extraction failed."""


class ImageExtractionError(IngestionError):
    """An embedded image could not be decoded."""


class PageType(str, Enum):
    """Automatic page classification."""

    TEXT_PAGE = "TEXT_PAGE"
    SCANNED_PAGE = "SCANNED_PAGE"
    TABLE_PAGE = "TABLE_PAGE"
    MIXED_PAGE = "MIXED_PAGE"


@dataclass(slots=True)
class ImageData:
    """An image embedded in a PDF page, plus its OCR result (if OCR was run)."""

    page_number: int
    index: int
    xref: int
    width: int
    height: int
    bbox: tuple[float, float, float, float] | None = None
    smask: int = 0  # xref of the soft mask (transparency), 0 if none
    ocr_text: str = ""
    ocr_confidence: float | None = None  # mean word confidence, 0-100
    ocr_applied: bool = False  # OCR was attempted on this image
    ocr_skipped_reason: str | None = None  # e.g. "too_small", "no_text", "low_confidence"


@dataclass(slots=True)
class TableData:
    """A table extracted from a page.

    ``rows`` holds the raw cell grid; the first row is treated as header.
    """

    page_number: int
    index: int
    rows: list[list[str]]
    bbox: tuple[float, float, float, float] | None = None

    @property
    def header(self) -> list[str]:
        return self.rows[0] if self.rows else []

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def to_pipe_text(self) -> str:
        """Render as ``col | col`` lines."""
        return "\n".join(" | ".join(row) for row in self.rows)

    def to_key_value_text(self) -> str:
        """Render each data row as ``Header: value`` lines (RAG-friendly)."""
        if len(self.rows) < 2:
            return self.to_pipe_text()
        header = self.rows[0]
        blocks: list[str] = []
        for row in self.rows[1:]:
            lines = [
                f"{header[i] if i < len(header) and header[i] else f'Column {i + 1}'}: {cell}"
                for i, cell in enumerate(row)
                if cell
            ]
            if lines:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


@dataclass(slots=True)
class PageContent:
    """All extracted content for a single page (1-indexed)."""

    page_number: int
    text: str = ""
    ocr_text: str = ""
    tables: list[TableData] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    page_type: PageType = PageType.TEXT_PAGE
    text_length: int = 0
    image_count: int = 0
    has_text: bool = False
    needs_ocr: bool = False
    ocr_applied: bool = False
    ocr_confidence: float | None = None  # 0-100 from Tesseract
    confidence: float = 1.0  # overall 0-1 extraction confidence
    final_text: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def images_with_text(self) -> list[ImageData]:
        """Embedded images that yielded OCR text."""
        return [img for img in self.images if img.ocr_text]


@dataclass(slots=True)
class Document:
    """A fully ingested document."""

    source_path: str
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Whole document with page markers, ready for chunking."""
        return "\n\n".join(
            f"[Page {p.page_number}]\n{p.final_text}" for p in self.pages if p.final_text
        )

    def get_page(self, page_number: int) -> PageContent | None:
        return next((p for p in self.pages if p.page_number == page_number), None)
