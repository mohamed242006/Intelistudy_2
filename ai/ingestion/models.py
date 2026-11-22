"""Canonical data models for the IntelliStudy ingestion pipeline (schema 2.0).

Every reader (PDF, DOCX, PPTX) produces the same structures:

* :class:`Document` - one ingested file: :class:`SourceInfo`, metadata, canonical statistics, pages.
* :class:`PageContent` - one PDF page, one PPTX slide, or the single logical page of a DOCX.
* :class:`Block` - the unit the chunker consumes (heading, paragraph, list item, table, image OCR,
  chart, speaker notes), in reading order, with its heading path and confidence.

The schema is additive: the PDF-era fields (``text``, ``ocr_text``, ``final_text``, ...) are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

SCHEMA_VERSION = "2.0"

Unit = Literal["page", "slide", "document"]
BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------- errors
class IngestionError(Exception):
    """Base class for ingestion failures."""


class UnsupportedFormatError(IngestionError):
    """The file type is not supported by any reader."""


class PDFParseError(IngestionError):
    """The PDF could not be opened or parsed."""


class DocxParseError(IngestionError):
    """The DOCX could not be opened or parsed."""


class PptxParseError(IngestionError):
    """The PPTX could not be opened or parsed."""


class OCRError(IngestionError):
    """OCR failed for a page."""


class TableExtractionError(IngestionError):
    """Table extraction failed."""


class ImageExtractionError(IngestionError):
    """An embedded image could not be decoded."""


# ---------------------------------------------------------------------- enums
class PageType(str, Enum):
    """Automatic page classification."""

    TEXT_PAGE = "TEXT_PAGE"
    SCANNED_PAGE = "SCANNED_PAGE"
    TABLE_PAGE = "TABLE_PAGE"
    MIXED_PAGE = "MIXED_PAGE"


class BlockType(str, Enum):
    """Kinds of canonical blocks."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE_OCR = "image_ocr"
    CHART = "chart"
    NOTES = "notes"


class SourceKind(str, Enum):
    """Where a block's text came from."""

    NATIVE = "native"  # text layer / document XML
    OCR_PAGE = "ocr_page"  # whole-page OCR of a scanned page
    OCR_IMAGE = "ocr_image"  # OCR of one embedded image


# ---------------------------------------------------------------- raw content
@dataclass(slots=True)
class ImageData:
    """An embedded image plus its OCR result (if OCR was run)."""

    page_number: int
    index: int
    xref: int = 0  # PDF only
    width: int = 0
    height: int = 0
    bbox: BBox | None = None
    smask: int = 0  # PDF soft-mask xref, 0 if none
    ocr_text: str = ""
    ocr_confidence: float | None = None  # mean word confidence, 0-100
    ocr_applied: bool = False  # OCR was attempted on this image
    ocr_skipped_reason: str | None = None  # e.g. "too_small", "no_text", "low_confidence"
    name: str = ""  # DOCX/PPTX part name or file name
    content_hash: str = ""  # sha1 of the image bytes (DOCX/PPTX)
    score: float = 1.0  # 0-1 item score used for page confidence


@dataclass(slots=True)
class TableData:
    """A table extracted from a page. ``rows`` is the cell grid; the first row is the header."""

    page_number: int
    index: int
    rows: list[list[str]]
    bbox: BBox | None = None

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
        """Render as ``col | col`` lines (pipes inside cells are escaped as ``/``)."""
        return "\n".join(" | ".join(c.replace("|", "/") for c in row) for row in self.rows)

    def row_key_value_text(self, row: list[str]) -> str:
        """Render one data row as ``Header: value`` lines."""
        header = self.header
        lines = [
            f"{header[i] if i < len(header) and header[i] else f'Column {i + 1}'}: {cell}"
            for i, cell in enumerate(row)
            if cell
        ]
        return "\n".join(lines)

    def to_key_value_text(self) -> str:
        """Render each data row as ``Header: value`` lines (RAG-friendly)."""
        if len(self.rows) < 2:
            return self.to_pipe_text()
        blocks = [self.row_key_value_text(row) for row in self.rows[1:]]
        return "\n\n".join(b for b in blocks if b)


# --------------------------------------------------------------------- blocks
@dataclass(slots=True)
class Block:
    """One unit of content in reading order.

    Attributes:
        type: Block kind.
        text: Display text (tables: pipe rendering).
        page_number: 1-based page / slide (always 1 for DOCX).
        order: Global position in the document (assigned by the pipeline).
        level: Heading level (0 = document title, 1 = top section) or list nesting depth.
        heading_path: Ancestor heading titles; a heading's own path ends with itself.
        list_marker: ``"•"``, ``"1."``, ``"a)"`` ... for list items.
        ref_index: Index of the table / image / chart on its page, when applicable.
        bbox: Position on the page (PDF points / EMU for PPTX), when known.
        confidence: 0-1 extraction confidence of this block.
        source: Where the text came from (native text or OCR).
        table: The table grid for ``TABLE`` blocks.
        metadata: Free-form extras (chart type, image name, ...).
    """

    type: BlockType
    text: str
    page_number: int
    order: int = 0
    level: int = 0
    heading_path: list[str] = field(default_factory=list)
    list_marker: str = ""
    ref_index: int | None = None
    bbox: BBox | None = None
    confidence: float = 1.0
    source: SourceKind = SourceKind.NATIVE
    table: TableData | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Plain-text rendering used for ``final_text`` and chunk display text."""
        if self.type is BlockType.HEADING:
            return f"{'#' * max(1, min(self.level + 1, 6))} {self.text}"
        if self.type is BlockType.LIST_ITEM:
            indent = "  " * max(self.level, 0)
            return f"{indent}{self.list_marker or '-'} {self.text}"
        if self.type is BlockType.TABLE:
            return f"[Table]\n{self.text}"
        if self.type is BlockType.IMAGE_OCR:
            return f"[Image text]\n{self.text}"
        if self.type is BlockType.CHART:
            return f"[Chart]\n{self.text}"
        if self.type is BlockType.NOTES:
            return f"[Speaker notes]\n{self.text}"
        return self.text


# ---------------------------------------------------------------------- pages
@dataclass(slots=True)
class PageContent:
    """All extracted content for a single page / slide (1-indexed)."""

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
    blocks: list[Block] = field(default_factory=list)
    size: tuple[float, float] | None = None  # width, height
    hidden: bool = False  # PPTX hidden slide
    title: str = ""  # slide title / first heading on the page

    @property
    def images_with_text(self) -> list[ImageData]:
        """Embedded images that yielded OCR text."""
        return [img for img in self.images if img.ocr_text]


@dataclass(slots=True)
class SourceInfo:
    """Identity and provenance of the ingested file."""

    doc_id: str  # sha256[:16] of the file bytes
    sha256: str
    file_name: str
    file_type: str  # "pdf" | "docx" | "pptx"
    size_bytes: int
    unit: Unit
    page_count: int
    parser: str
    ingested_at: str  # ISO-8601 UTC
    title: str = ""
    author: str = ""
    created: str = ""
    modified: str = ""
    language: str = ""


@dataclass(slots=True)
class Document:
    """A fully ingested document."""

    source_path: str
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    source: SourceInfo | None = None
    schema_version: str = SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)

    @property
    def unit(self) -> Unit:
        return self.source.unit if self.source else "page"

    @property
    def blocks(self) -> list[Block]:
        """All blocks of the document in reading order."""
        return [b for p in self.pages for b in p.blocks]

    @property
    def full_text(self) -> str:
        """Whole document with page markers."""
        label = {"page": "Page", "slide": "Slide", "document": "Page"}[self.unit]
        return "\n\n".join(
            f"[{label} {p.page_number}]\n{p.final_text}" for p in self.pages if p.final_text
        )

    def get_page(self, page_number: int) -> PageContent | None:
        return next((p for p in self.pages if p.page_number == page_number), None)
