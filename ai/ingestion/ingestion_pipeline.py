"""Ingestion pipeline - the single entry point for the rest of IntelliStudy.

Routes by extension (or MIME type) to the PDF, DOCX or PPTX reader and returns one canonical
:class:`~ai.ingestion.models.Document` (schema 2.0): blocks in reading order with heading paths,
source identity, metadata and canonical statistics.

PDF flow per page::

    PDFReader (L1) -> no text layer?   -> OCRProcessor.process        (whole page)
                   -> has images?      -> OCRProcessor.process_image  (each image; repeated logos skipped)
                   -> text layer       -> TableExtractor (L3) -> re-read text without table regions
                   -> DocumentBuilder.finalize_page (L4)

Usage::

    from ai.ingestion.ingestion_pipeline import IngestionPipeline

    document = IngestionPipeline().ingest("lecture.pdf")
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from ai.ingestion.cleaners import clean_text
from ai.ingestion.docs_reader import DocxReader
from ai.ingestion.document_builder import DocumentBuilder
from ai.ingestion.image_ocr import ImageOCRHelper
from ai.ingestion.models import (
    DocxParseError,
    Document,
    ImageExtractionError,
    IngestionError,
    OCRError,
    PageContent,
    PDFParseError,
    PptxParseError,
    TableExtractionError,
    UnsupportedFormatError,
)
from ai.ingestion.ocr_processor import OCRProcessor
from ai.ingestion.pdf_reader import PDFReader
from ai.ingestion.pptx_reader import PptxReader
from ai.ingestion.structure import assign_order_and_paths, compute_statistics, make_source_info, render_blocks
from ai.ingestion.table_extractor import TableExtractor

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".pptx": "pptx"}
MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}
_PDF_LOCK = threading.Lock()  # the PDF layers keep per-document state
_PARSE_ERRORS: dict[str, type[IngestionError]] = {"pdf": PDFParseError, "docx": DocxParseError, "pptx": PptxParseError}


def detect_file_type(path: str | Path, mime_type: str | None = None) -> str:
    """``"pdf"`` / ``"docx"`` / ``"pptx"`` from the MIME type or extension.

    Raises:
        UnsupportedFormatError: for anything else (including legacy .doc / .ppt).
    """
    if mime_type and mime_type.split(";")[0].strip().lower() in MIME_TYPES:
        return MIME_TYPES[mime_type.split(";")[0].strip().lower()]
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[suffix]
    raise UnsupportedFormatError(f"Unsupported file type '{suffix or mime_type}': {Path(path).name}")


class IngestionPipeline:
    """Orchestrates the readers. All collaborators are injectable.

    Args:
        reader: PDF reader (layer 1).
        ocr: OCR processor (created lazily, only when a page or image needs OCR).
        table_extractor: PDF table extractor (layer 3).
        builder: PDF page builder (layer 4).
        docx_reader / pptx_reader: Office readers (default ones share ``ocr``).
        enable_ocr: Full-page OCR of scanned PDF pages.
        enable_image_ocr: OCR of embedded images (all formats).
        enable_tables: PDF table extraction.
    """

    def __init__(
        self,
        reader: PDFReader | None = None,
        ocr: OCRProcessor | None = None,
        table_extractor: TableExtractor | None = None,
        builder: DocumentBuilder | None = None,
        docx_reader: DocxReader | None = None,
        pptx_reader: PptxReader | None = None,
        enable_ocr: bool = True,
        enable_image_ocr: bool = True,
        enable_tables: bool = True,
    ) -> None:
        self.reader = reader or PDFReader()
        self._ocr = ocr
        self.table_extractor = table_extractor or TableExtractor()
        self.builder = builder or DocumentBuilder(min_text_chars=self.reader.min_text_chars)
        self.enable_ocr = enable_ocr
        self.enable_image_ocr = enable_image_ocr
        self.enable_tables = enable_tables
        image_helper = ImageOCRHelper(_LazyOCR(lambda: self.ocr)) if enable_image_ocr else None
        self.docx_reader = docx_reader or DocxReader(ocr=image_helper)
        self.pptx_reader = pptx_reader or PptxReader(ocr=image_helper)

    @property
    def ocr(self) -> OCRProcessor:
        """Lazily create the OCR processor so text-only files never touch Tesseract."""
        if self._ocr is None:
            self._ocr = OCRProcessor()
        return self._ocr

    # ---------------------------------------------------------------- public
    def ingest(self, path: str | Path, mime_type: str | None = None) -> Document:
        """Ingest a PDF, DOCX or PPTX file.

        Raises:
            UnsupportedFormatError: unknown file type.
            PDFParseError / DocxParseError / PptxParseError: the file cannot be opened.
        """
        file_type = detect_file_type(path, mime_type)
        p = Path(path)
        if not p.is_file():
            raise _PARSE_ERRORS[file_type](f"File not found: {p}")
        logger.info("Starting ingestion (%s): %s", file_type, p.name)
        if file_type == "pdf":
            with _PDF_LOCK:
                pages, metadata = self._read_pdf(p)
            unit, parser = "page", "pymupdf+pdfplumber"
        elif file_type == "docx":
            pages, metadata = self.docx_reader.read(p)
            unit, parser = "document", self.docx_reader.parser_name
        else:
            pages, metadata = self.pptx_reader.read(p)
            unit, parser = "slide", self.pptx_reader.parser_name
        return self._finish(p, file_type, unit, parser, pages, metadata)

    def _finish(self, path: Path, file_type: str, unit: Any, parser: str,
                pages: list[PageContent], metadata: dict[str, Any]) -> Document:
        source = make_source_info(path, file_type, unit, len(pages), parser, metadata)
        document = Document(source_path=str(path), pages=pages, metadata=metadata, source=source)
        assign_order_and_paths(pages, reset_per_page=(unit == "slide"))
        document.statistics = compute_statistics(document)
        logger.info("Document built: %s blocks=%d", path.name, document.statistics["block_count"])
        return document

    # ------------------------------------------------------------------- PDF
    def _read_pdf(self, path: Path) -> tuple[list[PageContent], dict[str, Any]]:
        fitz_doc = self.reader.open(path)
        metadata = self.reader.extract_metadata(fitz_doc)
        pages: list[PageContent] = []
        if self._ocr is not None:
            self._ocr.clear_cache()
        try:
            self.reader.build_profile(fitz_doc)
            tables_ready = self._open_tables(path)
            for fitz_page in fitz_doc:
                pages.append(self._process_page(fitz_doc, fitz_page, tables_ready))
        finally:
            fitz_doc.close()
            self.table_extractor.close()
        return pages, metadata

    def _open_tables(self, path: Path) -> bool:
        if not self.enable_tables:
            return False
        try:
            self.table_extractor.open(path)
            return True
        except TableExtractionError as exc:
            logger.warning("Table layer disabled for this file: %s", exc)
            return False

    def _process_page(self, fitz_doc: Any, fitz_page: Any, tables_ready: bool) -> PageContent:
        number = fitz_page.number + 1
        try:
            page = self.reader.analyze_page(fitz_page)  # Layer 1
            if page.needs_ocr and self.enable_ocr:  # Layer 2a - whole page
                self._run_ocr(fitz_page, page)
            if self.enable_image_ocr and page.images and not page.ocr_applied:  # Layer 2b - images
                self._run_image_ocr(fitz_doc, page)
            if tables_ready and not page.ocr_applied:  # Layer 3 - needs a real text layer
                self._run_tables(fitz_page, page)
            return self.builder.finalize_page(page)  # Layer 4
        except Exception as exc:  # one bad page must never kill the document
            logger.exception("Page %d failed: %s", number, exc)
            failed = PageContent(page_number=number, confidence=0.0)
            failed.warnings.append(f"Page processing failed: {exc}")
            return failed

    def _run_ocr(self, fitz_page: Any, page: PageContent) -> None:
        try:
            result = self.ocr.process(fitz_page)
        except OCRError as exc:
            logger.error("%s", exc)
            page.warnings.append(str(exc))
            return
        page.ocr_text = result.text
        page.ocr_confidence = result.confidence
        page.ocr_applied = True
        page.blocks = []  # the (near-empty) text layer is replaced by the OCR text

    def _run_image_ocr(self, fitz_doc: Any, page: PageContent) -> None:
        """OCR each embedded image separately; repeated logos/banners are skipped."""
        repeated = self.reader.profile.repeated_xrefs
        for image in page.images:
            if image.xref in repeated:
                image.ocr_skipped_reason = "repeated_image"
                continue
            try:
                pil_image = self.reader.load_image(fitz_doc, image)
                result = self.ocr.process_image(pil_image, cache_key=image.xref)
            except (ImageExtractionError, OCRError) as exc:
                logger.warning("Image OCR skipped on page %d: %s", page.page_number, exc)
                page.warnings.append(str(exc))
                if isinstance(exc, OCRError) and not self.ocr.available:
                    return  # no engine => no point trying the remaining images
                continue
            image.ocr_applied = result.applied
            image.ocr_text = clean_text(result.text)
            image.ocr_confidence = result.confidence if result.applied else None
            image.ocr_skipped_reason = result.skipped_reason

    def _run_tables(self, fitz_page: Any, page: PageContent) -> None:
        try:
            page.tables = self.table_extractor.extract(page.page_number)
        except TableExtractionError as exc:
            logger.warning("%s", exc)
            page.warnings.append(str(exc))
            return
        if page.tables:
            boxes = [t.bbox for t in page.tables if t.bbox]
            page.blocks = self.reader.extract_blocks(fitz_page, exclude_bboxes=boxes)
            page.text = render_blocks(page.blocks)


class _LazyOCR:
    """Defers creating the OCR engine until an image actually needs OCR."""

    def __init__(self, factory: Callable[[], OCRProcessor]) -> None:
        self._factory = factory

    @property
    def available(self) -> bool:
        return self._factory().available

    def process_image(self, image: Any, cache_key: object | None = None) -> Any:
        return self._factory().process_image(image, cache_key)

