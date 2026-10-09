"""Hybrid PDF ingestion pipeline - the single entry point for the rest of IntelliStudy.

Flow per page::

    PDFReader (L1) -> page has no text layer?  -> OCRProcessor.process        (whole page)
                   -> else, page has images?   -> OCRProcessor.process_image  (each image)
                   -> else/also                -> TableExtractor (L3) -> re-read text without tables
                   -> DocumentBuilder.finalize_page (L4)

Usage::

    from ai.ingestion.ingestion_pipeline import IngestionPipeline

    document = IngestionPipeline().ingest("lecture.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai.ingestion.document_builder import DocumentBuilder
from ai.ingestion.models import (
    Document,
    ImageExtractionError,
    OCRError,
    PageContent,
    TableExtractionError,
)
from ai.ingestion.ocr_processor import OCRProcessor
from ai.ingestion.pdf_reader import PDFReader
from ai.ingestion.table_extractor import TableExtractor

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the four layers. All collaborators are injectable.

    Args:
        reader: Layer 1 PyMuPDF reader.
        ocr: Layer 2 OCR processor (created lazily, only if a page needs OCR).
        table_extractor: Layer 3 pdfplumber extractor.
        builder: Layer 4 document builder.
        enable_ocr: Disable to skip full-page OCR of scanned pages.
        enable_image_ocr: Disable to skip OCR of embedded images on pages that
            already have a text layer.
        enable_tables: Disable to skip table extraction entirely.
    """

    def __init__(
        self,
        reader: PDFReader | None = None,
        ocr: OCRProcessor | None = None,
        table_extractor: TableExtractor | None = None,
        builder: DocumentBuilder | None = None,
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

    @property
    def ocr(self) -> OCRProcessor:
        """Lazily create the OCR processor so text-only PDFs never touch Tesseract."""
        if self._ocr is None:
            self._ocr = OCRProcessor()
        return self._ocr

    # ---------------------------------------------------------------- public
    def ingest(self, path: str | Path) -> Document:
        """Run the full pipeline on a PDF and return a :class:`Document`.

        Raises:
            PDFParseError: if the file cannot be opened.
        """
        logger.info("Starting ingestion: %s", path)
        fitz_doc = self.reader.open(path)
        metadata = self.reader.extract_metadata(fitz_doc)
        pages: list[PageContent] = []
        if self._ocr is not None:
            self._ocr.clear_cache()  # image results are cached per document only
        try:
            tables_ready = self._open_tables(path)
            for fitz_page in fitz_doc:
                pages.append(self._process_page(fitz_doc, fitz_page, tables_ready))
        finally:
            fitz_doc.close()
            self.table_extractor.close()
        return self.builder.build(str(path), pages, metadata)

    # --------------------------------------------------------------- private
    def _open_tables(self, path: str | Path) -> bool:
        if not self.enable_tables:
            return False
        try:
            self.table_extractor.open(path)
            return True
        except TableExtractionError as exc:
            logger.warning("Table layer disabled for this file: %s", exc)
            return False

    def _process_page(self, fitz_doc, fitz_page, tables_ready: bool) -> PageContent:  # noqa: ANN001 - pymupdf types
        number = fitz_page.number + 1
        try:
            page = self.reader.analyze_page(fitz_page)  # Layer 1

            if page.needs_ocr and self.enable_ocr:  # Layer 2a - whole page, only when no text layer
                self._run_ocr(fitz_page, page)

            # Layer 2b - embedded images, only when the page was NOT OCR'd as a whole
            # (otherwise the image text is already in the page OCR => no duplicates).
            if self.enable_image_ocr and page.images and not page.ocr_applied:
                self._run_image_ocr(fitz_doc, page)

            if tables_ready and not page.ocr_applied:  # Layer 3 - needs a real text layer
                self._run_tables(fitz_page, page)

            return self.builder.finalize_page(page)  # Layer 4
        except Exception as exc:  # one bad page must never kill the document
            logger.exception("Page %d failed: %s", number, exc)
            failed = PageContent(page_number=number, confidence=0.0)
            failed.warnings.append(f"Page processing failed: {exc}")
            return failed

    def _run_ocr(self, fitz_page, page: PageContent) -> None:  # noqa: ANN001
        try:
            result = self.ocr.process(fitz_page)
            page.ocr_text = result.text
            page.ocr_confidence = result.confidence
            page.ocr_applied = True
        except OCRError as exc:
            logger.error("%s", exc)
            page.warnings.append(str(exc))

    def _run_image_ocr(self, fitz_doc, page: PageContent) -> None:  # noqa: ANN001
        """OCR each embedded image separately and store the result on the image."""
        for image in page.images:
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
            image.ocr_text = result.text
            image.ocr_confidence = result.confidence if result.applied else None
            image.ocr_skipped_reason = result.skipped_reason

    def _run_tables(self, fitz_page, page: PageContent) -> None:  # noqa: ANN001
        try:
            page.tables = self.table_extractor.extract(page.page_number)
        except TableExtractionError as exc:
            logger.warning("%s", exc)
            page.warnings.append(str(exc))
            return
        if page.tables:
            # Re-read the text with table regions removed so tables aren't duplicated.
            boxes = [t.bbox for t in page.tables if t.bbox]
            page.text = self.reader.extract_text(fitz_page, exclude_bboxes=boxes)
