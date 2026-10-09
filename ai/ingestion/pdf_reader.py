"""Layer 1 - PyMuPDF primary parser (refactored pdf_reader)."""

from __future__ import annotations

import io
import logging
import statistics
from pathlib import Path
from typing import Iterable

import pymupdf as fitz
from PIL import Image

from ai.ingestion.models import ImageData, ImageExtractionError, PageContent, PDFParseError
from ai.ingestion.cleaners import clean_text

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


class PDFReader:
    """Opens PDFs and extracts text, images and per-page metadata with PyMuPDF.

    Args:
        min_text_chars: Pages with fewer extractable characters than this are
            flagged as needing OCR.
        heading_size_ratio: A line is a heading if its font size is at least
            this multiple of the page's body font size.
    """

    def __init__(self, min_text_chars: int = 50, heading_size_ratio: float = 1.15) -> None:
        self.min_text_chars = min_text_chars
        self.heading_size_ratio = heading_size_ratio

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
        logger.info("Opened %s (%d pages)", p.name, doc.page_count)
        return doc

    @staticmethod
    def extract_metadata(doc: fitz.Document) -> dict[str, object]:
        """Return document-level metadata."""
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        meta["page_count"] = doc.page_count
        return meta

    # --------------------------------------------------------------- analyze
    def analyze_page(self, page: fitz.Page) -> PageContent:
        """Extract text/images for one page and compute detection metadata."""
        number = page.number + 1
        text = self.extract_text(page)
        images = self.extract_images(page)

        content = PageContent(page_number=number)
        content.text = text
        content.images = images
        content.image_count = len(images)
        content.text_length = len(text)
        content.has_text = content.text_length >= self.min_text_chars
        content.needs_ocr = not content.has_text
        logger.debug(
            "Page %d: %d chars, %d images, needs_ocr=%s",
            number, content.text_length, content.image_count, content.needs_ocr,
        )
        return content

    # ------------------------------------------------------------------ text
    def extract_text(self, page: fitz.Page, exclude_bboxes: Iterable[BBox] = ()) -> str:
        """Extract cleaned text, marking headings with ``## ``.

        Blocks whose centre lies inside any ``exclude_bboxes`` (e.g. tables) are
        skipped so that table content is not duplicated.
        """
        excluded = list(exclude_bboxes)
        data = page.get_text("dict", sort=True)
        blocks = [b for b in data.get("blocks", []) if b.get("type") == 0]

        sizes: list[float] = []
        for b in blocks:
            for ln in b["lines"]:
                for sp in ln["spans"]:
                    sizes.extend([sp["size"]] * max(len(sp["text"].strip()), 0))
        body_size = statistics.median(sizes) if sizes else 0.0

        out_blocks: list[str] = []
        for b in blocks:
            x0, y0, x1, y1 = b["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3] for ex in excluded):
                continue
            lines: list[str] = []
            for ln in b["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in ln["spans"]).strip()
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & 16 for s in spans)
                is_heading = (
                    body_size > 0
                    and len(line_text) <= 100
                    and not line_text.endswith((".", ",", ";"))
                    and (size >= body_size * self.heading_size_ratio or (bold and len(line_text) <= 80 and len(lines) == 0 and len(b["lines"]) == 1))
                )
                lines.append(f"## {line_text}" if is_heading else line_text)
            if lines:
                out_blocks.append("\n".join(lines))
        return clean_text("\n\n".join(out_blocks))

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
                bbox = (r.x0, r.y0, r.x1, r.y1)
            except Exception:  # bbox not available for some image types
                logger.debug("No bbox for image xref=%s on page %d", xref, page.number + 1)
            result.append(
                ImageData(page_number=page.number + 1, index=idx, xref=xref,
                          width=width, height=height, bbox=bbox, smask=smask)
            )
        return result

    @staticmethod
    def load_image(doc: fitz.Document, image: ImageData) -> Image.Image:
        """Decode one embedded image into an RGB PIL image (ready for OCR).

        Handles grayscale/CMYK colour spaces and transparency (alpha channel or
        soft mask) by compositing onto a white background.

        Raises:
            ImageExtractionError: if the image cannot be decoded.
        """
        try:
            pix = fitz.Pixmap(doc, image.xref)
            if pix.colorspace is None:  # stencil/mask-only image, nothing to read
                raise ImageExtractionError(f"Image xref={image.xref} has no colour data")
            if pix.colorspace.n != 3:  # gray, CMYK, ...
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img.load()

            if img.mode == "RGBA":
                alpha = img.getchannel("A")
                base = Image.new("RGB", img.size, "white")
                base.paste(img.convert("RGB"), mask=alpha)
                img = base
            else:
                img = img.convert("RGB")

            if image.smask:  # separate transparency mask
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
