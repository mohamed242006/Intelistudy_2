"""OCR of embedded images from raw bytes (DOCX / PPTX), with per-document caching and scoring.

Item scores feed page confidence:

* OCR produced text: 0.85-1.0, scaled by Tesseract's mean word confidence;
* intentional skip (tiny, thin, no text, low confidence): 0.60-0.85;
* unsupported format (EMF/WMF, ...): 0.70 - a mild skip, not a failure;
* decode or engine error: 0.0.

A block's own confidence is the engine confidence / 100, so the chunker can drop (< 0.3) or flag
(< 0.6) weak OCR text.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from ai.ingestion.models import OCRError

logger = logging.getLogger(__name__)

UNSUPPORTED_EXTENSIONS = frozenset({"emf", "wmf", "svg", "eps", "pict", "x-emf", "x-wmf"})
SKIP_SCORES = {"too_small": 0.85, "too_thin": 0.85, "no_text": 0.75, "low_confidence": 0.60}
UNSUPPORTED_SCORE = 0.70


class ImageOCREngine(Protocol):
    """What this module needs from :class:`~ai.ingestion.ocr_processor.OCRProcessor`."""

    available: bool

    def process_image(self, image: Image.Image, cache_key: object | None = None):  # noqa: ANN201
        ...


@dataclass(slots=True)
class ImageOCROutcome:
    text: str = ""
    engine_confidence: float | None = None  # 0-100
    score: float = 1.0  # item score for page confidence
    applied: bool = False
    skipped_reason: str | None = None
    warning: str | None = None
    content_hash: str = ""
    width: int = 0
    height: int = 0

    @property
    def block_confidence(self) -> float:
        return round(min((self.engine_confidence or 0.0) / 100.0, 1.0), 3)


class ImageOCRHelper:
    """Decodes image bytes, runs OCR once per distinct image (sha1) and scores the outcome."""

    def __init__(self, engine: ImageOCREngine | None) -> None:
        self.engine = engine
        self._cache: dict[str, ImageOCROutcome] = {}

    def reset(self) -> None:
        self._cache.clear()

    def seen(self, blob: bytes) -> bool:
        return hashlib.sha1(blob).hexdigest() in self._cache

    def ocr(self, blob: bytes, extension: str = "") -> ImageOCROutcome:
        digest = hashlib.sha1(blob).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        outcome = self._ocr_uncached(blob, extension.lower().lstrip("."))
        outcome.content_hash = digest
        self._cache[digest] = outcome
        return outcome

    def _ocr_uncached(self, blob: bytes, ext: str) -> ImageOCROutcome:
        if ext in UNSUPPORTED_EXTENSIONS:
            return ImageOCROutcome(score=UNSUPPORTED_SCORE, skipped_reason="unsupported_format")
        try:
            image = Image.open(io.BytesIO(blob))
            image.load()
            image = _to_rgb(image)
        except Exception as exc:  # PIL raises many types
            if ext in {"", "bin"}:
                return ImageOCROutcome(score=UNSUPPORTED_SCORE, skipped_reason="unsupported_format")
            return ImageOCROutcome(score=0.0, skipped_reason="decode_error", warning=f"Cannot decode image: {exc}")
        if self.engine is None or not getattr(self.engine, "available", False):
            return ImageOCROutcome(score=UNSUPPORTED_SCORE, skipped_reason="ocr_unavailable",
                                   width=image.width, height=image.height)
        try:
            result = self.engine.process_image(image)
        except OCRError as exc:
            return ImageOCROutcome(score=0.0, skipped_reason="ocr_error", warning=str(exc),
                                   width=image.width, height=image.height)
        outcome = ImageOCROutcome(
            text=result.text,
            engine_confidence=result.confidence if result.applied else None,
            applied=result.applied,
            skipped_reason=result.skipped_reason,
            width=image.width,
            height=image.height,
        )
        if result.text:
            outcome.score = round(0.85 + 0.15 * min(max(result.confidence, 0.0), 100.0) / 100.0, 3)
        else:
            outcome.score = SKIP_SCORES.get(result.skipped_reason or "", 0.75)
        return outcome


def _to_rgb(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white and convert to RGB."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        base = Image.new("RGB", rgba.size, "white")
        base.paste(rgba, mask=rgba.getchannel("A"))
        return base
    return image.convert("RGB")


def mean_score(scores: list[float]) -> float:
    """Page / slide confidence: mean of item scores, 1.0 when there are none."""
    return round(sum(scores) / len(scores), 3) if scores else 1.0
