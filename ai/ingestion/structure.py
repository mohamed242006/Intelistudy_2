"""Format-independent structure helpers: block ordering, heading paths, citations, statistics,
source identity."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ai.ingestion.models import (
    Block,
    BlockType,
    Document,
    PageContent,
    SourceInfo,
    Unit,
)

LOW_CONFIDENCE_THRESHOLD = 0.6
MAX_CITATION_HEADINGS = 3


# --------------------------------------------------------------------- source
def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_source_info(
    path: str | Path,
    file_type: str,
    unit: Unit,
    page_count: int,
    parser: str,
    metadata: dict[str, Any] | None = None,
) -> SourceInfo:
    """Build :class:`SourceInfo`; ``doc_id`` is the first 16 hex chars of the file's sha256."""
    p = Path(path)
    sha = file_sha256(p)
    meta = metadata or {}
    return SourceInfo(
        doc_id=sha[:16],
        sha256=sha,
        file_name=p.name,
        file_type=file_type,
        size_bytes=p.stat().st_size,
        unit=unit,
        page_count=page_count,
        parser=parser,
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        title=str(meta.get("title") or ""),
        author=str(meta.get("author") or ""),
        created=str(meta.get("created") or meta.get("creationDate") or ""),
        modified=str(meta.get("modified") or meta.get("modDate") or ""),
        language=str(meta.get("language") or ""),
    )


# ------------------------------------------------------------- heading paths
def assign_order_and_paths(pages: Sequence[PageContent], reset_per_page: bool = False) -> None:
    """Number blocks globally and fill ``heading_path`` (in place).

    A heading of level ``L`` closes every open heading of level ``>= L``. With ``reset_per_page``
    (slides) each page starts with an empty heading stack, so slides never inherit titles.
    """
    order = 0
    stack: list[tuple[int, str]] = []
    for page in pages:
        if reset_per_page:
            stack = []
        for block in page.blocks:
            block.order = order
            order += 1
            if block.type is BlockType.HEADING:
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
            block.heading_path = [title for _, title in stack]
        if not page.title:
            page.title = next((b.text for b in page.blocks if b.type is BlockType.HEADING), "")


# ----------------------------------------------------------------- citations
def _page_part(unit: Unit, page_start: int | None, page_end: int | None) -> str:
    if unit == "document" or page_start is None:
        return ""
    end = page_end if page_end is not None else page_start
    if unit == "slide":
        return f"slide {page_start}" if end == page_start else f"slides {page_start}-{end}"
    return f"p. {page_start}" if end == page_start else f"pp. {page_start}-{end}"


def citation_label(
    unit: Unit,
    page_start: int | None,
    heading_path: Iterable[str],
    page_end: int | None = None,
    max_headings: int = MAX_CITATION_HEADINGS,
) -> str:
    """``p. 4 > Caching > LRU``, ``slide 4 > Title`` or the heading path for DOCX.

    Only the last ``max_headings`` headings are kept (deeper is more specific).
    """
    path = [h for h in heading_path if h][-max_headings:] if max_headings > 0 else []
    parts = [_page_part(unit, page_start, page_end), *path]
    label = " > ".join(p for p in parts if p)
    return label or "document"


# ---------------------------------------------------------------- statistics
def compute_statistics(document: Document) -> dict[str, Any]:
    """Canonical statistics; identical keys for every format."""
    pages = document.pages
    blocks = document.blocks
    count = lambda t: sum(b.type is t for b in blocks)  # noqa: E731
    confs = [p.confidence for p in pages]
    return {
        "page_count": len(pages),
        "unit": document.unit,
        "block_count": len(blocks),
        "heading_count": count(BlockType.HEADING),
        "list_item_count": count(BlockType.LIST_ITEM),
        "table_count": count(BlockType.TABLE),
        "image_count": sum(p.image_count or len(p.images) for p in pages),
        "image_ocr_count": count(BlockType.IMAGE_OCR),
        "chart_count": count(BlockType.CHART),
        "notes_count": count(BlockType.NOTES),
        "hidden_pages": [p.page_number for p in pages if p.hidden],
        "ocr_pages": [p.page_number for p in pages if p.ocr_applied and p.ocr_text],
        "total_characters": sum(len(b.text) for b in blocks),
        "average_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "low_confidence_pages": [p.page_number for p in pages if p.confidence < LOW_CONFIDENCE_THRESHOLD],
        "warning_count": sum(len(p.warnings) for p in pages) + len(document.warnings),
    }


def render_blocks(blocks: Iterable[Block]) -> str:
    """Render blocks as readable text (one blank line between blocks)."""
    return "\n\n".join(b.render() for b in blocks if b.text)
