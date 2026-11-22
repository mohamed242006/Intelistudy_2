"""Layer 3 - table extraction with pdfplumber."""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Self

import pdfplumber

from ai.ingestion.models import TableData, TableExtractionError
from ai.ingestion.cleaners import cell_clean

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extracts tables page by page. Use as a context manager per document.

    Args:
        min_rows: Minimum rows (including header) for a grid to count as a table.
        min_cols: Minimum columns.
    """

    def __init__(self, min_rows: int = 2, min_cols: int = 2) -> None:
        self.min_rows = min_rows
        self.min_cols = min_cols
        self._pdf: pdfplumber.PDF | None = None

    def open(self, path: str | Path) -> Self:
        try:
            self._pdf = pdfplumber.open(path)
        except Exception as exc:
            raise TableExtractionError(f"pdfplumber cannot open '{path}': {exc}") from exc
        return self

    def close(self) -> None:
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def extract(self, page_number: int) -> list[TableData]:
        """Return tables on a 1-indexed page (empty list if none)."""
        if self._pdf is None:
            raise TableExtractionError("TableExtractor.open() must be called first")
        try:
            page = self._pdf.pages[page_number - 1]
            found = page.find_tables()
        except Exception as exc:
            raise TableExtractionError(f"Table detection failed on page {page_number}: {exc}") from exc

        tables: list[TableData] = []
        for found_table in found:
            try:
                raw = found_table.extract()
            except Exception as exc:
                logger.warning("Table extract failed on page %d: %s", page_number, exc)
                continue
            rows = normalize_grid([[cell_clean(c) for c in row] for row in raw])
            if len(rows) < self.min_rows or max((len(r) for r in rows), default=0) < self.min_cols:
                continue
            tables.append(
                TableData(page_number=page_number, index=len(tables), rows=rows,
                          bbox=tuple(found_table.bbox))  # type: ignore[arg-type]
            )
        if tables:
            logger.info("Page %d: %d table(s) extracted", page_number, len(tables))
        return tables


def normalize_grid(rows: list[list[str]], min_cols: int = 2) -> list[list[str]]:
    """Clean a raw cell grid.

    Drops empty rows and columns, then merges adjacent *complementary* columns (never both filled in
    the same row) - the spacer columns pdfplumber produces for merged / shaded cells.
    """
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    rows = [[r[i] for i in keep] for r in rows]
    changed = True
    while changed and rows and len(rows[0]) > min_cols:
        changed = False
        for i in range(len(rows[0]) - 1):
            if all(not (r[i] and r[i + 1]) for r in rows):
                rows = [r[:i] + [r[i] or r[i + 1]] + r[i + 2:] for r in rows]
                changed = True
                break
    return rows
