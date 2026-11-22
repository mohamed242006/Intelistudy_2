"""Chunking / indexing exceptions."""

from __future__ import annotations


class ChunkingError(Exception):
    """Invalid input or configuration for chunking."""


class ChunkValidationError(ChunkingError):
    """The produced chunks violate an invariant (oversize, duplicate id, dangling parent...)."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems[:10]) + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
        self.problems = problems


class IndexingError(Exception):
    """Embedding or storage failed while indexing a document."""


class VectorStoreError(Exception):
    """Invalid vector-store operation (bad filter, missing scope, wrong dimension...)."""
