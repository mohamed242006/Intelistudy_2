"""Scoped retrieval for IntelliStudy's indexed child chunks.

This module turns a natural-language question into query embeddings, asks the
configured vector store for scoped candidates, applies a calibrated relevance
gate, and optionally expands child hits with their parent passages.

Query rewriting, cross-encoder reranking, and answer generation belong in
higher layers. Keeping them out of this service makes retrieval measurable
and lets the same evidence contract serve Q&A, summaries, and quizzes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ai.chunking.interfaces import ParentStore, QueryHit, VectorStore
from ai.embedding.bge import Embedder

__all__ = [
    "RetrievalConfig",
    "RetrievedChunk",
    "RetrievalResult",
    "RetrievalService",
]


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Runtime controls for dense retrieval.

    min_score should be calibrated on a labeled development set. None keeps
    the gate disabled while the first retrieval benchmark is collected.
    """

    candidate_k: int = 20
    top_k: int = 5
    min_score: float | None = None
    include_parent_context: bool = True
    deduplicate_parents: bool = True

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        if self.min_score is not None and not math.isfinite(self.min_score):
            raise ValueError("min_score must be finite or None")


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A source passage ready for citation or prompt construction."""

    chunk_id: str
    document_uid: str
    score: float
    text: str
    context_text: str
    citation_label: str
    file_name: str
    parent_chunk_id: str = ""
    page_start: int | None = None
    page_end: int | None = None
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The complete, auditable result of one retrieval operation."""

    query: str
    status: str
    results: tuple[RetrievedChunk, ...]
    candidates_considered: int
    candidate_k: int
    min_score: float | None

    @property
    def evidence_found(self) -> bool:
        """Whether at least one passage passed the relevance gate."""
        return bool(self.results)


class RetrievalService:
    """Retrieve authorized evidence from the existing embedding and stores.

    The caller must provide the authenticated user's ID and the course ID.
    Those values are always passed to VectorStore.query and cannot be replaced
    by optional metadata filters.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        parent_store: ParentStore | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.parent_store = parent_store
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        course_id: str,
        document_uids: Sequence[str] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        include_parent_context: bool | None = None,
    ) -> RetrievalResult:
        """Return ranked passages for one authorized question.

        document_uids narrows the search to selected uploads. The vector store
        performs mandatory owner and course filtering. A result below min_score
        is withheld and produces status evidence_not_found.
        """
        self._require_text("query", query)
        self._require_text("user_id", user_id)
        self._require_text("course_id", course_id)

        requested_k = self.config.top_k if top_k is None else top_k
        if requested_k <= 0:
            raise ValueError("top_k must be positive")
        if requested_k > self.config.candidate_k:
            raise ValueError("top_k must not exceed configured candidate_k")

        threshold = self.config.min_score if min_score is None else min_score
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("min_score must be finite or None")

        query_vector = self.embedder.embed_query(query)
        candidates = self.vector_store.query(
            query_vector,
            user_id=user_id,
            course_id=course_id,
            k=self.config.candidate_k,
            document_uids=document_uids,
        )
        eligible = [
            hit for hit in candidates
            if self._eligible(hit, threshold)
        ]
        selected = self._select(
            eligible,
            requested_k,
            deduplicate_parents=self.config.deduplicate_parents,
        )

        use_parents = (
            self.config.include_parent_context
            if include_parent_context is None
            else include_parent_context
        )
        parents = self._load_parents(selected, user_id, course_id) if use_parents else {}
        results = tuple(self._to_result(hit, parents) for hit in selected)

        return RetrievalResult(
            query=query,
            status="evidence_found" if results else "evidence_not_found",
            results=results,
            candidates_considered=len(candidates),
            candidate_k=self.config.candidate_k,
            min_score=threshold,
        )

    @staticmethod
    def _require_text(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _eligible(hit: QueryHit, min_score: float | None) -> bool:
        try:
            score = float(hit.score)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score):
            return False
        return min_score is None or score >= min_score

    @staticmethod
    def _select(
        hits: Sequence[QueryHit],
        top_k: int,
        *,
        deduplicate_parents: bool,
    ) -> list[QueryHit]:
        selected: list[QueryHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            metadata = hit.metadata or {}
            document_uid = str(metadata.get("document_uid") or "")
            parent_id = str(metadata.get("parent_chunk_id") or "")
            key = (
                (document_uid, parent_id)
                if deduplicate_parents and parent_id
                else ("", hit.id)
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(hit)
            if len(selected) == top_k:
                break
        return selected

    def _load_parents(
        self,
        hits: Sequence[QueryHit],
        user_id: str,
        course_id: str,
    ) -> dict[tuple[str, str], str]:
        if self.parent_store is None:
            return {}

        requested: dict[str, set[str]] = {}
        for hit in hits:
            metadata = hit.metadata or {}
            document_uid = str(metadata.get("document_uid") or "")
            parent_id = str(metadata.get("parent_chunk_id") or "")
            if document_uid and parent_id:
                requested.setdefault(document_uid, set()).add(parent_id)

        parents: dict[tuple[str, str], str] = {}
        for document_uid, parent_ids in requested.items():
            records = self.parent_store.get(document_uid, sorted(parent_ids))
            for parent_id, parent in records.items():
                if parent.user_id == user_id and parent.course_id == course_id:
                    parents[(document_uid, parent_id)] = parent.text
        return parents

    @staticmethod
    def _to_result(
        hit: QueryHit,
        parents: dict[tuple[str, str], str],
    ) -> RetrievedChunk:
        metadata = dict(hit.metadata or {})
        document_uid = str(metadata.get("document_uid") or "")
        parent_id = str(metadata.get("parent_chunk_id") or "")
        context_text = parents.get((document_uid, parent_id), hit.document or "")
        return RetrievedChunk(
            chunk_id=str(metadata.get("chunk_id") or hit.id),
            document_uid=document_uid,
            score=float(hit.score),
            text=hit.document or "",
            context_text=context_text,
            citation_label=str(metadata.get("citation_label") or "document"),
            file_name=str(metadata.get("file_name") or ""),
            parent_chunk_id=parent_id,
            page_start=RetrievalService._optional_int(metadata.get("page_start")),
            page_end=RetrievalService._optional_int(metadata.get("page_end")),
            unit=str(metadata.get("unit") or ""),
            metadata=metadata,
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

