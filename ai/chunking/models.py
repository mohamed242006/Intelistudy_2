"""Chunk records produced by the chunking engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ai.chunking.errors import ChunkingError


@dataclass(frozen=True, slots=True)
class ChunkContext:
    """Who owns the chunks and how they will be embedded.

    ``document_uid`` identifies this upload for this user/course (two users uploading the same file
    get different ``document_uid`` values but the same ``doc_id``).
    """

    user_id: str
    course_id: str
    document_uid: str
    embedding_model: str

    def __post_init__(self) -> None:
        for name in ("user_id", "course_id", "document_uid", "embedding_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ChunkingError(f"ChunkContext.{name} must be a non-empty string")
            if ":" in value and name != "embedding_model":
                raise ChunkingError(f"ChunkContext.{name} must not contain ':'")


@dataclass(slots=True)
class ChildChunk:
    """The indexed unit: embedded and stored in the vector store."""

    chunk_id: str
    document_uid: str
    doc_id: str
    user_id: str
    course_id: str
    parent_chunk_id: str
    ordinal: int
    text: str  # display text shown to readers / the LLM
    embedding_text: str  # heading prefix + body (tables as ``Column: value``)
    token_count: int  # tokens of embedding_text
    heading_path: list[str]
    citation_label: str
    section_title: str
    unit: str
    page_start: int
    page_end: int
    block_start: int
    block_end: int
    block_types: list[str]
    kind: str  # text | table | image_ocr | notes | chart | mixed
    source_kind: str  # native | ocr_page | ocr_image | mixed
    ocr_confidence: float | None
    low_confidence: bool
    file_name: str
    file_type: str
    embedding_model: str
    chunker_version: str
    overlap_tokens: int = 0
    embedding_hash: str = ""
    record_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vector_id(self) -> str:
        return f"{self.document_uid}:{self.chunk_id}"

    def to_metadata(self) -> dict[str, str | int | float | bool]:
        """Flat scalar metadata for vector stores (lists as JSON, ``None`` dropped)."""
        meta: dict[str, str | int | float | bool] = {
            "chunk_id": self.chunk_id,
            "document_uid": self.document_uid,
            "doc_id": self.doc_id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "parent_chunk_id": self.parent_chunk_id,
            "ordinal": self.ordinal,
            "token_count": self.token_count,
            "heading_path": json.dumps(self.heading_path, ensure_ascii=False),
            "citation_label": self.citation_label,
            "section_title": self.section_title,
            "unit": self.unit,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "block_types": json.dumps(self.block_types),
            "kind": self.kind,
            "source_kind": self.source_kind,
            "low_confidence": self.low_confidence,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "embedding_model": self.embedding_model,
            "chunker_version": self.chunker_version,
            "overlap_tokens": self.overlap_tokens,
            "embedding_hash": self.embedding_hash,
            "record_hash": self.record_hash,
        }
        if self.ocr_confidence is not None:
            meta["ocr_confidence"] = self.ocr_confidence
        return meta

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParentChunk:
    """Expansion context for retrieval: consecutive children of one section. Never embedded."""

    parent_chunk_id: str
    document_uid: str
    doc_id: str
    user_id: str
    course_id: str
    ordinal: int
    text: str
    token_count: int
    heading_path: list[str]
    citation_label: str
    section_title: str
    unit: str
    page_start: int
    page_end: int
    child_ids: list[str]
    chunker_version: str
    record_hash: str = ""

    @property
    def store_key(self) -> str:
        return f"{self.document_uid}:{self.parent_chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParentChunk":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})  # type: ignore[attr-defined]


@dataclass(slots=True)
class ChunkingResult:
    children: list[ChildChunk]
    parents: list[ParentChunk]
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
