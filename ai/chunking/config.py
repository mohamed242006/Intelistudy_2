"""Chunking configuration and version."""

from __future__ import annotations

from dataclasses import dataclass

CHUNKER_VERSION = "1.3.0"
CHUNKER_MAJOR = CHUNKER_VERSION.split(".")[0]

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token budgets and filtering thresholds.

    Attributes:
        target_tokens: Preferred size of a child chunk's embedding text.
        hard_max_tokens: Absolute limit for the embedding text (BGE window is 512 incl. special tokens).
        overlap_tokens: Tail of the previous chunk repeated at the start of the next one when a
            section is split.
        heading_prefix_cap: Maximum tokens of the heading path prefixed to the embedding text.
        min_tokens: Sections below this merge with a neighbour.
        parent_max_tokens: Maximum size of a parent chunk.
        ocr_drop_below: Image-OCR blocks with lower confidence are dropped.
        ocr_flag_below: OCR blocks with lower confidence are kept and flagged ``low_confidence``.
        title_only_min_tokens: A slide with fewer body tokens (and no table/chart/notes/image text)
            is not indexed.
        index_hidden_slides: Index hidden PPTX slides.
    """

    target_tokens: int = 400
    hard_max_tokens: int = 480
    overlap_tokens: int = 60
    heading_prefix_cap: int = 60
    min_tokens: int = 80
    parent_max_tokens: int = 1600
    ocr_drop_below: float = 0.3
    ocr_flag_below: float = 0.6
    title_only_min_tokens: int = 6
    index_hidden_slides: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.hard_max_tokens:
            raise ValueError("target_tokens must be in (0, hard_max_tokens]")
        if self.heading_prefix_cap >= self.hard_max_tokens // 2:
            raise ValueError("heading_prefix_cap must be below half of hard_max_tokens")
        if not 0 <= self.overlap_tokens < self.target_tokens // 2:
            raise ValueError("overlap_tokens must be below half of target_tokens")
        if self.parent_max_tokens < self.hard_max_tokens:
            raise ValueError("parent_max_tokens must be >= hard_max_tokens")
