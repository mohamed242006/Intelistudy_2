"""Block selection and section building."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.chunking.config import ChunkingConfig
from ai.chunking.tokens import TokenCounter
from ai.ingestion.cleaners import has_content
from ai.ingestion.models import Block, BlockType, Document, SourceKind

RICH_TYPES = frozenset({BlockType.TABLE, BlockType.CHART, BlockType.NOTES, BlockType.IMAGE_OCR})


@dataclass(slots=True)
class SelectionReport:
    kept: list[Block]
    dropped_hidden: int = 0
    dropped_low_ocr: int = 0
    dropped_empty: int = 0
    dropped_title_only_pages: list[int] = field(default_factory=list)


class BlockSelector:
    """Decides which blocks reach the chunker.

    * hidden slides are excluded (unless configured otherwise);
    * blocks without letters or digits are dropped;
    * image-OCR blocks below ``ocr_drop_below`` confidence are dropped;
    * title-only slides (< ``title_only_min_tokens`` body tokens and no rich content) are dropped.
    """

    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        self.config = config
        self.counter = counter

    def select(self, document: Document) -> SelectionReport:
        report = SelectionReport(kept=[])
        for page in document.pages:
            if page.hidden and not self.config.index_hidden_slides:
                report.dropped_hidden += len(page.blocks)
                continue
            blocks: list[Block] = []
            for block in page.blocks:
                if not has_content(block.text):
                    report.dropped_empty += 1
                elif block.type is BlockType.IMAGE_OCR and block.confidence < self.config.ocr_drop_below:
                    report.dropped_low_ocr += 1
                elif block.source is not SourceKind.NATIVE and block.confidence < self.config.ocr_drop_below:
                    report.dropped_low_ocr += 1
                else:
                    blocks.append(block)
            if document.unit == "slide" and blocks and self._title_only(blocks):
                report.dropped_title_only_pages.append(page.page_number)
                continue
            report.kept.extend(blocks)
        return report

    def _title_only(self, blocks: list[Block]) -> bool:
        if any(b.type in RICH_TYPES for b in blocks):
            return False
        body = sum(self.counter.count(b.text) for b in blocks if b.type is not BlockType.HEADING)
        return body < self.config.title_only_min_tokens


@dataclass(slots=True)
class SectionPart:
    """The content under one heading."""

    heading_path: list[str]
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    """A chunking unit: one or more consecutive parts (several after small-section merging)."""

    heading_path: list[str]
    parts: list[SectionPart]
    page_key: int | None = None  # slide number when sections must not cross slides

    @property
    def title(self) -> str:
        return self.parts[0].title if self.parts else ""

    @property
    def blocks(self) -> list[Block]:
        return [b for part in self.parts for b in part.blocks]


def build_sections(blocks: list[Block], slide_bound: bool) -> list[Section]:
    """Start a section at every heading (and, for slides, at every slide boundary)."""
    sections: list[Section] = []
    current: Section | None = None
    for block in blocks:
        new_slide = slide_bound and current is not None and current.page_key != block.page_number
        if block.type is BlockType.HEADING or current is None or new_slide:
            if block.type is BlockType.HEADING:
                path, title = list(block.heading_path), block.text
            else:
                path, title = list(block.heading_path), (block.heading_path[-1] if block.heading_path else "")
            current = Section(heading_path=path, parts=[SectionPart(path, title)],
                              page_key=block.page_number if slide_bound else None)
            sections.append(current)
            if block.type is BlockType.HEADING:
                continue
        current.parts[-1].blocks.append(block)
    return sections


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def merge_small_sections(sections: list[Section], config: ChunkingConfig, counter: TokenCounter) -> list[Section]:
    """Merge sections below ``min_tokens`` into a neighbour when the result stays within the target.

    The merged section's heading path is the common prefix; the merged parts keep their titles as
    text lines (chunker 1.1.0 behaviour). Sections never merge across slides.
    """
    def tokens(section: Section) -> int:
        total = 0
        for part in section.parts:
            if part.title:
                total += counter.count(part.title)
            total += sum(counter.count(b.text) for b in part.blocks)
        return total

    def can_merge(a: Section, b: Section) -> bool:
        return a.page_key == b.page_key and tokens(a) + tokens(b) <= config.target_tokens

    def merge(a: Section, b: Section) -> Section:
        return Section(_common_prefix(a.heading_path, b.heading_path), a.parts + b.parts, a.page_key)

    merged: list[Section] = []
    for section in sections:
        if merged and tokens(merged[-1]) < config.min_tokens and can_merge(merged[-1], section):
            merged[-1] = merge(merged[-1], section)
        else:
            merged.append(section)
    if len(merged) >= 2 and tokens(merged[-1]) < config.min_tokens and can_merge(merged[-2], merged[-1]):
        last = merged.pop()
        merged[-1] = merge(merged[-1], last)
    return [s for s in merged if s.blocks]
