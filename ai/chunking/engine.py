"""ChunkingEngine = BlockSelector -> build_sections -> merge_small_sections -> ChunkBuilder
-> ParentBuilder -> ChunkValidator.

Usage::

    engine = ChunkingEngine()
    result = engine.run(document, ChunkContext(user_id, course_id, document_uid, EMBEDDING_MODEL))
    result.children  # indexed (embedded) chunks
    result.parents   # expansion context, never embedded

Chunk ids depend only on the chunk's own text (``c-{sha256[:12]}``, ``.{n}`` for repeated identical
text; parents ``p-{sha256[:12]}``), so an edit to a file leaves untouched chunks with the same id
and the same vector.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from statistics import mean

from ai.chunking.builder import ChunkBuilder, Draft
from ai.chunking.config import CHUNKER_VERSION, ChunkingConfig
from ai.chunking.errors import ChunkingError, ChunkValidationError
from ai.chunking.models import ChildChunk, ChunkContext, ChunkingResult, ParentChunk
from ai.chunking.sections import BlockSelector, build_sections, merge_small_sections
from ai.chunking.text import sha_hex
from ai.chunking.tokens import TokenCounter, default_token_counter
from ai.ingestion.models import Document, SourceKind
from ai.ingestion.structure import citation_label

logger = logging.getLogger(__name__)


def _unique_ids(prefix: str, texts: list[str]) -> list[str]:
    seen: Counter[str] = Counter()
    ids: list[str] = []
    for text in texts:
        base = f"{prefix}-{sha_hex(text, 12)}"
        n = seen[base]
        seen[base] += 1
        ids.append(base if n == 0 else f"{base}.{n}")
    return ids


class ChunkValidator:
    """Checks the invariants every indexed chunk set must satisfy."""

    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        self.config = config
        self.counter = counter

    def validate(self, children: list[ChildChunk], parents: list[ParentChunk]) -> list[str]:
        problems: list[str] = []
        ids = [c.chunk_id for c in children]
        if len(ids) != len(set(ids)):
            problems.append("duplicate child chunk ids")
        parent_ids = {p.parent_chunk_id for p in parents}
        if len(parent_ids) != len(parents):
            problems.append("duplicate parent chunk ids")
        parent_children = {cid for p in parents for cid in p.child_ids}
        for c in children:
            if not c.text.strip() or not c.embedding_text.strip():
                problems.append(f"{c.chunk_id}: empty text")
            n = self.counter.count(c.embedding_text)
            if n > self.config.hard_max_tokens:
                problems.append(f"{c.chunk_id}: {n} tokens > hard max {self.config.hard_max_tokens}")
            if n != c.token_count:
                problems.append(f"{c.chunk_id}: token_count {c.token_count} != {n}")
            if c.parent_chunk_id not in parent_ids:
                problems.append(f"{c.chunk_id}: dangling parent {c.parent_chunk_id}")
            if c.chunk_id not in parent_children:
                problems.append(f"{c.chunk_id}: not listed by its parent")
            if c.page_start > c.page_end or c.block_start > c.block_end:
                problems.append(f"{c.chunk_id}: inverted page/block range")
        for p in parents:
            if p.token_count > self.config.parent_max_tokens:
                problems.append(f"{p.parent_chunk_id}: parent {p.token_count} tokens > {self.config.parent_max_tokens}")
        return problems


class ChunkingEngine:
    """Turns a canonical :class:`Document` into child and parent chunks.

    Args:
        config: Token budgets and thresholds.
        tokenizer: Token counter (defaults to the BGE tokenizer, approximate fallback).
        strict: Raise :class:`ChunkValidationError` when validation finds problems.
    """

    def __init__(self, config: ChunkingConfig | None = None, tokenizer: TokenCounter | None = None,
                 strict: bool = True) -> None:
        self.config = config or ChunkingConfig()
        self.counter = tokenizer or default_token_counter()
        self.strict = strict
        self.selector = BlockSelector(self.config, self.counter)
        self.builder = ChunkBuilder(self.config, self.counter)
        self.validator = ChunkValidator(self.config, self.counter)

    def run(self, document: Document, context: ChunkContext) -> ChunkingResult:
        if document.source is None:
            raise ChunkingError("Document has no SourceInfo; ingest it with IngestionPipeline")
        source = document.source
        unit = source.unit
        selection = self.selector.select(document)
        sections = build_sections(selection.kept, slide_bound=(unit == "slide"))
        n_sections = len(sections)
        sections = merge_small_sections(sections, self.config, self.counter)

        drafts: list[Draft] = []
        for index, section in enumerate(sections):
            drafts.extend(self.builder.build(section, index))

        children = self._children(drafts, document, context)
        parents = self._parents(children, drafts, document, context)
        for child, draft in zip(children, drafts):
            child.record_hash = self._record_hash(child)
        warnings: list[str] = []
        problems = self.validator.validate(children, parents)
        if problems:
            if self.strict:
                raise ChunkValidationError(problems)
            warnings.extend(problems)

        tokens = [c.token_count for c in children]
        stats = {
            "chunker_version": CHUNKER_VERSION,
            "tokenizer": getattr(self.counter, "name", type(self.counter).__name__),
            "children": len(children),
            "parents": len(parents),
            "sections": n_sections,
            "sections_after_merge": len(sections),
            "blocks_in": len(document.blocks),
            "blocks_kept": len(selection.kept),
            "dropped_hidden": selection.dropped_hidden,
            "dropped_low_ocr": selection.dropped_low_ocr,
            "dropped_empty": selection.dropped_empty,
            "dropped_title_only_pages": selection.dropped_title_only_pages,
            "tokens_min": min(tokens, default=0),
            "tokens_mean": round(mean(tokens), 1) if tokens else 0,
            "tokens_max": max(tokens, default=0),
            "kinds": dict(Counter(c.kind for c in children)),
            "low_confidence_chunks": sum(c.low_confidence for c in children),
        }
        logger.info("Chunked %s: %d children, %d parents", source.file_name, len(children), len(parents))
        return ChunkingResult(children=children, parents=parents, warnings=warnings, statistics=stats)

    # ---------------------------------------------------------------- children
    def _children(self, drafts: list[Draft], document: Document, ctx: ChunkContext) -> list[ChildChunk]:
        source = document.source
        assert source is not None
        ids = _unique_ids("c", [d.display for d in drafts])
        children: list[ChildChunk] = []
        for ordinal, (draft, chunk_id) in enumerate(zip(drafts, ids)):
            blocks = draft.blocks
            pages = [b.page_number for b in blocks] or [1]
            orders = [b.order for b in blocks] or [0]
            ocr_blocks = [b for b in blocks if b.source is not SourceKind.NATIVE]
            sources = {b.source.value for b in blocks} or {SourceKind.NATIVE.value}
            ocr_conf = round(min(b.confidence for b in ocr_blocks), 3) if ocr_blocks else None
            kinds = draft.kinds or {"text"}
            child = ChildChunk(
                chunk_id=chunk_id,
                document_uid=ctx.document_uid,
                doc_id=source.doc_id,
                user_id=ctx.user_id,
                course_id=ctx.course_id,
                parent_chunk_id="",
                ordinal=ordinal,
                text=draft.display,
                embedding_text=draft.embedding_text,
                token_count=self.counter.count(draft.embedding_text),
                heading_path=draft.heading_path,
                citation_label=citation_label(source.unit, min(pages), draft.heading_path, max(pages)),
                section_title=draft.section_title,
                unit=source.unit,
                page_start=min(pages),
                page_end=max(pages),
                block_start=min(orders),
                block_end=max(orders),
                block_types=sorted({b.type.value for b in blocks}),
                kind=next(iter(kinds)) if len(kinds) == 1 else "mixed",
                source_kind=next(iter(sources)) if len(sources) == 1 else "mixed",
                ocr_confidence=ocr_conf,
                low_confidence=ocr_conf is not None and ocr_conf < self.config.ocr_flag_below,
                file_name=source.file_name,
                file_type=source.file_type,
                embedding_model=ctx.embedding_model,
                chunker_version=CHUNKER_VERSION,
                overlap_tokens=draft.overlap_tokens,
            )
            child.embedding_hash = sha_hex(f"{ctx.embedding_model}\n{child.embedding_text}", 32)
            children.append(child)
        return children

    # ----------------------------------------------------------------- parents
    def _parents(self, children: list[ChildChunk], drafts: list[Draft], document: Document,
                 ctx: ChunkContext) -> list[ParentChunk]:
        source = document.source
        assert source is not None
        groups: list[list[int]] = []
        current: list[int] = []
        used = 0
        for i, draft in enumerate(drafts):
            tokens = self.counter.count(draft.body)
            same_section = current and drafts[current[-1]].section_index == draft.section_index
            if current and (not same_section or used + tokens + 1 > self.config.parent_max_tokens):
                groups.append(current)
                current, used = [], 0
            current.append(i)
            used += tokens + 1
        if current:
            groups.append(current)

        texts = ["\n".join(drafts[i].body for i in group) for group in groups]
        ids = _unique_ids("p", texts)
        parents: list[ParentChunk] = []
        for ordinal, (group, text, parent_id) in enumerate(zip(groups, texts, ids)):
            members = [children[i] for i in group]
            first = drafts[group[0]]
            page_start = min(c.page_start for c in members)
            page_end = max(c.page_end for c in members)
            parent = ParentChunk(
                parent_chunk_id=parent_id,
                document_uid=ctx.document_uid,
                doc_id=source.doc_id,
                user_id=ctx.user_id,
                course_id=ctx.course_id,
                ordinal=ordinal,
                text=text,
                token_count=self.counter.count(text),
                heading_path=first.heading_path,
                citation_label=citation_label(source.unit, page_start, first.heading_path, page_end),
                section_title=first.section_title,
                unit=source.unit,
                page_start=page_start,
                page_end=page_end,
                child_ids=[c.chunk_id for c in members],
                chunker_version=CHUNKER_VERSION,
            )
            parent.record_hash = sha_hex(json.dumps(parent.to_dict(), sort_keys=True, ensure_ascii=False), 32)
            parents.append(parent)
            for child in members:
                child.parent_chunk_id = parent_id
        return parents

    @staticmethod
    def _record_hash(child: ChildChunk) -> str:
        meta = child.to_metadata()
        meta.pop("record_hash", None)
        return sha_hex(json.dumps(meta, sort_keys=True, ensure_ascii=False) + "\n" + child.text, 32)
