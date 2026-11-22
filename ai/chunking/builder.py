"""Section -> child-chunk drafts.

Inside a section, text blocks (paragraphs, list items, image text, merged titles) are packed up to
the target size; when a section is split, the next chunk repeats up to ``overlap_tokens`` of the
previous one. Tables, charts and speaker notes are atoms: each gets its own chunk and is only split
when it exceeds the hard limit (tables by rows with the header repeated; over-wide rows by cells;
unbreakable strings by characters). Every embedding text = heading prefix + body and stays within
``hard_max_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.chunking.config import ChunkingConfig
from ai.chunking.sections import Section
from ai.chunking.text import split_to_fit, tail_text, window_split
from ai.chunking.tokens import TokenCounter
from ai.ingestion.models import Block, BlockType, TableData

ATOM_TYPES = frozenset({BlockType.TABLE, BlockType.CHART, BlockType.NOTES})
_KIND = {
    BlockType.PARAGRAPH: "text", BlockType.LIST_ITEM: "text", BlockType.HEADING: "text",
    BlockType.TABLE: "table", BlockType.IMAGE_OCR: "image_ocr", BlockType.NOTES: "notes",
    BlockType.CHART: "chart",
}


@dataclass(slots=True)
class Unit:
    """A piece of section content: display text, embedding text and the blocks it came from."""

    display: str
    embed: str
    blocks: list[Block]
    atom: bool = False
    tokens: int = 0
    path: list[str] = field(default_factory=list)  # heading path of the part it came from


@dataclass(slots=True)
class Draft:
    """A child chunk before ids, citations and hashes are assigned."""

    section_index: int
    heading_path: list[str]
    section_title: str
    display: str  # full display text (with overlap)
    body: str  # display text without the overlap prefix (used for parents)
    embedding_text: str
    blocks: list[Block]
    overlap_tokens: int = 0
    kinds: set[str] = field(default_factory=set)


class ChunkBuilder:
    def __init__(self, config: ChunkingConfig, counter: TokenCounter) -> None:
        self.config = config
        self.counter = counter

    # ------------------------------------------------------------------ prefix
    def heading_prefix(self, heading_path: list[str]) -> str:
        """``A > B > C`` limited to ``heading_prefix_cap`` tokens (deepest headings kept)."""
        path = [h for h in heading_path if h]
        while path:
            prefix = " > ".join(path)
            if self.counter.count(prefix) <= self.config.heading_prefix_cap:
                return prefix
            if len(path) == 1:
                return split_to_fit(path[0], self.config.heading_prefix_cap, self.counter)[0]
            path = path[1:]
        return ""

    # ------------------------------------------------------------------- units
    def units(self, section: Section) -> list[Unit]:
        out: list[Unit] = []
        for part in section.parts:
            if part.title and part.heading_path != section.heading_path:
                pseudo = part.blocks[0] if part.blocks else None
                title_blocks = [pseudo] if pseudo is not None else []
                out.append(self._unit(part.title, part.title, title_blocks, atom=False, title_only=True))
            for block in part.blocks:
                if block.type is BlockType.TABLE and block.table is not None:
                    out.append(self._unit(block.render(), f"Table:\n{block.table.to_key_value_text()}", [block], True))
                elif block.type in ATOM_TYPES:
                    out.append(self._unit(block.render(), block.render(), [block], True))
                else:
                    out.append(self._unit(block.render(), block.render(), [block], False))
            for unit in out:
                if not unit.path:
                    unit.path = list(part.heading_path)
        return out

    def _unit(self, display: str, embed: str, blocks: list[Block], atom: bool, title_only: bool = False) -> Unit:
        unit = Unit(display=display, embed=embed, blocks=[] if title_only else blocks, atom=atom)
        unit.tokens = self.counter.count(embed)
        return unit

    # ------------------------------------------------------------------- build
    def build(self, section: Section, section_index: int) -> list[Draft]:
        prefix = self.heading_prefix(section.heading_path)
        prefix_tokens = self.counter.count(prefix) + 1 if prefix else 0
        budget = self.config.hard_max_tokens - prefix_tokens - 2
        target = max(min(self.config.target_tokens - prefix_tokens, budget), budget // 2)
        drafts: list[Draft] = []
        pending: list[Unit] = []
        pending_tokens = 0
        carry = ""  # overlap text from the previous text chunk
        carry_tokens = 0

        def emit(display: str, embed: str, blocks: list[Block], overlap: str = "", overlap_tokens: int = 0,
                 kinds: set[str] | None = None, paths: list[list[str]] | None = None) -> None:
            full_display = f"{overlap}\n{display}".strip() if overlap else display
            full_embed = f"{overlap}\n{embed}".strip() if overlap else embed
            embedding_text = f"{prefix}\n{full_embed}" if prefix else full_embed
            # Cite the chunk by the headings it actually contains (finer than a merged section's path).
            path = _common_prefix(paths) if paths else list(section.heading_path)
            if len(path) < len(section.heading_path):
                path = list(section.heading_path)
            title = path[-1] if path else section.title
            drafts.append(Draft(section_index, path, title, full_display, display,
                                embedding_text, _unique_blocks(blocks), overlap_tokens,
                                kinds or {_KIND[b.type] for b in blocks}))

        def flush(split: bool) -> None:
            nonlocal pending, pending_tokens, carry, carry_tokens
            real = [u for u in pending if u.display]
            if not real:
                pending, pending_tokens, carry, carry_tokens = [], 0, "", 0
                return
            display = "\n".join(u.display for u in real)
            embed = "\n".join(u.embed for u in real)
            blocks = [b for u in pending for b in u.blocks]
            kinds = {_KIND[b.type] for u in real for b in u.blocks} or {"text"}
            emit(display, embed, blocks, carry, carry_tokens, kinds, [u.path for u in real])
            carry, carry_tokens = "", 0
            pending, pending_tokens = [], 0
            if split and self.config.overlap_tokens:
                carry = tail_text(real[-1].display, self.config.overlap_tokens, self.counter)
                carry_tokens = self.counter.count(carry) if carry else 0
                if carry:  # the overlap belongs to the last block of the previous chunk
                    pending = [Unit("", "", real[-1].blocks, tokens=0)]

        for unit in units_or_empty(self.units(section)):
            if unit.atom:
                flush(split=False)
                for display, embed in self._split_atom(unit, budget):
                    emit(display, embed, unit.blocks, paths=[unit.path])
                continue
            if unit.tokens > budget - carry_tokens:
                flush(split=False)
                for tail, piece, overlap in window_split(unit.display, budget, self.config.overlap_tokens,
                                                         self.counter):
                    emit(piece, piece, unit.blocks, tail, overlap, paths=[unit.path])
                continue
            if pending_tokens + unit.tokens + len(pending) > target - carry_tokens and any(u.display for u in pending):
                flush(split=True)
            pending.append(unit)
            pending_tokens += unit.tokens
        if any(u.display for u in pending):
            flush(split=False)
        return [piece for d in drafts for piece in self._enforce_split(d, prefix, budget)]

    # ------------------------------------------------------------------- atoms
    def _split_atom(self, unit: Unit, budget: int) -> list[tuple[str, str]]:
        if unit.tokens <= budget:
            return [(unit.display, unit.embed)]
        block = unit.blocks[0]
        if block.type is BlockType.TABLE and block.table is not None:
            return self._split_table(block.table, budget)
        label = unit.display.split("\n", 1)[0] if unit.display.startswith("[") else ""
        body = unit.display.split("\n", 1)[1] if label and "\n" in unit.display else unit.display
        inner = budget - (self.counter.count(label) + 4 if label else 0)
        pieces = split_lines_to_fit(body, inner, self.counter)
        n = len(pieces)
        out = []
        for i, piece in enumerate(pieces, 1):
            head = f"{label[:-1]} part {i}/{n}]" if label.endswith("]") else label
            text = f"{head}\n{piece}" if head else piece
            out.append((text, text))
        return out

    def _split_table(self, table: TableData, budget: int) -> list[tuple[str, str]]:
        """Rows packed under the budget; the header is repeated in every part."""
        header = table.header
        header_line = " | ".join(header)
        rows = table.rows[1:] if len(table.rows) > 1 else table.rows
        groups: list[list[list[str]]] = []
        current: list[list[str]] = []
        overhead = self.counter.count(header_line) + 8
        used = overhead
        row_texts: list[tuple[list[str], str, int]] = []
        for row in rows:
            kv = table.row_key_value_text(row) if len(table.rows) > 1 else " | ".join(row)
            row_texts.append((row, kv, self.counter.count(kv) + 1))
        pieces: list[tuple[str, str]] = []
        for row, kv, n in row_texts:
            if n > budget - overhead:  # an over-wide row: split by cells / characters
                if current:
                    groups.append(current)
                    current, used = [], overhead
                for part in split_lines_to_fit(kv, budget - overhead, self.counter):
                    pieces.append((f"[Table row]\n{part}", f"Table:\n{part}"))
                continue
            if current and used + n > budget:
                groups.append(current)
                current, used = [], overhead
            current.append(row)
            used += n
        if current:
            groups.append(current)
        out: list[tuple[str, str]] = []
        total = len(groups)
        for i, group in enumerate(groups, 1):
            part = TableData(table.page_number, table.index, [header, *group] if len(table.rows) > 1 else group)
            label = f"[Table part {i}/{total}]" if total > 1 else "[Table]"
            out.append((f"{label}\n{part.to_pipe_text()}", f"Table:\n{part.to_key_value_text()}"))
        return out + pieces

    # -------------------------------------------------------------- guarantees
    def _enforce_split(self, draft: Draft, prefix: str, budget: int) -> list[Draft]:
        """Last line of defence: re-split a draft whose embedding text is over the hard limit."""
        if self.counter.count(draft.embedding_text) <= self.config.hard_max_tokens:
            return [draft]
        out: list[Draft] = []
        for piece in split_lines_to_fit(draft.display, budget, self.counter):
            embedding_text = f"{prefix}\n{piece}" if prefix else piece
            out.append(Draft(draft.section_index, draft.heading_path, draft.section_title, piece, piece,
                             embedding_text, draft.blocks, 0, draft.kinds))
        return out


def split_lines_to_fit(text: str, budget: int, counter: TokenCounter) -> list[str]:
    """Pack whole lines under ``budget``; lines that are too long are split by sentences/words/chars."""
    lines: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if counter.count(line) <= budget:
            lines.append(line)
        else:
            lines.extend(split_to_fit(line, budget, counter))
    out: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and counter.count("\n".join([*current, line])) > budget:
            out.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        out.append("\n".join(current))
    return out


def _common_prefix(paths: list[list[str]]) -> list[str]:
    out: list[str] = []
    for items in zip(*paths):
        if any(x != items[0] for x in items):
            break
        out.append(items[0])
    return out


def units_or_empty(units: list[Unit]) -> list[Unit]:
    return [u for u in units if u.display.strip()]


def _unique_blocks(blocks: list[Block]) -> list[Block]:
    seen: set[int] = set()
    out: list[Block] = []
    for block in blocks:
        if id(block) not in seen:
            seen.add(id(block))
            out.append(block)
    return out
