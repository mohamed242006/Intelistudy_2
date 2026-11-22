"""Chunking engine: limits, atoms, slides, OCR rules, stable ids, parents."""

from __future__ import annotations

import unittest

from ai.chunking.config import ChunkingConfig
from ai.chunking.engine import ChunkingEngine
from ai.chunking.errors import ChunkingError
from ai.chunking.models import ChunkContext
from ai.chunking.text import split_to_fit, tail_text
from ai.chunking.tokens import ApproxTokenCounter, default_token_counter
from ai.ingestion.ingestion_pipeline import IngestionPipeline
from ai.ingestion.models import Block, BlockType, Document, PageContent, SourceInfo, SourceKind, TableData
from ai.ingestion.structure import assign_order_and_paths
from tests.fixtures import FakeOCR, make_docx, make_pdf, make_pptx, tmpdir

COUNTER = default_token_counter()
CTX = ChunkContext("user-1", "course-1", "doc-1", "BAAI/bge-small-en-v1.5")


def synthetic(blocks_per_page: list[list[Block]], unit: str = "page") -> Document:
    pages = [PageContent(page_number=i + 1, blocks=blocks) for i, blocks in enumerate(blocks_per_page)]
    for page in pages:
        for b in page.blocks:
            b.page_number = page.page_number
    assign_order_and_paths(pages, reset_per_page=unit == "slide")
    src = SourceInfo("abcdef0123456789", "0" * 64, "x.pdf", "pdf", 1, unit, len(pages), "test",  # type: ignore[arg-type]
                     "2026-01-01T00:00:00+00:00")
    return Document("x.pdf", pages=pages, source=src)


def para(text: str) -> Block:
    return Block(BlockType.PARAGRAPH, text, 1)


def heading(text: str, level: int = 1) -> Block:
    return Block(BlockType.HEADING, text, 1, level=level)


SENTENCE = "The scheduler picks the next runnable thread according to its priority and recent CPU usage. "


class TextSplitTests(unittest.TestCase):
    def test_split_to_fit_respects_budget(self) -> None:
        text = SENTENCE * 40 + "x" * 3000
        pieces = split_to_fit(text, 50, COUNTER)
        self.assertTrue(all(COUNTER.count(p) <= 50 for p in pieces))
        self.assertEqual("".join(pieces).replace(" ", ""), text.replace(" ", ""))

    def test_tail_text(self) -> None:
        text = "First sentence here. Second sentence here. Third one."
        self.assertEqual(tail_text(text, 10, COUNTER), "Second sentence here. Third one.")
        self.assertEqual(tail_text("Only one.", 10, COUNTER), "")

    def test_approx_counter_overestimates(self) -> None:
        approx = ApproxTokenCounter()
        for text in (SENTENCE * 5, "internationalization of microservices", "x = 3.14159 * r^2"):
            self.assertGreaterEqual(approx.count(text) * 1.3, COUNTER.count(text))


class EngineTests(unittest.TestCase):
    def run_engine(self, doc: Document, **cfg: int) -> list:  # noqa: ANN401
        return ChunkingEngine(ChunkingConfig(**cfg), tokenizer=COUNTER).run(doc, CTX).children

    def test_long_section_split_with_overlap_and_limits(self) -> None:
        doc = synthetic([[heading("Scheduling"), para(SENTENCE * 60), para(SENTENCE * 3)]])
        result = ChunkingEngine(tokenizer=COUNTER).run(doc, CTX)
        children = result.children
        self.assertGreater(len(children), 2)
        self.assertTrue(all(c.token_count <= 480 for c in children))
        self.assertTrue(all(c.embedding_text.startswith("Scheduling\n") for c in children))
        self.assertTrue(any(c.overlap_tokens > 0 for c in children[1:]))
        self.assertTrue(all(c.citation_label == "p. 1 > Scheduling" for c in children))
        for parent in result.parents:
            self.assertLessEqual(parent.token_count, 1600)

    def test_table_is_atom_and_split_with_header(self) -> None:
        rows = [["Process", "State", "Priority"]] + [[f"proc-{i}", "ready", str(i)] for i in range(150)]
        table = TableData(1, 0, rows)
        doc = synthetic([[heading("Process Table"), para("Intro text before the table."),
                          Block(BlockType.TABLE, table.to_pipe_text(), 1, table=table), para("After.")]])
        children = self.run_engine(doc)
        tables = [c for c in children if c.kind == "table"]
        self.assertGreater(len(tables), 1)
        for c in tables:
            self.assertTrue(c.text.startswith("[Table part"))
            self.assertIn("Process | State | Priority", c.text)
            self.assertIn("Process: proc-", c.embedding_text)
            self.assertLessEqual(c.token_count, 480)
        self.assertFalse(any("Intro text" in c.text for c in tables))

    def test_wide_row_and_unbreakable_token(self) -> None:
        table = TableData(1, 0, [["Key", "Value"], ["blob", "word " * 900]])
        doc = synthetic([[Block(BlockType.TABLE, table.to_pipe_text(), 1, table=table), para("z" * 5000)]])
        children = self.run_engine(doc)
        self.assertTrue(all(c.token_count <= 480 for c in children))

    def test_small_sections_merge_and_keep_titles(self) -> None:
        doc = synthetic([[heading("Memory"), heading("Paging", 2), para("Pages are fixed-size blocks."),
                          heading("Segmentation", 2), para("Segments have variable size.")]])
        children = self.run_engine(doc)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].heading_path, ["Memory"])
        self.assertIn("Paging", children[0].text)
        self.assertIn("Segmentation", children[0].text)

    def test_low_confidence_ocr(self) -> None:
        weak = Block(BlockType.IMAGE_OCR, "blurry diagram text", 1, confidence=0.45, source=SourceKind.OCR_IMAGE)
        junk = Block(BlockType.IMAGE_OCR, "n0ise ~~", 1, confidence=0.2, source=SourceKind.OCR_IMAGE)
        doc = synthetic([[heading("Diagram"), para("The figure below shows the pipeline."), weak, junk]])
        result = ChunkingEngine(tokenizer=COUNTER).run(doc, CTX)
        text = " ".join(c.text for c in result.children)
        self.assertNotIn("n0ise", text)
        flagged = [c for c in result.children if c.low_confidence]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0].ocr_confidence, 0.45)
        self.assertEqual(result.statistics["dropped_low_ocr"], 1)

    def test_stable_ids_survive_edits(self) -> None:
        sections = [[heading(f"Topic {i}"), para(SENTENCE * 6 + f"Unique fact number {i}.")] for i in range(6)]
        original = self.run_engine(synthetic([[b for s in sections for b in s]]))
        edited_sections = sections[:2] + [[heading("Inserted"), para(SENTENCE * 6 + "Brand new.")]] + sections[2:]
        edited = self.run_engine(synthetic([[b for s in edited_sections for b in s]]))
        before = {c.chunk_id for c in original}
        after = {c.chunk_id for c in edited}
        self.assertEqual(len(before - after), 0)
        self.assertEqual(len(after - before), 1)

    def test_duplicate_text_gets_suffix(self) -> None:
        doc = synthetic([[heading("A"), para(SENTENCE * 6)], [heading("B"), para(SENTENCE * 6)]], unit="slide")
        ids = [c.chunk_id for c in self.run_engine(doc)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(ids[1].startswith(ids[0]) and ids[1].endswith(".1"))

    def test_context_validation(self) -> None:
        with self.assertRaises(ChunkingError):
            ChunkContext("", "c", "d", "m")
        with self.assertRaises(ChunkingError):
            ChunkContext("u:1", "c", "d", "m")


class RealFormatChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        d = tmpdir()
        pipeline = IngestionPipeline(ocr=FakeOCR())
        engine = ChunkingEngine(tokenizer=COUNTER)
        cls.results = {}
        for name, path in (("pdf", make_pdf(d / "a.pdf", with_image=True)), ("docx", make_docx(d / "b.docx")),
                           ("pptx", make_pptx(d / "c.pptx"))):
            cls.results[name] = engine.run(pipeline.ingest(path), CTX)

    def test_slides_never_cross_and_rules(self) -> None:
        result = self.results["pptx"]
        for c in result.children:
            self.assertEqual(c.page_start, c.page_end)
        text = " ".join(c.text for c in result.children)
        self.assertNotIn("KHOOR", text)  # hidden slide excluded
        self.assertEqual(result.statistics["dropped_title_only_pages"], [7])
        kinds = {c.kind for c in result.children}
        self.assertTrue({"table", "chart", "notes"} <= kinds)
        notes = next(c for c in result.children if c.kind == "notes")
        self.assertEqual(notes.citation_label, "slide 2 > Caesar Cipher")

    def test_docx_citations_use_heading_path(self) -> None:
        labels = [c.citation_label for c in self.results["docx"].children]
        self.assertTrue(all(not label.startswith("p.") for label in labels))
        # merged small sections are cited by the headings the chunk actually covers
        self.assertEqual(labels[0], "Distributed Systems Notes > Consensus")

    def test_every_child_has_parent_and_metadata(self) -> None:
        for result in self.results.values():
            parent_ids = {p.parent_chunk_id for p in result.parents}
            for c in result.children:
                self.assertIn(c.parent_chunk_id, parent_ids)
                meta = c.to_metadata()
                self.assertTrue(all(isinstance(v, (str, int, float, bool)) for v in meta.values()))
                self.assertEqual(len(c.embedding_hash), 32)

    def test_chunking_is_deterministic(self) -> None:
        d = tmpdir()
        pipeline = IngestionPipeline(ocr=FakeOCR())
        engine = ChunkingEngine(tokenizer=COUNTER)
        (d / "a").mkdir()
        (d / "b").mkdir()
        a = engine.run(pipeline.ingest(make_pdf(d / "a" / "x.pdf")), CTX)
        b = engine.run(pipeline.ingest(make_pdf(d / "b" / "x.pdf")), CTX)
        self.assertEqual([c.chunk_id for c in a.children], [c.chunk_id for c in b.children])
        # record_hash includes doc_id (file sha256), which differs because PyMuPDF stamps each save
        self.assertEqual([c.embedding_hash for c in a.children], [c.embedding_hash for c in b.children])


if __name__ == "__main__":
    unittest.main()
