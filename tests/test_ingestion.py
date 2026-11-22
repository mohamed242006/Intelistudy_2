"""Ingestion: cleaners, structure, PDF / DOCX / PPTX readers, pipeline routing."""

from __future__ import annotations

import unittest
from pathlib import Path

from ai.ingestion.cleaners import clean_inline, clean_text, has_content, normalize_unicode
from ai.ingestion.ingestion_pipeline import IngestionPipeline, detect_file_type
from ai.ingestion.models import (
    BlockType,
    DocxParseError,
    PDFParseError,
    PptxParseError,
    SourceKind,
    UnsupportedFormatError,
)
from ai.ingestion.structure import citation_label
from ai.ingestion.table_extractor import normalize_grid
from tests.fixtures import FakeOCR, make_blank_pdf, make_docx, make_pdf, make_pptx, tmpdir

CANONICAL_STAT_KEYS = {
    "page_count", "unit", "block_count", "heading_count", "list_item_count", "table_count", "image_count",
    "image_ocr_count", "chart_count", "notes_count", "hidden_pages", "ocr_pages", "total_characters",
    "average_confidence", "low_confidence_pages", "warning_count",
}


class CleanerTests(unittest.TestCase):
    def test_selective_nfkc_keeps_stem_notation(self) -> None:
        self.assertEqual(normalize_unicode("x² + H₂O = ½ ∈ ℝ"), "x² + H₂O = ½ ∈ ℝ")
        self.assertEqual(normalize_unicode("ﬁle ﬂow ＡＢＣ"), "file flow ABC")

    def test_invisible_and_surrogates_removed(self) -> None:
        self.assertEqual(clean_text("a​b­c﻿\ud800d‮"), "abcd")
        self.assertFalse(has_content("​​  ­"))

    def test_hyphen_repair_keeps_compounds(self) -> None:
        self.assertEqual(clean_inline("separa-\ntion of con-\ncerns"), "separation of concerns")
        self.assertEqual(clean_inline("Aspect-\nOriented and self-\ncontained"), "Aspect-Oriented and self-contained")

    def test_whitespace_normalised(self) -> None:
        self.assertEqual(clean_text("a \t  b\n\n\n\nc"), "a b\n\nc")


class StructureTests(unittest.TestCase):
    def test_citation_labels(self) -> None:
        self.assertEqual(citation_label("page", 4, ["Caching", "LRU"]), "p. 4 > Caching > LRU")
        self.assertEqual(citation_label("page", 4, ["A"], 5), "pp. 4-5 > A")
        self.assertEqual(citation_label("slide", 7, ["Intro"]), "slide 7 > Intro")
        self.assertEqual(citation_label("document", 1, ["Doc", "Sec"]), "Doc > Sec")
        self.assertEqual(citation_label("document", 1, []), "document")
        self.assertEqual(citation_label("page", 1, ["a", "b", "c", "d"]), "p. 1 > b > c > d")

    def test_normalize_grid_merges_spacer_columns(self) -> None:
        grid = [["", "Aspect", "", "Target"], ["Recall", "", "0.85", ""], ["", "", "", ""]]
        self.assertEqual(normalize_grid(grid), [["Aspect", "Target"], ["Recall", "0.85"]])


class PdfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = tmpdir()
        cls.ocr = FakeOCR()
        cls.doc = IngestionPipeline(ocr=cls.ocr).ingest(make_pdf(cls.dir / "notes.pdf", with_image=True))

    def test_source_and_statistics(self) -> None:
        src = self.doc.source
        assert src is not None
        self.assertEqual((src.file_type, src.unit, src.page_count), ("pdf", "page", 4))
        self.assertEqual(len(src.doc_id), 16)
        self.assertEqual(set(self.doc.statistics), CANONICAL_STAT_KEYS)

    def test_heading_levels_and_paths(self) -> None:
        page = self.doc.pages[1]
        headings = [(b.level, b.text) for b in page.blocks if b.type is BlockType.HEADING]
        self.assertEqual(headings, [(1, "Chapter 2 Memory Topic"), (2, "Section 2.1 Caching Basics")])
        para = next(b for b in page.blocks if b.type is BlockType.PARAGRAPH)
        self.assertEqual(para.heading_path, ["Chapter 2 Memory Topic", "Section 2.1 Caching Basics"])

    def test_running_header_and_page_numbers_removed(self) -> None:
        text = "\n".join(b.text for b in self.doc.blocks)
        self.assertNotIn("Lecture Notes", text)
        self.assertFalse(any(b.text.strip().isdigit() for b in self.doc.blocks))

    def test_bullets_wrapped_and_nested(self) -> None:
        items = [b for b in self.doc.pages[0].blocks if b.type is BlockType.LIST_ITEM]
        self.assertEqual(len(items), 3)
        self.assertEqual(items[1].text, "FIFO evicts entries in insertion order and is simple to implement in hardware.")
        self.assertEqual([i.level for i in items], [0, 0, 1])

    def test_table_block_not_duplicated_in_text(self) -> None:
        page = self.doc.pages[1]
        tables = [b for b in page.blocks if b.type is BlockType.TABLE]
        self.assertEqual(len(tables), 1)
        assert tables[0].table is not None
        self.assertEqual(tables[0].table.rows, [["Policy", "Hit rate"], ["LRU", "92%"], ["FIFO", "85%"]])
        prose = " ".join(b.text for b in page.blocks if b.type is not BlockType.TABLE)
        self.assertNotIn("92%", prose)

    def test_image_ocr_block(self) -> None:
        images = [b for b in self.doc.pages[0].blocks if b.type is BlockType.IMAGE_OCR]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].source, SourceKind.OCR_IMAGE)
        self.assertAlmostEqual(images[0].confidence, 0.91)

    def test_blank_page_is_not_a_failure(self) -> None:
        doc = IngestionPipeline(enable_ocr=False).ingest(make_blank_pdf(self.dir / "blank.pdf"))
        self.assertEqual(doc.statistics["block_count"], 0)
        self.assertEqual(doc.pages[0].warnings, [])


class DocxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = tmpdir()
        cls.ocr = FakeOCR()
        cls.doc = IngestionPipeline(ocr=cls.ocr).ingest(make_docx(cls.dir / "notes.docx"))

    def test_unit_and_metadata(self) -> None:
        assert self.doc.source is not None
        self.assertEqual(self.doc.source.unit, "document")
        self.assertEqual(self.doc.source.title, "Distributed Systems Notes")
        self.assertEqual(set(self.doc.statistics), CANONICAL_STAT_KEYS)

    def test_heading_levels(self) -> None:
        headings = [(b.level, b.text) for b in self.doc.blocks if b.type is BlockType.HEADING]
        self.assertEqual(headings[:3], [(0, "Distributed Systems Notes"), (1, "Consensus"), (2, "Raft")])
        raft_item = next(b for b in self.doc.blocks if b.text == "First step")
        self.assertEqual(raft_item.heading_path, ["Distributed Systems Notes", "Consensus", "Raft"])

    def test_lists(self) -> None:
        items = {b.text: b for b in self.doc.blocks if b.type is BlockType.LIST_ITEM}
        self.assertEqual(items["Leader election"].list_marker, "•")
        self.assertEqual(items["Safety"].level, 1)
        self.assertEqual((items["First step"].list_marker, items["Second step"].list_marker), ("1.", "2."))

    def test_table_in_document_order_with_merged_header(self) -> None:
        types = [b.type for b in self.doc.blocks]
        table_at = types.index(BlockType.TABLE)
        self.assertLess(table_at, [b.text for b in self.doc.blocks].index("Diagram"))
        table = self.doc.blocks[table_at].table
        assert table is not None
        self.assertEqual(table.rows[0], ["Algorithm", "Facts", "Facts"])
        self.assertEqual(table.rows[2], ["Raft", "2014", "Required"])

    def test_duplicate_image_ocr_once(self) -> None:
        self.assertEqual(sum(b.type is BlockType.IMAGE_OCR for b in self.doc.blocks), 1)
        self.assertEqual(self.ocr.calls, 1)

    def test_numbering_resolver_formats(self) -> None:
        from ai.ingestion.docs_reader import format_number
        self.assertEqual([format_number(n, f) for n, f in [(4, "lowerRoman"), (28, "upperLetter"), (3, "decimal")]],
                         ["iv", "AB", "3"])


class PptxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = tmpdir()
        cls.doc = IngestionPipeline(ocr=FakeOCR()).ingest(make_pptx(cls.dir / "deck.pptx"))

    def page(self, n: int):  # noqa: ANN201
        return self.doc.pages[n - 1]

    def test_one_page_per_slide(self) -> None:
        assert self.doc.source is not None
        self.assertEqual((self.doc.source.unit, len(self.doc.pages)), ("slide", 8))
        self.assertEqual(set(self.doc.statistics), CANONICAL_STAT_KEYS)

    def test_title_bullets_notes(self) -> None:
        blocks = self.page(2).blocks
        self.assertEqual((blocks[0].type, blocks[0].text), (BlockType.HEADING, "Caesar Cipher"))
        items = [(b.level, b.text) for b in blocks if b.type is BlockType.LIST_ITEM]
        self.assertEqual(items, [(0, "Shift every letter by a fixed key"), (1, "Only 25 possible keys"),
                                 (0, "Broken by brute force in seconds")])
        self.assertEqual(blocks[-1].type, BlockType.NOTES)

    def test_table_and_chart(self) -> None:
        table = next(b for b in self.page(3).blocks if b.type is BlockType.TABLE)
        self.assertIn("Caesar | 25", table.text)
        chart = next(b for b in self.page(4).blocks if b.type is BlockType.CHART)
        self.assertIn("Categories: E, T, A", chart.text)
        self.assertIn("E = 12.7", chart.text)

    def test_title_fallback_from_largest_font(self) -> None:
        first = self.page(5).blocks[0]
        self.assertEqual((first.type, first.text), (BlockType.HEADING, "Frequency Analysis"))
        self.assertEqual(first.metadata.get("title_source"), "fallback")

    def test_hidden_slide_flagged(self) -> None:
        self.assertTrue(self.page(6).hidden)
        self.assertEqual(self.doc.statistics["hidden_pages"], [6])

    def test_heading_paths_reset_per_slide(self) -> None:
        self.assertEqual(self.page(8).blocks[-1].heading_path, ["Cipher Wheel"])
        self.assertEqual(self.page(8).blocks[-1].type, BlockType.IMAGE_OCR)


class PipelineTests(unittest.TestCase):
    def test_routing(self) -> None:
        self.assertEqual(detect_file_type("a.PDF"), "pdf")
        self.assertEqual(detect_file_type("x.bin", "application/vnd.openxmlformats-officedocument."
                                                     "presentationml.presentation"), "pptx")
        for bad in ("a.doc", "a.ppt", "a.txt", "noext"):
            with self.assertRaises(UnsupportedFormatError):
                detect_file_type(bad)

    def test_missing_and_corrupt_files(self) -> None:
        d = tmpdir()
        pipeline = IngestionPipeline()
        with self.assertRaises(PDFParseError):
            pipeline.ingest(d / "missing.pdf")
        for name, error in (("bad.pdf", PDFParseError), ("bad.docx", DocxParseError), ("bad.pptx", PptxParseError)):
            path = Path(d / name)
            path.write_bytes(b"this is not a real file")
            with self.assertRaises(error):
                pipeline.ingest(path)
        empty = d / "empty.docx"
        empty.write_bytes(b"")
        with self.assertRaises(DocxParseError):
            pipeline.ingest(empty)

    def test_block_order_is_global_and_increasing(self) -> None:
        doc = IngestionPipeline(ocr=FakeOCR()).ingest(make_pptx(tmpdir() / "deck.pptx"))
        orders = [b.order for b in doc.blocks]
        self.assertEqual(orders, list(range(len(orders))))


if __name__ == "__main__":
    unittest.main()
