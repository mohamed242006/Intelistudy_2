"""Programmatic test documents (PDF, DOCX, PPTX) and fakes."""

from __future__ import annotations

import hashlib
import io
import math
import tempfile
from pathlib import Path
from typing import Sequence

import pymupdf as fitz
from PIL import Image, ImageDraw

from ai.ingestion.ocr_processor import OCRResult

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS = REPO_ROOT / "data" / "uploads"


class FakeOCR:
    """Stands in for OCRProcessor: every image 'contains' a fixed text."""

    available = True

    def __init__(self, text: str = "Diagram showing the request flow", confidence: float = 91.0) -> None:
        self.text = text
        self.confidence = confidence
        self.calls = 0

    def process_image(self, image: Image.Image, cache_key: object | None = None) -> OCRResult:
        self.calls += 1
        return OCRResult(self.text, self.confidence)

    def process(self, page: object) -> OCRResult:
        self.calls += 1
        return OCRResult(self.text, self.confidence)

    def clear_cache(self) -> None:
        pass


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    def __init__(self, dimension: int = 64, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.calls = 0
        self.texts_embedded = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for word in text.lower().split():
            h = int(hashlib.md5(word.strip(".,:;?!()").encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        if not any(v):
            v[0] = 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def png_bytes(text: str = "HELLO", size: tuple[int, int] = (400, 200)) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 80), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="intellistudy_test_"))


# ---------------------------------------------------------------------- PDF
LOREM = ("Caching keeps recently used data close to the processor so that repeated reads are fast. "
         "A cache hit avoids a slow trip to main memory, while a miss costs extra latency. ")


def make_pdf(path: Path, pages: int = 4, with_table: bool = True, with_image: bool = False) -> Path:
    doc = fitz.open()
    for n in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 40), "CSE447 Operating Systems - Lecture Notes", fontsize=9)  # running header
        page.insert_text((290, 820), f"{n}", fontsize=9)  # page number footer
        page.insert_text((72, 110), f"Chapter {n} Memory Topic", fontsize=20, fontname="hebo")
        page.insert_text((72, 150), f"Section {n}.1 Caching Basics", fontsize=15, fontname="hebo")
        y = 180
        for line in _wrap(LOREM * 2, 85):
            page.insert_text((72, y), line, fontsize=11)
            y += 14
        y += 10
        page.insert_text((80, y), "•", fontsize=11)
        page.insert_text((95, y), "LRU evicts the least recently used entry first", fontsize=11)
        y += 14
        page.insert_text((80, y), "•", fontsize=11)
        page.insert_text((95, y), "FIFO evicts entries in insertion order and is simple", fontsize=11)
        y += 14
        page.insert_text((95, y), "to implement in hardware.", fontsize=11)
        y += 14
        page.insert_text((110, y), "-", fontsize=11)
        page.insert_text((122, y), "Used by small embedded caches", fontsize=11)
        y += 30
        if with_table and n == 2:
            y = _draw_table(page, y, [["Policy", "Hit rate"], ["LRU", "92%"], ["FIFO", "85%"]])
        if with_image and n == 1:
            rect = fitz.Rect(72, y + 10, 272, y + 110)
            page.insert_image(rect, stream=png_bytes("CACHE DIAGRAM"))
    doc.set_toc([[1, f"Chapter {i} Memory Topic", i] for i in range(1, pages + 1)])
    doc.save(path)
    doc.close()
    return path


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _draw_table(page: fitz.Page, top: float, rows: list[list[str]]) -> float:
    col_w, row_h, left = 150, 22, 72
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(left + c * col_w, top + r * row_h, left + (c + 1) * col_w, top + (r + 1) * row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 5, rect.y0 + 15), cell, fontsize=10)
    return top + len(rows) * row_h + 20


def make_blank_pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------- DOCX
def make_docx(path: Path, with_image: bool = True, extra_paragraph: str | None = None) -> Path:
    from docx import Document

    d = Document()
    d.core_properties.title = "Distributed Systems Notes"
    d.core_properties.author = "Test Author"
    d.add_heading("Distributed Systems Notes", 0)
    d.add_heading("Consensus", 1)
    d.add_paragraph("Consensus lets a group of machines agree on one value even when some of them fail. "
                    "Paxos and Raft are the classic algorithms.")
    d.add_paragraph("Leader election", style="List Bullet")
    d.add_paragraph("Log replication", style="List Bullet")
    d.add_paragraph("Safety", style="List Bullet 2")
    d.add_heading("Raft", 2)
    d.add_paragraph("First step", style="List Number")
    d.add_paragraph("Second step", style="List Number")
    if extra_paragraph:
        d.add_paragraph(extra_paragraph)
    table = d.add_table(rows=3, cols=3)
    cells = [["Algorithm", "Year", "Leader"], ["Paxos", "1998", "Optional"], ["Raft", "2014", "Required"]]
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    merged = table.cell(0, 1).merge(table.cell(0, 2))
    merged.text = "Facts"
    if with_image:
        d.add_heading("Diagram", 1)
        d.add_picture(io.BytesIO(png_bytes("RAFT")))
        d.add_picture(io.BytesIO(png_bytes("RAFT")))  # duplicate image: indexed once
    d.add_heading("Summary", 1)
    d.add_paragraph("Raft was designed to be understandable.")
    d.save(path)
    return path


# --------------------------------------------------------------------- PPTX
def make_pptx(path: Path) -> Path:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches, Pt

    prs = Presentation()
    # 1 title slide
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Introduction to Security"
    s.placeholders[1].text = "Lecture 3 - classical ciphers and their weaknesses"
    # 2 title + bullets + notes
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Caesar Cipher"
    body = s.placeholders[1].text_frame
    body.text = "Shift every letter by a fixed key"
    p = body.add_paragraph()
    p.text = "Only 25 possible keys"
    p.level = 1
    p = body.add_paragraph()
    p.text = "Broken by brute force in seconds"
    s.notes_slide.notes_text_frame.text = "Mention that Julius Caesar used a shift of three."
    # 3 table
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Cipher Comparison"
    tbl = s.shapes.add_table(3, 2, Inches(1), Inches(2), Inches(6), Inches(2)).table
    for r, row in enumerate([["Cipher", "Key space"], ["Caesar", "25"], ["Vigenere", "26^m"]]):
        for c, value in enumerate(row):
            tbl.cell(r, c).text = value
    # 4 chart
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Letter Frequencies"
    data = CategoryChartData()
    data.categories = ["E", "T", "A"]
    data.add_series("English", (12.7, 9.1, 8.2))
    s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(2), Inches(6), Inches(4), data)
    # 5 blank slide: title from the largest explicit font (E2E-07)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    big = s.shapes.add_textbox(Inches(1), Inches(0.5), Inches(8), Inches(1)).text_frame
    big.text = "Frequency Analysis"
    big.paragraphs[0].runs[0].font.size = Pt(40)
    small = s.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(3)).text_frame
    small.text = "Count how often each ciphertext letter appears and match it to English frequencies."
    small.paragraphs[0].runs[0].font.size = Pt(18)
    # 6 hidden slide
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Hidden Answers"
    s.placeholders[1].text_frame.text = "The answer to exercise one is KHOOR"
    s._element.set("show", "0")
    # 7 title only
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Questions?"
    # 8 picture
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Cipher Wheel"
    s.shapes.add_picture(io.BytesIO(png_bytes("WHEEL")), Inches(1), Inches(2))
    prs.save(path)
    return path
