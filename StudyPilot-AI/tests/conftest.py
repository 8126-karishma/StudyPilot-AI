"""Shared fixtures. Nothing here downloads a model or needs network access."""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from studypilot.embeddings import TfidfEmbedder  # noqa: E402
from studypilot.pdf_reader import load_document  # noqa: E402
from studypilot.text_processor import Document, PageText  # noqa: E402

SAMPLE_TXT = ROOT / "data" / "sample_notes.txt"
SAMPLE_PDF = ROOT / "data" / "sample_notes.pdf"


def build_pdf(pages: list[str], **save_kwargs) -> bytes:
    """Create a small PDF in memory; an empty string makes a blank page."""
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            rc = page.insert_textbox(pymupdf.Rect(50, 50, 550, 790), text, fontsize=11, fontname="helv")
            assert rc >= 0, "fixture text does not fit on the page"
    data = pdf.tobytes(**save_kwargs)
    pdf.close()
    return data


@pytest.fixture
def pdf_factory():
    return build_pdf


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TXT.read_text(encoding="utf-8")


@pytest.fixture
def sample_doc() -> Document:
    return load_document("sample_notes.txt", SAMPLE_TXT.read_bytes())


@pytest.fixture
def sample_pdf_doc() -> Document:
    return load_document("sample_notes.pdf", SAMPLE_PDF.read_bytes())


@pytest.fixture
def small_doc() -> Document:
    return Document(
        name="small.txt",
        source_type="pdf",
        pages=[
            PageText(1, "Photosynthesis is the process plants use to turn sunlight, water and carbon dioxide "
                        "into glucose and oxygen. Chlorophyll is the green pigment that absorbs light energy."),
            PageText(2, "Mitosis is a type of cell division that produces two identical daughter cells. "
                        "Meiosis is a type of cell division that produces four cells with half the chromosomes."),
            PageText(3, "Gravity is a force that pulls objects with mass toward each other. "
                        "Friction is a force that resists the motion of surfaces sliding past each other."),
        ],
    )


@pytest.fixture
def tfidf() -> TfidfEmbedder:
    return TfidfEmbedder(min_score=0.10)


class FakeGenerator:
    """Stands in for a Hugging Face model: returns canned text and records prompts."""

    name = "fake-model"

    def __init__(self, reply: str | list[str] = "A fake answer."):
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


class BrokenGenerator:
    name = "broken-model"

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        raise RuntimeError("model exploded")


@pytest.fixture
def fake_generator():
    return FakeGenerator


@pytest.fixture
def broken_generator():
    return BrokenGenerator()
