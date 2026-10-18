"""Create data/sample_notes.pdf from data/sample_notes.txt (one PDF page per 'Part').

Usage: python scripts/make_sample_pdf.py
Requires PyMuPDF (already in requirements.txt). The text file is ASCII-only so the
built-in Helvetica font renders it correctly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "sample_notes.txt"
TARGET = ROOT / "data" / "sample_notes.pdf"


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    parts = [p.strip() for p in re.split(r"\n(?=Part \d+:)", text) if p.strip()]
    pdf = pymupdf.open()
    for part in parts:
        page = pdf.new_page()  # A4 by default
        rect = pymupdf.Rect(60, 60, page.rect.width - 60, page.rect.height - 60)
        fontsize = 11.0
        while page.insert_textbox(rect, part, fontsize=fontsize, fontname="helv") < 0:
            fontsize -= 0.5
            if fontsize < 6:
                raise RuntimeError("Part does not fit on a page")
    pdf.set_metadata({"title": "StudyPilot AI sample notes", "author": "StudyPilot AI"})
    pdf.save(TARGET, deflate=True)
    print(f"Wrote {TARGET} ({len(parts)} pages)")


if __name__ == "__main__":
    main()
