"""Text data structures, cleaning, sentence splitting and chunking."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Mrs.", "Ms.", "Fig.", "No.", "approx.")
_DOT = "\u2024"  # one-dot leader, used to protect abbreviations while splitting


@dataclass(frozen=True)
class PageText:
    """Cleaned text of one page (PDF) or section (TXT)."""

    number: int
    text: str


@dataclass
class Document:
    """A loaded study document made of numbered pages/sections."""

    name: str
    source_type: str  # "pdf" or "txt"
    pages: list[PageText] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for p in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def unit_name(self) -> str:
        return "Page" if self.source_type == "pdf" else "Section"

    def label(self, number: int) -> str:
        """Human-readable location, e.g. ``Page 3`` or ``Section 2``."""
        return f"{self.unit_name} {number}"

    @property
    def fingerprint(self) -> str:
        """Stable hash of the content; used to avoid recomputing unchanged documents."""
        digest = hashlib.sha256(self.name.encode("utf-8", "ignore"))
        for page in self.pages:
            digest.update(f"\x00{page.number}:".encode())
            digest.update(page.text.encode("utf-8", "ignore"))
        return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of text together with its source metadata."""

    chunk_id: int
    document: str
    page: int
    label: str
    text: str


def normalize_text(text: str) -> str:
    """Clean extracted text while keeping paragraph and heading structure.

    * Unicode NFKC normalisation, control characters and soft hyphens removed
    * words split by end-of-line hyphenation are re-joined
    * hard-wrapped lines inside a sentence are merged
    * runs of spaces / blank lines are collapsed
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)

    merged: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        previous = merged[-1] if merged else ""
        if previous and line and not previous.endswith((".", "!", "?", ":", ";")) and line[0].islower():
            merged[-1] = f"{previous} {line}"
        else:
            merged.append(line)
    cleaned = "\n".join(merged)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (lines are always treated as boundaries)."""
    if not text or not text.strip():
        return []
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", _DOT))
    sentences: list[str] = []
    for block in re.split(r"\n+", protected):
        for part in _SENTENCE_BOUNDARY.split(block):
            sentence = part.replace(_DOT, ".").strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` characters.

    Chunks prefer to end at a sentence boundary (then at whitespace), and each
    chunk starts roughly ``overlap`` characters before the previous one ended so
    that facts spanning a boundary are not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    length = len(text)
    start = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window_start = start + int(chunk_size * 0.6)
            sentence_cut = max(text.rfind(". ", window_start, end),
                               text.rfind("? ", window_start, end),
                               text.rfind("! ", window_start, end))
            if sentence_cut != -1:
                end = sentence_cut + 1
            else:
                best = max(text.rfind("\n", window_start, end), text.rfind(" ", window_start, end))
                if best != -1:
                    end = best
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        else:
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return chunks


def chunk_pages(document: Document, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk every page separately so each chunk keeps a correct page number."""
    chunks: list[Chunk] = []
    for page in document.pages:
        for piece in chunk_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(len(chunks), document.name, page.number, document.label(page.number), piece))
    return chunks


def split_into_sections(text: str, max_chars: int = 3000) -> list[str]:
    """Group paragraphs of a plain-text file into sections (TXT has no pages)."""
    sections: list[str] = []
    current = ""
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:  # one giant paragraph: cut it up
            if current:
                sections.append(current)
                current = ""
            sections.extend(chunk_text(paragraph, max_chars, 0))
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            sections.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        sections.append(current)
    return sections
