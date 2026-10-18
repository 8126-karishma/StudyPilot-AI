import pytest

from studypilot.text_processor import (
    Document, PageText, chunk_pages, chunk_text, normalize_text, split_into_sections, split_sentences,
)


def test_normalize_whitespace_control_chars_and_soft_hyphen():
    raw = "Hello\x00   world\u00a0\u00a0again\u00ad!\r\n\r\n\r\n\r\nNext   paragraph"
    assert normalize_text(raw) == "Hello world again!\n\nNext paragraph"


def test_normalize_joins_hyphenated_and_wrapped_lines():
    raw = "Machine learn-\ning models are trained on\ndata to find patterns.\nNew Heading\nAnother line."
    cleaned = normalize_text(raw)
    assert "learning models are trained on data to find patterns." in cleaned
    assert "\nNew Heading\n" in cleaned  # headings keep their own line


def test_normalize_empty_input():
    assert normalize_text("") == ""
    assert normalize_text("  \n \t ") == ""


def test_split_sentences_handles_abbreviations_and_lines():
    text = "Use tools, e.g. Python, for tasks. The value is 3.14 here! Next line\nHeading line"
    sentences = split_sentences(text)
    assert sentences[0] == "Use tools, e.g. Python, for tasks."
    assert "The value is 3.14 here!" in sentences
    assert sentences[-1] == "Heading line"


def test_chunk_text_respects_size_and_covers_everything():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(120))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"number {i} " in joined for i in range(120))


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    for before, after in zip(chunks, chunks[1:]):
        tail_words = before.split()[-3:]
        assert any(w in after.split()[:15] for w in tail_words), "neighbouring chunks should share text"


def test_chunk_text_short_empty_and_invalid():
    assert chunk_text("short text", 100, 10) == ["short text"]
    assert chunk_text("   ", 100, 10) == []
    with pytest.raises(ValueError):
        chunk_text("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)
    with pytest.raises(ValueError):
        chunk_text("abc", 10, -1)


def test_chunk_text_always_terminates_without_spaces():
    chunks = chunk_text("x" * 1000, 100, 90)
    assert "".join(chunks).count("x") >= 1000


def test_chunk_pages_keep_page_metadata():
    doc = Document("n.pdf", "pdf", [PageText(2, "Alpha beta gamma. " * 40), PageText(5, "Delta epsilon. " * 40)])
    chunks = chunk_pages(doc, 200, 40)
    assert {c.page for c in chunks} == {2, 5}
    assert all(c.document == "n.pdf" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(c.label == f"Page {c.page}" for c in chunks)
    assert all("Delta" not in c.text for c in chunks if c.page == 2)


def test_split_into_sections():
    text = "\n\n".join(["para " * 50] * 6)
    sections = split_into_sections(text, 600)
    assert len(sections) >= 2 and all(len(s) <= 600 for s in sections)
    assert split_into_sections("   ") == []


def test_document_fingerprint_changes_with_content():
    a = Document("a.txt", "txt", [PageText(1, "one")])
    b = Document("a.txt", "txt", [PageText(1, "two")])
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == Document("a.txt", "txt", [PageText(1, "one")]).fingerprint
