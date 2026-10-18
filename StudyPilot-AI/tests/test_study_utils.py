from studypilot.study_utils import (
    escape_markdown, estimate_reading_minutes, extract_definitions, extract_key_concepts,
    format_file_size, match_definition, top_words,
)
from studypilot.text_processor import Document, PageText


def test_format_file_size():
    assert format_file_size(0) == "0 B"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(5 * 1024 * 1024) == "5.0 MB"
    assert format_file_size(-5) == "0 B"


def test_reading_time():
    assert estimate_reading_minutes(0) == 0
    assert estimate_reading_minutes(10) == 1
    assert estimate_reading_minutes(1000) == 5


def test_escape_markdown():
    assert escape_markdown("a*b_c [x] $5 #tag <b>") == r"a\*b\_c \[x\] \$5 \#tag \<b\>"
    assert escape_markdown("") == ""


def test_definition_patterns():
    assert match_definition("A list is an ordered collection of items.") == ("list", "an ordered collection of items")
    term, definition = match_definition("In Python, a tuple is an immutable sequence of values.")
    assert term == "tuple" and "immutable" in definition
    assert match_definition("Overfitting refers to a model memorising noise in its data.")[0] == "Overfitting"
    assert match_definition("Entropy: a measure of disorder in a system or dataset.")[0] == "Entropy"


def test_non_definitions_are_ignored():
    for sentence in ["This is a good example of the idea.", "What is a list?", "It is a long way to go home today.",
                     "Part 1: Python Programming Basics", "The cat sat on the mat.",
                     "A cosine similarity close to 1 means the vectors align closely."]:
        assert match_definition(sentence) is None, sentence


def test_extract_definitions_with_pages(small_doc):
    defs = {d.term.lower(): d for d in extract_definitions(small_doc)}
    assert {"photosynthesis", "chlorophyll", "mitosis", "meiosis", "gravity", "friction"} <= set(defs)
    assert defs["mitosis"].page == 2 and defs["gravity"].page == 3
    assert defs["gravity"].sentence in small_doc.text


def test_key_concepts_on_sample_notes(sample_doc):
    concepts = extract_key_concepts(sample_doc, 12)
    terms = [c.term.lower() for c in concepts]
    assert len(concepts) == 12 and len(set(terms)) == 12
    assert "overfitting" in " ".join(terms) or "machine learning" in terms
    defined = [c for c in concepts if c.definition]
    assert defined and all(c.snippet in sample_doc.text for c in concepts)
    assert all(1 <= c.page <= sample_doc.page_count for c in concepts)


def test_key_concepts_short_document_fallback():
    doc = Document("tiny.txt", "txt", [PageText(1, "Neural networks learn patterns. Neural networks need data.")])
    concepts = extract_key_concepts(doc, 5)
    assert concepts and any("neural" in c.term.lower() for c in concepts)


def test_key_concepts_empty_or_stopword_only_document():
    assert extract_key_concepts(Document("e.txt", "txt", []), 5) == []
    assert extract_key_concepts(Document("s.txt", "txt", [PageText(1, "the and of to in it")]), 5) == []


def test_top_words():
    assert top_words("Python python PYTHON loops loops code", 2) == ["python", "loops"]
