import pytest

from studypilot.summarizer import extractive_points, extractive_summary, rank_sentences, summarize
from studypilot.text_processor import Document, PageText, split_sentences

from conftest import BrokenGenerator, FakeGenerator


def test_extractive_summary_uses_real_sentences_and_pages(sample_pdf_doc):
    result = summarize(sample_pdf_doc)
    assert result.method.startswith("Extractive") and not result.warnings
    source_sentences = {s for p in sample_pdf_doc.pages for s in split_sentences(p.text)}
    assert all(point.text in source_sentences for point in result.detailed)
    assert 4 <= len(result.detailed) <= 15
    assert len(split_sentences(result.short)) <= 6
    assert {p.page for p in result.detailed} >= {1, 2}  # covers several pages
    assert [(p.page) for p in result.detailed] == sorted(p.page for p in result.detailed)


def test_summary_prefers_definitions_over_headings(sample_doc):
    result = summarize(sample_doc)
    assert not any(p.text.startswith("Part ") for p in result.detailed)


def test_long_document_is_processed_in_chunks_quickly():
    pages = [PageText(i, " ".join(f"Topic {i} fact number {j} explains concept {i}-{j} in detail." for j in range(30)))
             for i in range(1, 81)]
    doc = Document("long.pdf", "pdf", pages)
    result = extractive_summary(doc, short_sentences=5)
    assert len(result.detailed) <= 15 and result.short


def test_extractive_points_edge_cases(small_doc):
    assert extractive_points(small_doc, 0) == []
    assert len(extractive_points(small_doc, 3)) == 3
    with pytest.raises(ValueError):
        extractive_points(Document("e.txt", "txt", []), 3)
    assert rank_sentences([]) == [] and rank_sentences(["Only one sentence here."]) == [1.0]


def test_model_summary_is_used_and_labelled(small_doc):
    result = summarize(small_doc, FakeGenerator("Plants and cells are studied."))
    assert result.short.startswith("Plants and cells are studied.") and "fake-model" in result.method
    assert result.detailed  # detailed points remain extractive


def test_model_failure_falls_back_with_warning(small_doc):
    result = summarize(small_doc, BrokenGenerator())
    assert result.short and result.warnings and result.method.startswith("Extractive")


def test_empty_model_output_falls_back(small_doc):
    result = summarize(small_doc, FakeGenerator(""))
    assert result.warnings and result.short
