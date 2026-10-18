import numpy as np
import pytest

from studypilot.config import Settings
from studypilot.embeddings import (
    EmbeddingUnavailableError, SentenceTransformerEmbedder, TfidfEmbedder, get_embedder,
)
from studypilot.retriever import Retriever
from studypilot.text_processor import chunk_pages


def test_retrieves_relevant_passage_with_page_metadata(small_doc, tfidf):
    retriever = Retriever(chunk_pages(small_doc, 300, 50), tfidf)
    results = retriever.search("What does chlorophyll absorb?", top_k=2)
    assert results and results[0].chunk.page == 1
    assert results[0].chunk.label == "Page 1" and results[0].chunk.document == "small.txt"
    results = retriever.search("difference between mitosis and meiosis", top_k=3)
    assert results[0].chunk.page == 2
    assert results == sorted(results, key=lambda r: -r.score)


def test_irrelevant_and_empty_queries_return_nothing(small_doc, tfidf):
    retriever = Retriever(chunk_pages(small_doc, 300, 50), tfidf)
    assert retriever.search("quantum chromodynamics lattice gauge") == []
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("gravity", top_k=0) == []


def test_retriever_requires_chunks(tfidf):
    with pytest.raises(ValueError):
        Retriever([], tfidf)


def test_min_score_override(small_doc, tfidf):
    retriever = Retriever(chunk_pages(small_doc, 300, 50), tfidf)
    assert retriever.search("gravity force", top_k=5, min_score=0.99) == []


def test_tfidf_requires_fit_and_nonempty_corpus():
    embedder = TfidfEmbedder()
    with pytest.raises(RuntimeError):
        embedder.encode(["hello"])
    with pytest.raises(ValueError):
        embedder.fit(["", "  "])


def test_tfidf_handles_only_stopwords():
    embedder = TfidfEmbedder()
    embedder.fit(["the and of", "to be or not"])
    assert embedder.encode(["the of"]).shape[0] == 1


class FakeModel:
    """Tiny deterministic 'neural' model: counts calls to prove caching works."""

    def __init__(self):
        self.calls = 0
        self.texts_encoded = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        self.texts_encoded += len(texts)
        vectors = np.array([[len(t) % 7 + 1, t.count("a") + 1, 1.0] for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_dense_embedder_caches_vectors():
    model = FakeModel()
    embedder = SentenceTransformerEmbedder("fake/model", model=model)
    first = embedder.encode(["alpha", "beta"])
    second = embedder.encode(["alpha", "beta", "gamma"])
    assert first.shape == (2, 3) and second.shape == (3, 3)
    assert model.texts_encoded == 3  # alpha and beta were not embedded twice
    assert embedder.encode([]).shape[0] == 0


def test_dense_embedder_works_with_retriever(small_doc):
    embedder = SentenceTransformerEmbedder("fake/model", min_score=0.0, model=FakeModel())
    retriever = Retriever(chunk_pages(small_doc, 300, 50), embedder)
    assert len(retriever.search("anything", top_k=2, min_score=0.0)) == 2


def test_falls_back_to_tfidf_when_neural_model_unavailable(monkeypatch):
    def boom(self):
        raise EmbeddingUnavailableError("the 'sentence-transformers' package is not installed")

    monkeypatch.setattr(SentenceTransformerEmbedder, "ensure_loaded", boom)
    embedder, note = get_embedder(Settings(), prefer_dense=True)
    assert isinstance(embedder, TfidfEmbedder)
    assert "unavailable" in note and "TF-IDF" in note


def test_neural_embeddings_can_be_disabled():
    embedder, note = get_embedder(Settings(use_dense_embeddings=False))
    assert isinstance(embedder, TfidfEmbedder) and "turned off" in note


def test_missing_sentence_transformers_package_is_handled(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbeddingUnavailableError):
        SentenceTransformerEmbedder("some/model").ensure_loaded()
