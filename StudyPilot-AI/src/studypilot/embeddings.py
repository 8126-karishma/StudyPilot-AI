"""Embedding back-ends used for semantic search.

Two interchangeable back-ends are provided:

* :class:`SentenceTransformerEmbedder` - dense neural embeddings (best quality,
  needs the optional ``sentence-transformers`` package and a one-time model download).
* :class:`TfidfEmbedder` - TF-IDF vectors from scikit-learn (no download, works
  everywhere). It is used automatically when the neural model is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the neural embedding model cannot be loaded."""


class Embedder(ABC):
    """Common interface: ``fit`` on the corpus, then ``encode`` texts to vectors."""

    name: str = "embedder"
    kind: str = "base"
    default_min_score: float = 0.0

    @abstractmethod
    def fit(self, texts: Sequence[str]) -> None:
        """Prepare the embedder for a corpus (no-op for pretrained models)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> Any:
        """Return a 2-D matrix (dense ndarray or scipy sparse) of L2-normalised vectors."""


class TfidfEmbedder(Embedder):
    """Keyword-weighted vectors; lightweight and fully offline."""

    kind = "tfidf"

    def __init__(self, min_score: float = 0.10) -> None:
        self.name = "TF-IDF keyword matching"
        self.default_min_score = min_score
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: Sequence[str]) -> None:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("Cannot fit an embedder on an empty corpus.")
        try:
            vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
            vectorizer.fit(texts)
        except ValueError:  # e.g. every word was a stop word
            vectorizer = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
            vectorizer.fit(texts)
        self._vectorizer = vectorizer

    def encode(self, texts: Sequence[str]) -> Any:
        if self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder must be fitted before encoding.")
        return self._vectorizer.transform(list(texts))


class SentenceTransformerEmbedder(Embedder):
    """Neural sentence embeddings with lazy loading and a per-text cache."""

    kind = "dense"

    def __init__(self, model_name: str, min_score: float = 0.30, model: Any | None = None) -> None:
        self.model_name = model_name
        self.name = f"Sentence Transformers ({model_name.split('/')[-1]})"
        self.default_min_score = min_score
        self._model = model  # a pre-built model can be injected (used by tests)
        self._cache: dict[str, np.ndarray] = {}

    def ensure_loaded(self) -> None:
        """Load the model now (downloads weights on the first run)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError("the 'sentence-transformers' package is not installed") from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # network errors, missing files, corrupt cache...
            logger.warning("Could not load %s: %s", self.model_name, exc)
            raise EmbeddingUnavailableError(f"the model '{self.model_name}' could not be loaded") from exc

    def fit(self, texts: Sequence[str]) -> None:  # pretrained: nothing to learn
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.ensure_loaded()
        texts = list(texts)
        keys = [hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if missing:
            vectors = self._model.encode(  # type: ignore[union-attr]
                list(missing.values()), normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, batch_size=32,
            )
            for key, vector in zip(missing.keys(), np.asarray(vectors, dtype=np.float32)):
                self._cache[key] = vector
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self._cache[k] for k in keys])


_DENSE_INSTANCES: dict[str, SentenceTransformerEmbedder] = {}


def get_embedder(settings: Settings, prefer_dense: bool | None = None) -> tuple[Embedder, str | None]:
    """Return ``(embedder, note)``; ``note`` explains a fallback (or is ``None``).

    The neural model is loaded once per process and shared; TF-IDF embedders are
    created fresh because they are fitted to one document at a time.
    """
    use_dense = settings.use_dense_embeddings if prefer_dense is None else prefer_dense
    if use_dense:
        embedder = _DENSE_INSTANCES.get(settings.embedding_model)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.min_similarity_dense)
        try:
            embedder.ensure_loaded()
            _DENSE_INSTANCES[settings.embedding_model] = embedder
            return embedder, None
        except EmbeddingUnavailableError as exc:
            note = f"Neural embeddings are unavailable ({exc}); using TF-IDF keyword matching instead."
    else:
        note = "Neural embeddings are turned off; using TF-IDF keyword matching."
    return TfidfEmbedder(settings.min_similarity_tfidf), note
