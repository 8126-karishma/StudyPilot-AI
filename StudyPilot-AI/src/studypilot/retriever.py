"""Semantic retrieval over document chunks using cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import Embedder
from .text_processor import Chunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity to the query."""

    chunk: Chunk
    score: float


class Retriever:
    """Indexes chunks once and answers similarity queries."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever without any chunks.")
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [c.text for c in self.chunks]
        embedder.fit(texts)
        self._matrix = embedder.encode(texts)

    def search(self, query: str, top_k: int = 4, min_score: float | None = None) -> list[SearchResult]:
        """Return up to ``top_k`` chunks whose similarity is at least ``min_score``.

        ``min_score=None`` uses the embedder's default threshold; pass ``0.0`` to
        disable filtering. An empty list means *nothing relevant was found*.
        """
        if not query or not query.strip() or top_k <= 0:
            return []
        threshold = self.embedder.default_min_score if min_score is None else min_score
        scores = cosine_similarity(self.embedder.encode([query]), self._matrix)[0]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(self.chunks[i], float(scores[i]))
            for i in order
            if scores[i] >= threshold and scores[i] > 0
        ]
