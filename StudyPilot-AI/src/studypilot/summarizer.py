"""Summarisation: a dependable extractive method plus an optional local model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .generation import GenerationUnavailableError, TextGenerator
from .study_utils import is_definition_sentence
from .text_processor import Document, chunk_text, split_sentences

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummaryPoint:
    """One summary sentence with the page/section it came from."""

    page: int
    label: str
    text: str


@dataclass
class SummaryResult:
    """Short + detailed summaries and how they were produced."""

    short: str
    detailed: list[SummaryPoint]
    method: str
    warnings: list[str] = field(default_factory=list)


def rank_sentences(sentences: list[str]) -> list[float]:
    """Score sentences by similarity to the block centroid; boost definitions."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]
    try:
        matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sentences)
    except ValueError:
        return [1.0] * len(sentences)
    centroid = np.asarray(matrix.mean(axis=0))
    similarity = cosine_similarity(matrix, centroid).ravel()
    scores: list[float] = []
    for sentence, value in zip(sentences, similarity):
        score = float(value)
        words = len(sentence.split())
        if is_definition_sentence(sentence):
            score *= 1.25
        if not sentence.rstrip().endswith((".", "!", "?")) and words < 12:
            score *= 0.2  # headings and fragments make poor summary sentences
        if words < 6:
            score *= 0.5
        elif words > 60:
            score *= 0.8
        scores.append(score)
    return scores


def extractive_points(document: Document, max_sentences: int) -> list[SummaryPoint]:
    """Pick the most informative sentences, processing the document block by block.

    Each page/section is ranked on its own (so long documents never need to be
    scored as one huge text) and every block can contribute, giving coverage.
    """
    if max_sentences <= 0:
        return []
    candidates: list[tuple[int, int, float, str, int]] = []  # block, position, score, text, page
    for block_index, page in enumerate(document.pages):
        sentences = split_sentences(page.text)
        for position, (sentence, score) in enumerate(zip(sentences, rank_sentences(sentences))):
            candidates.append((block_index, position, score, sentence, page.number))
    if not candidates:
        raise ValueError("The document contains no sentences to summarise.")

    chosen: set[tuple[int, int]] = set()
    seen_text: set[str] = set()

    def add(item: tuple[int, int, float, str, int]) -> bool:
        key = item[3].lower()
        if key in seen_text or len(chosen) >= max_sentences:
            return False
        seen_text.add(key)
        chosen.add((item[0], item[1]))
        return True

    best_per_block: dict[int, tuple[int, int, float, str, int]] = {}
    for item in candidates:
        if item[0] not in best_per_block or item[2] > best_per_block[item[0]][2]:
            best_per_block[item[0]] = item
    for item in sorted(best_per_block.values(), key=lambda it: -it[2])[:max_sentences]:
        add(item)
    for item in sorted(candidates, key=lambda it: -it[2]):
        if len(chosen) >= max_sentences:
            break
        add(item)

    picked = sorted((c for c in candidates if (c[0], c[1]) in chosen), key=lambda c: (c[0], c[1]))
    return [SummaryPoint(page=c[4], label=document.label(c[4]), text=c[3]) for c in picked]


def extractive_summary(document: Document, short_sentences: int = 4) -> SummaryResult:
    """Build short and detailed summaries without any language model."""
    total_sentences = sum(len(split_sentences(p.text)) for p in document.pages)
    detailed_count = min(max(short_sentences * 2, round(total_sentences * 0.12)), 15)
    short_points = extractive_points(document, short_sentences)
    detailed = extractive_points(document, detailed_count)
    return SummaryResult(
        short=" ".join(p.text for p in short_points),
        detailed=detailed,
        method="Extractive (key sentences selected from your notes; no language model used)",
    )


def abstractive_short_summary(
    document: Document, generator: TextGenerator, max_chunks: int = 12, chunk_chars: int = 1200
) -> str:
    """Summarise chunk by chunk with a local model, then summarise the summaries."""
    pieces: list[str] = []
    for page in document.pages:
        pieces.extend(chunk_text(page.text, chunk_chars, 0))
    if not pieces:
        raise ValueError("The document contains no text to summarise.")
    if len(pieces) > max_chunks:  # evenly sample very long documents to keep CPU time reasonable
        indexes = np.linspace(0, len(pieces) - 1, max_chunks).round().astype(int)
        pieces = [pieces[i] for i in sorted(set(indexes.tolist()))]
    partial = [
        generator.generate(f"Summarize the following study notes in two sentences:\n{piece}", max_new_tokens=96)
        for piece in pieces
    ]
    combined = " ".join(p for p in partial if p)
    if not combined:
        raise GenerationUnavailableError("the model returned an empty summary")
    if len(partial) > 1 and len(combined) > 600:
        combined = generator.generate(
            f"Summarize the following summaries in three sentences:\n{combined[:1800]}", max_new_tokens=128
        )
    if not combined.strip():
        raise GenerationUnavailableError("the model returned an empty summary")
    return combined.strip()


def summarize(document: Document, generator: TextGenerator | None = None, short_sentences: int = 4) -> SummaryResult:
    """Summarise a document, using the local model if given and falling back safely."""
    result = extractive_summary(document, short_sentences)
    if generator is None:
        return result
    try:
        result.short = abstractive_short_summary(document, generator)
        result.method = (
            f"Short summary generated by local model ({getattr(generator, 'name', 'model')}); "
            "detailed points are extractive key sentences"
        )
    except Exception as exc:  # any failure => keep the extractive result
        logger.warning("Model summarisation failed: %s", exc)
        result.warnings.append(
            "The language model could not produce a summary "
            f"({exc if isinstance(exc, GenerationUnavailableError) else 'unexpected error'}); "
            "showing the extractive summary instead."
        )
    return result
