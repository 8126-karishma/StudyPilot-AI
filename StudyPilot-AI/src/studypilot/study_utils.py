"""Study helpers: definitions, key concepts, markdown escaping and small formatters."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from .text_processor import split_sentences

if TYPE_CHECKING:  # pragma: no cover
    from .text_processor import Document

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being below
    between both but by can cannot could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more most my no nor not now of off
    on once only or other our out over own same she should so some such than that the their theirs them then
    there these they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours yourself""".split()
)
GENERIC_WORDS = frozenset(
    """example examples use used using uses called known often many one two three way ways type types thing things
    make makes made based like new help helps important usually may might simple common general different various
    several given need needs allow allows including include includes within without every much well first second
    step steps part parts set sets number numbers value values case cases result results often however also
    something someone want wants good better best""".split()
)
_BAD_TERM_STARTS = frozenset(
    "this that these those it they he she we you i there here which what who when if because however so but and "
    "or then also thus therefore for in on at by as with".split()
)
_BAD_TERM_WORDS = frozenset("can will does do has have may should would could not cannot".split())

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+#]*(?:-[A-Za-z0-9_+#]+)*")
_DEFINITION = re.compile(
    r"^(?P<term>[^.:;?!]{2,80}?)\s+"
    r"(?P<verb>(?:is|are)\s+(?=(?:a|an|the)\b)|is defined as\s+|are defined as\s+|can be defined as\s+|"
    r"is known as\s+|are known as\s+|is called\s+|are called\s+|refers to\s+|stands for\s+|means\s+)"
    r"(?P<rest>.{8,})$",
    re.IGNORECASE,
)
_COLON_DEFINITION = re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 \-/()+#]{1,50}):\s+(?P<rest>.{15,})$")
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_BULLET = re.compile(r"^[\-\u2022*\d.)\s]+(?=[A-Za-z])")
_MD_SPECIAL = re.compile(r"([\\`*_\[\]#>~$<|])")


@dataclass(frozen=True)
class Definition:
    """A definition found in the notes, with its exact source sentence."""

    term: str
    definition: str
    sentence: str
    page: int


@dataclass(frozen=True)
class Concept:
    """An important term with supporting evidence from the notes."""

    term: str
    score: float
    frequency: int
    page: int
    snippet: str
    definition: str | None = None


def format_file_size(num_bytes: int) -> str:
    """Format a byte count, e.g. ``1.5 MB``."""
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover


def estimate_reading_minutes(word_count: int, words_per_minute: int = 200) -> int:
    """Estimated reading time in whole minutes (at least 1 for non-empty text)."""
    if word_count <= 0:
        return 0
    return max(1, math.ceil(word_count / words_per_minute))


def escape_markdown(text: str) -> str:
    """Escape Markdown/LaTeX control characters in untrusted text before display."""
    return _MD_SPECIAL.sub(r"\\\1", text or "")


def iter_sentences(document: "Document") -> Iterator[tuple[int, str]]:
    """Yield ``(page_number, sentence)`` for the whole document."""
    for page in document.pages:
        for sentence in split_sentences(page.text):
            yield page.number, sentence


_LEADING_PHRASE_WORDS = frozenset(
    "in on for at by with when if while because after before during as although since within through".split()
)
_HEADING_TERM = re.compile(r"^(?:part|chapter|section|unit|lecture|topic|week|day|module)\b", re.IGNORECASE)


def _clean_term(raw: str) -> str | None:
    """Reduce the text before a defining verb to a clean term, or ``None`` if unsuitable."""
    segments = [seg.strip() for seg in raw.split(",") if seg.strip()]
    if not segments:
        return None
    # "In Python, a list ..." -> use the last segment; "RAG, often shortened to X, is ..." -> the first
    first_word = segments[0].split()[0].lower() if segments[0].split() else ""
    term = segments[-1] if first_word in _LEADING_PHRASE_WORDS else segments[0]
    term = _LEADING_ARTICLE.sub("", term.strip(" \"'()")).strip()
    words = term.split()
    if not 1 <= len(words) <= 6 or any(ch.isdigit() for ch in term):
        return None
    lowered = [w.lower().strip(".,;:") for w in words]
    if lowered[0] in _BAD_TERM_STARTS or any(w in _BAD_TERM_WORDS for w in lowered):
        return None
    if any(w in STOPWORDS and w != "of" for w in lowered):  # "close to", "and", "or" ... are not terms
        return None
    return term


def match_definition(sentence: str) -> tuple[str, str] | None:
    """Return ``(term, definition_text)`` if the sentence looks like a definition."""
    text = _BULLET.sub("", sentence.strip())
    if text.endswith("?") or len(text.split()) < 5:
        return None
    colon = _COLON_DEFINITION.match(text)
    if colon and not _HEADING_TERM.match(colon.group("term")):
        term = _clean_term(colon.group("term"))
        if term and len(colon.group("rest").split()) >= 5:
            return term, colon.group("rest").strip().rstrip(".")
    match = _DEFINITION.match(text)
    if not match:
        return None
    term = _clean_term(match.group("term"))
    if not term:
        return None
    rest = match.group("rest").strip().rstrip(".")
    if len(rest.split()) < 3:
        return None
    return term, rest


def is_definition_sentence(sentence: str) -> bool:
    """True when ``sentence`` defines something (used to favour it in summaries)."""
    return match_definition(sentence) is not None


def extract_definitions(document: "Document", max_items: int = 60) -> list[Definition]:
    """Find definition-style sentences; the first definition of each term wins."""
    found: dict[str, Definition] = {}
    for page, sentence in iter_sentences(document):
        matched = match_definition(sentence)
        if not matched:
            continue
        term, definition = matched
        key = term.lower()
        if key not in found:
            found[key] = Definition(term=term, definition=definition, sentence=sentence, page=page)
            if len(found) >= max_items:
                break
    return list(found.values())


def _is_candidate_word(word: str) -> bool:
    lowered = word.lower()
    if lowered in STOPWORDS or lowered in GENERIC_WORDS:
        return False
    return len(word) >= 3 or (word.isupper() and len(word) >= 2)


def extract_key_concepts(document: "Document", max_concepts: int = 12) -> list[Concept]:
    """Rank important terms/phrases using frequency, phrase length and definitions.

    Works for short documents too: with little text, single mentions are allowed.
    """
    definitions = extract_definitions(document)
    definition_by_key = {d.term.lower(): d for d in definitions}

    stats: dict[str, dict] = {}
    total_tokens = 0
    for page, sentence in iter_sentences(document):
        tokens = _TOKEN.findall(sentence)
        total_tokens += len(tokens)
        lowered = [t.lower() for t in tokens]
        seen_in_sentence: set[str] = set()
        for n in (1, 2, 3):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i : i + n]
                low = lowered[i : i + n]
                if not _is_candidate_word(gram[0]) or not _is_candidate_word(gram[-1]):
                    continue
                if any(w in STOPWORDS and w != "of" for w in low[1:-1]) or (n == 3 and "of" in (low[0], low[-1])):
                    continue
                if n >= 2 and any(w in STOPWORDS and w != "of" for w in low):
                    continue
                key = " ".join(low)
                entry = stats.setdefault(
                    key, {"count": 0, "n": n, "display": " ".join(gram), "page": page, "snippet": sentence}
                )
                if key not in seen_in_sentence:
                    entry["count"] += 1
                    seen_in_sentence.add(key)

    min_count = 1 if total_tokens < 120 else 2
    scored: dict[str, float] = {}
    for key, entry in stats.items():
        if entry["count"] >= min_count:
            scored[key] = entry["count"] * (1 + 0.7 * (entry["n"] - 1))
    for key in definition_by_key:
        if len(key.split()) <= 4:
            entry = stats.get(key)
            base = scored.get(key, 0.0)
            scored[key] = base + 6.0 + (entry["count"] if entry else 1)
            if entry is None:
                d = definition_by_key[key]
                stats[key] = {"count": 1, "n": len(key.split()), "display": d.term, "page": d.page,
                              "snippet": d.sentence}

    selected: list[str] = []
    for key, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0])):
        words = set(key.split())
        if any(words <= set(c.split()) or set(c.split()) <= words for c in selected):
            continue
        selected.append(key)
        if len(selected) >= max_concepts:
            break

    concepts: list[Concept] = []
    for key in selected:
        entry = stats[key]
        definition = definition_by_key.get(key)
        concepts.append(
            Concept(
                term=definition.term if definition else entry["display"],
                score=round(scored[key], 2),
                frequency=entry["count"],
                page=definition.page if definition else entry["page"],
                snippet=definition.sentence if definition else entry["snippet"],
                definition=definition.definition if definition else None,
            )
        )
    return concepts


def top_words(text: str, limit: int = 5) -> list[str]:
    """Most frequent non-stop-words (tiny helper used in tests/UI)."""
    counts = Counter(w.lower() for w in _TOKEN.findall(text) if _is_candidate_word(w))
    return [w for w, _ in counts.most_common(limit)]
