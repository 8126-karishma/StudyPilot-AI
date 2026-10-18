"""Centralised configuration for StudyPilot AI.

All tunable values live here. They can be overridden with environment
variables (optionally loaded from a local ``.env`` file); see ``.env.example``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only hit when the package is missing
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_NOTES_PATH = PROJECT_ROOT / "data" / "sample_notes.txt"
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
ENV_PREFIX = "STUDYPILOT_"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(ENV_PREFIX + name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logging.getLogger(__name__).warning("Ignoring invalid integer for %s%s", ENV_PREFIX, name)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # Upload safety limits
    max_file_mb: int = 10
    max_pages: int = 300
    max_chars: int = 1_500_000

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    # Minimum cosine similarity for a passage to count as "relevant".
    # These are heuristics, not tuned on a benchmark.
    min_similarity_tfidf: float = 0.10
    min_similarity_dense: float = 0.30

    # Optional model usage
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_dense_embeddings: bool = True
    use_generative_models: bool = False
    generator_model: str = "google/flan-t5-small"

    log_level: str = "INFO"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables (and ``.env`` if present)."""
        if load_dotenv is not None:
            load_dotenv(override=False)
        defaults = cls()
        chunk_size = _env_int("CHUNK_SIZE", defaults.chunk_size, minimum=200)
        overlap = _env_int("CHUNK_OVERLAP", defaults.chunk_overlap, minimum=0)
        if overlap >= chunk_size // 2:  # keep overlap sane so chunking always progresses
            overlap = chunk_size // 5
        return cls(
            max_file_mb=_env_int("MAX_FILE_MB", defaults.max_file_mb),
            max_pages=_env_int("MAX_PAGES", defaults.max_pages),
            max_chars=_env_int("MAX_CHARS", defaults.max_chars, minimum=1000),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            top_k=_env_int("TOP_K", defaults.top_k),
            embedding_model=_env_str("EMBEDDING_MODEL", defaults.embedding_model),
            use_dense_embeddings=_env_bool("USE_DENSE_EMBEDDINGS", defaults.use_dense_embeddings),
            use_generative_models=_env_bool("USE_GENERATIVE_MODELS", defaults.use_generative_models),
            generator_model=_env_str("GENERATOR_MODEL", defaults.generator_model),
            log_level=_env_str("LOG_LEVEL", defaults.log_level).upper(),
        )


def configure_logging(level: str = "INFO") -> None:
    """Configure basic logging once (safe to call repeatedly)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
