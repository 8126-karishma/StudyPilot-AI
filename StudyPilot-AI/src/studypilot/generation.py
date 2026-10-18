"""Optional local text generation (Hugging Face Transformers).

Everything here is optional: if ``transformers``/``torch`` are missing or the
model cannot be loaded, callers receive a :class:`GenerationUnavailableError`
and fall back to the extractive (no-LLM) implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from .config import Settings

logger = logging.getLogger(__name__)


class GenerationUnavailableError(RuntimeError):
    """The local language model cannot be loaded or used."""


class TextGenerator(Protocol):
    """Anything with ``name`` and ``generate(prompt, max_new_tokens)``."""

    name: str

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str: ...


class LocalTextGenerator:
    """Small seq2seq model (default: FLAN-T5 small) loaded lazily on first use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.name = model_name
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise GenerationUnavailableError(
                "the 'transformers' and 'torch' packages are not installed"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            logger.warning("Could not load generator %s: %s", self.model_name, exc)
            self._model = None
            raise GenerationUnavailableError(f"the model '{self.model_name}' could not be loaded") from exc

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        self._load()
        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2)
            return str(self._tokenizer.decode(output[0], skip_special_tokens=True)).strip()
        except GenerationUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Generation failed: %s", exc)
            raise GenerationUnavailableError("the language model failed while generating text") from exc


_GENERATORS: dict[str, LocalTextGenerator] = {}


def get_generator(settings: Settings) -> LocalTextGenerator:
    """Return the shared, lazily loaded generator for the configured model."""
    if settings.generator_model not in _GENERATORS:
        _GENERATORS[settings.generator_model] = LocalTextGenerator(settings.generator_model)
    return _GENERATORS[settings.generator_model]
