"""Safe loading of PDF and TXT uploads into :class:`Document` objects.

Uploads are treated as untrusted data: they are read from memory (never
executed, never written to disk), validated, size-limited and cleaned.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS, Settings
from .text_processor import Document, PageText, normalize_text, split_into_sections

logger = logging.getLogger(__name__)


class DocumentError(Exception):
    """Base class for problems with an uploaded document.

    ``user_message`` is always safe to show in the UI (no paths, no stack traces).
    """

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class UnsupportedFileTypeError(DocumentError):
    """The file extension is not .pdf or .txt."""


class FileTooLargeError(DocumentError):
    """The upload exceeds the configured size limit."""


class DocumentTooLongError(DocumentError):
    """The document has more pages or text than the configured limits."""


class EmptyDocumentError(DocumentError):
    """The file contains no usable text."""


class ScannedPDFError(EmptyDocumentError):
    """The PDF has pages but no selectable text (probably scanned images)."""


class PasswordProtectedError(DocumentError):
    """The PDF is encrypted."""


class CorruptDocumentError(DocumentError):
    """The file could not be parsed."""


def sanitize_filename(name: str) -> str:
    """Return a display-safe file name (no directories, control chars; max 120 chars)."""
    base = Path(str(name).replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return (base or "document")[:120]


def validate_upload(filename: str, size_bytes: int, settings: Settings | None = None) -> str:
    """Validate name and size; return the lower-cased extension (``.pdf``/``.txt``)."""
    settings = settings or Settings()
    extension = Path(sanitize_filename(filename)).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            "Unsupported file type. Please upload a PDF (.pdf) or plain-text (.txt) file."
        )
    if size_bytes <= 0:
        raise EmptyDocumentError("The uploaded file is empty.")
    if size_bytes > settings.max_file_bytes:
        raise FileTooLargeError(
            f"The file is larger than the {settings.max_file_mb} MB limit. "
            "Please upload a smaller file or split it into parts."
        )
    return extension


def read_pdf_bytes(data: bytes, name: str = "document.pdf", settings: Settings | None = None) -> Document:
    """Extract text page by page from PDF bytes, preserving page numbers."""
    settings = settings or Settings()
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise CorruptDocumentError("This file is not a valid PDF document.")
    try:
        import pymupdf  # imported lazily so TXT-only use works without it
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise CorruptDocumentError(
            "PDF support is unavailable because PyMuPDF is not installed."
        ) from exc

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("PyMuPDF could not open file: %s", type(exc).__name__)
        raise CorruptDocumentError("The PDF could not be opened. It may be damaged.") from exc

    pages: list[PageText] = []
    try:
        if pdf.needs_pass:
            raise PasswordProtectedError("This PDF is password-protected. Please remove the password and retry.")
        if pdf.page_count == 0:
            raise EmptyDocumentError("The PDF has no pages.")
        if pdf.page_count > settings.max_pages:
            raise DocumentTooLongError(
                f"The PDF has {pdf.page_count} pages; the limit is {settings.max_pages}. "
                "Please upload a shorter document."
            )
        total_chars = 0
        for index in range(pdf.page_count):
            try:
                raw = pdf.load_page(index).get_text("text")
            except Exception as exc:  # one bad page should not kill the whole file
                logger.warning("Skipping unreadable page %d: %s", index + 1, type(exc).__name__)
                raw = ""
            cleaned = normalize_text(raw)
            total_chars += len(cleaned)
            if total_chars > settings.max_chars:
                raise DocumentTooLongError(
                    "The document contains too much text to process safely. Please upload a shorter file."
                )
            if cleaned:
                pages.append(PageText(number=index + 1, text=cleaned))
    finally:
        pdf.close()

    if not pages:
        raise ScannedPDFError(
            "No selectable text was found. This looks like a scanned or image-only PDF. "
            "StudyPilot AI does not support OCR, so please upload a text-based PDF or a TXT file."
        )
    return Document(name=sanitize_filename(name), source_type="pdf", pages=pages)


def read_txt_bytes(data: bytes, name: str = "notes.txt", settings: Settings | None = None) -> Document:
    """Decode plain text and split it into numbered sections."""
    settings = settings or Settings()
    if b"\x00" in data[:4096]:
        raise CorruptDocumentError("This file does not look like plain text.")
    text = ""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    cleaned = normalize_text(text)
    if not cleaned:
        raise EmptyDocumentError("The text file contains no readable text.")
    if len(cleaned) > settings.max_chars:
        raise DocumentTooLongError(
            "The document contains too much text to process safely. Please upload a shorter file."
        )
    sections = split_into_sections(cleaned)
    pages = [PageText(number=i, text=section) for i, section in enumerate(sections, start=1)]
    return Document(name=sanitize_filename(name), source_type="txt", pages=pages)


def load_document(filename: str, data: bytes, settings: Settings | None = None) -> Document:
    """Validate and read an uploaded file (PDF or TXT) into a :class:`Document`."""
    settings = settings or Settings()
    extension = validate_upload(filename, len(data), settings)
    if extension == ".pdf":
        return read_pdf_bytes(data, filename, settings)
    return read_txt_bytes(data, filename, settings)
