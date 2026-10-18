import pymupdf
import pytest

from studypilot.config import Settings
from studypilot.pdf_reader import (
    CorruptDocumentError, DocumentTooLongError, EmptyDocumentError, FileTooLargeError,
    PasswordProtectedError, ScannedPDFError, UnsupportedFileTypeError, load_document,
    read_pdf_bytes, read_txt_bytes, sanitize_filename, validate_upload,
)

from conftest import SAMPLE_PDF


def test_pdf_text_and_page_numbers_are_preserved(pdf_factory):
    data = pdf_factory(["Alpha page one text.", "", "Gamma page three text."])
    doc = read_pdf_bytes(data, "notes.pdf")
    assert [p.number for p in doc.pages] == [1, 3]  # blank page 2 skipped, numbering kept
    assert "Alpha" in doc.pages[0].text and "Gamma" in doc.pages[1].text
    assert doc.label(3) == "Page 3"
    assert doc.source_type == "pdf"


def test_scanned_pdf_is_reported_without_ocr(pdf_factory):
    with pytest.raises(ScannedPDFError) as info:
        read_pdf_bytes(pdf_factory(["", ""]), "scan.pdf")
    assert "OCR" in info.value.user_message


def test_malformed_pdf_is_rejected_safely():
    with pytest.raises(CorruptDocumentError):
        read_pdf_bytes(b"%PDF-1.4 this is not really a pdf", "bad.pdf")
    with pytest.raises(CorruptDocumentError):
        read_pdf_bytes(b"plain text pretending to be a pdf", "fake.pdf")


def test_encrypted_pdf_is_rejected(pdf_factory):
    data = pdf_factory(["secret"], encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    with pytest.raises(PasswordProtectedError):
        read_pdf_bytes(data, "locked.pdf")


@pytest.mark.parametrize("name", ["virus.exe", "notes.docx", "noextension", "image.png", "archive.pdf.zip"])
def test_unsupported_file_types_are_rejected(name):
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(name, 100)


def test_empty_and_whitespace_files_are_rejected():
    with pytest.raises(EmptyDocumentError):
        load_document("empty.txt", b"")
    with pytest.raises(EmptyDocumentError):
        load_document("blank.txt", b"   \n\t \n  ")


def test_file_size_limit():
    settings = Settings(max_file_mb=1)
    with pytest.raises(FileTooLargeError):
        load_document("big.txt", b"a" * (2 * 1024 * 1024), settings)


def test_page_limit(pdf_factory):
    with pytest.raises(DocumentTooLongError):
        read_pdf_bytes(pdf_factory(["one", "two", "three"]), "long.pdf", Settings(max_pages=2))


def test_text_limit():
    with pytest.raises(DocumentTooLongError):
        read_txt_bytes(("word " * 500).encode(), "long.txt", Settings(max_chars=1000))


def test_txt_decoding_and_sections():
    assert read_txt_bytes("caf\u00e9 is a place".encode("utf-8")).pages[0].text.startswith("caf\u00e9")
    assert "caf\u00e9" in read_txt_bytes("caf\u00e9 is a place".encode("cp1252")).pages[0].text
    long_text = "\n\n".join(f"Paragraph {i}. " + "words " * 120 for i in range(12))
    doc = read_txt_bytes(long_text.encode())
    assert doc.page_count > 1 and doc.label(2) == "Section 2"


def test_binary_data_in_txt_is_rejected():
    with pytest.raises(CorruptDocumentError):
        read_txt_bytes(b"abc\x00\x01\x02binary")


def test_sanitize_filename_removes_paths_and_control_chars():
    assert sanitize_filename("../../etc/pass\x00wd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\Users\\me\\notes.pdf") == "notes.pdf"
    assert len(sanitize_filename("a" * 500 + ".txt")) == 120
    assert sanitize_filename("") == "document"


def test_bundled_sample_pdf_extracts():
    doc = load_document("sample_notes.pdf", SAMPLE_PDF.read_bytes())
    assert doc.page_count == 4
    assert "Overfitting" in doc.pages[1].text
    assert "cosine similarity" in doc.pages[3].text.lower()
