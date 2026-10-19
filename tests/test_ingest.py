"""Unit tests for the ingest module (Milestone 1)."""

from pathlib import Path

import pytest
import pymupdf

from lecturenotes.ingest.models import PDFSegment, TextSegment
from lecturenotes.ingest.pdf_extractor import extract_pdf, extract_pdf_from_bytes
from lecturenotes.ingest.text_extractor import extract_text, extract_text_from_string

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
SAMPLE_PDF_1 = SAMPLES_DIR / "sample_lecture_1.pdf"
SAMPLE_TXT_1 = SAMPLES_DIR / "sample_lecture_1.txt"
SAMPLE_TXT_2 = SAMPLES_DIR / "sample_lecture_2.txt"


class TestPDFExtraction:
    """Tests for PDF extraction using PyMuPDF."""

    def test_extract_pdf_success(self) -> None:
        assert SAMPLE_PDF_1.exists(), f"Sample PDF not found at {SAMPLE_PDF_1}"
        segments = extract_pdf(SAMPLE_PDF_1)
        assert len(segments) > 0
        assert all(isinstance(seg, PDFSegment) for seg in segments)
        assert all(seg.source_type == "pdf" for seg in segments)
        assert all(len(seg.text) > 0 for seg in segments)

        # Check pages are sequential and 1-indexed
        pages = [seg.page for seg in segments]
        assert pages == sorted(pages)
        assert pages[0] == 1

        # Check content keywords
        all_text = " ".join(seg.text for seg in segments)
        assert "Machine Learning" in all_text
        assert "supervised learning" in all_text

    def test_extract_pdf_from_bytes(self) -> None:
        pdf_bytes = SAMPLE_PDF_1.read_bytes()
        segments = extract_pdf_from_bytes(pdf_bytes)
        assert len(segments) > 0
        assert all(isinstance(seg, PDFSegment) for seg in segments)
        assert "Machine Learning" in segments[0].text

    def test_extract_pdf_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_pdf("non_existent_file.pdf")

    def test_extract_pdf_empty_document(self, tmp_path: Path) -> None:
        empty_pdf = tmp_path / "empty.pdf"
        doc = pymupdf.open()
        doc.new_page()  # blank page
        doc.save(str(empty_pdf))
        doc.close()

        with pytest.raises(ValueError, match="No extractable text found"):
            extract_pdf(empty_pdf, min_chars=10)


class TestTextExtraction:
    """Tests for plain text file extraction."""

    def test_extract_text_with_page_markers(self) -> None:
        assert SAMPLE_TXT_1.exists(), f"Sample text not found at {SAMPLE_TXT_1}"
        segments = extract_text(SAMPLE_TXT_1)
        assert len(segments) >= 5
        assert all(isinstance(seg, TextSegment) for seg in segments)
        assert all(seg.source_type == "text" for seg in segments)
        assert all(len(seg.text) > 0 for seg in segments)

        # First segment should contain title and Page 1
        assert "CS 4780" in segments[0].text
        assert segments[0].page == 1

        # Check Page 2 is present
        page_2_segs = [s for s in segments if s.page == 2]
        assert len(page_2_segs) == 1
        assert "supervised learning" in page_2_segs[0].text.lower()

    def test_extract_text_sample_2(self) -> None:
        assert SAMPLE_TXT_2.exists(), f"Sample text not found at {SAMPLE_TXT_2}"
        segments = extract_text(SAMPLE_TXT_2)
        assert len(segments) >= 5
        all_text = " ".join(seg.text for seg in segments)
        assert "Relational" in all_text or "relational" in all_text
        assert "ACID" in all_text

    def test_extract_text_plain_without_markers(self) -> None:
        raw = "This is paragraph 1 of the lecture notes.\n\nThis is paragraph 2 of the lecture notes."
        segments = extract_text_from_string(raw)
        assert len(segments) == 2
        assert segments[0].page == 1
        assert segments[1].page == 2

    def test_extract_text_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_text("non_existent_file.txt")

    def test_extract_text_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            extract_text_from_string("   \n\n  ")
