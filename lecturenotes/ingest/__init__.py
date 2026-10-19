"""Ingest package — public API."""

from lecturenotes.ingest.models import AudioSegment, PDFSegment, Segment, TextSegment
from lecturenotes.ingest.pdf_extractor import extract_pdf, extract_pdf_from_bytes
from lecturenotes.ingest.text_extractor import extract_text, extract_text_from_string

__all__ = [
    "AudioSegment",
    "PDFSegment",
    "Segment",
    "TextSegment",
    "extract_pdf",
    "extract_pdf_from_bytes",
    "extract_text",
    "extract_text_from_string",
]
