"""PDF text extraction using PyMuPDF (fitz).

Produces a normalized list of PDFSegment objects, one per page.
Pages with fewer than `min_chars` characters are skipped (likely image-only).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf as fitz

from lecturenotes.ingest.models import PDFSegment

logger = logging.getLogger(__name__)


def extract_pdf(
    pdf_path: str | Path,
    min_chars: int = 10,
) -> list[PDFSegment]:
    """Extract text from a PDF file, one segment per page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        min_chars: Pages with fewer extracted characters are skipped.
                   Defaults to 10 (filters blank / image-only pages).

    Returns:
        List of :class:`PDFSegment` objects in page order.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If the PDF has no extractable text at all.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Opening PDF: %s", path)
    doc: fitz.Document = fitz.open(str(path))
    try:
        return _extract_from_doc(doc, min_chars=min_chars, source_name=str(path))
    finally:
        doc.close()


def extract_pdf_from_bytes(
    pdf_bytes: bytes,
    min_chars: int = 10,
    source_name: str = "in-memory-pdf",
) -> list[PDFSegment]:
    """Extract text from PDF raw bytes (e.g. from UI file uploader)."""
    if not pdf_bytes:
        raise ValueError("PDF bytes are empty.")
    doc: fitz.Document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return _extract_from_doc(doc, min_chars=min_chars, source_name=source_name)
    finally:
        doc.close()


def _extract_from_doc(
    doc: fitz.Document,
    min_chars: int,
    source_name: str,
) -> list[PDFSegment]:
    """Internal helper to extract segments from an opened fitz Document."""
    segments: list[PDFSegment] = []
    total_pages = doc.page_count
    logger.info("PDF %s has %d pages", source_name, total_pages)

    for page_num in range(total_pages):
        page: fitz.Page = doc[page_num]
        raw_text: str = page.get_text("text")  # type: ignore[attr-defined]

        # Normalise whitespace: collapse multiple spaces / blank lines
        cleaned = _clean_text(raw_text)

        if len(cleaned) < min_chars:
            logger.debug(
                "Page %d skipped (only %d chars after cleaning).",
                page_num + 1,
                len(cleaned),
            )
            continue

        segments.append(PDFSegment(text=cleaned, page=page_num + 1))
        logger.debug("Page %d → %d chars", page_num + 1, len(cleaned))

    if not segments:
        raise ValueError(
            f"No extractable text found in '{source_name}'. "
            "The PDF may be image-only; consider running OCR first."
        )

    logger.info(
        "Extracted %d segments from %d pages in %s (skipped %d).",
        len(segments),
        total_pages,
        source_name,
        total_pages - len(segments),
    )
    return segments


def _clean_text(text: str) -> str:
    """Normalise extracted PDF text.

    * Strip leading/trailing whitespace per line.
    * Collapse runs of blank lines to a single newline.
    * Strip the final result.
    """
    lines = [line.strip() for line in text.splitlines()]
    # Remove consecutive blank lines
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned_lines).strip()
