"""Plain text lecture extraction.

Extracts text into TextSegment objects. If the text has 'Page X' markers,
it creates a TextSegment per page; otherwise it segments by paragraphs/sections.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lecturenotes.ingest.models import TextSegment

logger = logging.getLogger(__name__)

_PAGE_PATTERN = re.compile(r"^Page\s+(\d+)\b", re.IGNORECASE | re.MULTILINE)


def extract_text(
    text_path: str | Path,
    min_chars: int = 10,
) -> list[TextSegment]:
    """Extract segments from a plain text lecture file.

    Args:
        text_path: Absolute or relative path to the text file.
        min_chars: Minimum character length for a segment to be kept.

    Returns:
        List of TextSegment objects.
    """
    path = Path(text_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    return extract_text_from_string(raw_text, min_chars=min_chars)


def extract_text_from_string(
    raw_text: str,
    min_chars: int = 10,
) -> list[TextSegment]:
    """Extract segments from a string (e.g. from UI file uploader)."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Text content is empty.")

    splits = list(_PAGE_PATTERN.finditer(cleaned))
    segments: list[TextSegment] = []

    if splits:
        preamble = cleaned[: splits[0].start()].strip()
        for i, match in enumerate(splits):
            page_num = int(match.group(1))
            start_pos = match.end()
            end_pos = splits[i + 1].start() if i + 1 < len(splits) else len(cleaned)
            page_text = cleaned[start_pos:end_pos].strip()
            if i == 0 and preamble:
                page_text = f"{preamble}\n\n{page_text}".strip()

            if len(page_text) >= min_chars:
                segments.append(TextSegment(text=page_text, page=page_num))
    else:
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) >= min_chars]
        if not paragraphs:
            paragraphs = [cleaned]
        for idx, para in enumerate(paragraphs, start=1):
            segments.append(TextSegment(text=para, page=idx))

    if not segments:
        raise ValueError("No extractable segments found in text.")

    logger.info("Extracted %d text segments", len(segments))
    return segments
