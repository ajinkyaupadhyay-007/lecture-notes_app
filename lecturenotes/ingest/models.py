"""Pydantic models for the ingest module.

All downstream modules consume List[Segment] — a union of PDFSegment,
TextSegment, and AudioSegment — so the rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PDFSegment(BaseModel):
    """A contiguous block of text extracted from a single PDF page."""

    text: str = Field(..., description="Extracted text content.")
    page: int = Field(..., ge=1, description="1-indexed page number.")
    source_type: Literal["pdf"] = "pdf"


class TextSegment(BaseModel):
    """A block of text extracted from a plain text file."""

    text: str = Field(..., description="Extracted text content.")
    page: int = Field(1, ge=1, description="1-indexed section or page reference.")
    source_type: Literal["text"] = "text"


class AudioSegment(BaseModel):
    """A timed transcript segment produced by Whisper."""

    text: str = Field(..., description="Transcribed text content.")
    start: float = Field(..., ge=0.0, description="Segment start time in seconds.")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds.")
    source_type: Literal["audio"] = "audio"


# Union type used throughout the pipeline
Segment = PDFSegment | TextSegment | AudioSegment
