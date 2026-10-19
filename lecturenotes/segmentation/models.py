"""Pydantic models for topic segmentation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SentenceSpan(BaseModel):
    """An individual sentence with its origin metadata."""

    text: str = Field(..., description="Sentence text content.")
    sentence_idx: int = Field(..., ge=0, description="Global 0-indexed sentence order in lecture.")
    page: int | None = Field(default=None, ge=1, description="Origin page number (for PDF/text).")
    start: float | None = Field(default=None, ge=0.0, description="Start timestamp in seconds (for audio).")
    end: float | None = Field(default=None, ge=0.0, description="End timestamp in seconds (for audio).")
    source_type: Literal["pdf", "text", "audio"] = "pdf"

    @property
    def source_reference(self) -> str:
        """Formatted citation reference."""
        if self.start is not None and self.end is not None:
            m_s, s_s = divmod(int(self.start), 60)
            m_e, s_e = divmod(int(self.end), 60)
            return f"{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}"
        if self.page is not None:
            return f"Page {self.page}"
        return ""


class TopicSection(BaseModel):
    """A coherent topic segment composed of multiple sentences."""

    section_id: int = Field(..., ge=1, description="1-indexed section ID.")
    title: str = Field(..., description="Section title or topic name.")
    text: str = Field(..., description="Combined text of all sentences in the section.")
    sentences: list[SentenceSpan] = Field(default_factory=list, description="Sentences in this section.")
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    time_start: float | None = Field(default=None, ge=0.0)
    time_end: float | None = Field(default=None, ge=0.0)
    source_type: Literal["pdf", "text", "audio"] = "pdf"

    @property
    def source_reference(self) -> str:
        """Formatted source citation for the entire section."""
        if self.time_start is not None and self.time_end is not None:
            m_s, s_s = divmod(int(self.time_start), 60)
            m_e, s_e = divmod(int(self.time_end), 60)
            return f"[{m_s:02d}:{s_s:02d} - {m_e:02d}:{s_e:02d}]"
        if self.page_start is not None and self.page_end is not None:
            if self.page_start == self.page_end:
                return f"[Page {self.page_start}]"
            return f"[Pages {self.page_start}–{self.page_end}]"
        return ""
