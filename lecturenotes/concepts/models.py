"""Pydantic models for key concept extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class KeyConcept(BaseModel):
    """An extracted key concept or technical term."""

    term: str = Field(..., description="The keyphrase or entity text.")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (e.g. KeyBERT cosine similarity).")
    section_id: int = Field(..., ge=1, description="Originating topic section ID.")
    category: Literal["named_entity", "noun_chunk", "keyphrase"] = Field(
        default="keyphrase",
        description="Source category of the candidate phrase.",
    )
    source_sentence: str | None = Field(
        default=None,
        description="Example sentence from the section mentioning this concept.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float | int) -> float:
        """Clamp score to [0.0, 1.0] to handle negative cosine similarities or float inaccuracies."""
        return max(0.0, min(1.0, float(v)))


class SectionConcepts(BaseModel):
    """Collection of ranked key concepts for a topic section."""

    section_id: int = Field(..., ge=1)
    section_title: str = Field(...)
    concepts: list[KeyConcept] = Field(default_factory=list)
