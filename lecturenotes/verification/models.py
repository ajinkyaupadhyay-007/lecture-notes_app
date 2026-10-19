"""Pydantic models for NLI-based factuality verification results."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class VerificationLabel(str, Enum):
    """NLI classification outcome for a single claim."""

    ENTAILED = "entailed"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class VerificationResult(BaseModel):
    """Verification outcome for a single generated claim against its source span."""

    claim: str = Field(..., description="The generated sentence or answer being verified.")
    source_span: str = Field(..., description="The exact lecture passage used as evidence.")
    entailment_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that source_span entails the claim (cross-encoder score).",
    )
    label: VerificationLabel = Field(..., description="NLI label derived from entailment_score.")
    item_type: Literal["note", "definition", "flashcard", "quiz"] = Field(
        ..., description="The type of generated item being verified."
    )
    item_id: str = Field(
        ...,
        description="Identifier for the item (e.g. section_id, card_id, term, question_id).",
    )


class FactualityReport(BaseModel):
    """Aggregate factuality report for the full lecture output."""

    total_claims: int = Field(..., ge=0, description="Total number of claims verified.")
    entailed_count: int = Field(..., ge=0, description="Claims labelled ENTAILED.")
    neutral_count: int = Field(..., ge=0, description="Claims labelled NEUTRAL.")
    contradiction_count: int = Field(..., ge=0, description="Claims labelled CONTRADICTION.")
    factuality_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of claims that are ENTAILED (entailed_count / total_claims).",
    )
    results: list[VerificationResult] = Field(
        default_factory=list,
        description="Per-claim verification details.",
    )

    @classmethod
    def from_results(cls, results: list[VerificationResult]) -> "FactualityReport":
        """Build a FactualityReport by aggregating a list of VerificationResult objects."""
        total = len(results)
        entailed = sum(1 for r in results if r.label == VerificationLabel.ENTAILED)
        neutral = sum(1 for r in results if r.label == VerificationLabel.NEUTRAL)
        contradiction = sum(1 for r in results if r.label == VerificationLabel.CONTRADICTION)
        rate = entailed / total if total > 0 else 0.0
        return cls(
            total_claims=total,
            entailed_count=entailed,
            neutral_count=neutral,
            contradiction_count=contradiction,
            factuality_rate=rate,
            results=results,
        )
