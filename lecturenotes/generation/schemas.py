"""Pydantic v2 schemas for LLM-generated lecture notes, flashcards, and quiz items.

Every generated claim, summary item, and flashcard answer MUST include the exact source text span
it was generated from, enabling downstream factuality verification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SectionSummaryItem(BaseModel):
    """An individual takeaway or summary sentence derived from a topic section."""

    summary_sentence: str = Field(
        ...,
        description="A concise, factual takeaway sentence explaining a core concept in this section.",
    )
    source_span: str = Field(
        ...,
        description="The exact or near-exact sentence from the lecture text that supports this statement.",
    )


class SectionNote(BaseModel):
    """Structured notes for an individual topic section."""

    section_id: int = Field(..., ge=1, description="1-indexed topic section ID.")
    title: str = Field(..., description="Section title.")
    source_reference: str = Field(
        default="",
        description="Source reference, e.g. '[Page 1]' or '[Pages 2–3]' or '[01:23 - 04:50]'.",
    )
    summary_points: list[SectionSummaryItem] = Field(
        default_factory=list,
        description="Key bullet points / takeaways with source spans.",
    )


class ConceptDefinition(BaseModel):
    """A technical key concept and its grounded definition."""

    term: str = Field(..., description="The key concept term.")
    definition: str = Field(
        ...,
        description="Clear, accessible definition grounded in the lecture content.",
    )
    source_span: str = Field(
        ...,
        description="The source text sentence supporting or explaining this definition.",
    )


class Flashcard(BaseModel):
    """A review flashcard testing conceptual understanding."""

    card_id: int = Field(..., ge=1, description="1-indexed card identifier.")
    question: str = Field(..., description="Direct, clear question testing a specific concept.")
    answer: str = Field(
        ...,
        description="Factual, concise answer grounded strictly in the lecture text.",
    )
    source_span: str = Field(
        ...,
        description="The source sentence that justifies this answer.",
    )
    concept: str | None = Field(default=None, description="Related concept term if applicable.")


class MCQQuestion(BaseModel):
    """A 4-choice multiple choice question with distractors and explanations."""

    question_id: int = Field(..., ge=1, description="1-indexed question ID.")
    question: str = Field(..., description="Clear question stem.")
    options: dict[str, str] = Field(
        ...,
        description="Four choices mapped by keys 'A', 'B', 'C', 'D'.",
    )
    correct_answer: Literal["A", "B", "C", "D"] = Field(
        ...,
        description="The single letter of the correct answer.",
    )
    explanation: str = Field(
        ...,
        description="Explanation of why the correct option is true based on the text.",
    )
    distractor_explanations: dict[str, str] = Field(
        default_factory=dict,
        description="Explanations for why each wrong option is incorrect or a distractor.",
    )
    source_span: str = Field(
        ...,
        description="The sentence in the source text grounding the question and correct answer.",
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        default="Medium",
        description="Difficulty level of the question: Easy, Medium, or Hard.",
    )


class SectionNotesOutput(BaseModel):
    """Container schema for section notes generation."""

    section_id: int
    title: str
    summary_points: list[SectionSummaryItem]


class ConceptDefinitionsOutput(BaseModel):
    """Container schema for concept definitions generation."""

    definitions: list[ConceptDefinition]


class FlashcardsOutput(BaseModel):
    """Container schema for flashcards generation."""

    flashcards: list[Flashcard]


class QuizOutput(BaseModel):
    """Container schema for 5-question quiz generation."""

    quiz: list[MCQQuestion]


class LectureOutput(BaseModel):
    """Complete, consolidated output of the generation module."""

    lecture_title: str
    section_notes: list[SectionNote]
    concept_definitions: list[ConceptDefinition]
    flashcards: list[Flashcard]
    quiz: list[MCQQuestion]
