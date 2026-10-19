"""Generation package — public API."""

from lecturenotes.generation.generator import (
    generate_concept_definitions,
    generate_flashcards,
    generate_lecture_materials,
    generate_quiz,
    generate_section_note,
)
from lecturenotes.generation.llm_client import LLMClient
from lecturenotes.generation.schemas import (
    ConceptDefinition,
    Flashcard,
    LectureOutput,
    MCQQuestion,
    SectionNote,
    SectionSummaryItem,
)

__all__ = [
    "ConceptDefinition",
    "Flashcard",
    "LLMClient",
    "LectureOutput",
    "MCQQuestion",
    "SectionNote",
    "SectionSummaryItem",
    "generate_concept_definitions",
    "generate_flashcards",
    "generate_lecture_materials",
    "generate_quiz",
    "generate_section_note",
]
