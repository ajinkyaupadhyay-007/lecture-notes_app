"""Unit tests for the LLM generation module and Pydantic schemas (Milestone 4)."""

import pytest
from pydantic import ValidationError

from lecturenotes.concepts.models import KeyConcept
from lecturenotes.config import AppConfig, LLMConfig
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
from lecturenotes.segmentation.models import SentenceSpan, TopicSection


class TestPydanticSchemas:
    """Tests schema validation and source_span requirement."""

    def test_section_summary_item_requires_source_span(self) -> None:
        # Valid
        item = SectionSummaryItem(
            summary_sentence="Linear regression minimizes MSE.",
            source_span="Given features x and targets y, it fits a linear function by minimising MSE.",
        )
        assert item.source_span.startswith("Given features")

        # Missing source_span should fail validation
        with pytest.raises(ValidationError):
            SectionSummaryItem(summary_sentence="Just a claim without source.")  # type: ignore[call-arg]

    def test_flashcard_schema(self) -> None:
        fc = Flashcard(
            card_id=1,
            question="What is regularisation?",
            answer="A technique to reduce overfitting by adding a penalty term.",
            source_span="Regularisation is a technique to reduce overfitting by adding a penalty term to the loss function.",
            concept="regularisation",
        )
        assert fc.card_id == 1
        assert "penalty" in fc.answer

        # Missing source_span
        with pytest.raises(ValidationError):
            Flashcard(card_id=1, question="Q?", answer="A.")  # type: ignore[call-arg]

    def test_mcq_question_schema(self) -> None:
        mcq = MCQQuestion(
            question_id=1,
            question="What is the margin in SVMs?",
            options={
                "A": "Distance between hyperplane and nearest points",
                "B": "The number of folds in cross-validation",
                "C": "The learning rate in gradient descent",
                "D": "The degree of polynomial features",
            },
            correct_answer="A",
            explanation="The margin is the distance to the support vectors.",
            distractor_explanations={
                "B": "Relates to cross-validation, not SVMs.",
                "C": "Relates to gradient descent.",
                "D": "Relates to feature engineering.",
            },
            source_span="The margin is the distance between the hyperplane and the nearest training examples.",
        )
        assert mcq.correct_answer == "A"
        assert len(mcq.options) == 4

        # Invalid correct_answer letter
        with pytest.raises(ValidationError):
            MCQQuestion(
                question_id=1,
                question="Q",
                options={"A": "1", "B": "2", "C": "3", "D": "4"},
                correct_answer="E",  # type: ignore[arg-type]
                explanation="exp",
                source_span="source",
            )


class TestLLMClient:
    """Tests JSON parsing and provider dispatch."""

    def test_clean_and_parse_json_markdown_fences(self) -> None:
        raw = """```json
        {
            "status": "success",
            "count": 42
        }
        ```"""
        parsed = LLMClient._clean_and_parse_json(raw)
        assert parsed["status"] == "success"
        assert parsed["count"] == 42

    def test_clean_and_parse_json_with_surrounding_text(self) -> None:
        raw = """Here is the generated output:
        {"lecture": "CS101", "items": [1, 2]}
        Hope this helps!"""
        parsed = LLMClient._clean_and_parse_json(raw)
        assert parsed["lecture"] == "CS101"
        assert parsed["items"] == [1, 2]

    def test_clean_and_parse_json_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            LLMClient._clean_and_parse_json("Not a json at all")

    def test_unsupported_provider(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="unsupported_provider")  # type: ignore[arg-type]

        cfg = AppConfig()
        client = LLMClient(cfg)
        client.provider = "unknown"
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            client.generate_json("Test")


class TestGeneratorWithMockLLM:
    """Tests end-to-end generation orchestrator with deterministic mock provider."""

    @pytest.fixture
    def mock_client(self) -> LLMClient:
        cfg = AppConfig(llm=LLMConfig(provider="mock"))
        return LLMClient(cfg)

    @pytest.fixture
    def sample_sections(self) -> list[TopicSection]:
        s1 = TopicSection(
            section_id=1,
            title="Section 1: Supervised Learning",
            text="In supervised learning, we are given a training set of labelled examples.",
            sentences=[
                SentenceSpan(
                    text="In supervised learning, we are given a training set of labelled examples.",
                    sentence_idx=0,
                    page=1,
                )
            ],
            page_start=1,
            page_end=1,
            source_type="text",
        )
        return [s1]

    @pytest.fixture
    def sample_concepts(self) -> list[KeyConcept]:
        return [
            KeyConcept(
                term="supervised learning",
                score=0.85,
                section_id=1,
                category="noun_chunk",
                source_sentence="In supervised learning, we are given a training set of labelled examples.",
            ),
            KeyConcept(
                term="bias-variance trade-off",
                score=0.78,
                section_id=1,
                category="keyphrase",
                source_sentence="The bias-variance trade-off is a fundamental concept in machine learning.",
            ),
        ]

    def test_generate_section_note(self, sample_sections: list[TopicSection], mock_client: LLMClient) -> None:
        note = generate_section_note(sample_sections[0], mock_client)
        assert isinstance(note, SectionNote)
        assert note.section_id == 1
        assert len(note.summary_points) > 0
        assert all(len(p.source_span) > 0 for p in note.summary_points)

    def test_generate_concept_definitions(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        defs = generate_concept_definitions(sample_concepts, "Lecture text", mock_client)
        assert len(defs) > 0
        assert all(isinstance(d, ConceptDefinition) for d in defs)
        assert all(len(d.source_span) > 0 for d in defs)

    def test_generate_flashcards(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        cards = generate_flashcards("Lecture text", sample_concepts, mock_client, target_count=12)
        assert len(cards) >= 10
        assert all(isinstance(c, Flashcard) for c in cards)
        assert all(len(c.source_span) > 0 for c in cards)

    def test_generate_quiz(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        quiz = generate_quiz("Lecture text", sample_concepts, mock_client, question_count=5)
        assert len(quiz) == 5
        assert all(isinstance(q, MCQQuestion) for q in quiz)
        assert all(q.correct_answer in ["A", "B", "C", "D"] for q in quiz)
        assert all(len(q.source_span) > 0 for q in quiz)

    def test_generate_lecture_materials_end_to_end(
        self,
        sample_sections: list[TopicSection],
        sample_concepts: list[KeyConcept],
        mock_client: LLMClient,
    ) -> None:
        output = generate_lecture_materials(
            sections=sample_sections,
            concepts=sample_concepts,
            llm=mock_client,
            lecture_title="Intro ML",
        )
        assert isinstance(output, LectureOutput)
        assert output.lecture_title == "Intro ML"
        assert len(output.section_notes) == 1
        assert len(output.concept_definitions) >= 2
        assert len(output.flashcards) >= 10
        assert len(output.quiz) == 5
