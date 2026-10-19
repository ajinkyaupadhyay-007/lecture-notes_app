"""Unit tests for the key concept extraction module (Milestone 3)."""

from pathlib import Path

import pytest

from lecturenotes.concepts.extractor import (
    extract_concepts_for_section,
    extract_lecture_concepts,
    extract_spacy_candidates,
    get_all_unique_concepts,
)
from lecturenotes.concepts.models import KeyConcept, SectionConcepts
from lecturenotes.ingest.text_extractor import extract_text
from lecturenotes.segmentation.models import SentenceSpan, TopicSection
from lecturenotes.segmentation.texttiling import segment_lecture

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
SAMPLE_TXT_1 = SAMPLES_DIR / "sample_lecture_1.txt"


class TestSpacyCandidateExtraction:
    """Tests for spaCy candidate extraction."""

    def test_extract_spacy_candidates_noun_chunks_and_entities(self) -> None:
        text = (
            "Linear regression fits a linear function by minimising mean squared error loss. "
            "Gradient descent is an iterative optimisation algorithm."
        )
        candidates, category_map = extract_spacy_candidates(text)
        assert len(candidates) > 0

        # Check for key phrases
        cand_lower = [c.lower() for c in candidates]
        assert any("linear regression" in c for c in cand_lower)
        assert any("gradient descent" in c for c in cand_lower)
        assert any("squared error" in c for c in cand_lower)

    def test_extract_spacy_filters_stopwords(self) -> None:
        text = "This course covers the goal of it."
        candidates, _ = extract_spacy_candidates(text)
        cand_lower = [c.lower() for c in candidates]
        assert "this course" not in cand_lower
        assert "the goal" not in cand_lower
        assert "it" not in cand_lower


class TestConceptExtractionPerSection:
    """Tests for section-level KeyBERT extraction."""

    @pytest.fixture
    def sample_section(self) -> TopicSection:
        sentences = [
            SentenceSpan(
                text="The bias-variance trade-off is a fundamental concept in machine learning.",
                sentence_idx=0,
                page=3,
            ),
            SentenceSpan(
                text="Bias refers to error introduced by approximating a complex problem.",
                sentence_idx=1,
                page=3,
            ),
            SentenceSpan(
                text="High variance leads to overfitting on the training data.",
                sentence_idx=2,
                page=3,
            ),
        ]
        return TopicSection(
            section_id=1,
            title="Section 1: Bias-Variance",
            text=" ".join(s.text for s in sentences),
            sentences=sentences,
            page_start=3,
            page_end=3,
            source_type="text",
        )

    def test_extract_concepts_for_section(self, sample_section: TopicSection) -> None:
        concepts = extract_concepts_for_section(sample_section, top_n=5)
        assert len(concepts) > 0
        assert all(isinstance(c, KeyConcept) for c in concepts)
        assert all(c.section_id == 1 for c in concepts)
        assert all(0.0 <= c.score <= 1.0 for c in concepts)

        # Check that scores are sorted descending
        scores = [c.score for c in concepts]
        assert scores == sorted(scores, reverse=True)

        # Check term matches relevant concepts
        terms_lower = [c.term.lower() for c in concepts]
        assert any("bias" in t or "variance" in t or "trade-off" in t or "overfitting" in t for t in terms_lower)

        # Check source sentences are populated
        assert all(c.source_sentence is not None for c in concepts)


class TestLectureConceptsEndToEnd:
    """Integration test across entire sample lecture."""

    def test_extract_lecture_concepts_sample_1(self) -> None:
        segments = extract_text(SAMPLE_TXT_1)
        sections = segment_lecture(segments, threshold=0.20, window_size=2)
        assert len(sections) >= 3

        lecture_concepts = extract_lecture_concepts(sections, top_n_per_section=4)
        assert len(lecture_concepts) == len(sections)
        assert all(isinstance(sc, SectionConcepts) for sc in lecture_concepts)

        all_unique = get_all_unique_concepts(lecture_concepts)
        assert len(all_unique) >= 8

        all_terms_lower = [c.term.lower() for c in all_unique]
        # Should detect core ML concepts
        found_matches = sum(
            1
            for target in ["supervised learning", "linear regression", "gradient descent", "decision tree", "regularisation", "support vector"]
            if any(target in t for t in all_terms_lower)
        )
        assert found_matches >= 3, f"Expected core concepts found, got terms: {all_terms_lower}"
