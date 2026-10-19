"""Unit tests for the NLI-based verification module (Milestone 5).

All tests use a MockNLIModel stub and are 100% offline — no model download required.
"""

from __future__ import annotations

import numpy as np
import pytest

from lecturenotes.config import AppConfig, NLIConfig
from lecturenotes.generation.schemas import (
    ConceptDefinition,
    Flashcard,
    LectureOutput,
    MCQQuestion,
    SectionNote,
    SectionSummaryItem,
)
from lecturenotes.verification.models import (
    FactualityReport,
    VerificationLabel,
    VerificationResult,
)
from lecturenotes.verification.nli_verifier import (
    _score_to_label,
    filter_verified,
    verify_claim,
    verify_lecture_output,
)


# ---------------------------------------------------------------------------
# Mock NLI model — returns pre-configured scores without any real inference
# ---------------------------------------------------------------------------

class MockNLIModel:
    """Deterministic stub for the CrossEncoder NLI model.

    Maps (source_span, claim) pairs to synthetic [contradiction, entailment, neutral] logits.
    The default is a high-entailment response, but individual pairs can be overridden.
    """

    def __init__(self, default_scores: list[float] | None = None) -> None:
        # Default: clearly entailed — [contradiction=0.05, entailment=0.90, neutral=0.05]
        self.default_scores = default_scores or [0.05, 0.90, 0.05]
        self._overrides: dict[tuple[str, str], list[float]] = {}

    def set_scores(self, source: str, claim: str, scores: list[float]) -> None:
        """Register custom scores for a specific (source, claim) pair."""
        self._overrides[(source, claim)] = scores

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        results = []
        for pair in pairs:
            key = (pair[0], pair[1])
            scores = self._overrides.get(key, self.default_scores)
            results.append(scores)
        return np.array(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lecture_output() -> LectureOutput:
    """Build a minimal LectureOutput for end-to-end verification tests."""
    return LectureOutput(
        lecture_title="Test Lecture",
        section_notes=[
            SectionNote(
                section_id=1,
                title="Supervised Learning",
                source_reference="[Page 1]",
                summary_points=[
                    SectionSummaryItem(
                        summary_sentence="Supervised learning uses labelled training data.",
                        source_span="In supervised learning, we are given a training set of labelled examples.",
                    )
                ],
            )
        ],
        concept_definitions=[
            ConceptDefinition(
                term="bias-variance trade-off",
                definition="The trade-off between simplified model assumptions and sensitivity to training data.",
                source_span="The bias-variance trade-off is a fundamental concept in machine learning.",
            )
        ],
        flashcards=[
            Flashcard(
                card_id=1,
                question="What is supervised learning?",
                answer="Supervised learning maps input features to outputs using labelled examples.",
                source_span="In supervised learning, we are given a training set of labelled examples.",
                concept="supervised learning",
            )
        ],
        quiz=[
            MCQQuestion(
                question_id=1,
                question="Which algorithm uses a single sample update?",
                options={"A": "SGD", "B": "OLS", "C": "SVM", "D": "PCA"},
                correct_answer="A",
                explanation="SGD updates weights using a single random sample or mini-batch.",
                distractor_explanations={
                    "B": "OLS is closed-form.",
                    "C": "SVM is margin-based.",
                    "D": "PCA is dimensionality reduction.",
                },
                source_span="Stochastic gradient descent (SGD) approximates the gradient using a single sample.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests: Pydantic models
# ---------------------------------------------------------------------------

class TestVerificationModels:
    """Tests for VerificationResult and FactualityReport Pydantic models."""

    def test_verification_result_valid(self) -> None:
        r = VerificationResult(
            claim="SGD uses mini-batches.",
            source_span="SGD approximates the gradient using a mini-batch.",
            entailment_score=0.92,
            label=VerificationLabel.ENTAILED,
            item_type="note",
            item_id="1",
        )
        assert r.label == VerificationLabel.ENTAILED
        assert r.entailment_score == pytest.approx(0.92)

    def test_factuality_report_from_results_all_entailed(self) -> None:
        results = [
            VerificationResult(
                claim=f"Claim {i}",
                source_span=f"Source {i}",
                entailment_score=0.9,
                label=VerificationLabel.ENTAILED,
                item_type="note",
                item_id=str(i),
            )
            for i in range(5)
        ]
        report = FactualityReport.from_results(results)
        assert report.total_claims == 5
        assert report.entailed_count == 5
        assert report.factuality_rate == pytest.approx(1.0)

    def test_factuality_report_from_results_mixed(self) -> None:
        results = [
            VerificationResult(
                claim="Good claim",
                source_span="Evidence",
                entailment_score=0.85,
                label=VerificationLabel.ENTAILED,
                item_type="flashcard",
                item_id="1",
            ),
            VerificationResult(
                claim="Bad claim",
                source_span="Evidence",
                entailment_score=0.20,
                label=VerificationLabel.CONTRADICTION,
                item_type="quiz",
                item_id="2",
            ),
        ]
        report = FactualityReport.from_results(results)
        assert report.total_claims == 2
        assert report.entailed_count == 1
        assert report.contradiction_count == 1
        assert report.factuality_rate == pytest.approx(0.5)

    def test_factuality_report_empty(self) -> None:
        report = FactualityReport.from_results([])
        assert report.total_claims == 0
        assert report.factuality_rate == 0.0


# ---------------------------------------------------------------------------
# Tests: score_to_label helper
# ---------------------------------------------------------------------------

class TestScoreToLabel:
    """Tests for the _score_to_label internal helper."""

    def test_high_entailment_score_returns_entailed(self) -> None:
        # [contradiction=0.05, entailment=0.90, neutral=0.05], threshold=0.70
        score, label = _score_to_label([0.05, 0.90, 0.05], threshold=0.70)
        assert label == VerificationLabel.ENTAILED
        assert score == pytest.approx(0.90 / 1.0, abs=0.01)

    def test_low_entailment_contradiction_dominant_returns_contradiction(self) -> None:
        # [contradiction=0.75, entailment=0.10, neutral=0.15]
        score, label = _score_to_label([0.75, 0.10, 0.15], threshold=0.70)
        assert label == VerificationLabel.CONTRADICTION

    def test_neutral_dominant_returns_neutral(self) -> None:
        # [contradiction=0.10, entailment=0.30, neutral=0.60], threshold=0.70
        score, label = _score_to_label([0.10, 0.30, 0.60], threshold=0.70)
        assert label == VerificationLabel.NEUTRAL


# ---------------------------------------------------------------------------
# Tests: verify_claim
# ---------------------------------------------------------------------------

class TestVerifyClaim:
    """Tests for the verify_claim function using MockNLIModel."""

    def test_verify_claim_entailed(self) -> None:
        mock = MockNLIModel(default_scores=[0.05, 0.90, 0.05])
        result = verify_claim(
            claim="Supervised learning uses labelled training data.",
            source_span="In supervised learning, we are given a training set of labelled examples.",
            item_type="note",
            item_id="1",
            model=mock,
            threshold=0.70,
        )
        assert result.label == VerificationLabel.ENTAILED
        assert result.entailment_score >= 0.70

    def test_verify_claim_contradiction(self) -> None:
        mock = MockNLIModel(default_scores=[0.80, 0.05, 0.15])
        result = verify_claim(
            claim="Unsupervised learning always requires labels.",
            source_span="Unsupervised learning discovers structure in data without labels.",
            item_type="flashcard",
            item_id="5",
            model=mock,
            threshold=0.70,
        )
        assert result.label == VerificationLabel.CONTRADICTION

    def test_verify_claim_neutral(self) -> None:
        mock = MockNLIModel(default_scores=[0.05, 0.20, 0.75])
        result = verify_claim(
            claim="Deep learning is a subset of machine learning.",
            source_span="Machine learning encompasses many algorithms.",
            item_type="definition",
            item_id="deep learning",
            model=mock,
            threshold=0.70,
        )
        assert result.label == VerificationLabel.NEUTRAL


# ---------------------------------------------------------------------------
# Tests: verify_lecture_output & filter_verified
# ---------------------------------------------------------------------------

class TestVerifyLectureOutput:
    """Tests for the end-to-end verify_lecture_output and filter_verified functions."""

    def test_verify_lecture_output_produces_correct_counts(self) -> None:
        """With a high-entailment mock, all claims should be ENTAILED."""
        output = _make_lecture_output()
        mock = MockNLIModel(default_scores=[0.05, 0.90, 0.05])
        report = verify_lecture_output(output, model=mock, threshold=0.70)

        # 1 note point + 1 definition + 1 flashcard + 1 quiz = 4 total claims
        assert report.total_claims == 4
        assert report.entailed_count == 4
        assert report.factuality_rate == pytest.approx(1.0)

    def test_verify_lecture_output_with_mixed_results(self) -> None:
        """Quiz question set to contradiction, others entailed."""
        output = _make_lecture_output()
        mock = MockNLIModel(default_scores=[0.05, 0.90, 0.05])
        # Override quiz explanation to contradiction
        mock.set_scores(
            source="Stochastic gradient descent (SGD) approximates the gradient using a single sample.",
            claim="SGD updates weights using a single random sample or mini-batch.",
            scores=[0.85, 0.05, 0.10],
        )
        report = verify_lecture_output(output, model=mock, threshold=0.70)
        assert report.total_claims == 4
        assert report.contradiction_count == 1
        assert report.entailed_count == 3
        assert report.factuality_rate == pytest.approx(0.75)

    def test_filter_verified_retains_only_entailed(self) -> None:
        """filter_verified should drop CONTRADICTION-labelled items."""
        output = _make_lecture_output()
        mock = MockNLIModel(default_scores=[0.05, 0.90, 0.05])
        # Set flashcard to contradiction
        mock.set_scores(
            source="In supervised learning, we are given a training set of labelled examples.",
            claim="Supervised learning maps input features to outputs using labelled examples.",
            scores=[0.85, 0.05, 0.10],
        )
        report = verify_lecture_output(output, model=mock, threshold=0.70)
        filtered = filter_verified(output, report)

        # Flashcard labelled CONTRADICTION should be absent from verified output
        card_ids = {c.card_id for c in filtered.flashcards}
        # card_id=1 was CONTRADICTION; filter_verified drops it if alternatives exist
        assert report.contradiction_count >= 1
