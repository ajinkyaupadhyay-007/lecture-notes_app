"""Unit tests for Japanese support, Quiz difficulty, Video suggestions, Badges, and Challenges."""

import pytest

from lecturenotes.concepts.models import KeyConcept
from lecturenotes.config import AppConfig, LLMConfig
from lecturenotes.generation.generator import (
    generate_concept_definitions,
    generate_flashcards,
    generate_quiz,
    _fallback_quiz,
)
from lecturenotes.generation.llm_client import LLMClient
from lecturenotes.generation.schemas import MCQQuestion
from lecturenotes.recommendations.videos import get_concept_videos, VideoRecommendation
from lecturenotes.gamification.badges import (
    UserProgress,
    get_all_badges,
    get_all_challenges,
)
from lecturenotes.segmentation.models import SentenceSpan, TopicSection


@pytest.fixture
def mock_client() -> LLMClient:
    cfg = AppConfig(llm=LLMConfig(provider="mock"))
    return LLMClient(cfg)


@pytest.fixture
def sample_concepts() -> list[KeyConcept]:
    return [
        KeyConcept(
            term="linear regression",
            score=0.92,
            section_id=1,
            category="keyphrase",
            source_sentence="Linear regression fits a linear function by minimising the mean squared error loss.",
        ),
        KeyConcept(
            term="gradient descent",
            score=0.88,
            section_id=1,
            category="keyphrase",
            source_sentence="Gradient descent updates model parameters iteratively in the negative direction of the gradient.",
        ),
        KeyConcept(
            term="regularisation",
            score=0.85,
            section_id=2,
            category="keyphrase",
            source_sentence="Regularisation prevents overfitting by adding a penalty term to the empirical loss function.",
        ),
        KeyConcept(
            term="bias-variance trade-off",
            score=0.81,
            section_id=2,
            category="keyphrase",
            source_sentence="The bias-variance trade-off balances error from simplified assumptions against sensitivity to training variance.",
        ),
    ]


@pytest.fixture
def sample_section() -> TopicSection:
    text = "In supervised learning, we are given a training set of labelled examples. Linear regression fits a linear function by minimising the mean squared error loss."
    return TopicSection(
        section_id=1,
        title="Introduction to Supervised Learning",
        text=text,
        sentences=[
            SentenceSpan(text="In supervised learning, we are given a training set of labelled examples.", sentence_idx=0, page=1),
            SentenceSpan(text="Linear regression fits a linear function by minimising the mean squared error loss.", sentence_idx=1, page=1),
        ],
        source_reference="Slides 1-5",
    )


class TestJapaneseSupport:
    """Tests Japanese language generation in prompts, LLM client mock, and generator."""

    def test_japanese_mock_quiz(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        quiz = generate_quiz("Lecture text in Japanese", sample_concepts, mock_client, question_count=3, language="Japanese")
        assert len(quiz) == 3
        assert any("設問" in q.question or "学習" in q.question for q in quiz)
        assert all(isinstance(q, MCQQuestion) for q in quiz)

    def test_japanese_mock_flashcards(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        cards = generate_flashcards("Lecture text", sample_concepts, mock_client, target_count=5, language="Japanese")
        assert len(cards) >= 5
        assert any("復習カード" in c.question for c in cards)

    def test_japanese_mock_definitions(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        defs = generate_concept_definitions(sample_concepts, "Lecture text", mock_client, language="Japanese")
        assert len(defs) > 0
        assert any("トレードオフ" in d.term or "回帰" in d.term for d in defs)

    def test_japanese_fallback_quiz(self, sample_section: TopicSection, sample_concepts: list[KeyConcept]) -> None:
        quiz = _fallback_quiz([sample_section], sample_concepts, question_count=3, language="Japanese")
        assert len(quiz) == 3
        assert "次の説明に直接対応する概念はどれですか" in quiz[0].question


class TestQuizDifficulty:
    """Tests quiz difficulty selection and tagging."""

    def test_quiz_difficulty_easy(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        quiz = generate_quiz("Text", sample_concepts, mock_client, question_count=4, difficulty="Easy")
        assert all(q.difficulty in ("Easy", "Medium", "Hard") for q in quiz)
        assert any(q.difficulty == "Easy" for q in quiz)

    def test_quiz_difficulty_hard(self, sample_concepts: list[KeyConcept], mock_client: LLMClient) -> None:
        quiz = generate_quiz("Text", sample_concepts, mock_client, question_count=4, difficulty="Hard")
        assert any(q.difficulty == "Hard" for q in quiz)

    def test_fallback_quiz_difficulty(self, sample_section: TopicSection, sample_concepts: list[KeyConcept]) -> None:
        quiz = _fallback_quiz([sample_section], sample_concepts, question_count=3, difficulty="Hard")
        assert all(q.difficulty == "Hard" for q in quiz)


class TestVideoRecommendations:
    """Tests educational video recommendations in English and Japanese."""

    def test_curated_english_videos(self) -> None:
        videos = get_concept_videos("linear regression", language="English")
        assert len(videos) >= 2
        assert any("StatQuest" in v.channel or "MIT" in v.channel for v in videos)
        assert all(v.youtube_url.startswith("https://www.youtube.com/results?search_query=") for v in videos)

    def test_curated_japanese_videos(self) -> None:
        videos = get_concept_videos("linear regression", language="Japanese")
        assert len(videos) >= 1
        assert any("キカガク" in v.channel or "ヨビノリ" in v.channel for v in videos)

    def test_arbitrary_custom_concept_videos(self) -> None:
        videos = get_concept_videos("quantum entanglement", language="English")
        assert len(videos) >= 2
        assert all(isinstance(v, VideoRecommendation) for v in videos)
        assert "Quantum Entanglement" in videos[0].title


class TestGamificationAndBadges:
    """Tests UserProgress tracking, XP, Leveling, and Badge unlocks."""

    def test_initial_state(self) -> None:
        p = UserProgress()
        assert p.xp == 0
        assert p.quiz_streak == 0
        lvl = p.get_level_info()
        assert lvl["level"] == 1
        assert "Novice" in lvl["title"]

    def test_master_concept_and_badge_unlock(self) -> None:
        p = UserProgress()
        badges = p.mark_concept_completed("Linear Regression", total_concepts_count=3)
        assert "linear regression" in p.completed_concepts
        assert p.xp >= 50
        assert any(b.id == "first_concept" for b in badges)
        assert "first_concept" in p.unlocked_badge_ids

    def test_three_concepts_unlock(self) -> None:
        p = UserProgress()
        p.mark_concept_completed("c1")
        p.mark_concept_completed("c2")
        badges = p.mark_concept_completed("c3")
        assert any(b.id == "three_concepts" for b in badges)

    def test_quiz_streak_and_hard_badge(self) -> None:
        p = UserProgress()
        p.record_quiz_answer(1, is_correct=True, difficulty="Easy")
        p.record_quiz_answer(2, is_correct=True, difficulty="Medium")
        b3 = p.record_quiz_answer(3, is_correct=True, difficulty="Hard")

        assert p.quiz_streak == 3
        assert "streak_3" in p.unlocked_badge_ids
        assert "hard_mode" in p.unlocked_badge_ids

    def test_polyglot_badge_on_japanese(self) -> None:
        p = UserProgress()
        badges = p.record_language("Japanese")
        assert any(b.id == "polyglot" for b in badges)
        assert "polyglot" in p.unlocked_badge_ids

    def test_challenges_catalog(self) -> None:
        challenges = get_all_challenges()
        assert len(challenges) >= 4
        ids = [c.id for c in challenges]
        assert "speed_run" in ids
        assert "survival" in ids
        assert "hard_gauntlet" in ids

    def test_record_exam_result_awards_xp_and_ace_badge(self) -> None:
        p = UserProgress()
        initial_xp = p.xp
        badges = p.record_exam_result(
            score_pct=90,
            correct=9,
            total=10,
            difficulty="Mixed (1E+2M+2H)",
            elapsed_secs=245,
            xp_earned=90,
            lecture_title="Machine Learning Foundations",
        )
        assert len(p.exam_history) == 1
        entry = p.exam_history[0]
        assert entry["score_pct"] == 90
        assert entry["correct"] == 9
        assert entry["total"] == 10
        assert entry["difficulty"] == "Mixed (1E+2M+2H)"
        assert entry["lecture_title"] == "Machine Learning Foundations"
        assert "ts" in entry
        # Exam ace badge unlocked for >= 80%
        assert "exam_ace" in p.unlocked_badge_ids
        assert any(b.id == "exam_ace" for b in badges)
        assert p.xp >= initial_xp + 90 + 150  # 90 exam xp + 150 badge xp

    def test_exam_scholar_badge_on_three_exams(self) -> None:
        p = UserProgress()
        p.record_exam_result(60, 3, 5, "Easy", 60, 30)
        p.record_exam_result(70, 7, 10, "Medium", 150, 70)
        assert "exam_scholar" not in p.unlocked_badge_ids
        b3 = p.record_exam_result(75, 6, 8, "Hard", 200, 60)
        assert "exam_scholar" in p.unlocked_badge_ids
        assert any(b.id == "exam_scholar" for b in b3)
        assert len(p.exam_history) == 3

    def test_user_progress_reset_to_day1(self) -> None:
        p = UserProgress({"xp": 500, "daily_streak": 5, "streak": 5})
        p.reset_progress()
        assert p.xp == 0
        assert p.daily_streak == 1
        assert p.streak == 1
        assert p.get_level_info()["level"] == 1
        assert len(p.completed_concepts) == 0
        assert len(p.exam_history) == 0

    def test_exam_engine_expanded_mcqs_and_descriptive(self) -> None:
        from lecturenotes.generation.schemas import (
            LectureOutput,
            ConceptDefinition,
            Flashcard,
            MCQQuestion,
        )
        from lecturenotes.exam.exam_engine import (
            generate_expanded_mcqs,
            generate_descriptive_questions,
            evaluate_descriptive_answer,
        )

        dummy_output = LectureOutput(
            lecture_title="Optimization 101",
            section_notes=[],
            concept_definitions=[
                ConceptDefinition(
                    term="Gradient Descent",
                    definition="An iterative optimization algorithm for finding the local minimum of a differentiable function.",
                    source_span="Gradient descent iteratively updates parameters in the direction of steepest descent.",
                ),
                ConceptDefinition(
                    term="Learning Rate",
                    definition="A tuning parameter in an optimization algorithm that determines the step size at each iteration.",
                    source_span="The learning rate determines how far parameters move along the gradient.",
                ),
            ],
            flashcards=[
                Flashcard(
                    card_id=1,
                    question="What does gradient descent minimize?",
                    answer="It minimizes the loss function by iteratively updating model weights.",
                    source_span="It minimizes empirical risk.",
                    concept="Gradient Descent",
                ),
                Flashcard(
                    card_id=2,
                    question="What happens if the learning rate is too large?",
                    answer="The algorithm may overshoot the local minimum and diverge.",
                    source_span="Excessive step size causes divergence.",
                    concept="Learning Rate",
                ),
            ],
            quiz=[
                MCQQuestion(
                    question_id=1,
                    question="What is gradient descent?",
                    options={"A": "An optimization algorithm", "B": "A clustering method", "C": "A sorting algorithm", "D": "A database schema"},
                    correct_answer="A",
                    explanation="It is an optimization algorithm.",
                    source_span="Gradient descent minimizes loss.",
                    difficulty="Easy",
                ),
            ],
        )

        # 1. Expanded MCQs: requested 4 questions even though only 1 in quiz -> should synthesize 4
        expanded = generate_expanded_mcqs(dummy_output, target_count=4)
        assert len(expanded) == 4
        assert all(isinstance(q, MCQQuestion) for q in expanded)

        # 2. Descriptive questions: should generate 2 conceptual questions
        desc_qs = generate_descriptive_questions(dummy_output, count=2)
        assert len(desc_qs) == 2
        assert desc_qs[0].concept_term in ("Gradient Descent", "Learning Rate")
        assert len(desc_qs[0].key_points) > 0
        assert len(desc_qs[0].model_answer) > 20

        # 3. Descriptive rubric evaluation
        good_ans = (
            "Gradient descent is an iterative optimization algorithm that updates parameters "
            "in the direction of steepest descent to minimize the loss function."
        )
        eval_good = evaluate_descriptive_answer(good_ans, desc_qs[0])
        assert eval_good["score_pct"] >= 60
        assert eval_good["points_earned"] >= 6.0
        assert eval_good["word_count"] > 10

        short_ans = "It minimizes."
        eval_short = evaluate_descriptive_answer(short_ans, desc_qs[0])
        assert eval_short["score_pct"] == 0
        assert eval_short["passed"] is False

