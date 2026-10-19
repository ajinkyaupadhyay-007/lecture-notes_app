"""Unit tests for AI Tutor, concept visualizer, streak calculation, and new gamification badges."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from lecturenotes.chat.tutor import LectureTutor
from lecturenotes.concepts.visualizer import get_concept_visual
from lecturenotes.db.database import (
    init_db,
    record_login_streak,
    get_motivational_message,
    register_user,
    load_user_progress_db,
)
from lecturenotes.gamification.badges import (
    UserProgress,
    get_all_badges,
    get_all_challenges,
)
from lecturenotes.generation.schemas import ConceptDefinition, Flashcard, SectionNote, SectionSummaryItem


def test_lecture_tutor_basic():
    """Verify LectureTutor can provide grounded answers and deep dives."""
    concept_defs = [
        ConceptDefinition(
            term="Supervised Learning",
            definition="Machine learning approach using labeled training pairs (x, y).",
            source_span="In supervised learning, training examples are provided with ground truth labels.",
        )
    ]
    notes = [
        SectionNote(
            section_id=1,
            title="Introduction to ML",
            summary_points=[
                SectionSummaryItem(
                    summary_sentence="Supervised learning maps inputs to outputs using labeled data.",
                    source_span="In supervised learning, training examples are provided with ground truth labels.",
                )
            ],
        )
    ]
    flashcards = [
        Flashcard(
            card_id=1,
            question="What is supervised learning?",
            answer="Learning with labeled input-output pairs.",
            source_span="In supervised learning, training examples are provided with ground truth labels.",
            concept="Supervised Learning",
        )
    ]

    tutor = LectureTutor(
        lecture_title="Machine Learning Foundations",
        concept_definitions=concept_defs,
        section_notes=notes,
        flashcards=flashcards,
        language="English",
    )

    # 1. Answer question about concept
    ans = tutor.answer("What is supervised learning?")
    assert "Supervised Learning" in ans or "supervised learning" in ans.lower()
    assert len(ans) > 50

    # 2. Deep dive breakdown
    deep_dive = tutor.get_deep_dive("Supervised Learning")
    assert "concept" in deep_dive
    assert "intuition" in deep_dive
    assert "mechanism" in deep_dive
    assert "real_world" in deep_dive
    assert "takeaway" in deep_dive

    # 3. Starter questions
    starters = tutor.get_starter_questions()
    assert len(starters) >= 3


def test_lecture_tutor_japanese():
    """Verify Japanese language handling in tutor and deep dive."""
    concept_defs = [
        ConceptDefinition(
            term="勾配降下法",
            definition="関数の傾きを利用して最小値を探索する最適化アルゴリズム。",
            source_span="勾配降下法は反復的にパラメータを更新します。",
        )
    ]
    tutor = LectureTutor(
        lecture_title="機械学習の基礎",
        concept_definitions=concept_defs,
        language="Japanese",
    )
    ans = tutor.answer("勾配降下法とは何ですか？")
    assert "勾配降下法" in ans

    dd = tutor.get_deep_dive("勾配降下法")
    assert "直観" in dd["intuition"] or "例え" in dd["intuition"]


def test_concept_visualizer():
    """Verify visualizer provides SVG schematics for key machine learning concepts."""
    # 1. Neural network
    v_nn = get_concept_visual("Deep Neural Network Architecture")
    assert v_nn is not None
    assert "<svg" in v_nn["svg_html"]
    assert "Architecture" in v_nn["title"]

    # 2. Gradient descent
    v_gd = get_concept_visual("Gradient Descent Optimization")
    assert v_gd is not None
    assert "<svg" in v_gd["svg_html"]

    # 3. Overfitting
    v_of = get_concept_visual("Overfitting and Regularization")
    assert v_of is not None
    assert "<svg" in v_of["svg_html"]

    # 4. Classification
    v_cl = get_concept_visual("Supervised Classification")
    assert v_cl is not None
    assert "<svg" in v_cl["svg_html"]

    # 5. Generic concept fallback
    v_gen = get_concept_visual("Eigenvector Decomposition")
    assert v_gen is not None
    assert "<svg" in v_gen["svg_html"]


def test_streak_and_motivational_messages(tmp_path: Path):
    """Verify daily streak progression and motivational message generation."""
    db_file = tmp_path / "test_streak.db"
    init_db(db_file)

    # First login sets streak to 1
    s1, bs1, m1, is_new1 = record_login_streak(1, "demo", language="English", db_path=db_file)
    assert s1 == 1
    assert bs1 >= 1
    assert "title" in m1
    assert "quote" in m1
    assert "🔥" in m1["streak_subtitle"] or "✨" in m1["streak_subtitle"]

    # Same day login keeps streak = 1
    s2, bs2, m2, is_new2 = record_login_streak(1, "demo", language="English", db_path=db_file)
    assert s2 == 1
    assert not is_new2

    # Verify motivational message helper
    mot = get_motivational_message(3, "demo", language="English")
    assert "3-Day Streak" in mot["streak_subtitle"] or "🔥" in mot["streak_subtitle"]


def test_new_gamification_badges_and_challenges():
    """Verify new badges and challenges unlock when reaching target counts."""
    prog = UserProgress()

    # 1. Chatbot Inquirer badge (3 chat queries)
    prog.record_chat_query()
    prog.record_chat_query()
    assert "chatbot_curious" not in prog.unlocked_badge_ids
    new_b = prog.record_chat_query()
    assert "chatbot_curious" in prog.unlocked_badge_ids
    assert "tutor_dialogue" in prog.completed_challenges

    # 2. Deep Diver badge (3 concept deep dives)
    prog.record_deep_dive("Concept A")
    prog.record_deep_dive("Concept B")
    assert "concept_deep_dive" not in prog.unlocked_badge_ids
    new_b2 = prog.record_deep_dive("Concept C")
    assert "concept_deep_dive" in prog.unlocked_badge_ids
    assert "deep_dive_sprint" in prog.completed_challenges

    # 3. Visual Architect badge (3 visual diagrams)
    prog.record_visual_viewed("Neural Networks")
    prog.record_visual_viewed("Gradient Descent")
    assert "visual_learner" not in prog.unlocked_badge_ids
    new_b3 = prog.record_visual_viewed("Decision Trees")
    assert "visual_learner" in prog.unlocked_badge_ids
    assert "visual_mastery_quest" in prog.completed_challenges

    # 4. Daily Streak Titan (streak >= 3)
    prog.streak = 3
    prog._evaluate_badges()
    assert "streak_master" in prog.unlocked_badge_ids

    # 5. Verify total badges and challenges catalog
    badges = get_all_badges()
    assert len(badges) >= 17
    challenges = get_all_challenges()
    assert len(challenges) >= 7


def test_new_user_starts_at_level_1(tmp_path: Path):
    """Verify that any newly registered account starts strictly at Level 1 with 0 XP."""
    db_file = tmp_path / "test_fresh_user.db"
    init_db(db_file)

    ok, msg, user = register_user("freshstudent", "securepass123", db_path=db_file)
    assert ok is True
    assert user is not None

    # Also verify login streak right after registration does not change Level 1 / 0 XP
    from lecturenotes.db.database import record_login_streak
    record_login_streak(user["id"], user["username"], db_path=db_file)

    prog_data = load_user_progress_db(user["id"], db_path=db_file)
    user_prog = UserProgress(prog_data)
    lvl_info = user_prog.get_level_info()

    assert user_prog.xp == 0
    assert lvl_info["level"] == 1
    assert "Novice" in lvl_info["title"]
    assert user_prog.daily_streak == 1
    assert user_prog.quiz_streak == 0


def test_quiz_difficulty_generation():
    """Verify that generating Hard difficulty quiz produces Hard difficulty questions."""
    from lecturenotes.concepts.models import KeyConcept
    from lecturenotes.generation.generator import generate_quiz
    from lecturenotes.generation.llm_client import LLMClient

    mock_llm = LLMClient()
    mock_llm.provider = "mock"

    concepts = [
        KeyConcept(term="Overfitting", score=0.9, section_id=1, source_sentence="Overfitting occurs when a model learns training noise instead of true underlying signal."),
        KeyConcept(term="L1 Regularization", score=0.88, section_id=1, source_sentence="L1 regularisation penalizes absolute weights to drive parameters to zero."),
        KeyConcept(term="Gradient Descent", score=0.85, section_id=1, source_sentence="Gradient descent optimizes parameters by moving in the direction of steepest loss decrease."),
        KeyConcept(term="Supervised Learning", score=0.82, section_id=1, source_sentence="Supervised learning maps inputs to target outputs using labelled training data."),
    ]

    # Generate Hard quiz
    hard_quiz = generate_quiz(
        lecture_text="Overfitting and regularization lecture text...",
        concepts=concepts,
        llm=mock_llm,
        question_count=5,
        difficulty="Hard",
    )
    assert len(hard_quiz) == 5
    assert all(q.difficulty == "Hard" for q in hard_quiz)
    # Check that all 5 hard questions are distinct and unique
    assert len(set(q.question for q in hard_quiz)) == 5
    # Check that question stem reflects deeper nuance
    assert any("tradeoff" in q.question.lower() or "why" in q.question.lower() or "geometry" in q.explanation.lower() or "nuance" in q.question.lower() for q in hard_quiz)

    # Test error resilience fallback when LLM encounters network/quota failure
    class BrokenLLM:
        def generate_json(self, prompt, system_prompt=""):
            raise ConnectionError("Simulated LLM network disconnect")

    fallback_quiz = generate_quiz(
        lecture_text="Overfitting and regularization lecture text...",
        concepts=concepts,
        llm=BrokenLLM(),  # type: ignore
        question_count=5,
        difficulty="Hard",
    )
    assert len(fallback_quiz) == 5
    assert all(q.difficulty == "Hard" for q in fallback_quiz)

    # Generate Easy quiz
    easy_quiz = generate_quiz(
        lecture_text="Overfitting and regularization lecture text...",
        concepts=concepts,
        llm=mock_llm,
        question_count=5,
        difficulty="Easy",
    )
    assert len(easy_quiz) == 5
    assert all(q.difficulty == "Easy" for q in easy_quiz)
