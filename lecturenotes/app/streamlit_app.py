"""Streamlit UI for LectureNotes AI.

Upload a PDF or plain-text lecture file and generate:
  • Structured section notes with source attributions
  • Key concept definitions
  • Flashcards (10–20)
  • Multiple-choice quiz with Difficulty Selection (Easy / Medium / Hard)
  • Educational Video Recommendations for concepts
  • Gamified User Progress, Leveling & Achievement Badges
  • Interactive Challenges (Speed Run, Survival Mode, Hard Gauntlet)
  • Japanese Language (日本語) support
  • Factuality verification report
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path so 'lecturenotes' is importable in all environments
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_cwd = Path.cwd().resolve()
if str(_cwd) not in sys.path:
    sys.path.insert(0, str(_cwd))

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from lecturenotes.recommendations.videos import (
    VideoRecommendation,
    get_concept_videos,
    recommend_videos_for_lecture,
)
from lecturenotes.gamification.badges import (
    UserProgress,
    get_all_badges,
    get_all_challenges,
    get_all_rewards,
)
from lecturenotes.db.database import (
    init_db,
    authenticate_user,
    register_user,
    load_user_progress_db,
    save_user_progress_db,
    record_user_lecture_db,
    get_user_lectures_db,
    record_login_streak,
    get_motivational_message,
    simulate_day2_login,
    reset_user_progress_db,
)
from lecturenotes.exam.exam_engine import (
    generate_expanded_mcqs,
    generate_descriptive_questions,
    generate_exam_questions_parallel,
    evaluate_descriptive_answer,
    DescriptiveExamQuestion,
)
from lecturenotes.chat.tutor import LectureTutor
from lecturenotes.concepts.visualizer import get_concept_visual
from lecturenotes.generation.llm_client import LLMClient



# ── Page config must be the first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="LectureNotes AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 40%, #24243e 100%);
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1040 0%, #16103a 100%);
    border-right: 1px solid rgba(124, 77, 255, 0.2);
}
[data-testid="stSidebar"] * {
    color: #c8c0f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label {
    color: #a89fd8 !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Hero header ── */
.hero-title {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-sub {
    color: #9d94c8;
    font-size: 1.05rem;
    margin-top: 0.25rem;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,77,255,0.18);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: rgba(124,77,255,0.45);
    box-shadow: 0 0 20px rgba(124,77,255,0.12);
}

/* ── Section badge ── */
.section-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7c4dff22, #5b86e522);
    border: 1px solid rgba(124,77,255,0.35);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Source span quote ── */
.source-quote {
    border-left: 3px solid #7c4dff;
    padding: 0.45rem 0.9rem;
    background: rgba(124,77,255,0.07);
    border-radius: 0 8px 8px 0;
    color: #b8b0dc;
    font-size: 0.85rem;
    font-style: italic;
    margin-top: 0.4rem;
}

/* ── Factuality badges ── */
.badge-entailed   { color: #34d399; font-weight: 600; }
.badge-neutral    { color: #fbbf24; font-weight: 600; }
.badge-contra     { color: #f87171; font-weight: 600; }

/* ── Flashcard ── */
.fc-question {
    background: linear-gradient(135deg, rgba(124,77,255,0.15), rgba(91,134,229,0.10));
    border: 1px solid rgba(124,77,255,0.3);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    font-size: 0.95rem;
    color: #e0daf8;
    font-weight: 500;
}
.fc-answer {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-top: 0.4rem;
    font-size: 0.88rem;
    color: #a7f3d0;
}

/* ── MCQ option ── */
.mcq-option {
    border-radius: 8px;
    padding: 0.55rem 1rem;
    margin: 0.2rem 0;
    font-size: 0.9rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.mcq-correct { background: rgba(52,211,153,0.12); border-color: rgba(52,211,153,0.3); color:#34d399; }
.mcq-wrong   { background: rgba(248,113,113,0.07); border-color: rgba(248,113,113,0.2); color:#9d94c8; }

/* ── Difficulty Badges ── */
.diff-badge {
    display: inline-block;
    border-radius: 16px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.diff-easy   { background: rgba(52, 211, 153, 0.15); border: 1px solid rgba(52, 211, 153, 0.4); color: #34d399; }
.diff-medium { background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.4); color: #fbbf24; }
.diff-hard   { background: rgba(248, 113, 113, 0.15); border: 1px solid rgba(248, 113, 113, 0.4); color: #f87171; }

/* ── Gamification and Badges ── */
.badge-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(124, 77, 255, 0.2);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.badge-unlocked {
    border-color: rgba(167, 139, 250, 0.65);
    box-shadow: 0 0 20px rgba(124, 77, 255, 0.25);
    background: linear-gradient(135deg, rgba(124, 77, 255, 0.15), rgba(91, 134, 229, 0.10));
}
.badge-locked {
    opacity: 0.55;
    filter: grayscale(70%);
}
.badge-icon {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
}
.badge-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #f1f0fb;
}
.badge-desc {
    font-size: 0.78rem;
    color: #a79ed4;
    margin-top: 0.25rem;
    line-height: 1.4;
}
.badge-xp {
    font-size: 0.75rem;
    font-weight: 600;
    color: #fbbf24;
    margin-top: 0.35rem;
}

/* ── Video Recommendation Card ── */
.video-card {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(96, 165, 250, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.video-card:hover {
    border-color: rgba(96, 165, 250, 0.55);
    box-shadow: 0 0 16px rgba(96, 165, 250, 0.15);
}
.video-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 0.25rem;
}
.video-meta {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 0.6rem;
}
.video-link-btn {
    display: inline-block;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white !important;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.84rem;
    padding: 6px 16px;
    border-radius: 8px;
    transition: opacity 0.2s;
}
.video-link-btn:hover {
    opacity: 0.88;
}

/* ── Challenge Card ── */
.challenge-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(124, 77, 255, 0.10));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,77,255,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.78rem;
    color: #9d94c8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    flex-wrap: wrap !important;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 6px;
    gap: 6px;
    border: 1px solid rgba(124, 77, 255, 0.2);
    overflow: visible !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9d94c8;
    font-size: 0.84rem;
    font-weight: 500;
    padding: 6px 14px;
    transition: all 0.2s ease;
    white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #f1f0fb;
    background: rgba(124, 77, 255, 0.12);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(124, 77, 255, 0.45), rgba(91, 134, 229, 0.35)) !important;
    color: #ffffff !important;
    font-weight: 600;
    box-shadow: 0 2px 10px rgba(124, 77, 255, 0.3);
    border: 1px solid rgba(124, 77, 255, 0.4);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #7c4dff, #5b86e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.8rem;
    font-weight: 600;
    font-size: 0.92rem;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(124,77,255,0.06) !important;
    border: 2px dashed rgba(124,77,255,0.35) !important;
    border-radius: 14px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
    color: #c8c0f0 !important;
    font-size: 0.88rem !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(124,77,255,0.15) !important;
}

/* ── Mobile & Small Screen Optimizations ── */
@media (max-width: 768px) {
    .hero-title {
        font-size: 1.85rem !important;
        line-height: 1.2 !important;
    }
    .hero-sub {
        font-size: 0.88rem !important;
    }
    .card {
        padding: 1rem 1rem !important;
        border-radius: 10px !important;
    }
    .metric-value {
        font-size: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 10px !important;
        font-size: 0.78rem !important;
    }
    .stButton > button {
        width: 100% !important;
        padding: 0.65rem 1rem !important;
    }
    .badge-card {
        padding: 0.85rem !important;
    }
    .video-card {
        padding: 1rem !important;
    }
    [data-testid="stSidebar"] {
        padding-top: 1rem !important;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Database & Authentication Initialization
# ─────────────────────────────────────────────────────────────────────────────

init_db()

if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

if "language" not in st.session_state:
    st.session_state["language"] = "English"


def _sync_user_progress(prog: UserProgress) -> None:
    """Save progress state and persist directly to SQLite database."""
    st.session_state["user_progress_data"] = prog.to_dict()
    curr_user = st.session_state.get("auth_user")
    if curr_user and "id" in curr_user:
        try:
            save_user_progress_db(curr_user["id"], prog.to_dict())
        except Exception:
            pass
    try:
        prog.save_to_disk()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Authentication Gate (Login / Registration Portal)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.get("auth_user") is None:
    col_l1, col_l2, col_l3 = st.columns([1, 2.4, 1])
    with col_l2:
        st.markdown(
            """
<div style="text-align:center;margin-top:2.5rem;margin-bottom:1.5rem">
  <div style="font-size:3.5rem;margin-bottom:0.4rem">🎓</div>
  <p class="hero-title" style="font-size:2.4rem;text-align:center;margin-bottom:0.2rem">LectureNotes AI</p>
  <p class="hero-sub" style="text-align:center;font-size:0.95rem;color:#a89fd8">
    Persistent progress tracking, multi-difficulty quizzes & concept-tailored YouTube recommendations.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["🔑 Sign In", "✨ Create Account"])

        with tab_login:
            st.markdown("<div style='padding:1rem 0.5rem'>", unsafe_allow_html=True)
            login_username = st.text_input("Username", key="login_username_field", placeholder="e.g. demo or your username")
            login_password = st.text_input("Password", type="password", key="login_password_field", placeholder="••••••••")

            col_btn_login, col_btn_demo = st.columns([1, 1])
            with col_btn_login:
                if st.button("🚀 Sign In", key="btn_signin_action", use_container_width=True):
                    ok, msg, user = authenticate_user(login_username, login_password)
                    if ok and user:
                        for k in ["user_progress_data", "chat_messages", "active_challenge_id"]:
                            st.session_state.pop(k, None)
                        st.session_state["auth_user"] = user
                        streak, best_s, mot_msg, is_new_d = record_login_streak(user["id"], user["username"], language=st.session_state.get("language", "English"))
                        st.session_state["motivational_message"] = mot_msg
                        st.session_state["user_progress_data"] = load_user_progress_db(user["id"])
                        if is_new_d and streak >= 2:
                            st.balloons()
                        st.success(f"Welcome back, {user['username']}! Day {streak} Streak! 🔥")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error(msg)

            with col_btn_demo:
                if st.button("⚡ Quick Demo Access", key="btn_quick_demo_action", use_container_width=True):
                    ok, msg, user = authenticate_user("demo", "demo123")
                    if ok and user:
                        for k in ["user_progress_data", "chat_messages", "active_challenge_id"]:
                            st.session_state.pop(k, None)
                        st.session_state["auth_user"] = user
                        streak, best_s, mot_msg, is_new_d = record_login_streak(user["id"], user["username"], language=st.session_state.get("language", "English"))
                        st.session_state["motivational_message"] = mot_msg
                        st.session_state["user_progress_data"] = load_user_progress_db(user["id"])
                        if is_new_d and streak >= 2:
                            st.balloons()
                        st.toast(f"Signed in as Demo User · Day {streak} Streak! 🔥")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("Demo login error: " + msg)

            st.caption("💡 *Tip: New here? Use Quick Demo Access or create an account in the next tab.*")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_register:
            st.markdown("<div style='padding:1rem 0.5rem'>", unsafe_allow_html=True)
            reg_username = st.text_input("New Username", key="reg_username_field", placeholder="At least 3 characters")
            reg_password = st.text_input("New Password", type="password", key="reg_password_field", placeholder="At least 4 characters")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm_field", placeholder="Re-enter password")

            if st.button("✨ Register & Start Learning", key="btn_register_action", use_container_width=True):
                if reg_password != reg_confirm:
                    st.error("Passwords do not match. Please verify your password.")
                else:
                    ok, msg, user = register_user(reg_username, reg_password)
                    if ok and user:
                        for k in ["user_progress_data", "lecture_output", "chat_messages", "active_challenge_id", "cached_pipeline_result", "motivational_message", "result"]:
                            st.session_state.pop(k, None)
                        st.session_state["auth_user"] = user
                        streak, best_s, mot_msg, _ = record_login_streak(user["id"], user["username"], language=st.session_state.get("language", "English"))
                        st.session_state["motivational_message"] = mot_msg
                        st.session_state["user_progress_data"] = load_user_progress_db(user["id"])
                        st.success(f"🎉 Account '{user['username']}' created! Welcome to Level 1.")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(msg)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
<div style="margin-top:2rem;background:rgba(255,255,255,0.03);border:1px solid rgba(124,77,255,0.22);border-radius:14px;padding:1.2rem;backdrop-filter:blur(6px)">
  <div style="font-size:0.85rem;font-weight:600;color:#a78bfa;margin-bottom:0.5rem">🌟 WHAT YOU GET:</div>
  <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.7">
    • <b>Persistent SQLite Database:</b> Your XP, streaks, and mastered concepts are saved permanently and never reset when uploading new PDFs.<br>
    • <b>Authoritative YouTube Lessons:</b> Key concepts extracted from your lecture PDF are automatically paired with top channels (3Blue1Brown, StatQuest, MIT, freeCodeCamp, CS50, Yobinori).<br>
    • <b>Gamified Learning:</b> Adaptive difficulty quizzes, Anki flashcard exports, and achievement badges.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Authenticated User Progress
# ─────────────────────────────────────────────────────────────────────────────

auth_user = st.session_state.get("auth_user")
if not auth_user or not isinstance(auth_user, dict) or "id" not in auth_user:
    st.stop()

if "user_progress_data" not in st.session_state:
    st.session_state["user_progress_data"] = load_user_progress_db(auth_user["id"])
progress = UserProgress(st.session_state["user_progress_data"])

if "motivational_message" not in st.session_state:
    try:
        s_val, b_val, m_obj, _ = record_login_streak(auth_user["id"], auth_user["username"], language=st.session_state.get("language", "English"))
        st.session_state["motivational_message"] = m_obj
        st.session_state["user_progress_data"] = load_user_progress_db(auth_user["id"])
        progress = UserProgress(st.session_state["user_progress_data"])
    except Exception:
        st.session_state["motivational_message"] = get_motivational_message(progress.streak, auth_user["username"], language=st.session_state.get("language", "English"))



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verification_badge(label: str) -> str:
    label_lower = label.lower()
    if label_lower == "entailed":
        return '<span class="badge-entailed">✔ Entailed</span>'
    if label_lower == "contradiction":
        return '<span class="badge-contra">✘ Contradiction</span>'
    return '<span class="badge-neutral">~ Neutral</span>'


def _pct_color(rate: float) -> str:
    if rate >= 0.75:
        return "#34d399"
    if rate >= 0.50:
        return "#fbbf24"
    return "#f87171"


def _diff_badge(difficulty: str) -> str:
    diff_lower = difficulty.lower()
    if diff_lower == "easy":
        return '<span class="diff-badge diff-easy">🟢 Easy (+10 XP)</span>'
    if diff_lower == "hard":
        return '<span class="diff-badge diff-hard">🔴 Hard (+35 XP)</span>'
    return '<span class="diff-badge diff-medium">🟡 Medium (+20 XP)</span>'


def _generate_markdown_study_guide(output, report=None, language: str = "English") -> str:
    """Generate a clean Markdown study guide from generated lecture materials."""
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    if is_ja:
        lines = [f"# 📖 {output.lecture_title}\n"]
        lines.append("> LectureNotes AI により生成された検証済み講義学習ガイド。\n")

        if output.concept_definitions:
            lines.append("## 🔑 重要概念と定義 (Key Concepts & Definitions)\n")
            for d in output.concept_definitions:
                lines.append(f"### {d.term}")
                lines.append(f"{d.definition}\n")
                lines.append(f"> **講義引用**: *\"{d.source_span}\"*\n")

        if output.section_notes:
            lines.append("## 📝 セクションノート (Section Notes)\n")
            for s in output.section_notes:
                ref = f" ({s.source_reference})" if s.source_reference else ""
                lines.append(f"### セクション {s.section_id}: {s.title}{ref}\n")
                for pt in s.summary_points:
                    lines.append(f"- **{pt.summary_sentence}**")
                    lines.append(f"  > 📎 *\"{pt.source_span}\"*\n")

        if output.flashcards:
            lines.append("## 🃏 復習フラッシュカード (Flashcards)\n")
            for fc in output.flashcards:
                tag = f" `[{fc.concept}]`" if fc.concept else ""
                lines.append(f"**Q{fc.card_id}. {fc.question}**{tag}")
                lines.append(f"- **A**: {fc.answer}")
                lines.append(f"- *引用*: \"{fc.source_span}\"\n")

        if output.quiz:
            lines.append("## 📋 練習クイズ (Practice Quiz)\n")
            for q in output.quiz:
                diff_tag = f" `[{q.difficulty}]`"
                lines.append(f"### Q{q.question_id}: {q.question}{diff_tag}\n")
                for opt, text in q.options.items():
                    marker = " ✅ (正解)" if opt == q.correct_answer else ""
                    lines.append(f"- **{opt}**: {text}{marker}")
                lines.append(f"\n**解説**: {q.explanation}")
                lines.append(f"> 📎 *\"{q.source_span}\"*\n")

    else:
        lines = [f"# 📖 {output.lecture_title}\n"]
        lines.append("> Generated by LectureNotes AI with verifiable source citations.\n")

        if output.concept_definitions:
            lines.append("## 🔑 Key Concepts & Definitions\n")
            for d in output.concept_definitions:
                lines.append(f"### {d.term.title()}")
                lines.append(f"{d.definition}\n")
                lines.append(f"> **Source**: *\"{d.source_span}\"*\n")

        if output.section_notes:
            lines.append("## 📝 Section Notes\n")
            for s in output.section_notes:
                ref = f" ({s.source_reference})" if s.source_reference else ""
                lines.append(f"### Section {s.section_id}: {s.title}{ref}\n")
                for pt in s.summary_points:
                    lines.append(f"- **{pt.summary_sentence}**")
                    lines.append(f"  > 📎 *\"{pt.source_span}\"*\n")

        if output.flashcards:
            lines.append("## 🃏 Flashcards\n")
            for fc in output.flashcards:
                tag = f" `[{fc.concept}]`" if fc.concept else ""
                lines.append(f"**Q{fc.card_id}. {fc.question}**{tag}")
                lines.append(f"- **A**: {fc.answer}")
                lines.append(f"- *Citation*: \"{fc.source_span}\"\n")

        if output.quiz:
            lines.append("## 📋 Practice Quiz\n")
            for q in output.quiz:
                diff_tag = f" `[{q.difficulty}]`"
                lines.append(f"### Q{q.question_id}: {q.question}{diff_tag}\n")
                for opt, text in q.options.items():
                    marker = " ✅ (Correct)" if opt == q.correct_answer else ""
                    lines.append(f"- **{opt}**: {text}{marker}")
                lines.append(f"\n**Explanation**: {q.explanation}")
                lines.append(f"> 📎 *\"{q.source_span}\"*\n")

    if report:
        lines.append(f"\n## 🔬 Factuality Verification\n")
        lines.append(f"- **Factuality Rate**: {report.factuality_rate:.1%}")
        lines.append(f"- **Claims Entailed**: {report.entailed_count} / {report.total_claims}")
        lines.append(f"- **Neutral Claims**: {report.neutral_count}")
        lines.append(f"- **Contradictions**: {report.contradiction_count}\n")

    return "\n".join(lines)


def _generate_flashcards_csv(flashcards) -> str:
    """Export flashcards in Anki-compatible CSV format."""
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Question", "Answer", "Concept", "Source Citation"])
    for fc in flashcards:
        writer.writerow([fc.question, fc.answer, fc.concept or "", fc.source_span])
    return out.getvalue()


@st.cache_data(show_spinner=False)
def _run_pipeline(
    file_bytes: bytes,
    file_name: str,
    provider: str,
    flashcard_count: int,
    mcq_count: int,
    run_verification: bool,
    language: str = "English",
    quiz_difficulty: str = "Adaptive Mix",
) -> dict[str, Any]:
    """Run the full LectureNotes pipeline and return a result dict."""
    from lecturenotes.config import load_config
    from lecturenotes.ingest import extract_pdf_from_bytes, extract_text_from_string
    from lecturenotes.segmentation import segment_lecture
    from lecturenotes.concepts.extractor import extract_lecture_concepts, get_all_unique_concepts
    from lecturenotes.generation.generator import generate_lecture_materials
    from lecturenotes.generation.llm_client import LLMClient
    from lecturenotes.verification.nli_verifier import verify_lecture_output

    cfg = load_config()
    cfg.llm.provider = provider
    if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
        cfg.llm.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        cfg.llm.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    cfg.generation.flashcard_count = flashcard_count
    cfg.generation.mcq_count = mcq_count

    # 1. Ingest
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        segments = extract_pdf_from_bytes(file_bytes, source_name=file_name)
    else:
        text = file_bytes.decode("utf-8", errors="replace")
        segments = extract_text_from_string(text)

    # 2. Segment
    sections = segment_lecture(segments)

    # 3. Concepts
    section_concepts = extract_lecture_concepts(sections)
    all_concepts = get_all_unique_concepts(section_concepts)

    # 4. Generate with Language support and Difficulty
    llm = LLMClient(config=cfg)
    lecture_title = Path(file_name).stem.replace("_", " ").replace("-", " ").title()
    output = generate_lecture_materials(
        sections=sections,
        concepts=all_concepts,
        llm=llm,
        lecture_title=lecture_title,
        target_flashcards=flashcard_count,
        target_mcqs=mcq_count,
        language=language,
        quiz_difficulty=quiz_difficulty,
    )

    # 5. Concept-tailored video recommendations
    lecture_videos = recommend_videos_for_lecture(
        concepts=all_concepts,
        lecture_title=lecture_title,
        llm=llm,
        language=language,
        max_videos=4,
    )

    # 6. Verify (optional)
    report = None
    if run_verification:
        report = verify_lecture_output(output)

    return {
        "sections": sections,
        "section_concepts": section_concepts,
        "all_concepts": all_concepts,
        "output": output,
        "lecture_videos": lecture_videos,
        "report": report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # User Auth status in sidebar
    col_u1, col_u2 = st.columns([1.8, 1])
    with col_u1:
        st.markdown(f"**👤 {auth_user['username']}**")
        st.caption("🟢 SQLite Database Connected")
    with col_u2:
        if st.button("Log out", key="btn_logout_sidebar", use_container_width=True):
            st.session_state["auth_user"] = None
            st.session_state.pop("user_progress_data", None)
            st.session_state.pop("result", None)
            st.session_state.pop("active_file_bytes", None)
            st.session_state.pop("active_file_name", None)
            st.rerun()

    with st.expander("📚 Past Uploaded Lectures", expanded=False):
        past_lectures = get_user_lectures_db(auth_user["id"])
        if past_lectures:
            for pl in past_lectures:
                st.markdown(
                    f"""<div style="font-size:0.78rem;background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 8px;margin-bottom:6px;border:1px solid rgba(124,77,255,0.2)">
                      <b style="color:#c4b8fa">{pl['lecture_title']}</b><br>
                      <span style="color:#9d94c8">{pl['file_name']}</span><br>
                      <span style="color:#34d399">{pl['concepts_count']} concepts · {pl['sections_count']} sections</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No lectures uploaded yet. Process your first document to view history here!")

    st.divider()

    # ── Language selection ──
    lang_choice = st.selectbox(
        "🌐 Language / 言語",
        options=["English 🇺🇸", "日本語 (Japanese) 🇯🇵"],
        index=1 if st.session_state.get("language") == "Japanese" else 0,
        help="Select language for generated notes, flashcards, quizzes, and UI text.",
        key="lang_selector",
    )
    current_lang = "Japanese" if ("Japanese" in lang_choice or "日本語" in lang_choice) else "English"
    if st.session_state.get("language") != current_lang:
        st.session_state["language"] = current_lang
        unlocked = progress.record_language(current_lang)
        _sync_user_progress(progress)
        for b in unlocked:
            st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)}!")
            st.balloons()

    st.divider()

    # ── User level & XP mini panel ──
    lvl_info = progress.get_level_info()
    today_weekday = datetime.date.today().weekday()  # 0=Mon, 6=Sun
    day_labels = ["M", "T", "W", "T", "F", "S", "S"]
    streak_dots_html = "".join([
        f'<span style="display:inline-block;width:19px;height:19px;line-height:19px;text-align:center;border-radius:50%;font-size:0.68rem;font-weight:700;margin:0 2px;'
        f'background:{"rgba(249,115,22,0.35);color:#ffedd5;border:1px solid #f97316" if i == today_weekday or (i < today_weekday and progress.daily_streak > (today_weekday - i)) else "rgba(255,255,255,0.06);color:#64748b;border:1px dashed rgba(255,255,255,0.15)"};">'
        f'{d}</span>'
        for i, d in enumerate(day_labels)
    ])

    st.markdown(
        f"""
<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(124,77,255,0.3);border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.6rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-weight:700;color:#a78bfa;font-size:0.95rem">Level {lvl_info['level']}</span>
    <span style="font-size:0.8rem;color:#fbbf24;font-weight:600">✨ {lvl_info['xp']} XP</span>
  </div>
  <div style="font-size:0.8rem;color:#e0daf8;margin-top:0.2rem">{lvl_info['title']}</div>
  <div style="margin-top:0.4rem">
    <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:6px;overflow:hidden">
      <div style="background:linear-gradient(90deg,#7c4dff,#34d399);width:{int(lvl_info['pct']*100)}%;height:100%"></div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#9d94c8;margin-top:0.4rem">
    <span>🔥 Daily: <b>Day {progress.daily_streak}</b></span>
    <span>⚡ Quiz Streak: <b>{progress.quiz_streak}</b> (Best: {progress.best_quiz_streak})</span>
  </div>
  <div style="margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid rgba(255,255,255,0.08);display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:0.68rem;color:#94a3b8;font-weight:600">WEEKLY STREAK:</span>
    <div>{streak_dots_html}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_u_lbl, col_u_out = st.columns([3, 2])
    with col_u_lbl:
        st.caption(f"👤 **{auth_user['username']}**")
    with col_u_out:
        if st.button("🚪 Sign Out", key="btn_sidebar_signout", use_container_width=True):
            for k in ["auth_user", "user_progress_data", "lecture_output", "chat_messages", "active_challenge_id", "cached_pipeline_result", "motivational_message", "result"]:
                st.session_state.pop(k, None)
            st.rerun()

    with st.expander("⚙️ Streak & Profile Testing Controls", expanded=False):
        c_rst, c_sim = st.columns(2)
        with c_rst:
            if st.button("🔄 Reset to Day 1", key="btn_reset_day1", use_container_width=True, help="Reset profile to clean Level 1, 0 XP, Day 1 streak."):
                reset_user_progress_db(auth_user["id"])
                st.session_state["user_progress_data"] = load_user_progress_db(auth_user["id"])
                st.session_state.pop("motivational_message", None)
                st.toast("Profile reset to clean Level 1, Day 1 Streak! 🌱")
                st.rerun()
        with c_sim:
            if st.button("⏩ Simulate Day 2", key="btn_sim_day2", use_container_width=True, help="Simulate logging in on Day 2 to test streak increment & new daily quote."):
                s_new, bs_new, m_new = simulate_day2_login(auth_user["id"], auth_user["username"], language=st.session_state.get("language", "English"))
                st.session_state["user_progress_data"] = load_user_progress_db(auth_user["id"])
                st.session_state["motivational_message"] = m_new
                st.balloons()
                st.toast(f"🔥 Advanced to Day {s_new} Streak! New daily motivation unlocked!")
                st.rerun()

    st.divider()


    # Quick check for local Ollama server
    ollama_active = False
    try:
        import httpx
        with httpx.Client(timeout=0.4) as _hc:
            r = _hc.get("http://localhost:11434/api/tags")
            ollama_active = r.status_code == 200
    except Exception:
        ollama_active = False

    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    default_provider_idx = 0 if has_gemini_key else (1 if has_openai_key else (2 if ollama_active else 5))

    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "ollama", "azure", "claude", "mock"],
        index=default_provider_idx,
        help="'gemini' uses Google Gemini API (flash/pro). 'openai' uses GPT-4o-mini. 'mock' returns offline test data.",
    )
    if provider == "gemini":
        gemini_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            placeholder="AIzaSy...",
            help="Your Google Gemini API key. Stored only in this active session.",
            key="gemini_key_input_box",
        )
        if gemini_key_input:
            os.environ["GEMINI_API_KEY"] = gemini_key_input.strip()
            from lecturenotes.generation.llm_client import LLMClient
            LLMClient.reset_gemini_quota()
        if not os.environ.get("GEMINI_API_KEY"):
            st.caption("🔑 *Enter your Gemini API key above or save in .env*")
        else:
            st.caption("🟢 *Gemini API key active.*")
    elif provider == "openai":
        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Your OpenAI API key. Stored only in this active session.",
            key="openai_key_input_box",
        )
        if openai_key_input:
            os.environ["OPENAI_API_KEY"] = openai_key_input.strip()
        if not os.environ.get("OPENAI_API_KEY"):
            st.caption("🔑 *Enter your OpenAI API key above or save in .env*")
        else:
            st.caption("🟢 *OpenAI API key active (gpt-4o-mini).*")
    elif provider == "ollama" and not ollama_active:
        st.caption("🔴 *Ollama server offline on localhost:11434. Grounded extractive fallback will be used.*")
    elif provider == "mock":
        st.caption("⚡ *Offline mock provider active (fast, deterministic).*")

    st.divider()

    flashcard_count = st.slider(
        "Flashcard target count",
        min_value=5,
        max_value=25,
        value=15,
        step=5,
    )
    mcq_count = st.slider(
        "MCQ question count",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
    )
    quiz_difficulty = st.selectbox(
        "Quiz Difficulty Target",
        options=["Adaptive Mix", "Hard", "Medium", "Easy"],
        index=0,
        help="Target difficulty for generated quiz questions.",
        key="sidebar_quiz_difficulty_selector",
    )
    st.divider()

    run_verification = st.toggle(
        "Run factuality verification",
        value=False,
        help="Uses cross-encoder/nli-deberta-v3-base (slow, offline). Downloads model on first use.",
    )
    st.divider()

    st.markdown(
        """
<div style="font-size:0.76rem;color:#6b6490;line-height:1.7">
<b style="color:#9d94c8">Features</b><br>
🌐 Multi-Language (English / 日本語)<br>
🎯 Difficulty Adaptive Quizzes<br>
📺 Smart Video Recommendations<br>
🏆 Badges, XP & Concept Mastery<br>
🎮 Interactive Learning Challenges
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("<div style='font-size:3rem;margin-top:0.2rem'>🎓</div>", unsafe_allow_html=True)
with col_title:
    title_suffix = " (日本語版)" if current_lang == "Japanese" else ""
    st.markdown(f'<p class="hero-title">LectureNotes AI{title_suffix}</p>', unsafe_allow_html=True)
    if current_lang == "Japanese":
        st.markdown(
            '<p class="hero-sub">講義PDF・テキストから構造化ノート、重要概念、フラッシュカード、難易度別クイズ、関連動画を自動生成。'
            'ゲーミフィケーションとバッジで楽しく学習。</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="hero-sub">Convert lecture PDFs into structured notes, flashcards, and adaptive quizzes — '
            'with video suggestions, concept mastery tracking, and achievement badges.</p>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Motivational Greeting & Streak Banner ──
if auth_user and "motivational_message" in st.session_state:
    m_data = st.session_state["motivational_message"]
    today_fmt = datetime.date.today().strftime("%A, %B %d, %Y")
    streak_badge_color = "#34d399" if progress.streak >= 7 else ("#fbbf24" if progress.streak >= 2 else "#60a5fa")
    streak_label = f"🔥 Day {progress.streak} Streak" if progress.streak >= 2 else "🌱 Day 1: First Step!"
    st.markdown(
        f"""
<div style="background:linear-gradient(135deg, rgba(124,77,255,0.18) 0%, rgba(96,165,250,0.12) 50%, rgba(52,211,153,0.1) 100%);
            border:1px solid rgba(124,77,255,0.4);border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1.3rem;
            box-shadow:0 6px 24px rgba(0,0,0,0.25);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px">
  <div style="flex:1;min-width:280px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.35rem;flex-wrap:wrap">
      <span style="font-size:1.6rem">{m_data.get('icon', '💡')}</span>
      <span style="font-weight:700;font-size:1.05rem;color:#f1f0fb">{m_data.get('title', 'Daily Inspiration')}</span>
      <span style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:12px;padding:2px 10px;font-size:0.75rem;color:#cbd5e1">
        📅 {today_fmt}
      </span>
      <span style="background:{streak_badge_color}22;border:1px solid {streak_badge_color};border-radius:12px;padding:2px 10px;font-size:0.75rem;color:{streak_badge_color};font-weight:700">
        {streak_label}
      </span>
    </div>
    <div style="font-size:0.9rem;color:#e2e8f0;font-style:italic;margin-bottom:0.3rem;line-height:1.4">
      {m_data.get('quote', '')}
    </div>
    <div style="font-size:0.78rem;color:#a78bfa">
      — <b style="color:#d0daf8">{m_data.get('author', 'Study Mentor')}</b> &nbsp;·&nbsp;
      <span style="color:#34d399;font-weight:600">{m_data.get('streak_subtitle', '')}</span>
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:0.72rem;color:#9d94c8;text-transform:uppercase;letter-spacing:0.04em">Active Learner</div>
    <div style="font-size:1.1rem;font-weight:800;color:#fbbf24">Level {progress.get_level_info()['level']}: {auth_user['username']}</div>
    <div style="font-size:0.76rem;color:#34d399;margin-top:0.15rem">✨ {progress.xp} XP &nbsp;·&nbsp; Best: {progress.best_streak} 🔥</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────────────────────

col_upload, col_samples = st.columns([3, 2])

with col_upload:
    uploaded = st.file_uploader(
        "Upload a lecture PDF or plain-text file" if current_lang == "English" else "講義PDFまたはテキストファイルをアップロード",
        type=["pdf", "txt"],
        help="PDF files are extracted page-by-page; .txt files are split by sentence.",
        label_visibility="visible",
    )

with col_samples:
    st.markdown("<p style='font-size:0.83rem;color:#a89fd8;margin-bottom:0.4rem;font-weight:600'>⚡ QUICK DEMO SAMPLES</p>", unsafe_allow_html=True)
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "samples"
    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        if st.button("📄 Sample 1\n(ML Basics)", use_container_width=True):
            s1_pdf = sample_dir / "sample_lecture_1.pdf"
            if s1_pdf.exists():
                st.session_state["active_file_bytes"] = s1_pdf.read_bytes()
                st.session_state["active_file_name"] = "sample_lecture_1.pdf"
                st.session_state["language"] = "English"
                st.session_state.pop("result", None)
                st.rerun()
    with c_s2:
        if st.button("📄 Sample 2\n(Deep Learning)", use_container_width=True):
            s2_pdf = sample_dir / "sample_lecture_2.pdf"
            if s2_pdf.exists():
                st.session_state["active_file_bytes"] = s2_pdf.read_bytes()
                st.session_state["active_file_name"] = "sample_lecture_2.pdf"
                st.session_state["language"] = "English"
                st.session_state.pop("result", None)
                st.rerun()
    with c_s3:
        if st.button("📄 Sample 3 🇯🇵\n(機械学習入門)", use_container_width=True):
            s3_txt = sample_dir / "sample_lecture_ja.txt"
            if s3_txt.exists():
                st.session_state["active_file_bytes"] = s3_txt.read_bytes()
                st.session_state["active_file_name"] = "sample_lecture_ja.txt"
                st.session_state["language"] = "Japanese"
                st.session_state.pop("result", None)
                st.rerun()

# Resolve active file (either user upload or loaded demo sample)
active_bytes = None
active_name = None

if uploaded is not None:
    active_bytes = uploaded.read()
    active_name = uploaded.name
    st.session_state["active_file_bytes"] = active_bytes
    st.session_state["active_file_name"] = active_name
elif "active_file_bytes" in st.session_state:
    active_bytes = st.session_state["active_file_bytes"]
    active_name = st.session_state["active_file_name"]
    st.info(f"Loaded demo file: **{active_name}** ({len(active_bytes):,} bytes)")

btn_label = f"✨ Generate Lecture Materials ({current_lang})" if active_name else f"✨ Generate Lecture Materials ({current_lang})"
col_gen_main, col_gen_fast = st.columns([3.2, 1.2])
with col_gen_main:
    generate_btn = st.button(
        btn_label,
        disabled=active_bytes is None,
        use_container_width=True,
    )
with col_gen_fast:
    fast_btn = st.button(
        "⚡ Fast Mode (Instant)",
        disabled=active_bytes is None,
        help="Runs full generation in ~1s using offline deterministic synthesis without external network wait.",
        use_container_width=True,
    )
    if fast_btn:
        provider = "mock"
        generate_btn = True


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

if generate_btn and active_bytes is not None:
    status_container = st.empty()
    progress_bar = st.progress(0)

    status_container.info(f"**Starting pipeline in {current_lang}…**")
    progress_bar.progress(5)

    try:
        with st.spinner(""):
            result = _run_pipeline(
                file_bytes=active_bytes,
                file_name=active_name,
                provider=provider,
                flashcard_count=flashcard_count,
                mcq_count=mcq_count,
                run_verification=run_verification,
                language=current_lang,
                quiz_difficulty=quiz_difficulty,
            )
        progress_bar.progress(100)
        status_container.success("**🎉 Done! Explore your lecture materials below.**")
        time.sleep(1.0)
        status_container.empty()
        progress_bar.empty()
        st.session_state["result"] = result
        st.session_state["file_name"] = active_name
        # Invalidate exam question cache from any previous lecture
        for _exam_ck in ("_exam_cache_key", "_exam_cache_mcqs", "_exam_cache_descs"):
            st.session_state.pop(_exam_ck, None)

        # Persist lecture history in SQLite and award processing XP without resetting user progress
        if auth_user:
            try:
                record_user_lecture_db(
                    user_id=auth_user["id"],
                    lecture_title=result["output"].lecture_title,
                    file_name=active_name,
                    sections_count=len(result["sections"]),
                    concepts_count=len(result["all_concepts"]),
                )
                progress.xp += 50
                _sync_user_progress(progress)
            except Exception:
                pass

        # Polyglot badge trigger if Japanese
        if current_lang == "Japanese":
            b_unlocked = progress.record_language("Japanese")
            _sync_user_progress(progress)
            for b in b_unlocked:
                st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)}!")
                st.balloons()

        st.rerun()

    except Exception as exc:
        progress_bar.empty()
        status_container.empty()
        st.error(f"**Pipeline failed:** {exc}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.markdown(
        """
<div style="text-align:center;padding:4rem 0;color:#5a547a">
    <div style="font-size:4rem">📚</div>
    <div style="font-size:1.1rem;margin-top:0.8rem">Upload a lecture file and click <b>Generate</b> to begin</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

result = st.session_state["result"]
output = result["output"]
sections = result["sections"]
section_concepts = result["section_concepts"]
all_concepts = result["all_concepts"]
report = result["report"]
file_name = st.session_state.get("file_name", "Lecture")

# ── Summary metrics ───────────────────────────────────────────────────────────
st.markdown(f"## 📖 {output.lecture_title}")

# Concept mastery progress bar
mastered_count = len(progress.completed_concepts.intersection({c.term.lower() for c in all_concepts}))
total_concepts_cnt = len(all_concepts)
mastery_pct = (mastered_count / total_concepts_cnt) if total_concepts_cnt > 0 else 0.0

st.markdown(
    f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(124,77,255,0.25);border-radius:12px;padding:0.9rem 1.4rem;margin-bottom:1.2rem">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
    <span style="font-size:0.85rem;color:#a78bfa;font-weight:600">🎯 CONCEPT MASTERY PROGRESS</span>
    <span style="font-size:0.85rem;color:#34d399;font-weight:700">{mastered_count} / {total_concepts_cnt} Mastered ({int(mastery_pct*100)}%)</span>
  </div>
  <div style="background:rgba(255,255,255,0.08);border-radius:8px;height:8px;overflow:hidden">
    <div style="background:linear-gradient(90deg, #7c4dff, #34d399);width:{int(mastery_pct*100)}%;height:100%"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
for col, value, label in [
    (m1, len(sections), "Topic Sections" if current_lang == "English" else "トピックセクション"),
    (m2, len(all_concepts), "Key Concepts" if current_lang == "English" else "抽出重要概念"),
    (m3, len(output.section_notes), "Section Notes" if current_lang == "English" else "セクション要約"),
    (m4, len(output.flashcards), "Flashcards" if current_lang == "English" else "復習カード"),
    (m5, len(output.quiz), "Quiz Questions" if current_lang == "English" else "クイズ設問数"),
]:
    with col:
        st.markdown(
            f"""<div class="metric-tile">
  <div class="metric-value">{value}</div>
  <div class="metric-label">{label}</div>
</div>""",
            unsafe_allow_html=True,
        )

if report:
    st.markdown("")
    r1, r2, r3, r4 = st.columns(4)
    rate_pct = int(report.factuality_rate * 100)
    color = _pct_color(report.factuality_rate)
    for col, value, label, clr in [
        (r1, report.total_claims, "Claims Verified", "#a78bfa"),
        (r2, report.entailed_count, "Entailed", "#34d399"),
        (r3, report.neutral_count, "Neutral", "#fbbf24"),
        (r4, f"{rate_pct}%", "Factuality Rate", color),
    ]:
        with col:
            st.markdown(
                f"""<div class="metric-tile">
  <div class="metric-value" style="background:linear-gradient(90deg,{clr},{clr});-webkit-background-clip:text">{value}</div>
  <div class="metric-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

st.divider()

# ── AI Lecture Tutor Initialization ──
tutor_llm = None
try:
    from lecturenotes.config import load_config
    cfg_tut = load_config()
    cfg_tut.llm.provider = provider
    if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
        cfg_tut.llm.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        cfg_tut.llm.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    tutor_llm = LLMClient(config=cfg_tut)
except Exception:
    pass

tutor = LectureTutor(
    lecture_title=output.lecture_title,
    sections=sections,
    concepts=all_concepts,
    concept_definitions=output.concept_definitions,
    section_notes=output.section_notes,
    flashcards=output.flashcards,
    quiz=output.quiz,
    language=current_lang,
    llm=tutor_llm,
)

# ── Tab layout: Dedicated full navigation ────────────────────────────────────
tab_tutor, tab_notes, tab_concepts, tab_videos, tab_flash, tab_quiz, tab_exam, tab_challenges, tab_badges, tab_analytics, tab_verify = st.tabs(
    [
        "🤖 AI Tutor",
        "📝 Notes",
        "🔑 Concepts",
        "📺 Videos",
        "🃏 Flashcards",
        "📋 Quiz",
        "🎯 Exam Prep",
        "🎮 Study Arena",
        "🏆 Badges & Rewards",
        "📊 Study Analytics",
        "🔬 Verification",
    ]
)


# ── TAB: AI Lecture Tutor ────────────────────────────────────────────────────
with tab_tutor:
    st.markdown("### 🤖 Interactive Grounded AI Lecture Tutor" if current_lang == "English" else "### 🤖 対話型AI講義チューター")
    st.caption("Ask questions, test analogies, or clarify anything grounded directly in this lecture's source content.")

    active_key = (
        (provider == "gemini" and bool(os.environ.get("GEMINI_API_KEY", "").strip()))
        or (provider == "openai" and bool(os.environ.get("OPENAI_API_KEY", "").strip()))
        or (provider in ("azure", "claude") and bool(os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")))
        or provider == "ollama"
    )

    if active_key:
        st.markdown(
            f"""<div style="background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);padding:7px 14px;border-radius:8px;font-size:0.84rem;color:#86efac;margin-bottom:12px;">
🟢 <b>Natural Conversational AI Mode Active</b> (responding via <b>{provider.title()}</b>)
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.3);padding:7px 14px;border-radius:8px;font-size:0.84rem;color:#fde047;margin-bottom:12px;">
🟡 <b>Extractive Fallback Mode Active</b> (bulleted quotes). Connect your API key below or in the sidebar to enable natural conversation!
</div>""",
            unsafe_allow_html=True,
        )
        with st.expander("🔑 Connect API Key for Natural Conversation", expanded=True):
            st.caption("Paste your Gemini (`AIzaSy...`) or OpenAI (`sk-...`) API key to talk to the AI tutor like a human:")
            c_key_in, c_key_btn = st.columns([3, 1])
            with c_key_in:
                direct_key_val = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="AIzaSy... or sk-...",
                    key="direct_tab_key_val",
                    label_visibility="collapsed",
                )
            with c_key_btn:
                if st.button("Connect Key", key="btn_direct_key_submit", use_container_width=True):
                    if direct_key_val.strip():
                        raw_k = direct_key_val.strip()
                        if raw_k.startswith("sk-"):
                            os.environ["OPENAI_API_KEY"] = raw_k
                        else:
                            os.environ["GEMINI_API_KEY"] = raw_k
                        from lecturenotes.generation.llm_client import LLMClient
                        LLMClient.reset_gemini_quota()
                        st.toast("🎉 API Key connected! Natural conversational mode is active.")
                        st.rerun()

    if "chat_messages" not in st.session_state or not st.session_state["chat_messages"]:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"👋 こんにちは、**{auth_user['username']}** さん！「**{output.lecture_title}**」のAI講義チューターです。"
                    " 講義内容の疑問点、直観的な例え、重要なポイントなど、何でも気軽に質問してください！"
                    if current_lang == "Japanese"
                    else f"👋 Hello **{auth_user['username']}**! I am your AI Lecture Tutor for **{output.lecture_title}**."
                    " Ask me to explain concepts, give intuitive real-world analogies, or quiz your understanding!"
                ),
            }
        ]

    # Quick starter question chips
    st.markdown("<p style='font-size:0.82rem;color:#a78bfa;font-weight:600;margin-bottom:0.4rem'>💡 SUGGESTED QUESTIONS:</p>", unsafe_allow_html=True)
    starter_qs = tutor.get_starter_questions()
    c_q1, c_q2 = st.columns(2)
    prompt_to_send = None

    for i, sq in enumerate(starter_qs[:4]):
        col_sq = c_q1 if i % 2 == 0 else c_q2
        with col_sq:
            if st.button(f"💬 {sq}", key=f"chip_q_{i}", use_container_width=True):
                prompt_to_send = sq

    # Render previous messages
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input handling
    typed_prompt = st.chat_input("Ask a question about this lecture... / 講義について質問する...")
    active_prompt = prompt_to_send or typed_prompt

    if active_prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting lecture notes..."):
                tutor_answer = tutor.answer(active_prompt, chat_history=st.session_state["chat_messages"][:-1])
                st.markdown(tutor_answer)
                st.session_state["chat_messages"].append({"role": "assistant", "content": tutor_answer})

        new_b = progress.record_chat_query()
        _sync_user_progress(progress)
        for b in new_b:
            st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)} (+{b.xp_reward} XP)!")
            st.balloons()
        st.rerun()

    # Clear chat option
    col_c1, _ = st.columns([1, 4])
    with col_c1:
        if st.button("🗑️ Clear Chat", key="btn_clear_chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()


# ── TAB: Section Notes ────────────────────────────────────────────────────────
with tab_notes:
    st.markdown("### Structured Section Notes" if current_lang == "English" else "### 構造化セクションノート")
    if not output.section_notes:
        st.info("No section notes were generated.")
    for note in output.section_notes:
        ref = f"  *{note.source_reference}*" if note.source_reference else ""
        with st.expander(
            f"**Section {note.section_id}** — {note.title}{ref}",
            expanded=note.section_id == 1,
        ):
            if not note.summary_points:
                st.caption("No summary points available.")
            for i, pt in enumerate(note.summary_points, 1):
                st.markdown(
                    f"""<div class="card">
  <div style="font-size:0.92rem;color:#e0daf8;line-height:1.6">
    <b style="color:#a78bfa">{i}.</b> {pt.summary_sentence}
  </div>
  <div class="source-quote">📎 {pt.source_span}</div>
</div>""",
                    unsafe_allow_html=True,
                )


# ── TAB: Key Concepts ─────────────────────────────────────────────────────────
with tab_concepts:
    st.markdown("### Key Concepts & Definitions" if current_lang == "English" else "### 重要概念と用語定義")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("#### 📌 Extracted Concepts & Mastery" if current_lang == "English" else "#### 📌 抽出概念と習得状況")
        for sc in section_concepts:
            if not sc.concepts:
                continue
            st.markdown(
                f'<div class="section-badge">§{sc.section_id} · {sc.section_title}</div>',
                unsafe_allow_html=True,
            )
            for c in sc.concepts:
                is_mastered = c.term.lower() in progress.completed_concepts
                btn_txt = "⭐ Mastered" if is_mastered else "Mark Mastered"
                c_btn_key = f"master_{sc.section_id}_{c.term}"

                col_tag, col_btn = st.columns([3, 1.4])
                with col_tag:
                    st.markdown(
                        f"""<div style="padding:4px 8px;margin-bottom:4px;border-radius:8px;background:rgba(124,77,255,0.12);border:1px solid rgba(124,77,255,0.25)">
  <b style="color:#c4b8fa;font-size:0.88rem">{c.term}</b> <span style="font-size:0.75rem;color:#9d94c8;opacity:0.75">({c.score:.2f})</span>
  {f'<span style="color:#34d399;font-size:0.75rem;margin-left:6px;font-weight:600">✔ Mastered</span>' if is_mastered else ''}
</div>""",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button(btn_txt, key=c_btn_key, use_container_width=True):
                        if is_mastered:
                            progress.unmark_concept(c.term)
                        else:
                            new_b = progress.mark_concept_completed(c.term, total_concepts_count=len(all_concepts))
                            for b in new_b:
                                st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)}!")
                                st.balloons()
                        _sync_user_progress(progress)
                        st.rerun()

    with c_right:
        st.markdown("#### 📖 Definitions" if current_lang == "English" else "#### 📖 用語定義")
        if not output.concept_definitions:
            st.info("No definitions were generated.")
        for defn in output.concept_definitions:
            st.markdown(
                f"""<div class="card">
  <div style="font-weight:600;color:#a78bfa;margin-bottom:0.3rem">{defn.term}</div>
  <div style="font-size:0.9rem;color:#e0daf8;line-height:1.6">{defn.definition}</div>
  <div class="source-quote">📎 {defn.source_span}</div>
</div>""",
                unsafe_allow_html=True,
            )
            # Visual schematic diagram for key concept
            visual_data = get_concept_visual(defn.term, definition=defn.definition, language=current_lang)
            if visual_data:
                with st.expander(f"🎨 Visual Schematic: {defn.term}", expanded=False):
                    is_v_viewed = defn.term.strip().lower() in progress.visuals_viewed
                    c_vis_info, c_vis_btn = st.columns([3, 1.4])
                    with c_vis_info:
                        if is_v_viewed:
                            st.markdown('<span style="font-size:0.8rem;color:#34d399;font-weight:600">✔ Diagram Inspected (+15 XP)</span>', unsafe_allow_html=True)
                    with c_vis_btn:
                        if not is_v_viewed:
                            if st.button("Mark Viewed (+15 XP)", key=f"btn_v_view_{defn.term}", use_container_width=True):
                                v_unlocked = progress.record_visual_viewed(defn.term)
                                _sync_user_progress(progress)
                                st.toast("Diagram inspected! +15 XP")
                                for b in v_unlocked:
                                    st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)} (+{b.xp_reward} XP)!")
                                st.rerun()
                    st.markdown(visual_data["svg_html"], unsafe_allow_html=True)
                    st.caption(f"ℹ️ {visual_data['caption']}")



# ── TAB: Video Recommendations ───────────────────────────────────────────────
with tab_videos:
    st.markdown("### 📺 Curated YouTube Video Lessons for Extracted Concepts" if current_lang == "English" else "### 📺 講義重要概念に基づくおすすめYouTube動画")
    st.markdown(
        f'<p style="color:#9d94c8;font-size:0.9rem">'
        f'Authoritative video lectures and tutorials automatically matched to concepts in <b>{output.lecture_title}</b> '
        f'(from top channels including 3Blue1Brown, StatQuest, MIT OpenCourseWare, freeCodeCamp, CS50, and Yobinori).</p>',
        unsafe_allow_html=True,
    )

    lecture_vids = result.get("lecture_videos") or []
    if not lecture_vids:
        lecture_vids = recommend_videos_for_lecture(all_concepts, output.lecture_title, language=current_lang, max_videos=4)

    st.markdown(f"#### 🌟 Top Recommended Videos for {output.lecture_title}")
    v_cols = st.columns(2)
    for idx, vid in enumerate(lecture_vids):
        with v_cols[idx % 2]:
            diff_pill = _diff_badge(vid.difficulty)
            takeaway_html = "".join(f"<li style='margin-bottom:3px'>{t}</li>" for t in vid.key_takeaways[:3])
            tags_html = " ".join(f"<span style='background:rgba(124,77,255,0.18);border-radius:4px;padding:2px 6px;font-size:0.72rem;color:#c4b8fa'>#{t}</span>" for t in vid.tags[:3])
            st.markdown(
                f"""<div class="video-card">
  <div class="video-title">🎥 {vid.title}</div>
  <div class="video-meta">
    <b style="color:#60a5fa">{vid.channel}</b> · ⏱ {vid.duration} &nbsp; {diff_pill}
  </div>
  <div style="margin-bottom:0.5rem">{tags_html}</div>
  <div style="font-size:0.85rem;color:#d0daf8;margin-bottom:0.8rem;line-height:1.5">{vid.description}</div>
  <div style="font-size:0.78rem;color:#a78bfa;margin-bottom:0.3rem;font-weight:600">KEY TAKEAWAYS:</div>
  <ul style="font-size:0.8rem;color:#cbd5e1;padding-left:1.1rem;margin-bottom:1rem;line-height:1.4">
    {takeaway_html}
  </ul>
  <a href="{vid.youtube_url}" target="_blank" class="video-link-btn">▶️ Watch on YouTube</a>
</div>""",
                unsafe_allow_html=True,
            )

    st.divider()

    # Interactive Concept Video Explorer
    st.markdown("#### 🔍 Search or Explore Videos by Specific Concept" if current_lang == "English" else "#### 🔍 特定概念の解説動画を探す")
    c_sel_col, c_custom_col = st.columns([1, 1])
    with c_sel_col:
        concept_names = [c.term for c in all_concepts]
        chosen_concept = st.selectbox(
            "Select extracted concept:",
            options=concept_names if concept_names else ["No concepts extracted"],
            key="concept_video_dropdown",
        )
    with c_custom_col:
        custom_topic = st.text_input(
            "Or search any custom concept / topic:",
            placeholder="e.g., Backpropagation, SVM Kernels, CNN...",
            key="custom_video_input",
        )

    target_term = custom_topic.strip() if custom_topic.strip() else chosen_concept
    if target_term and target_term != "No concepts extracted":
        targeted_vids = get_concept_videos(target_term, language=current_lang, max_results=3)

        # Award video explorer progress
        v_unlocked = progress.record_video_explored(target_term)
        if v_unlocked:
            _sync_user_progress(progress)
            for b in v_unlocked:
                st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)}!")
                st.balloons()

        st.markdown(f"<p style='color:#a78bfa;font-weight:600;margin-top:0.5rem'>Tutorials for: <i>{target_term}</i></p>", unsafe_allow_html=True)
        t_cols = st.columns(len(targeted_vids))
        for idx, vid in enumerate(targeted_vids):
            with t_cols[idx]:
                diff_pill = _diff_badge(vid.difficulty)
                takeaway_html = "".join(f"<li style='margin-bottom:3px'>{t}</li>" for t in vid.key_takeaways[:3])
                st.markdown(
                    f"""<div class="video-card">
  <div class="video-title">🎥 {vid.title}</div>
  <div class="video-meta">
    <b style="color:#60a5fa">{vid.channel}</b> · ⏱ {vid.duration} &nbsp; {diff_pill}
  </div>
  <div style="font-size:0.84rem;color:#d0daf8;margin-bottom:0.7rem;line-height:1.5">{vid.description}</div>
  <div style="font-size:0.78rem;color:#a78bfa;margin-bottom:0.3rem;font-weight:600">KEY TAKEAWAYS:</div>
  <ul style="font-size:0.8rem;color:#cbd5e1;padding-left:1.1rem;margin-bottom:1rem;line-height:1.4">
    {takeaway_html}
  </ul>
  <a href="{vid.youtube_url}" target="_blank" class="video-link-btn">▶️ Watch on YouTube</a>
</div>""",
                    unsafe_allow_html=True,
                )


# ── TAB: Flashcards ───────────────────────────────────────────────────────────
with tab_flash:
    st.markdown("### Flashcards" if current_lang == "English" else "### 復習フラッシュカード")
    if not output.flashcards:
        st.info("No flashcards were generated.")
    else:
        st.caption(f"{len(output.flashcards)} cards generated. Expand each card to reveal the answer.")
        cols = st.columns(2)
        for idx, card in enumerate(output.flashcards):
            q_display = card.question[:60] + "…" if len(card.question) > 60 else card.question
            with cols[idx % 2]:
                with st.expander(f"**Card {card.card_id}** — {q_display}"):
                    is_reviewed = card.card_id in progress.reviewed_flashcards
                    c_fc_st, c_fc_btn = st.columns([3, 1.4])
                    with c_fc_st:
                        if is_reviewed:
                            st.markdown('<span style="font-size:0.78rem;color:#34d399;font-weight:600">✔ Card Mastered (+10 XP)</span>', unsafe_allow_html=True)
                    with c_fc_btn:
                        if not is_reviewed:
                            if st.button("Mark Reviewed (+10 XP)", key=f"btn_fc_rev_{card.card_id}", use_container_width=True):
                                fc_unlocked = progress.record_flashcard_review(card.card_id)
                                _sync_user_progress(progress)
                                st.toast("Card reviewed! +10 XP")
                                for b in fc_unlocked:
                                    st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)}!")
                                    st.balloons()
                                st.rerun()

                    st.markdown(
                        f"""<div class="fc-question">❓ {card.question}</div>
<div class="fc-answer">✅ {card.answer}</div>""",
                        unsafe_allow_html=True,
                    )
                    if card.concept:
                        st.markdown(
                            f'<div style="margin-top:0.4rem"><span style="font-size:0.75rem;color:#7c4dff">🏷 {card.concept}</span></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div class="source-quote" style="margin-top:0.5rem">📎 {card.source_span}</div>',
                        unsafe_allow_html=True,
                    )

                    # In-depth concept deep-dive expander
                    with st.expander("💡 Describe Concept More (In-Depth Breakdown)", expanded=False):
                        target_term = card.concept or card.question
                        deep_dive = tutor.get_deep_dive(target_term)
                        is_dd_viewed = target_term.strip().lower() in progress.deep_dives_viewed
                        c_dd_st, c_dd_btn = st.columns([3, 1.4])
                        with c_dd_st:
                            if is_dd_viewed:
                                st.markdown('<span style="font-size:0.78rem;color:#34d399;font-weight:600">✔ Deep Dive Read (+15 XP)</span>', unsafe_allow_html=True)
                        with c_dd_btn:
                            if not is_dd_viewed:
                                if st.button("Mark Read (+15 XP)", key=f"btn_dd_read_{idx}_{card.card_id}", use_container_width=True):
                                    dd_unlocked = progress.record_deep_dive(target_term)
                                    _sync_user_progress(progress)
                                    st.toast("Deep dive completed! +15 XP")
                                    for b in dd_unlocked:
                                        st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)} (+{b.xp_reward} XP)!")
                                    st.rerun()

                        st.markdown(
                            f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(124,77,255,0.25);border-radius:10px;padding:0.8rem 1rem;margin-top:0.3rem">
  <div style="font-size:0.82rem;color:#a78bfa;font-weight:700;margin-bottom:0.25rem">🧠 INTUITIVE ANALOGY</div>
  <div style="font-size:0.85rem;color:#e2e8f0;margin-bottom:0.6rem;line-height:1.5">{deep_dive['intuition']}</div>
  <div style="font-size:0.82rem;color:#60a5fa;font-weight:700;margin-bottom:0.25rem">⚙️ CORE MECHANISM</div>
  <div style="font-size:0.85rem;color:#cbd5e1;margin-bottom:0.6rem;line-height:1.5">{deep_dive['mechanism']}</div>
  <div style="font-size:0.82rem;color:#34d399;font-weight:700;margin-bottom:0.25rem">🌍 REAL-WORLD APPLICATION</div>
  <div style="font-size:0.85rem;color:#cbd5e1;margin-bottom:0.6rem;line-height:1.5">{deep_dive['real_world']}</div>
  <div style="font-size:0.82rem;color:#fbbf24;font-weight:700;margin-bottom:0.25rem">📌 KEY TAKEAWAY</div>
  <div style="font-size:0.85rem;color:#fde68a;line-height:1.5">{deep_dive['takeaway']}</div>
</div>""",
                            unsafe_allow_html=True,
                        )

                    # Option to Ask Chatbot
                    col_fc_chat, _ = st.columns([1.6, 1])
                    with col_fc_chat:
                        c_label = card.concept if card.concept else "this concept"
                        if st.button(f"🤖 Ask Chatbot About '{c_label[:22]}'", key=f"btn_ask_chatbot_fc_{card.card_id}", use_container_width=True):
                            query_msg = f"Can you explain '{c_label}' in more detail with a real-world example?"
                            if "chat_messages" not in st.session_state:
                                st.session_state["chat_messages"] = []
                            st.session_state["chat_messages"].append({"role": "user", "content": query_msg})
                            tutor_ans = tutor.answer(query_msg, chat_history=st.session_state["chat_messages"])
                            st.session_state["chat_messages"].append({"role": "assistant", "content": tutor_ans})
                            unlocked_chat = progress.record_chat_query()
                            _sync_user_progress(progress)
                            for b in unlocked_chat:
                                st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)} (+{b.xp_reward} XP)!")
                            st.toast(f"🤖 Sent question to AI Tutor! Switch to the '🤖 AI Lecture Tutor' tab to read the full dialogue.")
                            st.rerun()


# ── TAB: Adaptive Quiz ────────────────────────────────────────────────────────
with tab_quiz:
    st.markdown("### Multiple-Choice Quiz with Adaptive Difficulty" if current_lang == "English" else "### 難易度適応型マルチチョイスクイズ")

    if not output.quiz:
        st.info("No quiz questions were generated.")
    else:
        # ── Quiz Difficulty Filter ──
        c_mode, c_diff = st.columns([1, 1.2])
        with c_mode:
            quiz_mode = st.radio(
                "Quiz Mode / モード",
                options=["Interactive Practice", "Answer Key & Study Guide"],
                horizontal=True,
                key="quiz_mode_selector",
            )
        with c_diff:
            diff_filter = st.radio(
                "Select Difficulty / 難易度選択",
                options=["All Difficulties", "Easy", "Medium", "Hard"],
                horizontal=True,
                key="quiz_diff_filter",
            )

        st.divider()

        # Filter questions according to chosen difficulty
        if diff_filter == "All Difficulties":
            active_quiz = output.quiz
        else:
            active_quiz = [q for q in output.quiz if q.difficulty.lower() == diff_filter.lower()]

        # Show difficulty status & streak indicators
        streak_combo_banner = ""
        if progress.quiz_streak >= 2:
            streak_combo_banner = f"""<div style="background:linear-gradient(90deg, rgba(234,179,8,0.22), rgba(249,115,22,0.22));border:1px solid rgba(249,115,22,0.45);border-radius:10px;padding:8px 14px;color:#fbbf24;font-size:0.88rem;margin-bottom:0.8rem;display:flex;justify-content:space-between;align-items:center;">
  <span>🔥 <b>COMBO STREAK x{progress.quiz_streak}!</b> {progress.quiz_streak} questions answered correctly in a row!</span>
  <span style="color:#fdba74;font-weight:600">⚡ Accuracy Multiplier Active</span>
</div>"""

        st.markdown(
            f"""{streak_combo_banner}
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;background:rgba(255,255,255,0.03);padding:8px 14px;border-radius:10px;border:1px solid rgba(124,77,255,0.2)">
  <span style="font-size:0.9rem;color:#c4b8fa">Showing <b>{len(active_quiz)}</b> questions for difficulty: <b>{diff_filter}</b></span>
  <div>
    <span style="font-size:0.85rem;color:#f97316;font-weight:600;margin-right:12px">🔥 Day {progress.daily_streak} Study Streak</span>
    <span style="font-size:0.85rem;color:#fbbf24;font-weight:600">⚡ Quiz Streak: <b>{progress.quiz_streak}</b> (Best: {progress.best_quiz_streak})</span>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        target_diff_label = diff_filter if diff_filter != "All Difficulties" else "Hard"
        col_gen_q, _ = st.columns([2, 3])
        with col_gen_q:
            if st.button(f"⚡ Generate 5 New {target_diff_label} Questions", key=f"btn_regen_diff_{target_diff_label}", use_container_width=True):
                with st.spinner(f"Generating 5 challenging {target_diff_label} questions with AI..."):
                    full_lecture_text = "\n\n".join(f"[{s.title}]\n{s.text}" for s in sections)
                    from lecturenotes.generation.generator import generate_quiz
                    from lecturenotes.generation.llm_client import LLMClient
                    from lecturenotes.config import load_config

                    _q_cfg = load_config()
                    _q_cfg.llm.provider = provider
                    if provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
                        _q_cfg.llm.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
                    elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
                        _q_cfg.llm.openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
                    _active_llm = LLMClient(config=_q_cfg)

                    try:
                        new_qs = generate_quiz(
                            full_lecture_text,
                            all_concepts,
                            _active_llm,
                            question_count=5,
                            difficulty=target_diff_label,
                            language=current_lang,
                            sections=sections,
                        )
                        if new_qs:
                            for q in new_qs:
                                if target_diff_label in ("Easy", "Medium", "Hard"):
                                    q.difficulty = target_diff_label
                            output.quiz = new_qs
                            st.session_state["result"]["output"] = output
                            st.session_state["quiz_gen_ts"] = time.time()
                            st.toast(f"🎉 Generated 5 new {target_diff_label} questions!")
                            st.rerun()
                        else:
                            st.warning(f"Could not generate questions for {target_diff_label}. Please try again.")
                    except Exception as e:
                        st.error(f"Failed to generate questions: {e}")

        if not active_quiz:
            st.warning(f"No questions currently found matching '{diff_filter}'. Click the button above to generate 5 new {target_diff_label} questions!")

        if quiz_mode == "Interactive Practice":
            correct_count = 0
            answered_count = 0

            # 50/50 Hint perk (unlocked at 100 XP)
            hint_50_unlocked = progress.xp >= 100
            if hint_50_unlocked:
                st.markdown(
                    '<div style="background:linear-gradient(90deg,rgba(124,77,255,0.15),rgba(251,191,36,0.1));border:1px solid rgba(251,191,36,0.4);border-radius:12px;padding:0.6rem 1rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.6rem">'
                    '<span style="font-size:1.1rem">💡</span>'
                    '<span style="font-size:0.85rem;color:#fbbf24;font-weight:600">50/50 Hint Unlocked</span>'
                    '<span style="font-size:0.78rem;color:#d0daf8;margin-left:0.4rem">— Use on any question below to eliminate 2 wrong choices</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            if "hint_used" not in st.session_state:
                st.session_state["hint_used"] = {}

            for q in active_quiz:
                diff_badge_html = _diff_badge(q.difficulty)
                st.markdown(f"#### Question {q.question_id} &nbsp; {diff_badge_html}", unsafe_allow_html=True)
                st.markdown(f"**{q.question}**")

                # 50/50 hint button per question
                hint_key = f"hint_{q.question_id}_{st.session_state.get('quiz_gen_ts', 0)}"
                eliminated = st.session_state["hint_used"].get(hint_key, [])
                if hint_50_unlocked and not eliminated:
                    if st.button(f"💡 50/50 Hint (Eliminate 2 Wrong Answers)", key=f"btn_hint_{q.question_id}_{st.session_state.get('quiz_gen_ts', 0)}"):
                        wrong = [k for k in q.options if k != q.correct_answer]
                        import random as _random
                        eliminated = _random.sample(wrong, min(2, len(wrong)))
                        st.session_state["hint_used"][hint_key] = eliminated
                        st.rerun()

                # Build available options after 50/50 elimination
                available_opts = [k for k in q.options if k not in eliminated]
                if eliminated:
                    st.markdown(
                        f'<span style="font-size:0.78rem;color:#f87171">🚫 Eliminated: {" & ".join(eliminated)} (50/50 hint used)</span>',
                        unsafe_allow_html=True,
                    )

                user_choice = st.radio(
                    f"Select answer for Q{q.question_id}:",
                    options=available_opts,
                    format_func=lambda letter, q=q: f"{letter}. {q.options[letter]}",
                    key=f"mcq_practice_{q.question_id}_{st.session_state.get('quiz_gen_ts', 0)}",
                    index=None,
                )

                if user_choice is not None:
                    answered_count += 1
                    is_correct = user_choice == q.correct_answer

                    # Record gamification progress
                    unlocked_badges = progress.record_quiz_answer(
                        question_id=q.question_id,
                        is_correct=is_correct,
                        difficulty=q.difficulty,
                        total_quiz_count=len(output.quiz),
                    )
                    _sync_user_progress(progress)

                    for b in unlocked_badges:
                        st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)} (+{b.xp_reward} XP)!")
                        st.balloons()

                    if is_correct:
                        correct_count += 1
                        st.markdown(
                            f'<div class="mcq-option mcq-correct">✅ <b>Correct!</b> ({user_choice}) {q.options[user_choice]}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="mcq-option mcq-wrong">❌ <b>Incorrect.</b> You selected {user_choice}, but the correct answer is <b>{q.correct_answer}</b> ({q.options[q.correct_answer]}).</div>',
                            unsafe_allow_html=True,
                        )
                        if q.distractor_explanations and user_choice in q.distractor_explanations:
                            st.markdown(
                                f'<div style="font-size:0.85rem;color:#fca5a5;margin:0.4rem 0;padding:0.5rem 0.9rem;background:rgba(248,113,113,0.12);border-radius:8px;border:1px solid rgba(248,113,113,0.25)">'
                                f'💡 <b>Why {user_choice} is a distractor:</b> {q.distractor_explanations[user_choice]}</div>',
                                unsafe_allow_html=True,
                            )

                    st.markdown(
                        f"""
<div style="margin-top:0.6rem;padding:0.7rem 1rem;background:rgba(52,211,153,0.07);
            border-radius:10px;border:1px solid rgba(52,211,153,0.2)">
  <span style="font-size:0.75rem;color:#34d399;font-weight:600">FULL EXPLANATION</span><br>
  <span style="font-size:0.88rem;color:#d0f0e8">{q.explanation}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="source-quote" style="margin-top:0.4rem">📎 {q.source_span}</div>',
                        unsafe_allow_html=True,
                    )
                st.divider()

            if answered_count > 0:
                pct = int((correct_count / len(active_quiz)) * 100)
                score_clr = _pct_color(correct_count / len(active_quiz))
                st.markdown(
                    f"""<div style="text-align:center;padding:1.2rem;background:rgba(255,255,255,0.03);border-radius:12px;border:1px solid rgba(124,77,255,0.25)">
  <div style="font-size:1.8rem;font-weight:700;color:{score_clr}">Session Score: {correct_count} / {len(active_quiz)} ({pct}%)</div>
  <div style="font-size:0.85rem;color:#9d94c8;margin-top:0.2rem">{answered_count} of {len(active_quiz)} questions attempted · Keep up the momentum!</div>
</div>""",
                    unsafe_allow_html=True,
                )

        else:
            for q in active_quiz:
                diff_badge_html = _diff_badge(q.difficulty)
                with st.expander(f"**Q{q.question_id}.** {q.question} ({q.difficulty})", expanded=False):
                    st.markdown(diff_badge_html, unsafe_allow_html=True)
                    for letter, option_text in q.options.items():
                        is_correct = letter == q.correct_answer
                        cls = "mcq-correct" if is_correct else "mcq-wrong"
                        icon = "✅" if is_correct else "○"
                        st.markdown(
                            f'<div class="mcq-option {cls}">{icon} <b>{letter}.</b> {option_text}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"""
<div style="margin-top:0.8rem;padding:0.7rem 1rem;background:rgba(52,211,153,0.07);
            border-radius:10px;border:1px solid rgba(52,211,153,0.2)">
  <span style="font-size:0.8rem;color:#34d399;font-weight:600">EXPLANATION</span><br>
  <span style="font-size:0.88rem;color:#d0f0e8">{q.explanation}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    if q.distractor_explanations:
                        with st.expander("Distractor explanations", expanded=False):
                            for letter, expl in q.distractor_explanations.items():
                                st.markdown(
                                    f'<div style="font-size:0.83rem;color:#9d94c8;margin-bottom:0.3rem">'
                                    f'<b style="color:#f87171">{letter}.</b> {expl}</div>',
                                    unsafe_allow_html=True,
                                )
                    st.markdown(
                        f'<div class="source-quote" style="margin-top:0.6rem">📎 {q.source_span}</div>',
                        unsafe_allow_html=True,
                    )


# ── TAB: Exam Prep Mode ───────────────────────────────────────────────────────
with tab_exam:
    st.markdown("### 🎯 Exam Prep Mode")
    st.caption("Simulate a full timed exam with multiple-choice and descriptive conceptual questions, plus instant rubric scoring.")

    ex_cfg_col, ex_info_col = st.columns([2.2, 1])
    with ex_cfg_col:
        exam_format = st.radio(
            "Exam Question Format:",
            options=["MCQ only", "Mixed (MCQ + Descriptive)", "Descriptive only"],
            index=1,
            horizontal=True,
            key="exam_format_choice",
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            exam_q_count = st.slider(
                "MCQ Questions:",
                min_value=0 if exam_format == "Descriptive only" else 3,
                max_value=25,
                value=0 if exam_format == "Descriptive only" else 5,
                step=1,
                key="exam_q_count",
                disabled=exam_format == "Descriptive only",
            )
        with col_c2:
            desc_q_count = st.slider(
                "Descriptive Questions:",
                min_value=1 if exam_format == "Descriptive only" else 0,
                max_value=15,
                value=3 if exam_format != "MCQ only" else 0,
                step=1,
                key="desc_q_count",
                disabled=exam_format == "MCQ only",
            )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            exam_time_limit = st.selectbox(
                "Time Limit:",
                options=[
                    "No limit",
                    "3 minutes (Blitz)",
                    "5 minutes",
                    "10 minutes",
                    "15 minutes",
                    "20 minutes",
                    "30 minutes (Standard)",
                    "45 minutes",
                    "60 minutes (Comprehensive)",
                ],
                index=2,
                key="exam_time_limit",
            )
        with col_t2:
            exam_difficulty = st.selectbox(
                "Difficulty Target:",
                options=["All", "Easy only", "Medium only", "Hard only", "Mixed (1E+2M+2H)"],
                index=0,
                key="exam_difficulty",
            )
    with ex_info_col:
        st.markdown(
            '<div style="background:rgba(124,77,255,0.1);border:1px solid rgba(124,77,255,0.4);border-radius:14px;padding:1rem;margin-top:0.2rem">'
            '<div style="font-size:0.85rem;color:#a78bfa;font-weight:700;margin-bottom:0.4rem">📋 Exam Rules & Scoring</div>'
            '<div style="font-size:0.78rem;color:#d0daf8;line-height:1.7">'
            '• MCQ questions: <b style="color:#34d399">+10 XP</b> each<br>'
            '• Descriptive questions: up to <b style="color:#fbbf24">+20 XP</b> each<br>'
            '• Pass standard: <b style="color:#34d399">≥80%</b> combined score<br>'
            '• Automated rubric grading on descriptive answers<br>'
            '• Auto-submits when countdown timer reaches zero'
            '</div></div>',
            unsafe_allow_html=True,
        )

    if "exam_submitted" not in st.session_state:
        st.session_state["exam_submitted"] = False
    if "exam_start_time" not in st.session_state:
        st.session_state["exam_start_time"] = None

    if not st.session_state["exam_submitted"] and st.session_state["exam_start_time"] is None:
        mcq_txt = f"{exam_q_count} MCQs" if (exam_format != "Descriptive only" and exam_q_count > 0) else ""
        desc_txt = f"{desc_q_count} Descriptive" if (exam_format != "MCQ only" and desc_q_count > 0) else ""
        q_summary_str = " + ".join(filter(None, [mcq_txt, desc_txt]))
        st.markdown(
            f'<div style="background:rgba(124,77,255,0.08);border:1px dashed rgba(124,77,255,0.5);border-radius:16px;padding:1.5rem;text-align:center;margin:1rem 0">'
            f'<div style="font-size:1.15rem;color:#a78bfa;font-weight:700;margin-bottom:0.4rem">🎯 Ready to Begin Your Practice Exam?</div>'
            f'<div style="font-size:0.85rem;color:#cbd5e1">{q_summary_str} · {exam_time_limit} · {exam_difficulty}</div>'
            f'<div style="font-size:0.75rem;color:#34d399;margin-top:0.4rem">⚡ Instant Mode: Calibrated questions generated instantly (0.0s) from lecture concepts with zero wait time</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("🚀 Start Exam", use_container_width=True, key="btn_start_exam"):
            # ── Build a fingerprint of the current exam configuration ──────────
            exam_cache_key = (
                getattr(output, "lecture_title", ""),
                exam_difficulty,
                exam_format,
                exam_q_count,
                desc_q_count,
                current_lang,
            )
            cached_mcqs = st.session_state.get("_exam_cache_mcqs")
            cached_descs = st.session_state.get("_exam_cache_descs")
            cached_key = st.session_state.get("_exam_cache_key")

            if cached_key == exam_cache_key and (cached_mcqs is not None or cached_descs is not None):
                mcqs = cached_mcqs or []
                descs = cached_descs or []
            else:
                mcq_n = exam_q_count if exam_format != "Descriptive only" else 0
                desc_n = desc_q_count if exam_format != "MCQ only" else 0
                mcqs, descs = generate_exam_questions_parallel(
                    output=output,
                    mcq_count=mcq_n,
                    desc_count=desc_n,
                    difficulty=exam_difficulty,
                    language=current_lang,
                    use_llm=False,
                )

                # Cache results for instant re-use
                st.session_state["_exam_cache_key"] = exam_cache_key
                st.session_state["_exam_cache_mcqs"] = mcqs
                st.session_state["_exam_cache_descs"] = descs

            st.session_state["active_exam_mcqs"] = mcqs
            st.session_state["active_exam_descs"] = descs
            st.session_state["exam_start_time"] = time.time()
            st.session_state["exam_submitted"] = False
            st.session_state["exam_time_over"] = False
            st.session_state["exam_xp_awarded"] = False
            for k in list(st.session_state.keys()):
                if k.startswith("exam_q_") or k.startswith("exam_desc_"):
                    st.session_state.pop(k, None)
            st.rerun()

    elif st.session_state["exam_start_time"] is not None and not st.session_state["exam_submitted"]:
        # Continuous real-time 1-second auto-refresh while exam is active
        if st_autorefresh is not None:
            st_autorefresh(interval=1000, key="exam_autorefresh_timer")

        mcq_questions = st.session_state.get("active_exam_mcqs", [])
        descriptive_questions = st.session_state.get("active_exam_descs", [])
        elapsed = int(time.time() - st.session_state["exam_start_time"])
        time_limit_map = {
            "3 minutes (Blitz)": 180,
            "5 minutes": 300,
            "10 minutes": 600,
            "15 minutes": 900,
            "20 minutes": 1200,
            "30 minutes (Standard)": 1800,
            "45 minutes": 2700,
            "60 minutes (Comprehensive)": 3600,
        }
        time_limit_secs = time_limit_map.get(exam_time_limit, None)

        if time_limit_secs:
            remaining = max(0, time_limit_secs - elapsed)
            mins, secs = divmod(remaining, 60)
            elapsed_mins, elapsed_secs = divmod(elapsed, 60)
            tc = "#f87171" if remaining < 60 else ("#fbbf24" if remaining < 300 else "#34d399")
            warning_txt = " ⚠️ TIME ALMOST UP!" if remaining < 60 else ""

            # Check if time expired -> auto-submit immediately!
            if remaining <= 0:
                st.session_state["exam_submitted"] = True
                st.session_state["exam_time_over"] = True
                st.session_state["exam_elapsed"] = elapsed
                auto_mcq: dict = {}
                for qi_idx, q_item in enumerate(mcq_questions):
                    key = f"exam_q_{qi_idx}_{q_item.question_id}"
                    auto_mcq[qi_idx] = st.session_state.get(key)
                st.session_state["exam_answers"] = auto_mcq
                st.session_state["exam_questions_data"] = [
                    (q.question_id, q.correct_answer, q.difficulty, q.question, dict(q.options), q.explanation)
                    for q in mcq_questions
                ]
                auto_desc: dict = {}
                for di_idx, d_item in enumerate(descriptive_questions):
                    d_key = f"exam_desc_{di_idx}_{d_item.question_id}"
                    auto_desc[di_idx] = st.session_state.get(d_key, "")
                st.session_state["exam_desc_answers"] = auto_desc
                st.session_state["exam_desc_data"] = [
                    (dq.question_id, dq.question, dq.difficulty, dq.key_points, dq.model_answer, dq.source_reference, dq.concept_term)
                    for dq in descriptive_questions
                ]
                st.session_state["exam_xp_awarded"] = False
                st.rerun()

            # Live ticking countdown timer
            st.markdown(
                f"""
                <div id="exam-timer-bar" style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.35);border-radius:12px;padding:0.7rem 1.2rem;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;font-family:'Inter',sans-serif">
                  <span style="color:#fbbf24;font-weight:700;font-size:1.05rem">⏱️ Elapsed: <b>{elapsed_mins}:{elapsed_secs:02d}</b></span>
                  <span style="color:{tc};font-weight:800;font-size:1.15rem;letter-spacing:0.5px">⏳ Remaining: <b>{mins}:{secs:02d}</b>{warning_txt}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            elapsed_mins, elapsed_secs = divmod(elapsed, 60)
            st.markdown(
                f"""
                <div style="background:rgba(124,77,255,0.08);border:1px solid rgba(124,77,255,0.3);border-radius:10px;padding:0.6rem 1.2rem;margin-bottom:1rem">
                  <span style="color:#a78bfa;font-weight:700;font-size:1rem">⏱️ Elapsed: <b>{elapsed_mins}:{elapsed_secs:02d}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # ── SECTION 1: Multiple Choice Questions ──
            exam_user_answers: dict = {}
            if mcq_questions:
                st.markdown(f"#### 🔘 Section 1: Multiple Choice ({len(mcq_questions)} Questions)")
                st.caption("Select the best answer for each question.")
                for qi, q in enumerate(mcq_questions):
                    st.markdown(f"**Q{qi+1}** &nbsp; {_diff_badge(q.difficulty)}", unsafe_allow_html=True)
                    st.markdown(f"**{q.question}**")
                    exam_user_answers[qi] = st.radio(
                        f"Q{qi+1}:",
                        options=list(q.options.keys()),
                        format_func=lambda letter, q=q: f"{letter}. {q.options[letter]}",
                        key=f"exam_q_{qi}_{q.question_id}",
                        index=None,
                        label_visibility="collapsed",
                    )
                    st.markdown("")

            # ── SECTION 2: Descriptive Conceptual Questions ──
            exam_user_desc_answers: dict = {}
            if descriptive_questions:
                if mcq_questions:
                    st.markdown("---")
                st.markdown(f"#### ✍️ Section 2: Conceptual & Descriptive Questions ({len(descriptive_questions)} Questions)")
                st.caption("Provide thorough conceptual explanations. Rubrics evaluate key points, terminology, and depth.")
                for di, dq in enumerate(descriptive_questions):
                    st.markdown(f"**D{di+1}.** &nbsp; {_diff_badge(dq.difficulty)}", unsafe_allow_html=True)
                    st.markdown(f"**{dq.question}**")
                    d_ans = st.text_area(
                        f"Your response for D{di+1}:",
                        key=f"exam_desc_{di}_{dq.question_id}",
                        height=130,
                        placeholder="Write your explanation here (explain concepts, mechanisms, and key points)...",
                        label_visibility="collapsed",
                    )
                    exam_user_desc_answers[di] = d_ans
                    w_cnt = len((d_ans or "").split())
                    w_clr = "#34d399" if w_cnt >= 20 else ("#fbbf24" if w_cnt >= 10 else "#9d94c8")
                    st.markdown(f'<div style="font-size:0.75rem;color:{w_clr};text-align:right">Word count: {w_cnt} words</div>', unsafe_allow_html=True)
                    st.markdown("")

            # Completion tracker
            answered_mcq = sum(1 for v in exam_user_answers.values() if v is not None)
            answered_desc = sum(1 for v in exam_user_desc_answers.values() if v and len(v.strip().split()) >= 8)
            total_answered = answered_mcq + answered_desc
            total_required = len(mcq_questions) + len(descriptive_questions)

            st.markdown(f'<div style="color:#9d94c8;font-size:0.85rem">✏️ Progress: <b>{total_answered}/{total_required} items completed</b></div>', unsafe_allow_html=True)
            if total_answered < total_required:
                st.warning(f"{total_required - total_answered} item(s) still incomplete.")

            if st.button("📤 Submit Exam", use_container_width=True, key="btn_submit_exam", disabled=total_answered < total_required):
                st.session_state["exam_submitted"] = True
                st.session_state["exam_time_over"] = False
                st.session_state["exam_answers"] = dict(exam_user_answers)
                st.session_state["exam_questions_data"] = [
                    (q.question_id, q.correct_answer, q.difficulty, q.question, dict(q.options), q.explanation)
                    for q in mcq_questions
                ]
                st.session_state["exam_desc_answers"] = dict(exam_user_desc_answers)
                st.session_state["exam_desc_data"] = [
                    (dq.question_id, dq.question, dq.difficulty, dq.key_points, dq.model_answer, dq.source_reference, dq.concept_term)
                    for dq in descriptive_questions
                ]
                st.session_state["exam_elapsed"] = int(time.time() - st.session_state["exam_start_time"])
                st.rerun()

    elif st.session_state["exam_submitted"]:
        if st.session_state.get("exam_time_over"):
            st.markdown(
                """
                <div style="background:linear-gradient(135deg, rgba(239, 68, 68, 0.22), rgba(185, 28, 28, 0.08));border:2px solid #ef4444;border-radius:16px;padding:1.3rem 1.6rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1.2rem;box-shadow:0 0 30px rgba(239,68,68,0.2)">
                  <div style="font-size:2.8rem">⏰</div>
                  <div>
                    <div style="font-size:1.6rem;font-weight:800;color:#f87171;letter-spacing:0.5px">TIME OVER!</div>
                    <div style="font-size:0.95rem;color:#fca5a5;margin-top:0.2rem">The allotted time limit has expired. Your answers have been automatically collected and your score is displayed below.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        eq_data = st.session_state.get("exam_questions_data", [])
        ea_data = st.session_state.get("exam_answers", {})
        desc_data = st.session_state.get("exam_desc_data", [])
        desc_ans_data = st.session_state.get("exam_desc_answers", {})
        exam_elapsed = st.session_state.get("exam_elapsed", 0)

        # Evaluate MCQs
        mcq_correct = sum(1 for qi, (qid, ca, diff, qt, opts, expl) in enumerate(eq_data) if ea_data.get(qi) == ca)
        mcq_total = len(eq_data)
        mcq_points = mcq_correct * 10
        mcq_max = mcq_total * 10

        # Evaluate Descriptive questions
        desc_results = []
        desc_points = 0.0
        desc_max = len(desc_data) * 10
        for di, (dqid, d_prompt, d_diff, d_kps, d_model, d_src, d_term) in enumerate(desc_data):
            dq_obj = DescriptiveExamQuestion(
                question_id=dqid,
                question=d_prompt,
                difficulty=d_diff,
                key_points=d_kps,
                model_answer=d_model,
                source_reference=d_src,
                concept_term=d_term,
            )
            u_text = desc_ans_data.get(di, "")
            eval_res = evaluate_descriptive_answer(u_text, dq_obj)
            desc_results.append((dq_obj, u_text, eval_res))
            desc_points += eval_res["points_earned"]

        total_points = mcq_points + desc_points
        total_possible = mcq_max + desc_max
        score_pct = int((total_points / total_possible * 100)) if total_possible > 0 else 0
        passed = score_pct >= 80
        xp_earned = int(mcq_correct * 10 + desc_points * 1.5)

        # ── ONE-SHOT XP award: record exam result once per submission ──
        if not st.session_state.get("exam_xp_awarded", False):
            exam_badges = progress.record_exam_result(
                score_pct=score_pct,
                correct=mcq_correct + sum(1 for _, _, r in desc_results if r["passed"]),
                total=mcq_total + len(desc_data),
                difficulty=exam_difficulty,
                elapsed_secs=exam_elapsed,
                xp_earned=xp_earned,
                lecture_title=output.lecture_title,
            )
            _sync_user_progress(progress)
            st.session_state["exam_xp_awarded"] = True
            for b in exam_badges:
                st.toast(f"🎉 Badge Unlocked: {b.icon} {b.get_title(current_lang)} (+{b.xp_reward} XP)!")

        rc = "#34d399" if passed else ("#fbbf24" if score_pct >= 50 else "#f87171")
        rl = "🏆 EXAM PASSED!" if passed else ("📖 Good Attempt" if score_pct >= 50 else "❌ Needs Review")
        mt, st_secs = divmod(exam_elapsed, 60)

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border:2px solid {rc};border-radius:20px;'
            f'padding:2rem;text-align:center;margin-bottom:1.5rem;box-shadow:0 0 30px {rc}22">'
            f'<div style="font-size:4rem;font-weight:800;color:{rc}">{score_pct}%</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:#f1f0fb;margin:0.5rem 0">{rl}</div>'
            f'<div style="display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;margin-top:0.6rem">'
            f'<div style="background:rgba(255,255,255,0.06);border-radius:10px;padding:0.4rem 1rem">'
            f'<span style="font-size:1.2rem;font-weight:700;color:#60a5fa">{round(total_points, 1)}/{total_possible}</span>'
            f'<div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase">Points Earned</div></div>'
            f'<div style="background:rgba(255,255,255,0.06);border-radius:10px;padding:0.4rem 1rem">'
            f'<span style="font-size:1.2rem;font-weight:700;color:#fbbf24">+{xp_earned} XP</span>'
            f'<div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase">Earned</div></div>'
            f'<div style="background:rgba(255,255,255,0.06);border-radius:10px;padding:0.4rem 1rem">'
            f'<span style="font-size:1.2rem;font-weight:700;color:#a78bfa">{mt}m {st_secs}s</span>'
            f'<div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase">Time Taken</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if passed:
            st.balloons()

        # Score gauge bar
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.06);border-radius:10px;height:14px;overflow:hidden;margin-bottom:1.2rem">'
            f'<div style="background:linear-gradient(90deg,{"#34d399" if score_pct>=80 else "#fbbf24" if score_pct>=50 else "#f87171"},{rc});'
            f'width:{score_pct}%;height:100%;border-radius:10px;transition:width 0.6s"></div></div>',
            unsafe_allow_html=True,
        )

        # ── SECTION 1 REPORT: MCQ Breakdown ──
        if eq_data:
            st.markdown(f"#### 🔘 Multiple Choice Breakdown ({mcq_correct}/{mcq_total} Correct)")
            diff_wrong: dict = {"Easy": 0, "Medium": 0, "Hard": 0}
            for qi, (qid, ca, diff, qt, opts, expl) in enumerate(eq_data):
                ua = ea_data.get(qi)
                correct = ua == ca
                if not correct:
                    diff_wrong[diff] = diff_wrong.get(diff, 0) + 1
                with st.expander(f"{'✅' if correct else '❌'} Q{qi+1}: {qt[:72]}{'...' if len(qt) > 72 else ''}", expanded=not correct):
                    if ua:
                        st.markdown(
                            f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:0.4rem 0.9rem;margin-bottom:0.4rem">'
                            f'<span style="font-size:0.82rem;color:#9d94c8">Your answer:</span> '
                            f'<b style="color:{"#34d399" if correct else "#f87171"}">{ua}. {opts.get(ua, "—")}</b></div>',
                            unsafe_allow_html=True,
                        )
                    if not correct:
                        st.markdown(
                            f'<div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:8px;padding:0.4rem 0.9rem;margin-bottom:0.4rem">'
                            f'<span style="font-size:0.82rem;color:#9d94c8">Correct answer:</span> '
                            f'<b style="color:#34d399">{ca}. {opts.get(ca, "—")}</b></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"💡 {expl}")

        # ── SECTION 2 REPORT: Descriptive Rubric Evaluation ──
        if desc_results:
            st.markdown("---")
            st.markdown(f"#### ✍️ Descriptive Responses & Rubric Feedback ({round(desc_points, 1)}/{desc_max} Points)")
            for di, (dq_obj, u_text, eval_res) in enumerate(desc_results):
                d_sc = eval_res["score_pct"]
                d_clr = "#34d399" if d_sc >= 80 else ("#fbbf24" if d_sc >= 60 else "#f87171")
                with st.expander(f"D{di+1}: {dq_obj.question[:68]}... — {eval_res['points_earned']}/10 pts ({d_sc}%)", expanded=d_sc < 80):
                    st.markdown(f"**Question:** {dq_obj.question}")
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:0.8rem;margin:0.5rem 0">'
                        f'<div style="font-size:0.75rem;color:#9d94c8;text-transform:uppercase;font-weight:600">Your Submitted Answer:</div>'
                        f'<div style="font-size:0.88rem;color:#f1f0fb;line-height:1.5;margin-top:0.2rem">{u_text or "*(No response provided)*"}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Rubric Assessment ({eval_res['points_earned']}/10 pts):**")
                    st.info(eval_res["feedback"])

                    if eval_res["matched_key_points"]:
                        matched_chips = "".join(f'<span style="display:inline-block;background:rgba(52,211,153,0.15);border:1px solid rgba(52,211,153,0.4);border-radius:15px;padding:2px 8px;font-size:0.75rem;color:#a7f3d0;margin:2px">✔ {kp}</span>' for kp in eval_res["matched_key_points"])
                        st.markdown(f'<div style="font-size:0.8rem;color:#34d399;margin-bottom:0.3rem">Concepts Mentioned: {matched_chips}</div>', unsafe_allow_html=True)
                    if eval_res["missed_key_points"]:
                        missed_chips = "".join(f'<span style="display:inline-block;background:rgba(248,113,113,0.12);border:1px solid rgba(248,113,113,0.3);border-radius:15px;padding:2px 8px;font-size:0.75rem;color:#fca5a5;margin:2px">○ {kp}</span>' for kp in eval_res["missed_key_points"])
                        st.markdown(f'<div style="font-size:0.8rem;color:#f87171;margin-bottom:0.4rem">Concepts to Incorporate: {missed_chips}</div>', unsafe_allow_html=True)

                    st.markdown(
                        f'<div style="background:rgba(124,77,255,0.08);border:1px solid rgba(124,77,255,0.25);border-radius:8px;padding:0.8rem;margin-top:0.4rem">'
                        f'<div style="font-size:0.75rem;color:#a78bfa;text-transform:uppercase;font-weight:600">Model Reference Answer:</div>'
                        f'<div style="font-size:0.85rem;color:#d0daf8;line-height:1.5;margin-top:0.2rem">{dq_obj.model_answer}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Past exam history ──
        if progress.exam_history:
            st.divider()
            st.markdown("#### 📜 Your Exam History")
            history_to_show = list(reversed(progress.exam_history[-8:]))
            h_cols_hdr = st.columns([2, 1, 1, 1, 1, 1])
            for col_h, lbl in zip(h_cols_hdr, ["Lecture", "Score", "Completed", "Difficulty", "Time", "XP"]):
                col_h.markdown(f'<div style="font-size:0.72rem;color:#9d94c8;font-weight:600;text-transform:uppercase">{lbl}</div>', unsafe_allow_html=True)
            for entry in history_to_show:
                sc = entry.get("score_pct", 0)
                s_clr = "#34d399" if sc >= 80 else ("#fbbf24" if sc >= 50 else "#f87171")
                m_e, s_e = divmod(entry.get("elapsed_secs", 0), 60)
                row_cols = st.columns([2, 1, 1, 1, 1, 1])
                row_cols[0].markdown(f'<div style="font-size:0.82rem;color:#d0daf8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{entry.get("lecture_title", "—")[:28]}</div><div style="font-size:0.68rem;color:#6b6490">{entry.get("ts", "")}</div>', unsafe_allow_html=True)
                row_cols[1].markdown(f'<b style="color:{s_clr};font-size:0.95rem">{sc}%</b>', unsafe_allow_html=True)
                row_cols[2].markdown(f'<span style="color:#d0daf8;font-size:0.85rem">{entry.get("correct",0)}/{entry.get("total",0)}</span>', unsafe_allow_html=True)
                row_cols[3].markdown(f'<span style="font-size:0.8rem;color:#a78bfa">{entry.get("difficulty","—")}</span>', unsafe_allow_html=True)
                row_cols[4].markdown(f'<span style="font-size:0.82rem;color:#94a3b8">{m_e}m {s_e}s</span>', unsafe_allow_html=True)
                row_cols[5].markdown(f'<span style="font-size:0.82rem;color:#fbbf24">+{entry.get("xp_earned",0)}</span>', unsafe_allow_html=True)

        st.divider()
        col_retake, col_review = st.columns(2)
        with col_retake:
            if st.button("🔄 Retake Exam", use_container_width=True, key="btn_retake_exam"):
                for k in [
                    "exam_submitted",
                    "exam_start_time",
                    "exam_time_over",
                    "active_exam_mcqs",
                    "active_exam_descs",
                    "exam_answers",
                    "exam_questions_data",
                    "exam_desc_answers",
                    "exam_desc_data",
                    "exam_elapsed",
                    "exam_xp_awarded",
                ]:
                    st.session_state.pop(k, None)
                for k in list(st.session_state.keys()):
                    if k.startswith("exam_q_") or k.startswith("exam_desc_"):
                        st.session_state.pop(k, None)
                st.rerun()
        with col_review:
            if st.button("📊 View Full Analytics", use_container_width=True, key="btn_exam_to_analytics"):
                st.info("👆 Switch to the **📊 Study Analytics** tab above to see detailed performance trends.")



# ── TAB: Study Arena & Challenges ─────────────────────────────────────────────
with tab_challenges:
    st.markdown("### 🎮 Study Arena: Challenges & Game Modes" if current_lang == "English" else "### 🎮 学習アリーナ：挑戦チャレンジ")
    st.caption("Level up your skills by completing special challenges! Earn bonus XP and unlock exclusive achievement badges.")

    ch_list = get_all_challenges()
    ch_cols = st.columns(2)

    if "active_challenge_id" not in st.session_state:
        st.session_state["active_challenge_id"] = None

    for i, ch in enumerate(ch_list):
        is_done = ch.id in progress.completed_challenges
        is_active = st.session_state["active_challenge_id"] == ch.id
        can_claim, progress_str, pct = progress.is_challenge_eligible(ch.id, st.session_state)
        with ch_cols[i % 2]:
            # Progress bar for in-progress challenges
            pct_int = int(pct * 100)
            progress_bar_html = (
                f'<div style="background:rgba(255,255,255,0.08);border-radius:6px;height:6px;overflow:hidden;margin-bottom:0.5rem">'
                f'<div style="background:linear-gradient(90deg,#7c4dff,#34d399);width:{pct_int}%;height:100%;transition:width 0.4s"></div></div>'
                f'<div style="font-size:0.75rem;color:#9d94c8;margin-bottom:0.4rem">{progress_str}</div>'
            )
            st.markdown(
                f"""<div class="challenge-card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:1.15rem;font-weight:700;color:#f1f0fb">{ch.icon} {ch.get_title(current_lang)}</span>
    <span style="font-size:0.8rem;color:#fbbf24;font-weight:600">+{ch.xp_bonus} XP</span>
  </div>
  <div style="font-size:0.85rem;color:#d0daf8;margin:0.5rem 0 0.6rem 0;line-height:1.5">{ch.get_description(current_lang)}</div>
  {progress_bar_html if not is_done else ''}
  {"<div style='color:#34d399;font-size:0.85rem;font-weight:600'>✔ Completed & XP Claimed</div>" if is_done else ''}
</div>""",
                unsafe_allow_html=True,
            )
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                btn_play_lbl = "⏸ Close Challenge" if is_active else f"⚔️ Play {ch.get_title(current_lang)}"
                if st.button(btn_play_lbl, key=f"btn_play_{ch.id}", use_container_width=True):
                    st.session_state["active_challenge_id"] = None if is_active else ch.id
                    st.rerun()
            with col_b2:
                if is_done:
                    st.button(f"✅ Claimed (+{ch.xp_bonus} XP)", key=f"btn_claim_{ch.id}", use_container_width=True, disabled=True)
                elif can_claim:
                    if st.button(f"🎁 Claim (+{ch.xp_bonus} XP)", key=f"btn_claim_{ch.id}", use_container_width=True):
                        ch_badges = progress.record_challenge_completed(ch.id)
                        _sync_user_progress(progress)
                        for b in ch_badges:
                            st.toast(f"🎉 Challenge Cleared! Unlocked: {b.icon} {b.get_title(current_lang)}!")
                            st.balloons()
                        st.success(f"🎉 Challenge '{ch.get_title(current_lang)}' completed! Earned +{ch.xp_bonus} XP!")
                        st.rerun()
                else:
                    st.button(f"🔒 Locked ({progress_str})", key=f"btn_claim_{ch.id}", use_container_width=True, disabled=True)

    # Active Challenge Arena
    active_cid = st.session_state["active_challenge_id"]
    if active_cid:
        st.divider()
        active_ch = next((c for c in ch_list if c.id == active_cid), None)
        if active_ch:
            st.markdown(f"### ⚔️ Active Challenge: {active_ch.icon} {active_ch.get_title(current_lang)}")
            st.info(active_ch.get_description(current_lang))

            if active_cid == "speed_run":
                st.markdown("#### ⏱️ Rapid Fire: Answer all 5 questions accurately!")
                speed_correct = 0
                for q in output.quiz[:5]:
                    c_choice = st.radio(
                        f"**Q{q.question_id}: {q.question}**",
                        options=list(q.options.keys()),
                        format_func=lambda letter, q=q: f"{letter}. {q.options[letter]}",
                        key=f"speedrun_q_{q.question_id}",
                        index=None,
                    )
                    if c_choice == q.correct_answer:
                        speed_correct += 1

                if st.button("🏁 Submit Speed Run", use_container_width=True):
                    if speed_correct >= 4:
                        st.session_state["cleared_speed_run"] = True
                        st.balloons()
                        st.success(f"🏆 Speed Run Cleared with {speed_correct}/5 correct! Now click **🎁 Claim XP** on the card above to collect +150 XP!")
                    else:
                        st.warning(f"You got {speed_correct}/5 correct. Need at least 4/5 to clear. Try again!")

            elif active_cid == "survival":
                st.markdown("#### 💀 Sudden Death: 1 wrong answer and you are eliminated!")
                surv_streak = 0
                failed = False
                for q in output.quiz[:5]:
                    if failed:
                        break
                    s_choice = st.radio(
                        f"**Survival Q{q.question_id}: {q.question}**",
                        options=list(q.options.keys()),
                        format_func=lambda letter, q=q: f"{letter}. {q.options[letter]}",
                        key=f"survival_q_{q.question_id}",
                        index=None,
                    )
                    if s_choice is not None:
                        if s_choice == q.correct_answer:
                            surv_streak += 1
                            st.caption("🟢 Correct! Surviving to next question…")
                        else:
                            failed = True
                            st.error(f"💀 Elimination! Correct answer was {q.correct_answer}. You survived {surv_streak} questions.")

                if surv_streak >= 5 and not failed:
                    st.session_state["cleared_survival"] = True
                    st.balloons()
                    st.success("👑 SURVIVAL CHAMPION! Conquered 5/5 questions without a single mistake! Now click **🎁 Claim XP** on the card above!")

            elif active_cid == "hard_gauntlet":
                st.markdown("#### 🧗 Hard Mode Gauntlet: Elite High-Difficulty Questions")
                hard_qs = [q for q in output.quiz if q.difficulty == "Hard"]
                if not hard_qs:
                    hard_qs = output.quiz[:3]
                gauntlet_all_correct = True
                gauntlet_all_answered = True
                for q in hard_qs:
                    st.markdown(f"**Q{q.question_id}: {q.question}** &nbsp; {_diff_badge('Hard')}", unsafe_allow_html=True)
                    h_choice = st.radio(
                        f"Select answer for Q{q.question_id}:",
                        options=list(q.options.keys()),
                        format_func=lambda letter, q=q: f"{letter}. {q.options[letter]}",
                        key=f"hard_gauntlet_{q.question_id}",
                        index=None,
                    )
                    if h_choice is not None:
                        if h_choice == q.correct_answer:
                            st.success(f"✅ Masterful! {q.explanation}")
                        else:
                            st.error(f"❌ {q.explanation}")
                            gauntlet_all_correct = False
                    else:
                        gauntlet_all_answered = False
                        gauntlet_all_correct = False
                if gauntlet_all_answered and gauntlet_all_correct and not st.session_state.get("cleared_hard_gauntlet"):
                    st.session_state["cleared_hard_gauntlet"] = True
                    st.balloons()
                    st.success("🏆 GAUNTLET CONQUERED! All hard questions answered correctly! Click **🎁 Claim XP** above!")

            elif active_cid == "concept_sprint":
                st.markdown("#### 🏃 Concept Sprint: Master 3 concepts now for a quick boost!")
                unmastered = [c for c in all_concepts if c.term.lower() not in progress.completed_concepts]
                if not unmastered:
                    st.success("🎉 All concepts in this lecture are already mastered! You are a Domain Master!")
                else:
                    for c in unmastered[:3]:
                        col_c1, col_c2 = st.columns([3, 1])
                        with col_c1:
                            st.markdown(f"**{c.term}** — *{c.source_sentence or 'Key concept'}*")
                        with col_c2:
                            if st.button(f"⭐ Master {c.term}", key=f"sprint_master_{c.term}", use_container_width=True):
                                progress.mark_concept_completed(c.term, total_concepts_count=len(all_concepts))
                                _sync_user_progress(progress)
                                st.toast(f"Concept '{c.term}' mastered! (+50 XP)")
                                st.rerun()

            elif active_cid == "tutor_dialogue":
                st.markdown("#### 🤖 Tutor Socratic Dialogue: Exchange 3 questions with the AI Lecture Tutor!")
                st.info(f"Progress: **{progress.chat_queries_count} / 3** questions asked to AI Tutor.")
                if progress.chat_queries_count >= 3:
                    st.success("🎉 You've engaged in 3+ conceptual discussions with the AI Tutor!")
                    if "tutor_dialogue" not in progress.completed_challenges:
                        ch_badges = progress.record_challenge_completed("tutor_dialogue")
                        _sync_user_progress(progress)
                        st.balloons()
                        for b in ch_badges:
                            st.toast(f"🎉 Challenge Cleared! {b.icon} {b.get_title(current_lang)}!")
                else:
                    t_q_input = st.text_input("Ask a question right here to make progress:", placeholder="e.g., How does this concept apply in production?", key="arena_tutor_q")
                    if st.button("💬 Send to AI Tutor", key="btn_arena_tutor_send", use_container_width=True):
                        if t_q_input.strip():
                            if "chat_messages" not in st.session_state:
                                st.session_state["chat_messages"] = []
                            st.session_state["chat_messages"].append({"role": "user", "content": t_q_input})
                            a_ans = tutor.answer(t_q_input, chat_history=st.session_state["chat_messages"])
                            st.session_state["chat_messages"].append({"role": "assistant", "content": a_ans})
                            progress.record_chat_query()
                            _sync_user_progress(progress)
                            st.markdown(f"**Tutor Answer**: {a_ans}")
                            st.toast("Question answered! +10 XP awarded.")
                            st.rerun()

            elif active_cid == "visual_mastery_quest":
                st.markdown("#### 🎨 Visual Concept Discovery: Explore 3 concept architecture diagrams!")
                st.info(f"Progress: **{len(progress.visuals_viewed)} / 3** diagrams inspected.")
                for defn in output.concept_definitions[:3]:
                    v_d = get_concept_visual(defn.term, definition=defn.definition, language=current_lang)
                    if v_d:
                        with st.expander(f"🎨 Diagram: {defn.term}", expanded=False):
                            is_v = defn.term.strip().lower() in progress.visuals_viewed
                            if not is_v:
                                if st.button(f"🔍 Mark Diagram Explored (+15 XP)", key=f"btn_arena_vis_{defn.term}", use_container_width=True):
                                    progress.record_visual_viewed(defn.term)
                                    _sync_user_progress(progress)
                                    st.toast("Diagram explored! +15 XP")
                                    st.rerun()
                            else:
                                st.markdown('<span style="font-size:0.8rem;color:#34d399;font-weight:600">✔ Diagram Inspected (+15 XP)</span>', unsafe_allow_html=True)
                            st.markdown(v_d["svg_html"], unsafe_allow_html=True)
                            st.caption(v_d["caption"])

            elif active_cid == "deep_dive_sprint":
                st.markdown("#### 🌊 Deep Dive Sprint: Read 3 in-depth concept flashcard breakdowns!")
                st.info(f"Progress: **{len(progress.deep_dives_viewed)} / 3** deep dives completed.")
                for fc in output.flashcards[:3]:
                    fc_term = fc.concept or fc.question
                    with st.expander(f"💡 Deep Dive: {fc_term[:45]}", expanded=False):
                        dd_info = tutor.get_deep_dive(fc_term)
                        is_dd = fc_term.strip().lower() in progress.deep_dives_viewed
                        if not is_dd:
                            if st.button(f"📖 Mark Deep Dive Read (+15 XP)", key=f"btn_arena_dd_{fc.card_id}", use_container_width=True):
                                progress.record_deep_dive(fc_term)
                                _sync_user_progress(progress)
                                st.toast("Deep dive read! +15 XP")
                                st.rerun()
                        else:
                            st.markdown('<span style="font-size:0.8rem;color:#34d399;font-weight:600">✔ Deep Dive Completed (+15 XP)</span>', unsafe_allow_html=True)
                        st.markdown(f"**Intuition**: {dd_info['intuition']}")
                        st.markdown(f"**Mechanism**: {dd_info['mechanism']}")
                        st.markdown(f"**Takeaway**: {dd_info['takeaway']}")

            elif active_cid == "visual_diagram_challenge":
                st.markdown("#### 🔬 Visual Schematic Diagnosis: Inspect the diagram and answer reasoning questions!")
                # Pick a concept with a visual diagram
                vdc_concepts = output.concept_definitions[:3] if output.concept_definitions else []
                if not vdc_concepts:
                    st.warning("No concept definitions found. Process a lecture first.")
                else:
                    vdc_def = vdc_concepts[0]
                    vdc_visual = get_concept_visual(vdc_def.term, definition=vdc_def.definition, language=current_lang)
                    if vdc_visual:
                        st.markdown(f"**Target Concept:** `{vdc_def.term}`")
                        st.markdown(vdc_visual["svg_html"], unsafe_allow_html=True)
                        st.caption(vdc_visual["caption"])
                    else:
                        st.markdown(
                            f'<div style="background:rgba(124,77,255,0.1);border:1px dashed rgba(124,77,255,0.5);border-radius:12px;padding:1.2rem;text-align:center;font-size:1.5rem">🔬 {vdc_def.term}</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown("**Answer these 3 visual reasoning questions about the diagram:**")

                    vdc_q1 = st.radio(
                        "Q1: What does the diagram primarily illustrate?",
                        options=[
                            f"The internal architecture and data flow of {vdc_def.term}",
                            "A comparison between two unrelated technologies",
                            "A historical timeline of events",
                            "A geographic map of data centers",
                        ],
                        key="vdc_q1",
                        index=None,
                    )
                    vdc_q2 = st.radio(
                        "Q2: In the diagram, what best describes the direction of information flow?",
                        options=[
                            "Random and non-deterministic",
                            "Left-to-right or top-to-bottom, representing sequential processing",
                            "Only vertical, representing layer depth",
                            "Circular with no clear start or end",
                        ],
                        key="vdc_q2",
                        index=None,
                    )
                    vdc_q3 = st.radio(
                        f"Q3: The concept '{vdc_def.term}' is best described as:",
                        options=[
                            vdc_def.definition[:120] if vdc_def.definition else f"A core mechanism of {vdc_def.term}",
                            "A deprecated technique replaced by simpler rule-based systems",
                            "A hardware specification for GPU memory allocation",
                            "An unrelated networking protocol",
                        ],
                        key="vdc_q3",
                        index=None,
                    )

                    vdc_correct_answers = (
                        f"The internal architecture and data flow of {vdc_def.term}",
                        "Left-to-right or top-to-bottom, representing sequential processing",
                        vdc_def.definition[:120] if vdc_def.definition else f"A core mechanism of {vdc_def.term}",
                    )

                    if st.button("🔬 Submit Visual Reasoning Answers", use_container_width=True, key="btn_vdc_submit"):
                        if None in (vdc_q1, vdc_q2, vdc_q3):
                            st.warning("Please answer all 3 questions before submitting.")
                        else:
                            score = sum([
                                vdc_q1 == vdc_correct_answers[0],
                                vdc_q2 == vdc_correct_answers[1],
                                vdc_q3 == vdc_correct_answers[2],
                            ])
                            if score == 3:
                                st.session_state["cleared_visual_diagram_challenge"] = True
                                st.balloons()
                                st.success("🔬 VISUAL DIAGNOSIS COMPLETE! Perfect score — 3/3! Click **🎁 Claim XP** above to collect +160 XP!")
                            elif score == 2:
                                st.warning(f"⚠️ {score}/3 correct. You need all 3 right to pass. Review the diagram carefully and try again!")
                            else:
                                st.error(f"❌ {score}/3 correct. Study the diagram more carefully — look at the flow direction and labels!")

            elif active_cid == "concept_mindmap_challenge":
                st.markdown("#### 🧩 Concept Pipeline & Hierarchy Mapping Challenge!")
                st.info("Arrange the core concepts from this lecture into their correct computational/logical pipeline order.")

                mindmap_concepts = [c.term for c in all_concepts[:6]] if all_concepts else []
                if len(mindmap_concepts) < 3:
                    st.warning("Need at least 3 concepts from the lecture. Process a lecture first.")
                else:
                    # Show the concepts as draggable-style cards
                    st.markdown(
                        '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(124,77,255,0.3);border-radius:14px;padding:1rem;margin-bottom:1rem">'
                        '<div style="font-size:0.85rem;color:#9d94c8;margin-bottom:0.6rem;font-weight:600">📋 CONCEPTS FROM YOUR LECTURE:</div>'
                        + "".join(
                            f'<span style="display:inline-block;background:rgba(124,77,255,0.2);border:1px solid rgba(124,77,255,0.5);'
                            f'border-radius:20px;padding:0.2rem 0.7rem;margin:0.2rem;font-size:0.85rem;color:#e0d7ff">{c}</span>'
                            for c in mindmap_concepts
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Map each stage of the pipeline:**")
                    stage_opts = ["— Select a concept —"] + mindmap_concepts

                    mm_stage1 = st.selectbox(
                        "🟣 Stage 1 — Foundation / Input (what forms the basis of the topic):",
                        options=stage_opts,
                        key="mm_stage1",
                    )
                    mm_stage2 = st.selectbox(
                        "🔵 Stage 2 — Core Processing / Mechanism (the key transformation or algorithm):",
                        options=stage_opts,
                        key="mm_stage2",
                    )
                    mm_stage3 = st.selectbox(
                        "🟢 Stage 3 — Output / Generalization (the result, prediction, or outcome):",
                        options=stage_opts,
                        key="mm_stage3",
                    )

                    # Render live pipeline visualization
                    s1_lbl = mm_stage1 if mm_stage1 != "— Select a concept —" else "?"
                    s2_lbl = mm_stage2 if mm_stage2 != "— Select a concept —" else "?"
                    s3_lbl = mm_stage3 if mm_stage3 != "— Select a concept —" else "?"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;margin:1rem 0;flex-wrap:wrap">'
                        f'<div style="background:rgba(124,77,255,0.25);border:2px solid #7c4dff;border-radius:10px;padding:0.5rem 1rem;font-size:0.9rem;color:#e0d7ff;font-weight:600">{s1_lbl}</div>'
                        f'<div style="font-size:1.4rem;color:#7c4dff">➜</div>'
                        f'<div style="background:rgba(96,165,250,0.2);border:2px solid #60a5fa;border-radius:10px;padding:0.5rem 1rem;font-size:0.9rem;color:#dbeafe;font-weight:600">{s2_lbl}</div>'
                        f'<div style="font-size:1.4rem;color:#60a5fa">➜</div>'
                        f'<div style="background:rgba(52,211,153,0.2);border:2px solid #34d399;border-radius:10px;padding:0.5rem 1rem;font-size:0.9rem;color:#d1fae5;font-weight:600">{s3_lbl}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if st.button("🧩 Verify My Pipeline Mapping", use_container_width=True, key="btn_mm_verify"):
                        chosen = {mm_stage1, mm_stage2, mm_stage3}
                        if "— Select a concept —" in chosen:
                            st.warning("Please select a concept for each stage before verifying.")
                        elif len(chosen) < 3:
                            st.error("Each stage must map to a DIFFERENT concept. No duplicates allowed!")
                        else:
                            # Check: all 3 chosen concepts are from the lecture
                            valid_selections = all(s in mindmap_concepts for s in chosen)
                            if valid_selections:
                                st.session_state["cleared_concept_mindmap_challenge"] = True
                                st.balloons()
                                st.success(
                                    f"🧩 PIPELINE MASTERED! Your mapping: **{mm_stage1}** ➜ **{mm_stage2}** ➜ **{mm_stage3}** — "
                                    f"A logically valid conceptual pipeline! Click **🎁 Claim XP** above to collect +140 XP!"
                                )
                            else:
                                st.error("Please only use concepts from the lecture list above!")


# ── TAB: Badges & Progress ───────────────────────────────────────────────────
with tab_badges:
    st.markdown("### 🏆 User Progress, Leveling & Achievements" if current_lang == "English" else "### 🏆 進捗・レベル・実績バッジ")

    lvl_info = progress.get_level_info()
    st.markdown(
        f"""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(124,77,255,0.3);border-radius:16px;padding:1.4rem;margin-bottom:1.5rem">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <span style="font-size:1.8rem;font-weight:700;color:#f1f0fb">Level {lvl_info['level']}: {lvl_info['title']}</span>
      <div style="font-size:0.9rem;color:#9d94c8;margin-top:0.2rem">Total XP: <b style="color:#fbbf24">{lvl_info['xp']} XP</b> · Best Streak: <b style="color:#f87171">{progress.best_streak} 🔥</b></div>
    </div>
    <div style="text-align:right">
      <div style="font-size:1.4rem;font-weight:700;color:#34d399">{len(progress.unlocked_badge_ids)} / {len(get_all_badges())}</div>
      <div style="font-size:0.75rem;color:#9d94c8;text-transform:uppercase">Badges Unlocked</div>
    </div>
  </div>
  <div style="margin-top:0.8rem">
    <div style="background:rgba(255,255,255,0.08);border-radius:10px;height:10px;overflow:hidden">
      <div style="background:linear-gradient(90deg, #7c4dff, #34d399);width:{int(lvl_info['pct']*100)}%;height:100%"></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎖️ Badges Showcase" if current_lang == "English" else "#### 🎖️ 獲得バッジ一覧")

    all_b = get_all_badges()
    b_cols = st.columns(4)

    for idx, b in enumerate(all_b):
        is_unlocked = b.id in progress.unlocked_badge_ids
        cls = "badge-card badge-unlocked" if is_unlocked else "badge-card badge-locked"
        status_txt = f"<span style='color:#34d399;font-weight:600'>✔ Unlocked (+{b.xp_reward} XP)</span>" if is_unlocked else f"<span style='color:#94a3b8'>🔒 Locked (+{b.xp_reward} XP)</span>"

        with b_cols[idx % 4]:
            st.markdown(
                f"""<div class="{cls}">
  <div class="badge-icon">{b.icon}</div>
  <div class="badge-title">{b.get_title(current_lang)}</div>
  <div class="badge-desc">{b.get_description(current_lang)}</div>
  <div class="badge-xp">{status_txt}</div>
</div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    # ── XP Rewards Vault & Study Perks ─────────────────────────────────────────
    st.divider()
    st.markdown("### 🎁 XP Rewards Vault & Study Perks" if current_lang == "English" else "### 🎁 XP報酬ヴォールト＆学習特典")
    st.caption("Earn XP from challenges, quizzes, and badges to unlock powerful study perks and rewards.")

    all_rewards = get_all_rewards()
    rw_cols = st.columns(len(all_rewards))
    for ri, rw in enumerate(all_rewards):
        unlocked = progress.xp >= rw.required_xp
        claimed = rw.id in progress.claimed_rewards
        with rw_cols[ri]:
            card_bg = "rgba(52,211,153,0.08)" if unlocked else "rgba(255,255,255,0.03)"
            card_border = "rgba(52,211,153,0.5)" if unlocked else "rgba(255,255,255,0.1)"
            lock_icon = "🔓" if unlocked else "🔒"
            xp_color = "#34d399" if unlocked else "#94a3b8"
            st.markdown(
                f'<div style="background:{card_bg};border:1px solid {card_border};border-radius:14px;padding:1rem;text-align:center;min-height:160px">'
                f'<div style="font-size:2rem">{rw.icon}</div>'
                f'<div style="font-size:0.85rem;font-weight:700;color:#f1f0fb;margin:0.4rem 0">{rw.get_title(current_lang)}</div>'
                f'<div style="font-size:0.72rem;color:#9d94c8;margin-bottom:0.5rem;line-height:1.4">{rw.get_description(current_lang)}</div>'
                f'<div style="font-size:0.78rem;color:{xp_color};font-weight:600">{lock_icon} {rw.required_xp} XP required</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if unlocked and not claimed:
                if st.button(f"🎁 Activate", key=f"btn_rw_{rw.id}", use_container_width=True):
                    progress.claim_reward(rw.id)
                    _sync_user_progress(progress)
                    st.success(f"✨ {rw.get_title(current_lang)} activated!")
                    st.rerun()
            elif claimed:
                st.button("✅ Active", key=f"btn_rw_{rw.id}", use_container_width=True, disabled=True)
            else:
                xp_needed = rw.required_xp - progress.xp
                st.button(f"🔒 Need {xp_needed} more XP", key=f"btn_rw_{rw.id}", use_container_width=True, disabled=True)

    # ── Active Perk Controls ─────────────────────────────────────────────────────
    active_perks = progress.claimed_rewards
    if active_perks:
        st.markdown("#### ✨ Your Active Perks")

        # Theme switcher
        if "reward_themes" in active_perks:
            theme_choice = st.selectbox(
                "🎨 Active Study Theme:",
                options=["Default (Cosmic Violet)", "Cyberpunk Neon", "Emerald Scholar", "Solar Amber"],
                key="active_theme_selector",
            )
            theme_css_map = {
                "Cyberpunk Neon": "linear-gradient(135deg, #0d0d1a 0%, #120029 50%, #001f0d 100%)",
                "Emerald Scholar": "linear-gradient(135deg, #071a14 0%, #0d2b1f 50%, #0a2218 100%)",
                "Solar Amber": "linear-gradient(135deg, #1a1000 0%, #2a1a00 50%, #1a1200 100%)",
            }
            if theme_choice in theme_css_map:
                st.markdown(
                    f'<style>.stApp {{background: {theme_css_map[theme_choice]} !important;}}</style>',
                    unsafe_allow_html=True,
                )
                st.success(f"🎨 Theme applied: **{theme_choice}**")

        # Podcast script generator
        if "reward_audio_podcast" in active_perks:
            st.markdown("---")
            st.markdown("**🎙️ Lecture Podcast Script Generator**")
            if st.button("🎙️ Generate 2-Speaker Podcast Script for this Lecture", key="btn_gen_podcast", use_container_width=True):
                concepts_summary = ", ".join([c.term for c in all_concepts[:8]]) if all_concepts else "key lecture concepts"
                podcast_script = f"""🎙️ **Lecture Podcast: {concepts_summary}**

---
**[HOST]:** Welcome back to Study Deep! Today we're diving into: {concepts_summary}.

**[EXPERT]:** That's right. Let's start with the fundamentals — {all_concepts[0].term if all_concepts else 'our first concept'}.
{f'This is essentially: {all_concepts[0].definition[:200]}' if all_concepts and all_concepts[0].definition else ''}

**[HOST]:** So how does this connect to {all_concepts[1].term if len(all_concepts) > 1 else 'the bigger picture'}?

**[EXPERT]:** Great question. Think of it as a pipeline — {' feeds into '.join([c.term for c in all_concepts[:4]]) if len(all_concepts) >= 4 else 'each concept building on the last'}.
The key insight is that mastering these fundamentals unlocks everything else.

**[HOST]:** And for students preparing for exams?

**[EXPERT]:** Focus on: {concepts_summary}. Understand the *why* behind each mechanism, not just the *what*.

**[HOST]:** Amazing insights! Thanks for listening to Study Deep. Keep learning, keep growing! 🚀
---"""
                st.markdown(podcast_script)
                st.download_button(
                    "⬇️ Download Podcast Script (.txt)",
                    data=podcast_script,
                    file_name="lecture_podcast_script.txt",
                    mime="text/plain",
                    key="btn_download_podcast",
                )

        # Mastery Certificate
        if "reward_mastery_certificate" in active_perks:
            st.markdown("---")
            st.markdown("**📜 Verified Certificate of Lecture Mastery**")
            username = st.session_state.get("username", "Learner")
            concepts_count = len(progress.completed_concepts)
            badges_count = len(progress.unlocked_badge_ids)
            challenges_count = len(progress.completed_challenges)
            cert_html = f"""
<div style="background:linear-gradient(135deg,#1a1040,#24243e);border:3px solid #fbbf24;border-radius:20px;padding:2.5rem;text-align:center;max-width:700px;margin:auto">
  <div style="font-size:2.5rem">📜</div>
  <div style="font-size:0.8rem;color:#fbbf24;letter-spacing:0.2em;text-transform:uppercase;margin:0.3rem 0">Official Certificate of Mastery</div>
  <div style="font-size:1.8rem;font-weight:700;color:#f1f0fb;margin:0.8rem 0">LectureNotes AI</div>
  <div style="font-size:0.9rem;color:#9d94c8;margin-bottom:0.8rem">This certifies that</div>
  <div style="font-size:1.5rem;font-weight:700;color:#a78bfa;margin:0.5rem 0">{username}</div>
  <div style="font-size:0.85rem;color:#d0daf8;margin:0.8rem 0;line-height:1.8">
    has demonstrated mastery by earning <b style="color:#fbbf24">{progress.xp} XP</b>,<br/>
    completing <b style="color:#34d399">{challenges_count} challenges</b>,
    mastering <b style="color:#60a5fa">{concepts_count} concepts</b>,<br/>
    and unlocking <b style="color:#f472b6">{badges_count} achievement badges</b>.
  </div>
  <div style="font-size:0.75rem;color:#64748b;margin-top:1rem">Issued: {datetime.date.today().strftime('%B %d, %Y')} · Verified by LectureNotes AI</div>
</div>"""
            st.markdown(cert_html, unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Certificate (HTML)",
                data=cert_html,
                file_name="mastery_certificate.html",
                mime="text/html",
                key="btn_download_cert",
            )


# ── TAB: Study Analytics Dashboard ────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Study Analytics Dashboard")
    st.caption("Track your learning performance, consistency, and growth over time.")

    # ── Derived stats ──
    total_answered = len(progress.quiz_answers)
    total_correct = sum(1 for a in progress.quiz_answers if a.get("is_correct"))
    accuracy_pct = int((total_correct / total_answered * 100)) if total_answered > 0 else 0
    concepts_done = len(progress.completed_concepts)
    exams_taken = len(progress.exam_history)
    best_exam_score = max((e.get("score_pct", 0) for e in progress.exam_history), default=0)

    # ── KPI tiles row ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_style = "background:rgba(255,255,255,0.04);border:1px solid rgba(124,77,255,0.3);border-radius:14px;padding:0.9rem 0.5rem;text-align:center"
    kpi_items = [
        (k1, progress.xp, "Total XP", "#fbbf24"),
        (k2, f"{accuracy_pct}%", "Quiz Accuracy", "#34d399"),
        (k3, total_answered, "Q's Answered", "#60a5fa"),
        (k4, concepts_done, "Concepts Mastered", "#f472b6"),
        (k5, exams_taken, "Exams Taken", "#a78bfa"),
        (k6, f"{best_exam_score}%", "Best Exam Score", "#f97316"),
    ]
    for col, val, label, clr in kpi_items:
        with col:
            st.markdown(
                f'<div style="{kpi_style}">'
                f'<div style="font-size:1.65rem;font-weight:800;color:{clr}">{val}</div>'
                f'<div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase;margin-top:0.2rem">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Row 1: Accuracy by Difficulty + XP Level progress ──
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 🎯 Accuracy by Difficulty")
        diff_stats = {"Easy": {"correct": 0, "total": 0}, "Medium": {"correct": 0, "total": 0}, "Hard": {"correct": 0, "total": 0}}
        for a in progress.quiz_answers:
            d = a.get("difficulty", "Medium")
            if d in diff_stats:
                diff_stats[d]["total"] += 1
                if a.get("is_correct"):
                    diff_stats[d]["correct"] += 1

        if total_answered > 0:
            for diff_name, stats in diff_stats.items():
                t = stats["total"]
                c = stats["correct"]
                pct = int(c / t * 100) if t > 0 else 0
                bar_color = {"Easy": "#34d399", "Medium": "#fbbf24", "Hard": "#f87171"}.get(diff_name, "#7c4dff")
                bar_bg = {"Easy": "rgba(52,211,153,0.12)", "Medium": "rgba(251,191,36,0.12)", "Hard": "rgba(248,113,113,0.12)"}.get(diff_name, "rgba(124,77,255,0.12)")
                icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(diff_name, "⚪")
                st.markdown(
                    f'<div style="background:{bar_bg};border:1px solid {bar_color}33;border-radius:10px;padding:0.65rem 0.9rem;margin-bottom:0.6rem">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:0.35rem">'
                    f'<span style="font-size:0.88rem;color:#d0daf8;font-weight:600">{icon} {diff_name}</span>'
                    f'<span style="font-size:0.88rem;color:{bar_color};font-weight:700">{c}/{t} &nbsp;·&nbsp; {pct}%</span></div>'
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:8px;height:10px;overflow:hidden">'
                    f'<div style="background:linear-gradient(90deg,{bar_color}aa,{bar_color});width:{pct}%;height:100%;border-radius:8px;transition:width 0.5s"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            # Streamlit native chart (accuracy %)
            try:
                import pandas as pd
                chart_data = {
                    "Difficulty": list(diff_stats.keys()),
                    "Accuracy %": [int(s["correct"] / s["total"] * 100) if s["total"] > 0 else 0 for s in diff_stats.values()],
                    "Attempted": [s["total"] for s in diff_stats.values()],
                }
                df_chart = pd.DataFrame(chart_data)
                st.bar_chart(df_chart.set_index("Difficulty")["Accuracy %"], color="#7c4dff", height=160)
            except Exception:
                pass
        else:
            st.info("📋 Answer some quiz questions to see your accuracy breakdown!")

    with col_chart2:
        st.markdown("#### 📈 XP Progress to Next Level")
        lvl_info = progress.get_level_info()
        current_lvl = lvl_info["level"]
        lvl_pct = int(lvl_info["pct"] * 100)
        level_thresholds = [0, 100, 300, 600, 1000, 1500]
        xp_for_next = level_thresholds[current_lvl] if current_lvl < len(level_thresholds) else progress.xp

        # Level milestone bars
        for lv in range(1, min(current_lvl + 3, len(level_thresholds))):
            lv_pct = 100 if lv < current_lvl else (lvl_pct if lv == current_lvl else 0)
            lv_color = "#7c4dff" if lv < current_lvl else ("#fbbf24" if lv == current_lvl else "rgba(255,255,255,0.1)")
            lv_label = f"Lv {lv}" + (" ✓" if lv < current_lvl else (" ← You are here" if lv == current_lvl else ""))
            st.markdown(
                f'<div style="margin-bottom:0.55rem"><div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">'
                f'<span style="font-size:0.82rem;color:#9d94c8">{lv_label}</span>'
                f'<span style="font-size:0.82rem;color:{lv_color};font-weight:600">{lv_pct}%</span></div>'
                f'<div style="background:rgba(255,255,255,0.07);border-radius:6px;height:9px;overflow:hidden">'
                f'<div style="background:{lv_color};width:{lv_pct}%;height:100%;border-radius:6px"></div></div></div>',
                unsafe_allow_html=True,
            )
        xp_needed = max(0, xp_for_next - progress.xp)
        st.markdown(
            f'<div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.3);border-radius:10px;padding:0.6rem 0.9rem;margin-top:0.5rem">'
            f'<span style="font-size:0.82rem;color:#9d94c8">XP to Level {current_lvl + 1}: </span>'
            f'<b style="color:#fbbf24">{xp_needed} XP needed</b>'
            f'<div style="font-size:0.78rem;color:#a78bfa;margin-top:0.15rem">Current total: {progress.xp} XP · Level {current_lvl}: {lvl_info["title"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Row 2: Exam History trend + Activity Breakdown ──
    col_exam_hist, col_activity = st.columns([3, 2])

    with col_exam_hist:
        st.markdown("#### 🎯 Exam Score History")
        if progress.exam_history:
            try:
                import pandas as pd
                exam_df_data = {
                    "Attempt": [f"#{i+1} {e.get('ts','')[-5:]}" for i, e in enumerate(progress.exam_history[-10:])],
                    "Score %": [e.get("score_pct", 0) for e in progress.exam_history[-10:]],
                }
                df_exam = pd.DataFrame(exam_df_data).set_index("Attempt")
                st.line_chart(df_exam["Score %"], color="#a78bfa", height=180)
            except Exception:
                pass

            # History table
            st.markdown('<div style="font-size:0.8rem;color:#9d94c8;font-weight:600;margin:0.4rem 0">Recent Exams (latest first):</div>', unsafe_allow_html=True)
            for entry in reversed(progress.exam_history[-6:]):
                sc = entry.get("score_pct", 0)
                s_clr = "#34d399" if sc >= 80 else ("#fbbf24" if sc >= 50 else "#f87171")
                m_e, s_e = divmod(entry.get("elapsed_secs", 0), 60)
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
                    f'border-radius:8px;padding:0.45rem 0.8rem;margin-bottom:0.35rem;display:flex;justify-content:space-between;align-items:center">'
                    f'<div><span style="font-size:0.82rem;color:#d0daf8;font-weight:600">{entry.get("lecture_title","—")[:30]}</span>'
                    f'<span style="font-size:0.7rem;color:#6b6490;margin-left:0.5rem">{entry.get("ts","")}</span></div>'
                    f'<div style="display:flex;gap:0.7rem;align-items:center">'
                    f'<b style="color:{s_clr};font-size:0.95rem">{sc}%</b>'
                    f'<span style="font-size:0.75rem;color:#a78bfa">{entry.get("difficulty","—")}</span>'
                    f'<span style="font-size:0.75rem;color:#94a3b8">{m_e}m {s_e}s</span>'
                    f'<span style="font-size:0.75rem;color:#fbbf24">+{entry.get("xp_earned",0)} XP</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            if exams_taken > 0:
                avg_exam = int(sum(e.get("score_pct",0) for e in progress.exam_history) / exams_taken)
                pass_count = sum(1 for e in progress.exam_history if e.get("score_pct",0) >= 80)
                st.markdown(
                    f'<div style="background:rgba(124,77,255,0.1);border:1px solid rgba(124,77,255,0.3);border-radius:10px;padding:0.6rem 1rem;margin-top:0.5rem;display:flex;gap:1.5rem">'
                    f'<div><div style="font-size:1.1rem;font-weight:700;color:#a78bfa">{avg_exam}%</div><div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase">Avg Score</div></div>'
                    f'<div><div style="font-size:1.1rem;font-weight:700;color:#34d399">{pass_count}/{exams_taken}</div><div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase">Passed</div></div>'
                    f'<div><div style="font-size:1.1rem;font-weight:700;color:#fbbf24">{best_exam_score}%</div><div style="font-size:0.68rem;color:#9d94c8;text-transform:uppercase">Best Score</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.03);border:1px dashed rgba(124,77,255,0.3);border-radius:14px;padding:2rem;text-align:center">'
                '<div style="font-size:2rem">🎯</div>'
                '<div style="font-size:0.9rem;color:#9d94c8;margin-top:0.5rem">No exams taken yet.</div>'
                '<div style="font-size:0.8rem;color:#6b6490">Head to the <b>🎯 Exam Prep</b> tab to take your first exam!</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    with col_activity:
        st.markdown("#### 🧩 Learning Activity Breakdown")
        activities = [
            ("🃏", "Flashcards Reviewed", len(progress.reviewed_flashcards), "#60a5fa", 10),
            ("📺", "Videos Explored", len(progress.explored_videos), "#f472b6", 5),
            ("🎨", "Diagrams Inspected", len(progress.visuals_viewed), "#fbbf24", 5),
            ("🌊", "Deep Dives Read", len(progress.deep_dives_viewed), "#34d399", 5),
            ("💬", "AI Tutor Queries", progress.chat_queries_count, "#a78bfa", 10),
            ("🏆", "Challenges Done", len(progress.completed_challenges), "#f97316", 8),
        ]
        total_activities = sum(count for _, _, count, _, _ in activities)
        for icon, label, count, color, max_target in activities:
            bar_w = min(100, int(count / max_target * 100))
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem">'
                f'<span style="font-size:1.1rem;width:1.5rem">{icon}</span>'
                f'<div style="flex:1">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:0.15rem">'
                f'<span style="font-size:0.78rem;color:#d0daf8">{label}</span>'
                f'<span style="font-size:0.8rem;color:{color};font-weight:700">{count}</span></div>'
                f'<div style="background:rgba(255,255,255,0.07);border-radius:5px;height:7px;overflow:hidden">'
                f'<div style="background:{color};width:{bar_w}%;height:100%;border-radius:5px"></div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.6rem 0.9rem;margin-top:0.4rem;text-align:center">'
            f'<div style="font-size:0.72rem;color:#9d94c8;text-transform:uppercase">Total Learning Actions</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#a78bfa">{total_activities}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Row 3: Streak Calendar + Concept Mastery ──
    col_streak, col_mastery = st.columns(2)

    with col_streak:
        st.markdown("#### 🔥 Study Streak Overview")
        streak_cols = st.columns(7)
        day_labels_an = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        today_weekday_an = datetime.date.today().weekday()
        for di in range(7):
            day_active = di <= today_weekday_an and progress.daily_streak > 0
            is_today = di == today_weekday_an
            box_bg = "linear-gradient(135deg,rgba(124,77,255,0.8),rgba(52,211,153,0.6))" if day_active else "rgba(255,255,255,0.05)"
            border = "rgba(124,77,255,0.7)" if is_today else ("rgba(52,211,153,0.4)" if day_active else "rgba(255,255,255,0.1)")
            dot = "🔥" if day_active else "○"
            with streak_cols[di]:
                st.markdown(
                    f'<div style="background:{box_bg};border:1px solid {border};border-radius:10px;padding:0.4rem 0.2rem;text-align:center">'
                    f'<div style="font-size:0.9rem">{dot}</div>'
                    f'<div style="font-size:0.65rem;color:#d0daf8;font-weight:{"700" if is_today else "400"}">{day_labels_an[di]}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.35);border-radius:10px;padding:0.6rem;text-align:center">'
                f'<div style="font-size:1.5rem;font-weight:700;color:#f87171">Day {progress.daily_streak} 🔥</div>'
                f'<div style="font-size:0.7rem;color:#9d94c8;text-transform:uppercase">Current Streak</div></div>',
                unsafe_allow_html=True,
            )
        with c_s2:
            st.markdown(
                f'<div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.35);border-radius:10px;padding:0.6rem;text-align:center">'
                f'<div style="font-size:1.5rem;font-weight:700;color:#fbbf24">Day {progress.best_streak} 🏆</div>'
                f'<div style="font-size:0.7rem;color:#9d94c8;text-transform:uppercase">Best Streak</div></div>',
                unsafe_allow_html=True,
            )

    with col_mastery:
        st.markdown("#### 🎓 Concept Mastery Status")
        total_lec_concepts = len(all_concepts)
        mastered_set = {c.term.lower() for c in all_concepts if c.term.lower() in progress.completed_concepts}
        mastered_cnt = len(mastered_set)
        remaining_cnt = total_lec_concepts - mastered_cnt
        mastery_pct_val = int(mastered_cnt / total_lec_concepts * 100) if total_lec_concepts > 0 else 0

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(124,77,255,0.25);border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:0.8rem">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">'
            f'<span style="font-size:0.85rem;color:#a78bfa;font-weight:600">This Lecture</span>'
            f'<span style="font-size:0.9rem;color:#34d399;font-weight:700">{mastered_cnt}/{total_lec_concepts} mastered ({mastery_pct_val}%)</span></div>'
            f'<div style="background:rgba(255,255,255,0.08);border-radius:8px;height:10px;overflow:hidden">'
            f'<div style="background:linear-gradient(90deg,#7c4dff,#34d399);width:{mastery_pct_val}%;height:100%"></div></div></div>',
            unsafe_allow_html=True,
        )

        if mastered_set:
            st.markdown('<div style="font-size:0.78rem;color:#34d399;font-weight:600;margin-bottom:0.3rem">✔ Mastered:</div>', unsafe_allow_html=True)
            chips_html = "".join(
                f'<span style="display:inline-block;background:rgba(52,211,153,0.15);border:1px solid rgba(52,211,153,0.4);'
                f'border-radius:20px;padding:2px 10px;font-size:0.75rem;color:#a7f3d0;margin:2px">{c.term}</span>'
                for c in all_concepts if c.term.lower() in progress.completed_concepts
            )
            st.markdown(chips_html, unsafe_allow_html=True)

        remaining_concepts = [c for c in all_concepts if c.term.lower() not in progress.completed_concepts]
        if remaining_concepts:
            st.markdown('<div style="font-size:0.78rem;color:#9d94c8;font-weight:600;margin-top:0.5rem;margin-bottom:0.3rem">📌 Still to master:</div>', unsafe_allow_html=True)
            rem_chips = "".join(
                f'<span style="display:inline-block;background:rgba(124,77,255,0.12);border:1px solid rgba(124,77,255,0.3);'
                f'border-radius:20px;padding:2px 10px;font-size:0.75rem;color:#c4b8fa;margin:2px">{c.term}</span>'
                for c in remaining_concepts[:8]
            )
            if len(remaining_concepts) > 8:
                rem_chips += f'<span style="font-size:0.72rem;color:#6b6490;margin-left:0.3rem">+{len(remaining_concepts)-8} more</span>'
            st.markdown(rem_chips, unsafe_allow_html=True)

    st.divider()

    # ── Weak Area Detector & Intelligent Study Plan ──
    st.markdown("#### 🔍 Weak Area Detector & AI Study Plan")
    st.caption("Automated diagnostic of learning gaps across quizzes, practice exams, flashcards, and concept retention.")

    # Diagnostic signals
    wrong_answers_an = [a for a in progress.quiz_answers if not a.get("is_correct")]
    hard_w = sum(1 for a in wrong_answers_an if a.get("difficulty") == "Hard")
    med_w = sum(1 for a in wrong_answers_an if a.get("difficulty") == "Medium")
    easy_w = sum(1 for a in wrong_answers_an if a.get("difficulty") == "Easy")

    exams_below_target = [e for e in progress.exam_history if e.get("score_pct", 0) < 80]
    total_exams = len(progress.exam_history)
    avg_exam_score = int(sum(e.get("score_pct", 0) for e in progress.exam_history) / total_exams) if total_exams > 0 else 0

    unmastered_concepts = [c for c in all_concepts if c.term.lower() not in progress.completed_concepts]
    unmastered_count = len(unmastered_concepts)

    flashcards_reviewed_cnt = len(progress.reviewed_flashcards)
    visuals_viewed_cnt = len(progress.visuals_viewed)
    tutor_queries_cnt = progress.chat_queries_count

    # Build diagnostic findings
    findings = []

    if hard_w > 0:
        findings.append({
            "severity": "critical",
            "icon": "🧗",
            "title": f"{hard_w} Hard Question{'s' if hard_w > 1 else ''} Missed",
            "desc": "Struggles with advanced edge cases, derivations, or complex conceptual combinations.",
            "action": "Play the 🎮 Hard Gauntlet Challenge and review deep dives for unmastered terms.",
        })

    if med_w > 0:
        findings.append({
            "severity": "warning",
            "icon": "📖",
            "title": f"{med_w} Medium Question{'s' if med_w > 1 else ''} Missed",
            "desc": "Incomplete recall of standard conceptual definitions and intermediate relationships.",
            "action": "Flip all flashcards and ask the 🤖 AI Tutor to explain differences between similar concepts.",
        })

    if easy_w > 0:
        findings.append({
            "severity": "warning",
            "icon": "⚠️",
            "title": f"{easy_w} Foundational Question{'s' if easy_w > 1 else ''} Missed",
            "desc": "Core terminology or elementary formulas need immediate review.",
            "action": "Re-read the primary definitions in the 🔑 Key Concepts tab before retaking quizzes.",
        })

    if total_exams > 0 and avg_exam_score < 80:
        findings.append({
            "severity": "warning",
            "icon": "🎯",
            "title": f"Exam Pass Rate Below Target (Avg: {avg_exam_score}%)",
            "desc": f"{len(exams_below_target)} out of {total_exams} exam attempts scored below the 80% passing standard.",
            "action": "Take an untimed 10-question practice exam to focus on accuracy over speed.",
        })

    if unmastered_count > 0:
        findings.append({
            "severity": "info",
            "icon": "🔑",
            "title": f"{unmastered_count} Concept{'s' if unmastered_count > 1 else ''} Not Yet Mastered",
            "desc": "Completing all concepts unlocks the Domain Master achievement (+250 XP).",
            "action": "Check off mastered terms in the 🔑 Concepts tab as you understand them.",
        })

    if flashcards_reviewed_cnt < 3:
        findings.append({
            "severity": "info",
            "icon": "🃏",
            "title": "Low Flashcard Active Recall",
            "desc": f"Only {flashcards_reviewed_cnt} flashcard(s) flipped so far. Active recall strengthens long-term memory.",
            "action": "Review at least 5 flashcards to unlock the Memory Maestro badge (+80 XP).",
        })

    if visuals_viewed_cnt < 2:
        findings.append({
            "severity": "info",
            "icon": "🎨",
            "title": "Visual Diagrams Untapped",
            "desc": f"Only {visuals_viewed_cnt} concept diagram(s) inspected. Diagrams build strong mental models.",
            "action": "Inspect 3+ concept diagrams to unlock the Visual Architect badge (+70 XP).",
        })

    if tutor_queries_cnt < 2:
        findings.append({
            "severity": "info",
            "icon": "🤖",
            "title": "AI Tutor Available",
            "desc": f"Only {tutor_queries_cnt} question(s) asked to the AI Tutor. Dialogue clarifies tricky questions.",
            "action": "Ask the AI Tutor 3 questions to unlock the Socratic Inquirer badge (+80 XP).",
        })

    # Render findings and recommendations
    col_findings, col_actions = st.columns([3, 2])

    with col_findings:
        st.markdown("##### 🚨 Diagnostic Findings")
        if not findings or (not wrong_answers_an and not exams_below_target and unmastered_count == 0):
            st.markdown(
                f'<div style="background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.4);border-radius:14px;padding:1.5rem;text-align:center">'
                f'<div style="font-size:2.5rem">🏆</div>'
                f'<div style="font-size:1.15rem;font-weight:700;color:#34d399;margin-top:0.4rem">All Systems Green!</div>'
                f'<div style="font-size:0.84rem;color:#d0daf8;margin-top:0.3rem">'
                f'Outstanding job! You have zero detected weak spots, solid quiz accuracy, and high mastery. '
                f'Try the <b>🎮 Hard Gauntlet Challenge</b> or export your verified study notes.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            sev_colors = {
                "critical": ("#f87171", "rgba(248,113,113,0.12)", "rgba(248,113,113,0.4)"),
                "warning": ("#fbbf24", "rgba(251,191,36,0.12)", "rgba(251,191,36,0.4)"),
                "info": ("#60a5fa", "rgba(96,165,250,0.1)", "rgba(96,165,250,0.3)"),
            }
            for f in findings[:5]:
                clr, bg, bdr = sev_colors.get(f["severity"], ("#a78bfa", "rgba(124,77,255,0.1)", "rgba(124,77,255,0.3)"))
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {bdr};border-left:4px solid {clr};'
                    f'border-radius:0 12px 12px 0;padding:0.75rem 1rem;margin-bottom:0.6rem">'
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem">'
                    f'<span style="font-size:1.1rem">{f["icon"]}</span>'
                    f'<span style="color:{clr};font-weight:700;font-size:0.88rem">{f["title"]}</span></div>'
                    f'<div style="font-size:0.78rem;color:#d0daf8;line-height:1.4">{f["desc"]}</div>'
                    f'<div style="font-size:0.75rem;color:{clr};margin-top:0.35rem;font-weight:600">💡 Recommended: {f["action"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with col_actions:
        st.markdown("##### 🎯 Recommended Study Sequence")
        steps = [
            ("1", "Review Weak Concepts", "Read definitions for missed terms in 🔑 Concepts", "#f87171" if (hard_w or med_w or easy_w) else "#34d399"),
            ("2", "Active Recall Session", "Flip at least 5 flashcards to cement retention", "#fbbf24" if flashcards_reviewed_cnt < 5 else "#34d399"),
            ("3", "AI Dialogue", "Clarify tricky questions with the 🤖 AI Tutor", "#60a5fa" if tutor_queries_cnt < 3 else "#34d399"),
            ("4", "Practice Exam", "Test exam readiness with a 10-minute timed run", "#a78bfa" if total_exams == 0 or avg_exam_score < 80 else "#34d399"),
            ("5", "Arena Gauntlet", "Compete in Survival or Hard Gauntlet mode", "#f97316"),
        ]
        for step_num, step_title, step_desc, step_color in steps:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
                f'border-radius:10px;padding:0.65rem 0.9rem;margin-bottom:0.5rem;display:flex;gap:0.75rem;align-items:center">'
                f'<div style="background:{step_color}22;border:1px solid {step_color};color:{step_color};'
                f'width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;'
                f'font-size:0.75rem;font-weight:700;flex-shrink:0">{step_num}</div>'
                f'<div><div style="font-size:0.82rem;color:#f1f0fb;font-weight:600">{step_title}</div>'
                f'<div style="font-size:0.73rem;color:#9d94c8;line-height:1.3">{step_desc}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── TAB: Verification ─────────────────────────────────────────────────────────
with tab_verify:
    st.markdown("### Factuality Verification Report" if current_lang == "English" else "### 事実性検証レポート")
    if report is None:
        st.info(
            "Factuality verification was not enabled. "
            "Turn on **Run factuality verification** in the sidebar and regenerate."
        )
    else:
        rate_pct = int(report.factuality_rate * 100)
        color = _pct_color(report.factuality_rate)
        st.markdown(
            f"""
<div style="text-align:center;padding:1.5rem;background:rgba(255,255,255,0.03);
            border-radius:16px;border:1px solid rgba(124,77,255,0.2);margin-bottom:1.5rem">
  <div style="font-size:3.5rem;font-weight:700;color:{color}">{rate_pct}%</div>
  <div style="font-size:0.9rem;color:#9d94c8;margin-top:0.3rem">
    Overall Factuality Rate ({report.entailed_count}/{report.total_claims} claims entailed)
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.progress(float(report.factuality_rate))

        st.divider()

        types = sorted({r.item_type for r in report.results})
        selected_type = st.radio(
            "Filter by type",
            options=["all"] + types,
            horizontal=True,
            label_visibility="visible",
        )

        filtered = (
            report.results
            if selected_type == "all"
            else [r for r in report.results if r.item_type == selected_type]
        )

        st.caption(f"Showing {len(filtered)} of {len(report.results)} verification results")

        for res in filtered:
            badge_html = _verification_badge(res.label)
            st.markdown(
                f"""<div class="card" style="padding:1rem 1.3rem">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
    <span style="font-size:0.78rem;color:#7c4dff;text-transform:uppercase;letter-spacing:0.05em">{res.item_type} · {res.item_id}</span>
    <span>{badge_html} &nbsp;<span style="font-size:0.8rem;color:#9d94c8">{res.entailment_score:.3f}</span></span>
  </div>
  <div style="font-size:0.88rem;color:#e0daf8;margin-bottom:0.4rem"><b>Claim:</b> {res.claim}</div>
  <div class="source-quote">📎 {res.source_span}</div>
</div>""",
                unsafe_allow_html=True,
            )


# ── Export Study Materials ────────────────────────────────────────────────────
st.divider()
st.markdown("### 💾 Export Study Materials" if current_lang == "English" else "### 💾 学習教材のエクスポート")
st.caption("Download your generated notes, flashcards, and quiz for offline studying or Anki.")

c_exp1, c_exp2, c_exp3 = st.columns(3)
stem_name = Path(file_name).stem if file_name else "lecture_materials"

with c_exp1:
    md_content = _generate_markdown_study_guide(output, report, language=current_lang)
    st.download_button(
        label="📥 Download Study Guide (.md)",
        data=md_content,
        file_name=f"{stem_name}_{current_lang}_Study_Guide.md",
        mime="text/markdown",
        use_container_width=True,
    )

with c_exp2:
    csv_content = _generate_flashcards_csv(output.flashcards)
    st.download_button(
        label="🃏 Export Anki Flashcards (.csv)",
        data=csv_content,
        file_name=f"{stem_name}_flashcards.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c_exp3:
    import json
    json_content = json.dumps(output.model_dump(), indent=2, ensure_ascii=False)
    st.download_button(
        label="📋 Export Raw JSON (.json)",
        data=json_content,
        file_name=f"{stem_name}_{current_lang}_materials.json",
        mime="application/json",
        use_container_width=True,
    )
