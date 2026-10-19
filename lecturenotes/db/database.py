"""SQLite database management for user authentication and persistent progress tracking."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Any

DB_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_DB_PATH = DB_DIR / "lecturenotes.db"


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create and return a SQLite connection with row_factory set to Row."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Initialize SQLite database tables and default demo user."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            # Users table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # User progress table (persists XP, streaks, badges, completed concepts)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_progress (
                    user_id INTEGER PRIMARY KEY,
                    xp INTEGER DEFAULT 0,
                    streak INTEGER DEFAULT 0,
                    best_streak INTEGER DEFAULT 0,
                    progress_json TEXT DEFAULT '{}',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                );
                """
            )

            # User lectures table (history of processed lectures)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_lectures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    lecture_title TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    sections_count INTEGER DEFAULT 0,
                    concepts_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                );
                """
            )

            # Create default demo user if not exists
            cur = conn.execute("SELECT id FROM users WHERE username = ?", ("demo",))
            if not cur.fetchone():
                salt = secrets.token_hex(16)
                p_hash = hashlib.sha256((salt + "demo123").encode("utf-8")).hexdigest()
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                    ("demo", p_hash, salt),
                )
                demo_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO user_progress (user_id, xp, streak, best_streak, progress_json) VALUES (?, ?, ?, ?, ?)",
                    (demo_id, 0, 1, 1, json.dumps({"studied_languages": ["English"], "xp": 0, "daily_streak": 1, "best_daily_streak": 1})),
                )
    finally:
        conn.close()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash password with sha256 and salt."""
    if not salt:
        salt = secrets.token_hex(16)
    p_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return p_hash, salt


def register_user(
    username: str,
    password: str,
    db_path: Path | str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Register a new user account and initialize their persistent progress."""
    clean_user = username.strip()
    if not clean_user or len(clean_user) < 3:
        return False, "Username must be at least 3 characters long.", None
    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long.", None

    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute("SELECT id FROM users WHERE username = ?", (clean_user,))
            if cur.fetchone():
                return False, f"Username '{clean_user}' is already registered.", None

            p_hash, salt = hash_password(password)
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (clean_user, p_hash, salt),
            )
            user_id = cur.lastrowid

            # Initialize user progress: clean Level 1 (0 XP)
            initial_progress = {
                "completed_concepts": [],
                "quiz_answers": [],
                "reviewed_flashcards": [],
                "explored_videos": [],
                "xp": 0,  # Level 1 (0 XP)
                "streak": 1,
                "best_streak": 1,
                "daily_streak": 1,
                "best_daily_streak": 1,
                "quiz_streak": 0,
                "best_quiz_streak": 0,
                "unlocked_badge_ids": [],
                "completed_challenges": [],
                "studied_languages": ["English"],
            }
            conn.execute(
                "INSERT INTO user_progress (user_id, xp, streak, best_streak, progress_json) VALUES (?, ?, ?, ?, ?)",
                (user_id, 0, 1, 1, json.dumps(initial_progress)),
            )

            return True, "Registration successful!", {"id": user_id, "username": clean_user}
    except Exception as exc:
        return False, f"Registration error: {exc}", None
    finally:
        conn.close()


def authenticate_user(
    username: str,
    password: str,
    db_path: Path | str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Authenticate an existing user with username and password."""
    clean_user = username.strip()
    if not clean_user or not password:
        return False, "Please provide both username and password.", None

    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (clean_user,),
        )
        row = cur.fetchone()
        if not row:
            return False, "Invalid username or password.", None

        expected_hash = hashlib.sha256((row["salt"] + password).encode("utf-8")).hexdigest()
        if expected_hash != row["password_hash"]:
            return False, "Invalid username or password.", None

        return True, "Login successful!", {"id": row["id"], "username": row["username"]}
    finally:
        conn.close()


def load_user_progress_db(
    user_id: int,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load persistent user progress from SQLite."""
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT xp, streak, best_streak, progress_json FROM user_progress WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return {
                "completed_concepts": [],
                "quiz_answers": [],
                "reviewed_flashcards": [],
                "explored_videos": [],
                "xp": 0,
                "streak": 0,
                "best_streak": 0,
                "unlocked_badge_ids": [],
                "completed_challenges": [],
                "studied_languages": ["English"],
            }

        data = json.loads(row["progress_json"]) if row["progress_json"] else {}
        data["xp"] = row["xp"]
        data["streak"] = row["streak"]
        data["best_streak"] = row["best_streak"]
        return data
    finally:
        conn.close()


def save_user_progress_db(
    user_id: int,
    progress_dict: dict[str, Any],
    db_path: Path | str | None = None,
) -> None:
    """Save or update user progress in SQLite."""
    conn = get_db_connection(db_path)
    try:
        xp = int(progress_dict.get("xp", 0))
        streak = int(progress_dict.get("streak", 0))
        best_streak = int(progress_dict.get("best_streak", streak))
        raw_json = json.dumps(progress_dict)

        with conn:
            conn.execute(
                """
                INSERT INTO user_progress (user_id, xp, streak, best_streak, progress_json, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    xp = excluded.xp,
                    streak = excluded.streak,
                    best_streak = excluded.best_streak,
                    progress_json = excluded.progress_json,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (user_id, xp, streak, best_streak, raw_json),
            )
    finally:
        conn.close()


def record_user_lecture_db(
    user_id: int,
    lecture_title: str,
    file_name: str,
    sections_count: int,
    concepts_count: int,
    db_path: Path | str | None = None,
) -> None:
    """Save a processed lecture to user's history and award processing XP."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO user_lectures (user_id, lecture_title, file_name, sections_count, concepts_count)
                VALUES (?, ?, ?, ?, ?);
                """,
                (user_id, lecture_title, file_name, sections_count, concepts_count),
            )
    finally:
        conn.close()


def get_user_lectures_db(
    user_id: int,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve history of uploaded and processed lectures for a user."""
    conn = get_db_connection(db_path)
    try:
        cur = conn.execute(
            """
            SELECT id, lecture_title, file_name, sections_count, concepts_count, created_at
            FROM user_lectures
            WHERE user_id = ?
            ORDER BY id DESC, created_at DESC
            LIMIT 15;
            """,
            (user_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_motivational_message(streak: int, username: str, language: str = "English") -> dict[str, str]:
    """Generate a personalized daily motivational message and quote based on the calendar day and user streak."""
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")
    today = datetime.date.today()
    day_of_year = today.timetuple().tm_yday

    quotes_en = [
        {"title": "Consistency is King", "quote": "“We are what we repeatedly do. Excellence, then, is not an act, but a habit.”", "author": "Will Durant", "icon": "⚡"},
        {"title": "Power of Daily Growth", "quote": "“Success is the sum of small efforts, repeated day in and day out.”", "author": "Robert Collier", "icon": "🌱"},
        {"title": "Unstoppable Curiosity", "quote": "“The beautiful thing about learning is that no one can take it away from you.”", "author": "B.B. King", "icon": "🧠"},
        {"title": "Compound Knowledge", "quote": "“An investment in knowledge pays the best interest.”", "author": "Benjamin Franklin", "icon": "📚"},
        {"title": "Focused Momentum", "quote": "“Small disciplines repeated with consistency every day lead to great achievements.”", "author": "John C. Maxwell", "icon": "🔥"},
        {"title": "The Art of Mastery", "quote": "“Live as if you were to die tomorrow. Learn as if you were to live forever.”", "author": "Mahatma Gandhi", "icon": "✨"},
        {"title": "Deep Understanding", "quote": "“I would rather have questions that can't be answered than answers that can't be questioned.”", "author": "Richard Feynman", "icon": "🔬"},
        {"title": "Continuous Iteration", "quote": "“It does not matter how slowly you go as long as you do not stop.”", "author": "Confucius", "icon": "🐢"},
        {"title": "Intellectual Bravery", "quote": "“The mind is not a vessel to be filled, but a fire to be kindled.”", "author": "Plutarch", "icon": "💡"},
        {"title": "Resilience in Problem Solving", "quote": "“It's not that I'm so smart, it's just that I stay with problems longer.”", "author": "Albert Einstein", "icon": "🧩"},
        {"title": "Clarity of Thought", "quote": "“Simplicity is prerequisite for reliability.”", "author": "Edsger W. Dijkstra", "icon": "🎯"},
        {"title": "Growth Mindset", "quote": "“The expert in anything was once a beginner.”", "author": "Helen Hayes", "icon": "🌟"},
        {"title": "Action Precedes Motivation", "quote": "“You don't have to be great to start, but you have to start to be great.”", "author": "Zig Ziglar", "icon": "🚀"},
        {"title": "Wisdom & Practice", "quote": "“Knowledge without practice is like a book on the back of an ass.”", "author": "Persian Proverb", "icon": "📖"},
    ]

    quotes_ja = [
        {"title": "継続は力なり", "quote": "「日々の小さな努力の積み重ねが、やがて大きな成果を生む。」", "author": "学習の格言", "icon": "🌱"},
        {"title": "知的好奇心の扉", "quote": "「学べば学ぶほど、自分が何も知らなかったことに気づく。」", "author": "アインシュタイン", "icon": "🧠"},
        {"title": "習慣の力", "quote": "「優れた結果は、日々の確かな習慣から生まれる。」", "author": "ウィル・デュラント", "icon": "⚡"},
        {"title": "思考の深まり", "quote": "「簡単な言葉で説明できなければ、本当に理解したとは言えない。」", "author": "リチャード・ファインマン", "icon": "💡"},
        {"title": "一歩ずつの成長", "quote": "「千里の道も一歩から。」", "author": "老子", "icon": "🔥"},
    ]

    # Select quote stably based on calendar day of year + streak offset so it changes every day
    pool = quotes_ja if is_ja else quotes_en
    selected = pool[(day_of_year + streak) % len(pool)]

    if streak >= 7:
        streak_sub = "💎 Diamond 7-Day Streak Achieved! Your learning velocity is unmatched!" if not is_ja else "💎 7日連続学習達成！知識が完全に定着しつつあります！"
    elif streak >= 2:
        streak_sub = f"🔥 Day {streak} Streak! You came back strong — momentum is building!" if not is_ja else f"🔥 {streak}日連続ストリーク！習慣が確実に定着しています！"
    else:
        streak_sub = "✨ Day 1: A fresh journey begins today. Take the first step!" if not is_ja else "✨ 1日目：新たな学習の旅がここから始まります！"


    return {
        "title": selected["title"],
        "quote": selected["quote"],
        "author": selected["author"],
        "icon": selected["icon"],
        "streak_subtitle": streak_sub,
    }


def record_login_streak(
    user_id: int,
    username: str,
    language: str = "English",
    db_path: Path | str | None = None,
) -> tuple[int, int, dict[str, str], bool]:
    """Calculate and persist user's daily login streak. Returns (streak, best_streak, motivation_msg, is_new_day)."""
    conn = get_db_connection(db_path)
    today = datetime.date.today()
    today_str = today.isoformat()
    yesterday_str = (today - datetime.timedelta(days=1)).isoformat()

    try:
        cur = conn.execute("SELECT xp, streak, best_streak, progress_json FROM user_progress WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            streak = 1
            best_streak = 1
            prog_data = {"last_login_date": today_str, "streak": 1, "best_streak": 1, "xp": 0}
            is_new_day = True
        else:
            prog_data = json.loads(row["progress_json"]) if row["progress_json"] else {}
            last_date = prog_data.get("last_login_date", "")
            curr_streak = int(row["streak"] or prog_data.get("streak", 1))
            best_streak = int(row["best_streak"] or prog_data.get("best_streak", curr_streak))

            if last_date == today_str:
                streak = max(1, curr_streak)
                is_new_day = False
            elif last_date == yesterday_str:
                streak = curr_streak + 1
                best_streak = max(best_streak, streak)
                is_new_day = True
            else:
                streak = 1
                best_streak = max(best_streak, 1)
                is_new_day = True

            prog_data["last_login_date"] = today_str
            prog_data["streak"] = streak
            prog_data["best_streak"] = best_streak
            # Always sync xp in prog_data with row['xp']
            prog_data["xp"] = int(row["xp"] or 0)

        with conn:
            conn.execute(
                """
                UPDATE user_progress
                SET streak = ?, best_streak = ?, progress_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?;
                """,
                (streak, best_streak, json.dumps(prog_data), user_id),
            )

        motivation = get_motivational_message(streak, username, language=language)
        return streak, best_streak, motivation, is_new_day
    finally:
        conn.close()


def simulate_day2_login(
    user_id: int,
    username: str,
    language: str = "English",
    db_path: Path | str | None = None,
) -> tuple[int, int, dict[str, str]]:
    """Simulate user logging in on Day 2 by rewinding last_login_date to yesterday and triggering streak increment."""
    conn = get_db_connection(db_path)
    today = datetime.date.today()
    yesterday_str = (today - datetime.timedelta(days=1)).isoformat()
    try:
        cur = conn.execute("SELECT progress_json FROM user_progress WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        prog_data = json.loads(row["progress_json"]) if (row and row["progress_json"]) else {}
        prog_data["last_login_date"] = yesterday_str
        prog_data["streak"] = 1
        with conn:
            conn.execute(
                "UPDATE user_progress SET streak = 1, progress_json = ? WHERE user_id = ?",
                (json.dumps(prog_data), user_id),
            )
    finally:
        conn.close()

    # Now calculate login streak as if logging in from yesterday -> Day 2!
    streak, best_s, mot, _ = record_login_streak(user_id, username, language=language, db_path=db_path)
    return streak, best_s, mot


def reset_user_progress_db(
    user_id: int,
    db_path: Path | str | None = None,
) -> None:
    """Reset user progress in DB to clean Level 1 (0 XP, Streak 1)."""
    conn = get_db_connection(db_path)
    today_str = datetime.date.today().isoformat()
    clean_prog = {
        "completed_concepts": [],
        "quiz_answers": [],
        "reviewed_flashcards": [],
        "explored_videos": [],
        "xp": 0,
        "streak": 1,
        "best_streak": 1,
        "daily_streak": 1,
        "best_daily_streak": 1,
        "quiz_streak": 0,
        "best_quiz_streak": 0,
        "unlocked_badge_ids": [],
        "completed_challenges": [],
        "studied_languages": ["English"],
        "chat_queries_count": 0,
        "deep_dives_viewed": [],
        "visuals_viewed": [],
        "claimed_rewards": [],
        "exam_history": [],
        "last_login_date": today_str,
    }
    try:
        with conn:
            conn.execute(
                """
                UPDATE user_progress
                SET xp = 0, streak = 1, best_streak = 1, progress_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?;
                """,
                (json.dumps(clean_prog), user_id),
            )
    finally:
        conn.close()


