"""Unit tests for SQLite database user authentication and persistent progress tracking."""

from pathlib import Path
import pytest

from lecturenotes.db.database import (
    authenticate_user,
    get_user_lectures_db,
    init_db,
    load_user_progress_db,
    record_user_lecture_db,
    register_user,
    save_user_progress_db,
    simulate_day2_login,
    reset_user_progress_db,
)
from lecturenotes.gamification.badges import UserProgress



@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_lecturenotes.db"
    init_db(db_file)
    return db_file


class TestDatabaseAuthAndProgress:
    """Tests registration, authentication, progress persistence, and lecture history."""

    def test_init_creates_demo_user(self, temp_db: Path) -> None:
        ok, msg, user = authenticate_user("demo", "demo123", db_path=temp_db)
        assert ok is True
        assert user is not None
        assert user["username"] == "demo"

    def test_register_and_authenticate_success(self, temp_db: Path) -> None:
        ok, msg, user = register_user("alice", "secret456", db_path=temp_db)
        assert ok is True
        assert user is not None
        assert user["username"] == "alice"

        # Authenticate with right password
        auth_ok, auth_msg, auth_user = authenticate_user("alice", "secret456", db_path=temp_db)
        assert auth_ok is True
        assert auth_user["id"] == user["id"]

        # Authenticate with wrong password
        fail_ok, fail_msg, fail_user = authenticate_user("alice", "wrongpassword", db_path=temp_db)
        assert fail_ok is False
        assert fail_user is None

    def test_register_duplicate_fails(self, temp_db: Path) -> None:
        register_user("bob", "secret123", db_path=temp_db)
        ok, msg, user = register_user("bob", "anotherpass", db_path=temp_db)
        assert ok is False
        assert "already registered" in msg

    def test_progress_persistence_and_update(self, temp_db: Path) -> None:
        ok, msg, user = register_user("carol", "pass1234", db_path=temp_db)
        user_id = user["id"]

        progress_data = load_user_progress_db(user_id, db_path=temp_db)
        assert progress_data["xp"] >= 0

        # Update progress
        progress_data["xp"] = 350
        progress_data["streak"] = 5
        progress_data["completed_concepts"] = ["gradient descent", "backpropagation"]
        save_user_progress_db(user_id, progress_data, db_path=temp_db)

        # Reload and verify
        reloaded = load_user_progress_db(user_id, db_path=temp_db)
        assert reloaded["xp"] == 350
        assert reloaded["streak"] == 5
        assert "gradient descent" in reloaded["completed_concepts"]
        assert "backpropagation" in reloaded["completed_concepts"]

    def test_user_progress_class_db_sync(self, temp_db: Path) -> None:
        ok, msg, user = register_user("david", "pass1234", db_path=temp_db)
        user_id = user["id"]

        prog = UserProgress({"xp": 200, "streak": 3, "completed_concepts": ["regularisation"]})
        save_user_progress_db(user_id, prog.to_dict(), db_path=temp_db)

        loaded_prog = UserProgress(load_user_progress_db(user_id, db_path=temp_db))
        assert loaded_prog.xp == 200
        assert loaded_prog.streak == 3
        assert "regularisation" in loaded_prog.completed_concepts

    def test_record_and_get_user_lectures(self, temp_db: Path) -> None:
        ok, msg, user = register_user("eva", "pass1234", db_path=temp_db)
        user_id = user["id"]

        record_user_lecture_db(user_id, "CS101 Intro", "lecture1.pdf", 5, 12, db_path=temp_db)
        record_user_lecture_db(user_id, "Deep Learning Lecture 2", "lecture2.pdf", 8, 20, db_path=temp_db)

        lectures = get_user_lectures_db(user_id, db_path=temp_db)
        assert len(lectures) == 2
        assert lectures[0]["lecture_title"] == "Deep Learning Lecture 2"
        assert lectures[1]["lecture_title"] == "CS101 Intro"

    def test_user_progress_exam_history_db_sync(self, temp_db: Path) -> None:
        ok, msg, user = register_user("felix", "pass1234", db_path=temp_db)
        user_id = user["id"]

        prog = UserProgress()
        prog.record_exam_result(95, 9, 10, "Hard only", 300, 90, "Neural Networks")
        save_user_progress_db(user_id, prog.to_dict(), db_path=temp_db)

        loaded_prog = UserProgress(load_user_progress_db(user_id, db_path=temp_db))
        assert len(loaded_prog.exam_history) == 1
        assert loaded_prog.exam_history[0]["score_pct"] == 95
        assert loaded_prog.exam_history[0]["lecture_title"] == "Neural Networks"
        assert "exam_ace" in loaded_prog.unlocked_badge_ids

    def test_day2_login_simulation_advances_streak(self, temp_db: Path) -> None:
        ok, msg, user = register_user("grace", "pass1234", db_path=temp_db)
        user_id = user["id"]

        # Initial login on Day 1
        prog_day1 = load_user_progress_db(user_id, db_path=temp_db)
        assert prog_day1["streak"] == 1

        # Simulate logging in on Day 2
        streak, best_s, mot = simulate_day2_login(user_id, "grace", db_path=temp_db)
        assert streak == 2
        assert best_s >= 2
        assert "Day 2" in mot["streak_subtitle"] or "2日" in mot["streak_subtitle"]

        prog_day2 = load_user_progress_db(user_id, db_path=temp_db)
        assert prog_day2["streak"] == 2

    def test_reset_user_progress_to_day1(self, temp_db: Path) -> None:
        ok, msg, user = register_user("helen", "pass1234", db_path=temp_db)
        user_id = user["id"]

        # Advance streak and XP
        save_user_progress_db(user_id, {"xp": 600, "streak": 4, "best_streak": 4}, db_path=temp_db)
        assert load_user_progress_db(user_id, db_path=temp_db)["xp"] == 600

        # Reset to Day 1
        reset_user_progress_db(user_id, db_path=temp_db)
        reset_prog = load_user_progress_db(user_id, db_path=temp_db)
        assert reset_prog["xp"] == 0
        assert reset_prog["streak"] == 1
        assert reset_prog["best_streak"] == 1


