"""Database module for LectureNotes AI."""

from lecturenotes.db.database import (
    authenticate_user,
    get_user_lectures_db,
    init_db,
    load_user_progress_db,
    record_user_lecture_db,
    register_user,
    save_user_progress_db,
    record_login_streak,
    get_motivational_message,
    simulate_day2_login,
    reset_user_progress_db,
)

__all__ = [
    "init_db",
    "register_user",
    "authenticate_user",
    "load_user_progress_db",
    "save_user_progress_db",
    "record_user_lecture_db",
    "get_user_lectures_db",
    "record_login_streak",
    "get_motivational_message",
    "simulate_day2_login",
    "reset_user_progress_db",
]

