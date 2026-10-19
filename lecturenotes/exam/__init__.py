"""Exam preparation engine for LectureNotes AI."""

from lecturenotes.exam.exam_engine import (
    DescriptiveExamQuestion,
    generate_expanded_mcqs,
    generate_descriptive_questions,
    generate_exam_questions_parallel,
    evaluate_descriptive_answer,
)

__all__ = [
    "DescriptiveExamQuestion",
    "generate_expanded_mcqs",
    "generate_descriptive_questions",
    "generate_exam_questions_parallel",
    "evaluate_descriptive_answer",
]
