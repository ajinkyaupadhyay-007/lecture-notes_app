"""Prompt templates for structured LLM generation.

All prompts enforce strict JSON responses and require attributing every statement,
definition, flashcard answer, and quiz question to an exact source_span from the lecture.
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You are an expert academic tutor and NLP lecture note specialist.
Your task is to convert lecture transcripts into highly structured, factual learning materials.
CRITICAL RULES:
1. Output ONLY valid, parseable JSON. Do not include markdown code block formatting (```json) or conversational text.
2. Ground every claim, takeaway, definition, and answer strictly in the provided text.
3. Every generated item MUST include a "source_span" field containing the exact sentence from the text that proves or justifies it. Do not invent facts."""


def _language_instruction(language: str) -> str:
    """Helper to return target language guidance."""
    lang_lower = language.lower()
    if lang_lower in ("japanese", "ja", "nihongo", "日本語"):
        return (
            "LANGUAGE REQUIREMENT: Generate all content (sentences, definitions, questions, choices, "
            "and explanations) strictly in Japanese (日本語). Use accurate technical Japanese vocabulary "
            "(e.g., 教師あり学習, 勾配降下法, 過学習, 正則化). Keep exact factual fidelity to the source."
        )
    return "LANGUAGE REQUIREMENT: Output in clear, fluent English."


def build_section_notes_prompt(
    section_id: int,
    section_title: str,
    section_text: str,
    language: str = "English",
) -> str:
    """Prompt for generating concise summary bullet points for a topic section."""
    lang_instr = _language_instruction(language)
    schema_example = {
        "section_id": section_id,
        "title": section_title,
        "summary_points": [
            {
                "summary_sentence": "Linear regression fits a linear function by minimising the mean squared error loss.",
                "source_span": "Given features x and targets y, it fits a linear function by minimising the mean squared error (MSE) loss.",
            }
        ],
    }

    return f"""Analyze the following topic section from a lecture and generate 2 to 4 clear, factual summary takeaway points.
Every summary point must have a 'summary_sentence' and the exact 'source_span' from the section text that supports it.
{lang_instr}

SECTION ID: {section_id}
SECTION TITLE: {section_title}
SECTION TEXT:
\"\"\"{section_text}\"\"\"

Return a JSON object conforming exactly to this schema:
{json.dumps(schema_example, indent=2)}"""


def build_definitions_prompt(
    terms: list[str],
    lecture_text: str,
    language: str = "English",
) -> str:
    """Prompt for generating grounded definitions for extracted technical terms."""
    terms_str = ", ".join(f"'{t}'" for t in terms)
    lang_instr = _language_instruction(language)
    schema_example = {
        "definitions": [
            {
                "term": "bias-variance trade-off",
                "definition": "The trade-off between error from model assumptions (bias) and sensitivity to training data fluctuations (variance).",
                "source_span": "Bias refers to error introduced by approximating a complex real-world problem with a simplified model.",
            }
        ]
    }

    return f"""Define the following technical concepts strictly based on the provided lecture context.
Only define terms that are explained or grounded in the text. For each term, provide a clear educational 'definition' and the supporting 'source_span'.
{lang_instr}

TERMS TO DEFINE:
{terms_str}

LECTURE CONTEXT:
\"\"\"{lecture_text}\"\"\"

Return a JSON object conforming exactly to this schema:
{json.dumps(schema_example, indent=2)}"""


def build_flashcards_prompt(
    lecture_text: str,
    concepts: list[str],
    target_count: int = 15,
    language: str = "English",
) -> str:
    """Prompt for generating 10 to 20 conceptual flashcards."""
    concepts_hint = ", ".join(concepts[:15]) if concepts else "all core topics"
    lang_instr = _language_instruction(language)
    schema_example = {
        "flashcards": [
            {
                "card_id": 1,
                "question": "What is the primary difference between L1 and L2 regularisation?",
                "answer": "L1 regularisation (Lasso) shrinks some weights to exactly zero producing sparse models, whereas L2 (Ridge) shrinks weights towards zero without zeroing them.",
                "source_span": "L1 regularisation (Lasso) adds λ||w||_1, which can drive some weights to exactly zero, producing sparse models.",
                "concept": "regularisation",
            }
        ]
    }

    return f"""Generate between {max(10, target_count - 2)} and {min(20, target_count + 2)} review flashcards for students studying this lecture.
Focus on testing conceptual understanding, definitions, key algorithms, and tradeoffs.
Key concepts to cover: {concepts_hint}.
CRITICAL: Every flashcard 'answer' must be strictly factual, and 'source_span' MUST contain the exact sentence from the lecture text justifying the answer.
{lang_instr}

LECTURE TEXT:
\"\"\"{lecture_text}\"\"\"

Return a JSON object conforming exactly to this schema:
{json.dumps(schema_example, indent=2)}"""


def build_quiz_prompt(
    lecture_text: str,
    concepts: list[str],
    question_count: int = 5,
    difficulty: str = "Medium",
    language: str = "English",
) -> str:
    """Prompt for generating multiple choice quiz with distractors, explanations, and difficulty level."""
    concepts_hint = ", ".join(concepts[:10]) if concepts else "core concepts"
    lang_instr = _language_instruction(language)

    diff_normalized = difficulty.capitalize() if difficulty else "Medium"
    if "adaptive" in diff_normalized.lower() or "mix" in diff_normalized.lower():
        diff_guidance = (
            "DIFFICULTY LEVEL: ADAPTIVE MIX. Provide a structured progression of difficulties: "
            "Early questions should be Easy (foundational definitions), middle questions should be Medium (application & comparisons), "
            "and later questions should be Hard (deep edge-case tradeoffs and analytical discrimination). "
            "Set the 'difficulty' field of each question to 'Easy', 'Medium', or 'Hard' accordingly."
        )
        diff_field_req = "A 'difficulty' field set to 'Easy', 'Medium', or 'Hard' matching the question's challenge level."
    elif diff_normalized == "Easy":
        diff_guidance = (
            "DIFFICULTY LEVEL: EASY. Focus on direct recall of foundational definitions, terminology, "
            "and basic principles. Distractors should be clear and distinct."
        )
        diff_field_req = "A 'difficulty' field set to 'Easy'."
    elif diff_normalized == "Hard":
        diff_guidance = (
            "DIFFICULTY LEVEL: HARD. Focus on deep analytical nuances, algorithm tradeoffs, edge-case behavior, "
            "and mathematical or structural comparisons. Distractors should be realistic and require subtle discrimination."
        )
        diff_field_req = "A 'difficulty' field set to 'Hard'."
    else:
        diff_guidance = (
            "DIFFICULTY LEVEL: MEDIUM. Focus on standard conceptual understanding, practical application, "
            "and comparing core techniques."
        )
        diff_field_req = "A 'difficulty' field set to 'Medium'."

    schema_example = {
        "quiz": [
            {
                "question_id": 1,
                "question": "Which optimisation algorithm updates parameters using a single random sample or mini-batch?",
                "options": {
                    "A": "Stochastic gradient descent (SGD)",
                    "B": "Ordinary least squares",
                    "C": "Maximum-margin hyperplane",
                    "D": "Kernel ridge regression",
                },
                "correct_answer": "A",
                "explanation": "SGD approximates the gradient using a single sample or mini-batch, making it computationally efficient.",
                "distractor_explanations": {
                    "B": "Ordinary least squares is a closed-form solution, not a mini-batch gradient algorithm.",
                    "C": "Maximum-margin hyperplane is an SVM optimization objective, not an iterative mini-batch optimizer.",
                    "D": "Kernel ridge regression solves a regularized kernel system, not an iterative SGD variant.",
                },
                "source_span": "Stochastic gradient descent (SGD) approximates the full gradient using a single random sample or a mini-batch.",
                "difficulty": "Hard" if diff_normalized == "Hard" else ("Easy" if diff_normalized == "Easy" else "Medium"),
            }
        ]
    }

    return f"""Generate exactly {question_count} high-quality, conceptual multiple-choice questions (MCQs) for this lecture.
Cover topics including: {concepts_hint}.
{diff_guidance}
{lang_instr}

Each question MUST have:
- Four realistic options ('A', 'B', 'C', 'D')
- A single 'correct_answer'
- An 'explanation' of why the correct choice is right
- 'distractor_explanations' for each incorrect option explaining why it is a plausible distractor but incorrect
- A 'source_span' quoting the exact sentence from the text supporting the answer.
- {diff_field_req}

LECTURE TEXT:
\"\"\"{lecture_text}\"\"\"

Return a JSON object conforming exactly to this schema:
{json.dumps(schema_example, indent=2)}"""
