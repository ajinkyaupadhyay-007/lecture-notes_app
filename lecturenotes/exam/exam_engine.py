"""Exam Preparation Engine for LectureNotes AI.

Provides expanded MCQ question generation, descriptive (conceptual/short-answer)
question generation, and automated rubric-based descriptive answer evaluation.
"""

from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal
from pydantic import BaseModel, Field

from lecturenotes.generation.schemas import LectureOutput, MCQQuestion

logger = logging.getLogger(__name__)


def _get_default_llm_client() -> Any:
    """Lazily load configured LLMClient if Gemini or other provider key is present."""
    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return None
    try:
        from lecturenotes.config import load_config
        from lecturenotes.generation.llm_client import LLMClient
        cfg = load_config()
        return LLMClient(cfg)
    except Exception as e:
        logger.debug("Could not initialize default LLMClient: %s", e)
        return None


def _generate_creative_llm_mcqs(
    output: LectureOutput,
    target_count: int,
    difficulty: str = "Hard",
    language: str = "English",
    llm_client: Any = None,
) -> list[MCQQuestion]:
    """Use Gemini / LLM to generate creative, deeply challenging, graduate-level questions."""
    client = llm_client or _get_default_llm_client()
    if not client or target_count <= 0:
        return []

    concepts = getattr(output, "concept_definitions", []) or []
    # Limit context to 8 concepts / 4 sections to keep prompts lean and fast
    concepts_text = "\n".join(
        [f"- {c.term}: {c.definition[:70]}" for c in concepts[:8]]
    )
    sections = getattr(output, "section_notes", []) or []
    sections_text = "\n".join(
        [f"- {getattr(s, 'title', getattr(s, 'heading', 'Section'))}" for s in sections[:4]]
    )
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    prompt = (
        f"You are a distinguished university professor designing an exceptionally difficult, graduate-level exam.\n"
        f"Topic / Title: {output.lecture_title}\n"
        f"Core Lecture Concepts:\n{concepts_text}\n"
        f"Lecture Outline:\n{sections_text}\n\n"
        f"Target Difficulty: {difficulty} (MUST be genuinely difficult, rigorous, and intellectually creative)\n"
        f"Target Language: {'Japanese' if is_ja else 'English'}\n"
        f"Required Question Count: {target_count}\n\n"
        f"STRICT DESIGN PRINCIPLES FOR MAXIMUM DIFFICULTY:\n"
        f"1. Zero simple recall: Absolutely DO NOT ask for basic definitions or terminology identification.\n"
        f"2. Complex scenarios: Frame questions around applied architectural trade-offs, theoretical boundary conditions, failure mode diagnoses under severe stress/distribution shift, or conflicting constraints.\n"
        f"3. High-quality distractors: Every distractor (A, B, C, D) must be a plausible, sophisticated statement representing a subtle misconception or overlooked constraint.\n"
        f"4. Rigorous explanation: Provide an exhaustive breakdown proving why the correct choice holds and exactly why the other choices fail.\n\n"
        f"Output MUST be a JSON object with this exact schema:\n"
        f'{{\n  "questions": [\n    {{\n      "question": "Scenario or problem statement...",\n      "options": {{\n        "A": "Option text...",\n        "B": "Option text...",\n        "C": "Option text...",\n        "D": "Option text..."\n      }},\n      "correct_answer": "A",\n      "explanation": "Detailed theoretical breakdown...",\n      "difficulty": "Hard"\n    }}\n  ]\n}}'
    )

    try:
        res = client.generate_json(
            prompt,
            system_prompt="You are an expert assessment examiner generating rigorous, creative, difficult exam questions in JSON.",
        )
        q_list = res.get("questions", [])
        if not isinstance(q_list, list):
            return []

        parsed: list[MCQQuestion] = []
        for i, qd in enumerate(q_list):
            if not isinstance(qd, dict):
                continue
            q_text = str(qd.get("question", "")).strip()
            opts = qd.get("options", {})
            ca = str(qd.get("correct_answer", "A")).strip().upper()
            expl = str(qd.get("explanation", "")).strip()
            diff = qd.get("difficulty", "Hard")
            if diff not in ("Easy", "Medium", "Hard"):
                diff = "Hard"
            if q_text and isinstance(opts, dict) and len(opts) >= 4:
                clean_opts = {k.upper(): str(v).strip() for k, v in opts.items() if k.upper() in ("A", "B", "C", "D")}
                if len(clean_opts) == 4 and ca in clean_opts:
                    parsed.append(
                        MCQQuestion(
                            question_id=i + 1,
                            question=q_text,
                            options=clean_opts,
                            correct_answer=ca,  # type: ignore
                            explanation=expl,
                            source_span=str(qd.get("source_span") or output.lecture_title or "Lecture source reference"),
                            difficulty=diff,
                        )
                    )
        return parsed[:target_count]
    except Exception as e:
        logger.warning("LLM MCQ generation failed, falling back to deterministic: %s", e)
        return []


def _generate_creative_llm_descriptive(
    output: LectureOutput,
    count: int,
    difficulty: str = "Hard",
    language: str = "English",
    llm_client: Any = None,
) -> list[DescriptiveExamQuestion]:
    """Use Gemini / LLM to generate creative, deeply analytical descriptive questions."""
    client = llm_client or _get_default_llm_client()
    if not client or count <= 0:
        return []

    concepts = getattr(output, "concept_definitions", []) or []
    # Limit context to 8 concepts / 4 sections for faster, leaner prompts
    concepts_text = "\n".join(
        [f"- {c.term}: {c.definition[:70]}" for c in concepts[:8]]
    )
    sections = getattr(output, "section_notes", []) or []
    sections_text = "\n".join(
        [f"- {getattr(s, 'title', getattr(s, 'heading', 'Section'))}" for s in sections[:4]]
    )
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    prompt = (
        f"You are a distinguished university professor designing an exceptionally difficult written examination.\n"
        f"Topic / Title: {output.lecture_title}\n"
        f"Core Lecture Concepts:\n{concepts_text}\n"
        f"Lecture Outline:\n{sections_text}\n\n"
        f"Target Difficulty: {difficulty} (Genuinely difficult, creative, and requiring high-level synthesis)\n"
        f"Target Language: {'Japanese' if is_ja else 'English'}\n"
        f"Required Question Count: {count}\n\n"
        f"STRICT DESIGN PRINCIPLES FOR MAXIMUM DIFFICULTY:\n"
        f"1. Require multi-dimensional analytical critique: Ask the student to contrast opposing paradigms, critique fundamental assumptions under edge cases, or diagnose systemic bottlenecks.\n"
        f"2. Applied problem solving: Formulate scenarios requiring the student to propose and justify an optimal mitigation architecture under conflicting computational or resource constraints.\n"
        f"3. Provide 4-6 specific key technical terms and mechanisms that must be addressed in an exemplary response.\n"
        f"4. Provide an authoritative, exhaustive model answer (120-220 words).\n\n"
        f"Output MUST be a JSON object with this exact schema:\n"
        f'{{\n  "questions": [\n    {{\n      "question": "In-depth prompt requiring critique...",\n      "difficulty": "Hard",\n      "key_points": ["point 1", "point 2", "point 3", "point 4"],\n      "model_answer": "Comprehensive reference answer...",\n      "concept_term": "Target Concept"\n    }}\n  ]\n}}'
    )

    try:
        res = client.generate_json(
            prompt,
            system_prompt="You are an expert assessment examiner generating creative, challenging descriptive exam questions in JSON.",
        )
        q_list = res.get("questions", [])
        if not isinstance(q_list, list):
            return []

        parsed: list[DescriptiveExamQuestion] = []
        for i, qd in enumerate(q_list):
            if not isinstance(qd, dict):
                continue
            q_text = str(qd.get("question", "")).strip()
            kps = qd.get("key_points", [])
            if not isinstance(kps, list):
                kps = [str(kps)]
            model_ans = str(qd.get("model_answer", "")).strip()
            diff = qd.get("difficulty", "Hard")
            if diff not in ("Easy", "Medium", "Hard"):
                diff = "Hard"
            ct = str(qd.get("concept_term", "")).strip()
            if q_text and model_ans:
                parsed.append(
                    DescriptiveExamQuestion(
                        question_id=i + 1,
                        question=q_text,
                        difficulty=diff,
                        key_points=[str(kp).strip() for kp in kps if str(kp).strip()],
                        model_answer=model_ans,
                        concept_term=ct,
                    )
                )
        return parsed[:count]
    except Exception as e:
        logger.warning("LLM descriptive generation failed, falling back to deterministic: %s", e)
        return []


class DescriptiveExamQuestion(BaseModel):
    """A descriptive conceptual question with key points and model answer."""

    question_id: int = Field(..., ge=1, description="1-indexed question ID.")
    question: str = Field(..., description="Conceptual or analytical question prompt.")
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        default="Medium",
        description="Difficulty level: Easy, Medium, or Hard.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Core concepts or points that should be mentioned in a complete answer.",
    )
    model_answer: str = Field(..., description="High-quality reference/model answer.")
    source_reference: str = Field(default="", description="Source span or lecture reference.")
    concept_term: str = Field(default="", description="Target concept term.")


def generate_expanded_mcqs(
    output: LectureOutput,
    target_count: int = 10,
    difficulty: str = "All",
    language: str = "English",
    llm_client: Any = None,
    use_llm: bool = False,
) -> list[MCQQuestion]:
    """Return up to target_count MCQ questions with genuine difficulty calibration.

    If difficulty is 'Hard only', questions emphasize constraints, failure modes,
    algorithmic tradeoffs, and boundary conditions with rigorous distractors.
    When use_llm is True, calls LLM for creative generation.
    When use_llm is False (default Instant Mode), synthesizes calibrated questions
    instantly (<0.01s) directly from lecture concepts with zero network delay.
    """
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")
    is_hard_mode = difficulty in ("Hard only", "Hard")
    is_easy_mode = difficulty in ("Easy only", "Easy")
    is_medium_mode = difficulty in ("Medium only", "Medium")

    llm_mcqs: list[MCQQuestion] = []
    if use_llm:
        llm_mcqs = _generate_creative_llm_mcqs(
            output=output,
            target_count=target_count,
            difficulty=difficulty,
            language=language,
            llm_client=llm_client,
        )
        if llm_mcqs and len(llm_mcqs) >= target_count:
            for idx, q in enumerate(llm_mcqs):
                q.question_id = idx + 1
            return llm_mcqs[:target_count]

    pool: list[MCQQuestion] = list(llm_mcqs)
    for q in getattr(output, "quiz", []) or []:
        if is_hard_mode and q.difficulty != "Hard":
            continue
        if is_easy_mode and q.difficulty != "Easy":
            continue
        if is_medium_mode and q.difficulty != "Medium":
            continue
        if not any(pq.question.lower().strip() == q.question.lower().strip() for pq in pool):
            pool.append(q)

    # If we already have enough questions in the filtered pool
    if len(pool) >= target_count:
        return pool[:target_count]

    existing_stems = {q.question.lower().strip() for q in pool}
    synthesized: list[MCQQuestion] = []
    q_id_counter = len(pool) + 1

    concepts = getattr(output, "concept_definitions", []) or []
    flashcards = getattr(output, "flashcards", []) or []
    sections = getattr(output, "section_notes", []) or []

    # ── 1. HARD MODE SYNTHESIS: Constraints, Trade-offs & Failure Modes ──
    if is_hard_mode:
        for c in concepts:
            if len(pool) + len(synthesized) >= target_count:
                break

            # Distractors pool from other concepts
            other_terms = [other.term for other in concepts if other.term.lower() != c.term.lower()]
            d_term1 = other_terms[0] if len(other_terms) > 0 else "alternative models"
            d_term2 = other_terms[1] if len(other_terms) > 1 else "heuristic baselines"

            # Hard Archetype A: Algorithmic Trade-off / Boundary Condition
            stem_a = (
                f"講義内容に基づき、概念「{c.term}」を適用する際の重要な制約・アルゴリズム上のトレードオフ、または前提条件として最も適切なものはどれですか？"
                if is_ja
                else f"Based on the lecture, which fundamental algorithmic tradeoff, constraint, or boundary condition governs '{c.term}'?"
            )
            if stem_a.lower().strip() not in existing_stems:
                correct_a = (
                    f"「{c.definition[:80]}…」という運用前提・理論的性質を満たす必要がある"
                    if is_ja
                    else f"It is bounded by the operational principle: \"{c.definition[:85]}...\""
                )
                distractors_a = [
                    f"「{d_term1}」と同等の完全な損失凸性および大域的収束性を無条件で仮定する"
                    if is_ja
                    else f"It assumes identical loss convexity to {d_term1} without parameter sensitivity",
                    f"「{d_term2}」における正則化ハイパーパラメータを完全に排除して過学習を無視する"
                    if is_ja
                    else f"It eliminates regularization parameters unlike {d_term2}, ignoring generalization limits",
                    "局所最適解に陥ることなく大域的最適性を無条件で保証する"
                    if is_ja
                    else "It unconditionally guarantees global empirical risk minimization without sample bias",
                ]
                choices_a = [correct_a] + distractors_a
                random.shuffle(choices_a)
                c_letters = ["A", "B", "C", "D"]
                opts_a = {c_letters[i]: ch for i, ch in enumerate(choices_a)}
                corr_letter_a = c_letters[choices_a.index(correct_a)]

                synthesized.append(
                    MCQQuestion(
                        question_id=q_id_counter,
                        question=stem_a,
                        options=opts_a,
                        correct_answer=corr_letter_a,  # type: ignore
                        explanation=f"'{c.term}' requires understanding its foundational boundary condition: {c.source_span or c.definition}",
                        source_span=c.source_span or c.definition,
                        difficulty="Hard",
                    )
                )
                existing_stems.add(stem_a.lower().strip())
                q_id_counter += 1

            if len(pool) + len(synthesized) >= target_count:
                break

            # Hard Archetype B: Contradiction / Invalid Assertion
            stem_b = (
                f"「{c.term}」に関する次の記述のうち、講義の理論的原則と矛盾する（誤っている）ものはどれですか？"
                if is_ja
                else f"Which of the following assertions regarding '{c.term}' is THEORETICALLY INVALID or contradictory to the lecture's principles?"
            )
            if stem_b.lower().strip() not in existing_stems:
                invalid_statement = (
                    f"「{c.term}」は前提条件を一切考慮せず任意のスケーリングで線形計算量を維持する"
                    if is_ja
                    else f"'{c.term}' eliminates all computational bottlenecks and scales linearly without domain constraints"
                )
                valid_stmt1 = (
                    f"「{c.definition[:70]}…」のメカニズムによって機能する"
                    if is_ja
                    else f"Operates fundamentally through: {c.definition[:75]}"
                )
                valid_stmt2 = (
                    f"講義テキスト「{c.source_span[:70] or c.term}…」に基づく実証的根拠を持つ"
                    if is_ja
                    else f"Is supported by lecture evidence: \"{c.source_span[:70] or c.definition[:70]}...\""
                )
                valid_stmt3 = (
                    f"「{d_term1}」とは異なる固有の理論的役割を担っている"
                    if is_ja
                    else f"Maintains a distinct architectural role compared to {d_term1}"
                )
                choices_b = [invalid_statement, valid_stmt1, valid_stmt2, valid_stmt3]
                random.shuffle(choices_b)
                c_letters = ["A", "B", "C", "D"]
                opts_b = {c_letters[i]: ch for i, ch in enumerate(choices_b)}
                corr_letter_b = c_letters[choices_b.index(invalid_statement)]

                synthesized.append(
                    MCQQuestion(
                        question_id=q_id_counter,
                        question=stem_b,
                        options=opts_b,
                        correct_answer=corr_letter_b,  # type: ignore
                        explanation=f"The incorrect assertion claims unconstrained scaling. In reality: {c.definition}",
                        source_span=c.source_span or c.definition,
                        difficulty="Hard",
                    )
                )
                existing_stems.add(stem_b.lower().strip())
                q_id_counter += 1

    # ── 2. GENERAL / MEDIUM / EASY SYNTHESIS ──
    # Synthesize from flashcards
    all_answers = [f.answer.strip() for f in flashcards if f.answer.strip()]
    for fc in flashcards:
        if len(pool) + len(synthesized) >= target_count:
            break
        if fc.question.lower().strip() in existing_stems:
            continue

        correct_ans = fc.answer.strip()
        other_answers = [a for a in all_answers if a != correct_ans]
        while len(other_answers) < 3:
            other_answers.append(f"Alternative hypothesis {len(other_answers) + 1}")
        distractors = random.sample(other_answers, min(3, len(other_answers)))

        choices = [correct_ans] + distractors
        random.shuffle(choices)
        choice_letters = ["A", "B", "C", "D"]
        options = {letter: choices[i] for i, letter in enumerate(choice_letters)}
        correct_letter = choice_letters[choices.index(correct_ans)]

        if is_hard_mode:
            assigned_diff: Literal["Easy", "Medium", "Hard"] = "Hard"
            q_stem = f"Analyze the following: {fc.question} What is the rigorously grounded principle?"
        elif is_easy_mode:
            assigned_diff = "Easy"
            q_stem = fc.question
        else:
            assigned_diff = "Medium"
            q_stem = fc.question

        synthesized.append(
            MCQQuestion(
                question_id=q_id_counter,
                question=q_stem,
                options=options,
                correct_answer=correct_letter,  # type: ignore
                explanation=f"Based on the lecture: {fc.answer}",
                source_span=fc.source_span,
                difficulty=assigned_diff,
            )
        )
        existing_stems.add(q_stem.lower().strip())
        q_id_counter += 1

    # Synthesize from concepts if still need more
    for c in concepts:
        if len(pool) + len(synthesized) >= target_count:
            break
        stem = (
            f"何の説明ですか: 「{c.definition}」"
            if is_ja
            else f"Which concept is defined as: \"{c.definition}\"?"
        )
        if stem.lower().strip() in existing_stems:
            continue

        correct_term = c.term.title()
        other_terms = [other.term.title() for other in concepts if other.term.lower() != c.term.lower()]
        while len(other_terms) < 3:
            other_terms.append(f"Concept {len(other_terms) + 1}")
        distractors = random.sample(other_terms, min(3, len(other_terms)))

        choices = [correct_term] + distractors
        random.shuffle(choices)
        choice_letters = ["A", "B", "C", "D"]
        options = {letter: choices[i] for i, letter in enumerate(choice_letters)}
        correct_letter = choice_letters[choices.index(correct_term)]

        assigned_diff = "Hard" if is_hard_mode else ("Easy" if is_easy_mode else "Medium")
        synthesized.append(
            MCQQuestion(
                question_id=q_id_counter,
                question=stem,
                options=options,
                correct_answer=correct_letter,  # type: ignore
                explanation=f"'{c.term}' is defined as: {c.definition}",
                source_span=c.source_span,
                difficulty=assigned_diff,
            )
        )
        existing_stems.add(stem.lower().strip())
        q_id_counter += 1

    combined = pool + synthesized
    return combined[:target_count]


def generate_descriptive_questions(
    output: LectureOutput,
    count: int = 3,
    difficulty: str = "Medium",
    language: str = "English",
    llm_client: Any = None,
    use_llm: bool = False,
) -> list[DescriptiveExamQuestion]:
    """Generate in-depth conceptual and descriptive questions from concepts and sections.

    Supports generating up to 20+ distinct questions across concepts, sections,
    comparative trade-offs, failure mode analyses, and applied scenarios.
    When use_llm is True, calls LLM for creative question generation.
    When use_llm is False (default Instant Mode), synthesizes calibrated questions
    instantly (<0.01s) directly from lecture concepts with zero network delay.
    """
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")
    is_hard_mode = difficulty in ("Hard only", "Hard")
    is_easy_mode = difficulty in ("Easy only", "Easy")

    llm_desc: list[DescriptiveExamQuestion] = []
    if use_llm:
        llm_desc = _generate_creative_llm_descriptive(
            output=output,
            count=count,
            difficulty=difficulty,
            language=language,
            llm_client=llm_client,
        )
        if llm_desc and len(llm_desc) >= count:
            for idx, dq in enumerate(llm_desc):
                dq.question_id = idx + 1
            return llm_desc[:count]

    concepts = getattr(output, "concept_definitions", []) or []
    sections = getattr(output, "section_notes", []) or []
    flashcards = getattr(output, "flashcards", []) or []

    questions: list[DescriptiveExamQuestion] = list(llm_desc or [])
    seen_prompts: set[str] = {dq.question.lower().strip() for dq in questions}
    q_id = len(questions) + 1

    diff_label: Literal["Easy", "Medium", "Hard"] = "Hard" if is_hard_mode else ("Easy" if is_easy_mode else "Medium")

    # ── ARCHETYPE 1: Concept Mechanistic Analysis & Theoretical Depth ──
    for c in concepts:
        if len(questions) >= count:
            break

        words = [w.strip(".,;:()\"'").lower() for w in c.definition.split() if len(w) > 4]
        key_words = list(dict.fromkeys(words))[:4]
        key_pts = [c.term.lower()] + key_words

        if is_hard_mode:
            key_pts += ["tradeoff", "limitation", "assumption"]
            if is_ja:
                prompt = (
                    f"【高度論述・トレードオフ分析】講義における「{c.term}」について批判的に論述しなさい。"
                    f"基礎となる理論的前提条件、計算量と精度のトレードオフを論じ、"
                    f"エッジケースやリソース制約下での具体的な課題と対策を述べなさい（100〜180字程度）。"
                )
                model_ans = (
                    f"「{c.term}」は次のように定義されます: {c.definition}。"
                    f"理論的前提として、入力データの分布と計算リソースの整合性が求められます。"
                    f"トレードオフとしては、精度の向上に伴う計算オーバーヘッドが挙げられ、"
                    f"エッジケースでは正則化や境界条件の適切な設定による対策が必要です。講義典拠: {c.source_span}。"
                )
            else:
                prompt = (
                    f"【Advanced Analytical Critique】 Critically analyze the role of '{c.term}' in this lecture. "
                    f"Explain its foundational theoretical assumptions, assess the core trade-offs between "
                    f"computational overhead and accuracy, and formulate an actionable mitigation strategy for when "
                    f"operating under resource-constrained or adversarial edge cases (100–180 words)."
                )
                model_ans = (
                    f"'{c.term}' is formally defined as: {c.definition}. "
                    f"Foundational assumptions require consistent operational conditions and valid feature representation. "
                    f"The critical trade-off lies between computational complexity and empirical fidelity. "
                    f"To mitigate edge-case failure, practitioners must enforce strict boundary validation and adaptive regularization. "
                    f"Lecture evidence: \"{c.source_span}\"."
                )
        elif is_easy_mode:
            if is_ja:
                prompt = f"「{c.term}」の用語を自分の言葉で定義し、講義における主な役割を簡潔に説明しなさい（30〜60字）。"
                model_ans = f"「{c.term}」は「{c.definition}」を意味し、講義における基礎的な役割を担います。"
            else:
                prompt = f"Define the concept '{c.term}' in your own words. Explain its primary purpose and role in the lecture (30–60 words)."
                model_ans = f"'{c.term}' is defined as: {c.definition}. Its primary role is to serve as a foundational mechanism in this subject."
        else:
            if is_ja:
                prompt = (
                    f"【記述問題】講義における「{c.term}」の概念的役割と具体的な仕組みについて説明してください。"
                    f"その定義と、講義内でどのような課題を解決するために用いられるかを論述しなさい。"
                )
                model_ans = f"「{c.term}」は次のように定義されます: {c.definition}。講義テキストによると、{c.source_span}。"
            else:
                prompt = (
                    f"Explain the mechanism, definition, and significance of '{c.term}' in the context of this lecture. "
                    f"Describe the specific problem it addresses and how it operates in practice."
                )
                model_ans = (
                    f"'{c.term}' is defined as: {c.definition}. "
                    f"According to the lecture text, '{c.source_span}', which demonstrates its role in practical application."
                )

        if prompt not in seen_prompts:
            questions.append(
                DescriptiveExamQuestion(
                    question_id=q_id,
                    question=prompt,
                    difficulty=diff_label,
                    key_points=key_pts,
                    model_answer=model_ans,
                    source_reference=c.source_span,
                    concept_term=c.term,
                )
            )
            seen_prompts.add(prompt)
            q_id += 1

    # ── ARCHETYPE 2: Section Synthesis & Theoretical Bottlenecks ──
    for sec in sections:
        if len(questions) >= count:
            break
        summary_points = getattr(sec, "summary_points", []) or []
        takeaways = [sp.summary_sentence for sp in summary_points[:2]]
        joined_takeaways = " ".join(takeaways) if takeaways else sec.title

        if is_hard_mode:
            key_pts = [w.lower() for w in sec.title.split() if len(w) > 3] + ["bottleneck", "constraint", "architecture"]
            if is_ja:
                prompt = (
                    f"【理論統合・制約分析】セクション「{sec.title}」の論点に基づき、提示された方法論の核心を統合しなさい。"
                    f"本アプローチを大規模環境に適用した際に生じる理論的ボトルネックおよび制約を評価し論述しなさい。"
                )
                model_ans = (
                    f"セクション「{sec.title}」の要点は次の通りです: {joined_takeaways}。"
                    f"理論的ボトルネックとしてはスケーラビリティとリソース競合が想定され、適切な最適化が必要です。"
                )
            else:
                prompt = (
                    f"【Theoretical Synthesis & Bottleneck Analysis】 Based on section '{sec.title}', synthesize the core methodologies presented. "
                    f"Analyze the primary theoretical bottlenecks and constraints when scaling this approach, and critique potential architectural failure points (100–180 words)."
                )
                model_ans = (
                    f"In '{sec.title}', the core takeaways are: {joined_takeaways}. "
                    f"Key bottlenecks involve computational scaling and assumption validity under distribution shifts."
                )
        else:
            key_pts = [w.lower() for w in sec.title.split() if len(w) > 3] + ["analysis", "mechanism"]
            if is_ja:
                prompt = (
                    f"【記述・応用問題】セクション「{sec.title}」の論点に基づき、"
                    f"主要な知見がどのように講義全体の理解に寄与するかを要約・論述しなさい。"
                )
                model_ans = f"セクション「{sec.title}」の要点は次の通りです: {joined_takeaways}"
            else:
                prompt = (
                    f"Based on the topic section '{sec.title}', synthesize the primary theoretical principles "
                    f"presented and explain why they are critical for understanding the subject matter."
                )
                model_ans = f"In '{sec.title}', the core takeaways are: {joined_takeaways}"

        if prompt not in seen_prompts:
            questions.append(
                DescriptiveExamQuestion(
                    question_id=q_id,
                    question=prompt,
                    difficulty=diff_label,
                    key_points=key_pts,
                    model_answer=model_ans,
                    source_reference=sec.source_reference,
                    concept_term=sec.title,
                )
            )
            seen_prompts.add(prompt)
            q_id += 1

    # ── ARCHETYPE 3: Concept Comparative Analysis & Trade-offs ──
    for i in range(len(concepts) - 1):
        if len(questions) >= count:
            break
        c1, c2 = concepts[i], concepts[i + 1]

        if is_hard_mode:
            key_pts = [c1.term.lower(), c2.term.lower(), "tradeoff", "comparison", "workload"]
            if is_ja:
                prompt = (
                    f"【比較批評・アーキテクチャのトレードオフ】「{c1.term}」と「{c2.term}」について厳密な比較分析を行いなさい。"
                    f"どのようなワークロード特性や運用制約下で一方が他方に対して優位性を持つか、"
                    f"およびそれに伴うリソースオーバーヘッドと複雑性のトレードオフを論じなさい。"
                )
                model_ans = (
                    f"「{c1.term}」({c1.definition})と「{c2.term}」({c2.definition})の比較において、"
                    f"前者は特定の制約下での精度や一貫性に優れる一方、後者は柔軟性や効率性に利点があります。"
                    f"ワークロードに応じた適切な選定とオーバーヘッドの管理が不可欠です。"
                )
            else:
                prompt = (
                    f"【Comparative Critique & Architectural Trade-offs】 Formulate a rigorous comparative analysis between '{c1.term}' and '{c2.term}'. "
                    f"Under what specific workload characteristics or operational constraints is '{c1.term}' preferred over '{c2.term}', "
                    f"and what consequential trade-offs in computational overhead and complexity are incurred?"
                )
                model_ans = (
                    f"Comparing '{c1.term}' ({c1.definition}) with '{c2.term}' ({c2.definition}): "
                    f"'{c1.term}' excels under regimes demanding rigorous consistency, whereas '{c2.term}' provides advantages in adaptability. "
                    f"The core trade-off involves balancing structural complexity against operational latency."
                )
        else:
            key_pts = [c1.term.lower(), c2.term.lower(), "compare", "difference"]
            if is_ja:
                prompt = f"「{c1.term}」と「{c2.term}」を比較対照し、それぞれの動作原理と講義における位置づけの違いを説明しなさい。"
                model_ans = f"「{c1.term}」は{c1.definition}であり、「{c2.term}」は{c2.definition}です。"
            else:
                prompt = f"Compare and contrast '{c1.term}' with '{c2.term}'. How do their mechanisms differ, and what distinct roles do they play in the lecture?"
                model_ans = f"'{c1.term}' is {c1.definition}, whereas '{c2.term}' is {c2.definition}."

        if prompt not in seen_prompts:
            questions.append(
                DescriptiveExamQuestion(
                    question_id=q_id,
                    question=prompt,
                    difficulty=diff_label,
                    key_points=key_pts,
                    model_answer=model_ans,
                    source_reference=c1.source_span,
                    concept_term=f"{c1.term} vs {c2.term}",
                )
            )
            seen_prompts.add(prompt)
            q_id += 1

    # ── ARCHETYPE 4: Failure Mode & Edge-Case Engineering ──
    for c in concepts:
        if len(questions) >= count:
            break
        key_pts = [c.term.lower(), "failure", "mitigation", "resilience"]
        if is_ja:
            prompt = (
                f"【障害モード・耐障害性分析】「{c.term}」を実装したシステムが、講義で述べられた想定外の入力データやノイズに直面した場合を想定しなさい。"
                f"理論的な破綻症状を診断し、講義の原則に基づいた具体的な是正・緩和プロトコルを提案しなさい。"
            )
            model_ans = (
                f"「{c.term}」の障害時は、{c.definition}の仮定が崩壊することで性能劣化が生じます。"
                f"是正プロトコルとして、入力の正規化、例外境界の監視、およびフォールバック機構を導入する必要があります。"
            )
        else:
            prompt = (
                f"【Failure Mode & Resilience Analysis】 Suppose an engineer implements a pipeline relying on '{c.term}', "
                f"but encounters input conditions violating core assumptions. Diagnose the likely theoretical failure symptoms "
                f"and propose a concrete mitigation protocol grounded in the lecture."
            )
            model_ans = (
                f"When '{c.term}' fails, violations of its core principle ({c.definition}) trigger systematic errors. "
                f"A robust mitigation protocol includes strict sanity checking, bounded fallback rules, and adaptive regularization."
            )

        if prompt not in seen_prompts:
            questions.append(
                DescriptiveExamQuestion(
                    question_id=q_id,
                    question=prompt,
                    difficulty=diff_label,
                    key_points=key_pts,
                    model_answer=model_ans,
                    source_reference=c.source_span,
                    concept_term=c.term,
                )
            )
            seen_prompts.add(prompt)
            q_id += 1

    # ── ARCHETYPE 5: Flashcard Applied Problem Solving ──
    for fc in flashcards:
        if len(questions) >= count:
            break
        key_pts = [w.lower() for w in fc.question.split() if len(w) > 4][:3] + ["mechanism", "application"]
        if is_ja:
            prompt = (
                f"【実践応用・問題解決】講義における問い「{fc.question}」について、単なる回答を超えて、"
                f"なぜその結論に至るのかの理論的根拠と、実務における応用上の留意点を論述しなさい。"
            )
            model_ans = f"核心的な回答は「{fc.answer}」です。これは講義の原則({fc.source_span})に基づいています。"
        else:
            prompt = (
                f"【Applied Problem Solving】 Regarding the lecture topic: \"{fc.question}\", explain the deep theoretical rationale "
                f"behind the solution and discuss practical implications for implementation."
            )
            model_ans = f"The core answer is: {fc.answer}. This is justified by the lecture rationale: \"{fc.source_span}\"."

        if prompt not in seen_prompts:
            questions.append(
                DescriptiveExamQuestion(
                    question_id=q_id,
                    question=prompt,
                    difficulty=diff_label,
                    key_points=key_pts,
                    model_answer=model_ans,
                    source_reference=fc.source_span,
                    concept_term=fc.concept or "Core Principles",
                )
            )
            seen_prompts.add(prompt)
            q_id += 1

    return questions[:count]


def evaluate_descriptive_answer(
    user_answer: str,
    question: DescriptiveExamQuestion,
) -> dict[str, Any]:
    """Evaluate a user's written response against expected key points and difficulty standards.

    Applies calibrated rubrics: Hard questions require higher substance, explicit trade-off
    or constraint discussions, and deeper vocabulary coverage.
    """
    clean_answer = (user_answer or "").strip()
    words = clean_answer.split()
    word_count = len(words)
    is_hard = question.difficulty == "Hard"

    # Minimum threshold based on difficulty
    min_words = 15 if is_hard else 8
    if word_count < min_words:
        msg = (
            f"Answer is too brief for an advanced analytical question ({word_count} words; minimum {min_words} words required). "
            f"Please provide an in-depth critique addressing theoretical assumptions, trade-offs, and failure modes."
            if is_hard
            else f"Answer is too brief ({word_count} words; minimum {min_words} words required). Please provide a complete conceptual explanation."
        )
        return {
            "score_pct": 0,
            "points_earned": 0,
            "max_points": 10,
            "word_count": word_count,
            "passed": False,
            "matched_key_points": [],
            "missed_key_points": question.key_points,
            "feedback": msg,
        }

    lower_ans = clean_answer.lower()
    matched = [kp for kp in question.key_points if kp.lower() in lower_ans]
    missed = [kp for kp in question.key_points if kp.lower() not in lower_ans]

    # Length score
    target_words = 50.0 if is_hard else 35.0
    len_score = min(35, int((word_count / target_words) * 35))

    # Concept coverage score: up to 55 points
    coverage_ratio = len(matched) / len(question.key_points) if question.key_points else 1.0
    coverage_score = int(coverage_ratio * 55)

    # Analytical depth bonus for Hard questions: up to 10 points
    depth_bonus = 0
    analytical_terms = [
        "tradeoff", "trade-off", "constraint", "limitation", "bottleneck",
        "assumption", "mitigation", "overhead", "failure", "efficiency",
        "complexity", "adversarial", "boundary", "トレードオフ", "制約", "前提", "ボトルネック"
    ]
    if is_hard:
        found_analytical = [t for t in analytical_terms if t in lower_ans]
        if found_analytical:
            depth_bonus = min(10, len(found_analytical) * 3)
    else:
        depth_bonus = 10  # Full bonus granted for standard/easy

    score_pct = min(100, len_score + coverage_score + depth_bonus)
    points_earned = round((score_pct / 100.0) * 10, 1)
    passed = score_pct >= 60

    if score_pct >= 85:
        feedback = f"🌟 Outstanding response! You demonstrated advanced conceptual rigor ({len(matched)} key points covered) with thorough analytical depth."
    elif score_pct >= 60:
        advice = f" To achieve full marks on this {question.difficulty.lower()} question, also address: {', '.join(missed[:2])}." if missed else ""
        feedback = f"👍 Solid attempt ({len(matched)} key points covered).{advice}"
    else:
        advice = f" Key concepts to include: {', '.join(missed[:3])}." if missed else ""
        feedback = f"⚠️ Needs greater analytical depth ({word_count} words). Make sure to elaborate on mechanisms, trade-offs, and failure mitigations.{advice}"

    return {
        "score_pct": score_pct,
        "points_earned": points_earned,
        "max_points": 10,
        "word_count": word_count,
        "passed": passed,
        "matched_key_points": matched,
        "missed_key_points": missed,
        "feedback": feedback,
    }


def generate_exam_questions_parallel(
    output: LectureOutput,
    mcq_count: int = 5,
    desc_count: int = 3,
    difficulty: str = "All",
    language: str = "English",
    llm_client: Any = None,
    use_llm: bool = False,
) -> tuple[list[MCQQuestion], list[DescriptiveExamQuestion]]:
    """Generate MCQ and descriptive questions.

    When use_llm is False (default Instant Mode):
        Generates calibrated questions instantly (<0.01s) directly from lecture
        concepts, definitions, and takeaways with zero API wait time.
    When use_llm is True:
        Runs parallel LLM generation calls using a thread pool.

    Returns:
        (mcqs, descriptives) — both lists may be empty if generation fails.
    """
    if not use_llm:
        mcqs = (
            generate_expanded_mcqs(
                output, mcq_count, difficulty, language, None, use_llm=False
            )
            if mcq_count > 0
            else []
        )
        descs = (
            generate_descriptive_questions(
                output, desc_count, difficulty, language, None, use_llm=False
            )
            if desc_count > 0
            else []
        )
        return mcqs, descs

    mcqs: list[MCQQuestion] = []
    descs: list[DescriptiveExamQuestion] = []

    tasks: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        if mcq_count > 0:
            tasks["mcq"] = executor.submit(
                generate_expanded_mcqs,
                output,
                mcq_count,
                difficulty,
                language,
                llm_client,
                True,
            )
        if desc_count > 0:
            tasks["desc"] = executor.submit(
                generate_descriptive_questions,
                output,
                desc_count,
                difficulty,
                language,
                llm_client,
                True,
            )

        for key, future in tasks.items():
            try:
                result = future.result()
                if key == "mcq":
                    mcqs = result
                else:
                    descs = result
            except Exception as exc:
                logger.warning("Parallel exam generation task '%s' failed: %s", key, exc)

    return mcqs, descs
