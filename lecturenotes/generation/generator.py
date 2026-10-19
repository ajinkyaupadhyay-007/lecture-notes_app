"""Orchestrator for structured lecture material generation.

Wires together prompts, LLM client, and Pydantic validation schemas.
"""

from __future__ import annotations

import logging
from typing import Sequence

from lecturenotes.concepts.models import KeyConcept
from lecturenotes.config import load_config
from lecturenotes.generation.llm_client import LLMClient
from lecturenotes.generation.prompts import (
    build_definitions_prompt,
    build_flashcards_prompt,
    build_quiz_prompt,
    build_section_notes_prompt,
)
from lecturenotes.generation.schemas import (
    ConceptDefinition,
    ConceptDefinitionsOutput,
    Flashcard,
    FlashcardsOutput,
    LectureOutput,
    MCQQuestion,
    QuizOutput,
    SectionNote,
    SectionNotesOutput,
    SectionSummaryItem,
)
from lecturenotes.segmentation.models import TopicSection

logger = logging.getLogger(__name__)


def generate_section_note(
    section: TopicSection,
    llm: LLMClient,
    language: str = "English",
) -> SectionNote:
    """Generate structured summary points with source spans for a single topic section."""
    prompt = build_section_notes_prompt(
        section_id=section.section_id,
        section_title=section.title,
        section_text=section.text,
        language=language,
    )
    raw_data = llm.generate_json(prompt)
    validated = SectionNotesOutput.model_validate(raw_data)

    return SectionNote(
        section_id=section.section_id,
        title=section.title,
        source_reference=section.source_reference,
        summary_points=validated.summary_points,
    )


def generate_concept_definitions(
    concepts: Sequence[KeyConcept],
    lecture_text: str,
    llm: LLMClient,
    max_terms: int = 15,
    language: str = "English",
) -> list[ConceptDefinition]:
    """Generate definitions with source spans for the top extracted technical concepts."""
    top_terms = [c.term for c in concepts[:max_terms]]
    if not top_terms:
        return []

    prompt = build_definitions_prompt(top_terms, lecture_text, language=language)
    raw_data = llm.generate_json(prompt)
    validated = ConceptDefinitionsOutput.model_validate(raw_data)
    return validated.definitions


def generate_flashcards(
    lecture_text: str,
    concepts: Sequence[KeyConcept],
    llm: LLMClient,
    target_count: int | None = None,
    language: str = "English",
) -> list[Flashcard]:
    """Generate 10 to 20 flashcards with answers and source spans."""
    cfg = load_config()
    count = target_count or cfg.generation.flashcard_count
    concept_names = [c.term for c in concepts]

    prompt = build_flashcards_prompt(lecture_text, concept_names, target_count=count, language=language)
    raw_data = llm.generate_json(prompt)
    validated = FlashcardsOutput.model_validate(raw_data)
    return validated.flashcards


def generate_quiz(
    lecture_text: str,
    concepts: Sequence[KeyConcept],
    llm: LLMClient,
    question_count: int | None = None,
    difficulty: str = "Medium",
    language: str = "English",
    sections: list[TopicSection] | None = None,
) -> list[MCQQuestion]:
    """Generate a multiple choice quiz with distractors, explanations, and specified difficulty."""
    cfg = load_config()
    count = question_count or cfg.generation.mcq_count
    concept_names = [c.term for c in concepts]
    diff_norm = difficulty.capitalize() if difficulty else "Medium"

    try:
        prompt = build_quiz_prompt(
            lecture_text,
            concept_names,
            question_count=count,
            difficulty=diff_norm,
            language=language,
        )
        raw_data = llm.generate_json(prompt)
        validated = QuizOutput.model_validate(raw_data)
        if validated.quiz:
            if diff_norm in ("Easy", "Medium", "Hard"):
                for q in validated.quiz:
                    q.difficulty = diff_norm
            return validated.quiz
    except Exception as exc:
        logger.warning("generate_quiz failed (%s); generating fallback quiz", exc)

    fallback_quiz = _fallback_quiz(
        sections=sections or [],
        concepts=concepts,
        question_count=count,
        difficulty=diff_norm,
        language=language,
    )
    if diff_norm in ("Easy", "Medium", "Hard"):
        for q in fallback_quiz:
            q.difficulty = diff_norm
    return fallback_quiz


def generate_lecture_materials(
    sections: list[TopicSection],
    concepts: list[KeyConcept],
    llm: LLMClient | None = None,
    lecture_title: str = "Lecture Notes",
    target_flashcards: int | None = None,
    target_mcqs: int | None = None,
    language: str = "English",
    quiz_difficulty: str = "Medium",
) -> LectureOutput:
    """Execute complete generation pipeline producing validated LectureOutput."""
    client = llm or LLMClient()
    logger.info(
        "Generating lecture materials for %d sections, %d concepts (lang: %s, difficulty: %s)",
        len(sections),
        len(concepts),
        language,
        quiz_difficulty,
    )

    # Combine text for holistic prompts
    full_lecture_text = "\n\n".join(f"[{s.title}]\n{s.text}" for s in sections)

    # 1. Section notes in parallel
    def _gen_single_note(sec: TopicSection) -> SectionNote:
        try:
            return generate_section_note(sec, client, language=language)
        except Exception as e:
            logger.warning("LLM call failed for section %d (%s); using extractive fallback", sec.section_id, e)
            fallback_point = SectionSummaryItem(
                summary_sentence=sec.sentences[0].text if sec.sentences else "Summary unavailable.",
                source_span=sec.sentences[0].text if sec.sentences else "",
            )
            return SectionNote(
                section_id=sec.section_id,
                title=sec.title,
                source_reference=sec.source_reference,
                summary_points=[fallback_point],
            )

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(6, max(1, len(sections)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        section_notes = list(executor.map(_gen_single_note, sections))

    # 2, 3, 4. Concept definitions, Flashcards, and Quiz in parallel
    tf = target_flashcards or (client.config.generation.flashcard_count if hasattr(client, "config") else 15)
    tq = target_mcqs or (client.config.generation.mcq_count if hasattr(client, "config") else 5)

    def _fetch_definitions() -> list[ConceptDefinition]:
        try:
            return generate_concept_definitions(concepts, full_lecture_text, client, language=language)
        except Exception as e:
            logger.warning("Failed to generate definitions (%s); generating fallback definitions", e)
            return _fallback_concept_definitions(concepts, language=language)

    def _fetch_flashcards() -> list[Flashcard]:
        try:
            cards = generate_flashcards(full_lecture_text, concepts, client, target_count=tf, language=language)
            return cards if cards else _fallback_flashcards(sections, concepts, target_count=tf, language=language)
        except Exception as e:
            logger.warning("Failed to generate flashcards (%s); generating fallback flashcards", e)
            return _fallback_flashcards(sections, concepts, target_count=tf, language=language)

    def _fetch_quiz() -> list[MCQQuestion]:
        try:
            q = generate_quiz(full_lecture_text, concepts, client, question_count=tq, difficulty=quiz_difficulty, language=language)
            return q if q else _fallback_quiz(sections, concepts, question_count=tq, difficulty=quiz_difficulty, language=language)
        except Exception as e:
            logger.warning("Failed to generate quiz (%s); generating fallback quiz", e)
            return _fallback_quiz(sections, concepts, question_count=tq, difficulty=quiz_difficulty, language=language)

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_defs = pool.submit(_fetch_definitions)
        fut_fc = pool.submit(_fetch_flashcards)
        fut_quiz = pool.submit(_fetch_quiz)

        definitions = fut_defs.result()
        flashcards = fut_fc.result()
        quiz = fut_quiz.result()

    return LectureOutput(
        lecture_title=lecture_title,
        section_notes=section_notes,
        concept_definitions=definitions,
        flashcards=flashcards,
        quiz=quiz,
    )


def _fallback_concept_definitions(
    concepts: Sequence[KeyConcept],
    max_terms: int = 15,
    language: str = "English",
) -> list[ConceptDefinition]:
    """Extractive fallback for concept definitions using identified source sentences."""
    definitions: list[ConceptDefinition] = []
    seen = set()
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    for c in concepts[:max_terms]:
        if c.term.lower() in seen:
            continue
        seen.add(c.term.lower())
        source = c.source_sentence or f"Key concept: {c.term}."
        if is_ja:
            defn_text = f"「{c.term}」は講義内で次のように説明されています: {source.strip()}"
        else:
            defn_text = f"{c.term.capitalize()} is discussed as follows: {source.strip()}"

        definitions.append(
            ConceptDefinition(
                term=c.term,
                definition=defn_text,
                source_span=source.strip(),
            )
        )
    return definitions


def _fallback_flashcards(
    sections: list[TopicSection],
    concepts: Sequence[KeyConcept],
    target_count: int = 15,
    language: str = "English",
) -> list[Flashcard]:
    """Deterministic fallback flashcards created from extracted concepts and section sentences."""
    cards: list[Flashcard] = []
    card_id = 1
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    for c in concepts:
        if card_id > target_count:
            break
        if not c.source_sentence or len(c.source_sentence.strip()) < 15:
            continue

        q_text = f"「{c.term}」について講義ではどのように説明されていますか？" if is_ja else f"What does the lecture explain regarding '{c.term}'?"
        cards.append(
            Flashcard(
                card_id=card_id,
                question=q_text,
                answer=c.source_sentence.strip(),
                source_span=c.source_sentence.strip(),
                concept=c.term,
            )
        )
        card_id += 1

    if card_id <= target_count:
        for s in sections:
            for sent in s.sentences:
                if card_id > target_count:
                    break
                text = sent.text.strip()
                if len(text) < 25:
                    continue
                q_text = f"セクション「{s.title}」の要点:" if is_ja else f"Key takeaway from '{s.title}':"
                cards.append(
                    Flashcard(
                        card_id=card_id,
                        question=q_text,
                        answer=text,
                        source_span=text,
                        concept=s.title,
                    )
                )
                card_id += 1

    return cards


def _fallback_quiz(
    sections: list[TopicSection],
    concepts: Sequence[KeyConcept],
    question_count: int = 5,
    difficulty: str = "Medium",
    language: str = "English",
) -> list[MCQQuestion]:
    """Deterministic fallback multiple-choice quiz questions with grounded source spans and difficulty."""
    quiz: list[MCQQuestion] = []
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")
    diff_norm = difficulty.capitalize() if difficulty else "Medium"

    valid_concepts = [c for c in concepts if getattr(c, "source_sentence", None) and len(c.source_sentence.strip()) > 15]
    if len(valid_concepts) < 4 and concepts:
        valid_concepts = list(concepts)
        for c in valid_concepts:
            if not getattr(c, "source_sentence", None) or len(c.source_sentence.strip()) <= 15:
                c.source_sentence = f"The fundamental principle of {c.term} serves as a critical mechanism in this lecture domain."

    if len(valid_concepts) >= 4:
        for q_idx in range(question_count):
            c_idx = q_idx % len(valid_concepts)
            target_c = valid_concepts[c_idx]
            other_c = [c for j, c in enumerate(valid_concepts) if j != c_idx]
            distractors = other_c[:3]

            curr_diff = diff_norm if diff_norm in ("Easy", "Hard") else ("Easy" if q_idx == 0 else ("Hard" if q_idx >= 3 else "Medium"))

            if curr_diff == "Hard":
                if is_ja:
                    q_prompt = f"講義内容に基づき、概念「{target_c.term}」を適用する際の重要な制約・アルゴリズム上のトレードオフ、または前提条件として最も適切なものはどれですか？"
                    expl = f"「{target_c.term}」の適用には次の厳密な性質が関与します: {target_c.source_sentence}"
                    d_expl = {
                        "B": f"「{distractors[0].term if len(distractors) > 0 else '選択肢B'}」の性質と混同しています。",
                        "C": f"「{distractors[1].term if len(distractors) > 1 else '選択肢C'}」の仮定を誤って適用しています。",
                        "D": "講義テキストで支持されない非現実的な仮定です。",
                    }
                    opts = {
                        "A": f"「{target_c.source_sentence[:75]}…」という性質を満たす必要がある",
                        "B": f"「{distractors[0].term if len(distractors) > 0 else '手法B'}」と同等の凸最適化収束特性を仮定する",
                        "C": f"「{distractors[1].term if len(distractors) > 1 else '手法C'}」におけるハイパーパラメータ正則化を完全に排除する",
                        "D": "局所最適解に陥ることなく大域的最適性を無条件で保証する",
                    }
                else:
                    q_prompt = f"Based on the lecture, which fundamental algorithmic tradeoff, constraint, or boundary condition governs '{target_c.term}'?"
                    expl = f"Applying '{target_c.term}' requires understanding its foundational boundary condition: {target_c.source_sentence}"
                    d_expl = {
                        "B": f"Incorrectly assumes convergence properties of '{distractors[0].term if len(distractors) > 0 else 'Option B'}'.",
                        "C": f"Conflates structural constraints with '{distractors[1].term if len(distractors) > 1 else 'Option C'}'.",
                        "D": "Asserts an invalid unconstrained optimization assumption.",
                    }
                    opts = {
                        "A": f"It is bounded by the operational principle: \"{target_c.source_sentence[:80]}...\"",
                        "B": f"It assumes identical loss convexity to {distractors[0].term if len(distractors) > 0 else 'competing approaches'}",
                        "C": f"It eliminates regularization parameters unlike {distractors[1].term if len(distractors) > 1 else 'traditional models'}",
                        "D": "It unconditionally guarantees global empirical risk minimization without sample bias",
                    }
            elif is_ja:
                q_prompt = f"次の説明に直接対応する概念はどれですか: 「{target_c.source_sentence}」"
                expl = f"「{target_c.term}」はこの講義記述によって定義されている中心概念です。"
                d_expl = {
                    "B": f"「{distractors[0].term if len(distractors) > 0 else '選択肢B'}」は別のトピックを指します。",
                    "C": f"「{distractors[1].term if len(distractors) > 1 else '選択肢C'}」は異なる概念です。",
                    "D": f"「{distractors[2].term if len(distractors) > 2 else '選択肢D'}」はこの記述に該当しません。",
                }
                opts = {
                    "A": target_c.term.title(),
                    "B": distractors[0].term.title() if len(distractors) > 0 else "Decision Trees",
                    "C": distractors[1].term.title() if len(distractors) > 1 else "Neural Networks",
                    "D": distractors[2].term.title() if len(distractors) > 2 else "Support Vector Machines",
                }
            else:
                q_prompt = f"Which concept is directly associated with the statement: \"{target_c.source_sentence}\"?"
                expl = f"'{target_c.term}' is the concept defined by this statement in the lecture."
                d_expl = {
                    "B": f"'{distractors[0].term if len(distractors) > 0 else 'Option B'}' refers to another topic.",
                    "C": f"'{distractors[1].term if len(distractors) > 1 else 'Option C'}' is a distinct concept.",
                    "D": f"'{distractors[2].term if len(distractors) > 2 else 'Option D'}' does not describe this statement.",
                }
                opts = {
                    "A": target_c.term.title(),
                    "B": distractors[0].term.title() if len(distractors) > 0 else "Decision Trees",
                    "C": distractors[1].term.title() if len(distractors) > 1 else "Neural Networks",
                    "D": distractors[2].term.title() if len(distractors) > 2 else "Support Vector Machines",
                }

            quiz.append(
                MCQQuestion(
                    question_id=q_idx + 1,
                    question=q_prompt,
                    options=opts,
                    correct_answer="A",
                    explanation=expl,
                    distractor_explanations=d_expl,
                    source_span=target_c.source_sentence,
                    difficulty=curr_diff,
                )
            )
        return quiz

    # Alternatively generate from section sentences
    all_sentences = [
        sent.text.strip()
        for s in sections
        for sent in s.sentences
        if len(sent.text.strip()) > 30
    ]
    for i in range(min(question_count, max(1, len(all_sentences)))):
        target = all_sentences[i] if i < len(all_sentences) else "Supervised learning uses labelled data."
        other = [s for j, s in enumerate(all_sentences) if j != i][:3]
        while len(other) < 3:
            other.append(f"Alternative hypothesis {len(other) + 1}")

        curr_diff = diff_norm if diff_norm in ("Easy", "Hard") else ("Easy" if i == 0 else "Medium")
        if is_ja:
            q_stem = "講義テキストによって直接裏付けられている記述はどれですか？"
            expl = "選択肢Aは講義の検証済みテキストに直接記述されています。"
            d_expl = {
                "B": "これは別の文脈からの記述です。",
                "C": "この選択肢は該当セクションで支持されていません。",
                "D": "この選択肢は異なる主張です。",
            }
        else:
            q_stem = "Which statement is directly supported by the lecture text?"
            expl = "Option A is directly stated in the verified lecture text."
            d_expl = {
                "B": "This represents a distinct statement from elsewhere in the material.",
                "C": "This option is not supported by the relevant section context.",
                "D": "This statement is an alternative claim.",
            }

        quiz.append(
            MCQQuestion(
                question_id=i + 1,
                question=q_stem,
                options={
                    "A": target,
                    "B": other[0],
                    "C": other[1],
                    "D": other[2],
                },
                correct_answer="A",
                explanation=expl,
                distractor_explanations=d_expl,
                source_span=target,
                difficulty=curr_diff,
            )
        )
    return quiz

