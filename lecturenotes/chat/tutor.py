"""Interactive Grounded AI Lecture Tutor.

Enables interactive student Q&A, conceptual deep-dives, and Socratic clarifications
grounded strictly in the uploaded lecture materials.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lecturenotes.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class LectureTutor:
    """An interactive learning assistant grounded in the active lecture's notes and concepts."""

    def __init__(
        self,
        lecture_title: str,
        sections: list[Any] | None = None,
        concepts: list[Any] | None = None,
        concept_definitions: list[Any] | None = None,
        section_notes: list[Any] | None = None,
        flashcards: list[Any] | None = None,
        quiz: list[Any] | None = None,
        language: str = "English",
        llm: LLMClient | None = None,
    ) -> None:
        self.lecture_title = lecture_title or "Lecture"
        self.sections = sections or []
        self.concepts = concepts or []
        self.concept_definitions = concept_definitions or []
        self.section_notes = section_notes or []
        self.flashcards = flashcards or []
        self.quiz = quiz or []
        self.language = language
        self.is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")
        self.llm = llm

    def _build_lecture_context(self, max_chars: int = 5000) -> str:
        """Compile a compact, high-signal knowledge context from generated lecture materials."""
        parts: list[str] = [f"LECTURE TITLE: {self.lecture_title}"]

        if self.concept_definitions:
            parts.append("\nKEY CONCEPTS & DEFINITIONS:")
            for d in self.concept_definitions[:12]:
                term = getattr(d, "term", "")
                defn = getattr(d, "definition", "")
                source = getattr(d, "source_span", "")
                parts.append(f"- {term}: {defn} (Source: \"{source}\")")

        if self.section_notes:
            parts.append("\nSECTION SUMMARY NOTES:")
            for s in self.section_notes[:6]:
                title = getattr(s, "title", f"Section {getattr(s, 'section_id', '')}")
                ref = getattr(s, "source_reference", "")
                pts = getattr(s, "summary_points", [])
                pt_texts = "; ".join(getattr(pt, "summary_sentence", "") for pt in pts[:3])
                parts.append(f"- [{title}{f' {ref}' if ref else ''}]: {pt_texts}")

        if self.flashcards:
            parts.append("\nREPRESENTATIVE FLASHCARDS:")
            for fc in self.flashcards[:5]:
                q = getattr(fc, "question", "")
                a = getattr(fc, "answer", "")
                parts.append(f"- Q: {q} | A: {a}")

        full_text = "\n".join(parts)
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n...(truncated for context limit)"
        return full_text

    def answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a pedagogical, factual response to a student question."""
        clean_q = question.strip()
        if not clean_q:
            return (
                "講義について何でも質問してください！"
                if self.is_ja
                else "Please ask any question about this lecture! I'm here to explain concepts, give analogies, or quiz your recall."
            )

        # 1. Try LLM generation if available
        if self.llm and getattr(self.llm, "provider", "").lower() != "mock":
            try:
                context_str = self._build_lecture_context()
                history_str = ""
                if chat_history:
                    recent = chat_history[-4:]
                    hist_lines = [f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in recent]
                    history_str = "\nPREVIOUS CONVERSATION:\n" + "\n".join(hist_lines) + "\n"

                lang_instruction = (
                    "Respond exclusively in natural, polite Japanese (日本語)."
                    if self.is_ja
                    else "Respond in clear, engaging English."
                )

                system_prompt = (
                    "You are the LectureNotes AI Tutor, an encouraging, pedagogically skilled study coach. "
                    "Your goal is to answer student questions based strictly on the provided lecture materials. "
                    "Provide clear intuitive explanations, highlight core formulas or rules, use illuminating real-world analogies, "
                    "and always cite the relevant source concept or lecture section when possible. "
                    f"{lang_instruction}"
                )

                user_prompt = (
                    f"LECTURE CONTEXT:\n{context_str}\n"
                    f"{history_str}\n"
                    f"STUDENT QUESTION:\n{clean_q}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Provide a direct, crystal-clear explanation.\n"
                    "2. Include an intuitive analogy if conceptual.\n"
                    "3. Ground your explanation in the lecture content.\n"
                    "4. Add a quick bulleted takeaway or summary at the end."
                )

                if hasattr(self.llm, "generate_text"):
                    resp = self.llm.generate_text(user_prompt, system_prompt=system_prompt)
                    if resp and len(resp.strip()) > 20:
                        return resp.strip()
            except Exception as exc:
                logger.warning("LLM text generation failed in LectureTutor; falling back to grounded extractive answering: %s", exc)

        # 2. Deterministic Grounded Extractive Answering Fallback
        return self._extractive_grounded_answer(clean_q)

    def _extractive_grounded_answer(self, question: str) -> str:
        """Generate a grounded, high-signal answer using rule-based concept matching."""
        q_lower = question.lower()
        q_tokens = set(re.findall(r"\w+", q_lower))

        matched_defns = []
        for d in self.concept_definitions:
            term = getattr(d, "term", "").lower()
            term_tokens = set(re.findall(r"\w+", term))
            if term in q_lower or (term_tokens and term_tokens.issubset(q_tokens)):
                matched_defns.append(d)

        matched_notes = []
        for s in self.section_notes:
            for pt in getattr(s, "summary_points", []):
                sent = getattr(pt, "summary_sentence", "").lower()
                overlap = len(set(re.findall(r"\w+", sent)).intersection(q_tokens))
                if overlap >= 2:
                    matched_notes.append((overlap, s, pt))
        matched_notes.sort(key=lambda x: x[0], reverse=True)

        matched_cards = []
        for fc in self.flashcards:
            q_text = getattr(fc, "question", "").lower()
            overlap = len(set(re.findall(r"\w+", q_text)).intersection(q_tokens))
            if overlap >= 2 or (getattr(fc, "concept", "") and getattr(fc, "concept", "").lower() in q_lower):
                matched_cards.append((overlap, fc))
        matched_cards.sort(key=lambda x: x[0], reverse=True)

        if self.is_ja:
            lines: list[str] = []
            if matched_defns:
                d = matched_defns[0]
                lines.append(f"### 💡 **{d.term}** の解説\n")
                lines.append(f"{d.definition}\n")
                lines.append(f"> 📎 **講義引用**: *\"{d.source_span}\"*\n")
            elif matched_notes:
                _, s, pt = matched_notes[0]
                lines.append(f"### 💡 講義セクション「{s.title}」より\n")
                lines.append(f"**要約**: {pt.summary_sentence}\n")
                lines.append(f"> 📎 *\"{pt.source_span}\"*\n")
            elif matched_cards:
                _, fc = matched_cards[0]
                lines.append(f"### 💡 関連フラッシュカード (Q: {fc.question})\n")
                lines.append(f"**回答**: {fc.answer}\n")
                lines.append(f"> 📎 *\"{fc.source_span}\"*\n")
            else:
                lines.append(f"### 📚 「{self.lecture_title}」に関するまとめ\n")
                lines.append("この講義で抽出された重要概念は以下の通りです:\n")
                for d in self.concept_definitions[:4]:
                    lines.append(f"- **{d.term}**: {d.definition[:90]}…")
                lines.append("\n気になる用語や概念があれば、その名前を指定して質問してください！")

            lines.append("\n---\n**✨ 学習のヒント**: 疑問点があればさらに詳しく質問したり、フラッシュカードで復習してみましょう。")
            return "\n".join(lines)

        else:
            lines = []
            if matched_defns:
                d = matched_defns[0]
                lines.append(f"### 💡 Core Concept: **{d.term.title()}**\n")
                lines.append(f"{d.definition}\n")
                lines.append(f"> 📎 **Lecture Citation**: *\"{d.source_span}\"*\n")
                lines.append("#### 🔍 Intuitive Breakdown & Context:")
                lines.append(f"- **What it represents**: A foundational component in understanding *{self.lecture_title}*.")
                lines.append("- **Practical takeaway**: Pay close attention to how this principle interacts with the rest of the lecture's data and mechanics.")
            elif matched_notes:
                _, s, pt = matched_notes[0]
                lines.append(f"### 💡 From Section: **{s.title}**\n")
                lines.append(f"{pt.summary_sentence}\n")
                lines.append(f"> 📎 **Source Reference**: *\"{pt.source_span}\"*\n")
                lines.append("#### 📌 Key Takeaway:")
                lines.append(f"- This point is integral to the overarching discussion of *{self.lecture_title}*.")
            elif matched_cards:
                _, fc = matched_cards[0]
                lines.append(f"### 💡 Review Insight (Q: {fc.question})\n")
                lines.append(f"**Direct Answer**: {fc.answer}\n")
                lines.append(f"> 📎 **Citation**: *\"{fc.source_span}\"*\n")
            else:
                lines.append(f"### 📚 Overview of **{self.lecture_title}**\n")
                lines.append("Here are the top key concepts covered in this document:\n")
                for d in self.concept_definitions[:4]:
                    lines.append(f"- **{d.term.title()}**: {d.definition[:100]}…")
                lines.append("\nFeel free to ask about any specific concept, section, or request an intuitive analogy!")

            lines.append("\n---\n*💡 **Study Tip**: Ask me to give you a real-world analogy or test your understanding with a quick challenge question!*")
            return "\n".join(lines)

    def get_deep_dive(self, term_or_concept: str) -> dict[str, str]:
        """Generate a structured, in-depth breakdown of a concept for flashcard expansion."""
        target = (term_or_concept or "").strip()
        defn_match = next(
            (d for d in self.concept_definitions if d.term.lower() == target.lower()),
            None,
        )
        if not defn_match:
            defn_match = next(
                (d for d in self.concept_definitions if target.lower() in d.term.lower() or d.term.lower() in target.lower()),
                None,
            )

        definition_text = getattr(defn_match, "definition", f"Key topic in {self.lecture_title}.") if defn_match else f"A foundational mechanism in {self.lecture_title}."
        source_span = getattr(defn_match, "source_span", "Grounded in lecture text.") if defn_match else "Source material citation."

        if self.is_ja:
            return {
                "concept": target,
                "intuition": f"「{target}」は、日常の例えで言えば、経験や反復を通じて最適な判断を下すための『指標や判断基準』に相当します。",
                "mechanism": f"{definition_text} 講義では具体的な数式や入力データの関係性とともに解説されています。",
                "real_world": f"自動運転、推薦アルゴリズム、画像認識など、現代のAI・情報技術の幅広い現場で実際に活用されています。",
                "takeaway": "試験や実践では、前提条件と何を入力とし何を出力するかを明確に区別して覚えることがポイントです。",
                "source_span": source_span,
            }
        else:
            return {
                "concept": target.title(),
                "intuition": f"Think of **{target.title()}** like a tailored compass or rulebook: it provides a structured way to evaluate data, make decisions, and guide system behavior.",
                "mechanism": f"{definition_text} Within the lecture context, this mechanism forms an essential bridge connecting raw inputs to verifiable predictions or outputs.",
                "real_world": f"Widely applied across machine learning pipelines, production analytics, search ranking, and intelligent system automation.",
                "takeaway": f"Focus on understanding the core inputs, the transformation process, and why this concept was introduced in {self.lecture_title}.",
                "source_span": source_span,
            }

    def get_starter_questions(self) -> list[str]:
        """Generate tailored starter questions based on the active lecture content."""
        concept_terms = [getattr(c, "term", "") for c in self.concepts if getattr(c, "term", "")]
        c1 = concept_terms[0].title() if len(concept_terms) > 0 else "the core topic"
        c2 = concept_terms[1].title() if len(concept_terms) > 1 else "key concepts"

        if self.is_ja:
            return [
                f"「{c1}」について、初心者向けに分かりやすい例えで説明してください。",
                f"この講義で最も重要なポイント（結論）は何ですか？",
                f"「{c1}」と「{c2}」の違いは何ですか？",
                "この講義内容から、3問の確認クイズを出題してください！",
            ]
        else:
            return [
                f"Can you explain '{c1}' using an intuitive real-world analogy?",
                "What is the single most important takeaway from this lecture?",
                f"What is the key difference between '{c1}' and '{c2}'?",
                "Quiz me with 3 rapid questions to test my understanding!",
            ]
