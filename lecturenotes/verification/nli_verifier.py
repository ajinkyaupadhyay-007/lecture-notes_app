"""NLI-based factuality verifier using cross-encoder/nli-deberta-v3-base.

Checks whether each generated claim (summary_sentence, flashcard answer, definition,
quiz explanation) is entailed by its attributed source_span using a cross-encoder NLI model.
Strictly offline — no LLM calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lecturenotes.config import load_config
from lecturenotes.generation.schemas import (
    ConceptDefinition,
    Flashcard,
    LectureOutput,
    MCQQuestion,
    SectionNote,
)
from lecturenotes.verification.models import (
    FactualityReport,
    VerificationLabel,
    VerificationResult,
)

logger = logging.getLogger(__name__)

_nli_model: Any = None


def _resolve_nli_model_path(model_name: str) -> str:
    """Return local HuggingFace snapshot path for the NLI model if available.

    Falls back to the original model_name (HF Hub identifier) if no local snapshot
    is found, which will trigger a network download on first use.
    """
    # Normalise model name to HF Hub cache directory name
    safe_name = model_name.replace("/", "--")
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hf_cache / f"models--{safe_name}"
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            resolved = str(snapshots[-1])
            logger.info("NLI model resolved to local snapshot: %s", resolved)
            return resolved
    logger.info("NLI model not cached locally; will attempt download: %s", model_name)
    return model_name


def get_nli_model(model_name: str | None = None) -> Any:
    """Load or return cached CrossEncoder NLI model.

    Args:
        model_name: HuggingFace model identifier. Defaults to config value.

    Returns:
        A sentence_transformers CrossEncoder instance.
    """
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        if not model_name:
            model_name = load_config().nli.model
        resolved = _resolve_nli_model_path(model_name)
        logger.info("Loading NLI cross-encoder: %s", resolved)
        # num_labels=3: [contradiction, entailment, neutral] — DeBERTa NLI label order
        _nli_model = CrossEncoder(resolved, num_labels=3)
    return _nli_model


def _score_to_label(scores: list[float], threshold: float) -> tuple[float, VerificationLabel]:
    """Convert raw NLI logit scores to (entailment_prob, VerificationLabel).

    DeBERTa NLI label order: [contradiction=0, entailment=1, neutral=2].

    Args:
        scores: Softmax probabilities for [contradiction, entailment, neutral].
        threshold: Minimum entailment probability to classify as ENTAILED.

    Returns:
        Tuple of (entailment_score, VerificationLabel).
    """
    import numpy as np

    probs = np.array(scores, dtype=float)
    probs = probs / probs.sum()  # ensure normalised

    entailment_prob = float(probs[1])  # index 1 = entailment for DeBERTa NLI

    if entailment_prob >= threshold:
        return entailment_prob, VerificationLabel.ENTAILED
    # Contradiction dominates if contradiction prob > neutral prob
    if float(probs[0]) > float(probs[2]):
        return entailment_prob, VerificationLabel.CONTRADICTION
    return entailment_prob, VerificationLabel.NEUTRAL


def verify_claim(
    claim: str,
    source_span: str,
    item_type: str,
    item_id: str,
    *,
    model: Any | None = None,
    threshold: float | None = None,
) -> VerificationResult:
    """Run NLI cross-encoder to verify that source_span entails the claim.

    Args:
        claim: The generated sentence / answer to verify.
        source_span: The exact lecture text attributed as evidence.
        item_type: One of "note", "definition", "flashcard", "quiz".
        item_id: Identifier string for the item.
        model: Pre-loaded CrossEncoder (loaded lazily if None).
        threshold: Entailment threshold (loaded from config if None).

    Returns:
        A VerificationResult with score and label.
    """
    if model is None:
        model = get_nli_model()
    if threshold is None:
        threshold = load_config().nli.entailment_threshold

    # CrossEncoder expects (premise, hypothesis) — premise is the source span
    raw_scores = model.predict([[source_span, claim]])
    scores_list = raw_scores[0].tolist() if hasattr(raw_scores[0], "tolist") else list(raw_scores[0])

    entailment_score, label = _score_to_label(scores_list, threshold)

    return VerificationResult(
        claim=claim,
        source_span=source_span,
        entailment_score=entailment_score,
        label=label,
        item_type=item_type,  # type: ignore[arg-type]
        item_id=item_id,
    )


def verify_lecture_output(
    output: LectureOutput,
    *,
    model: Any | None = None,
    threshold: float | None = None,
) -> FactualityReport:
    """Verify all generated claims in a LectureOutput against their source spans.

    Iterates over:
    - section_notes → each summary_sentence vs its source_span
    - concept_definitions → definition vs source_span
    - flashcards → answer vs source_span
    - quiz → explanation vs source_span

    Args:
        output: The fully generated LectureOutput.
        model: Pre-loaded NLI CrossEncoder (loaded lazily if None).
        threshold: Entailment threshold (loaded from config if None).

    Returns:
        A FactualityReport aggregating all VerificationResult objects.
    """
    if model is None:
        model = get_nli_model()
    if threshold is None:
        threshold = load_config().nli.entailment_threshold

    results: list[VerificationResult] = []

    # 1. Section notes
    for note in output.section_notes:
        for point in note.summary_points:
            try:
                result = verify_claim(
                    claim=point.summary_sentence,
                    source_span=point.source_span,
                    item_type="note",
                    item_id=str(note.section_id),
                    model=model,
                    threshold=threshold,
                )
                results.append(result)
            except Exception as exc:
                logger.warning("Verification failed for note %s: %s", note.section_id, exc)

    # 2. Concept definitions
    for defn in output.concept_definitions:
        try:
            result = verify_claim(
                claim=defn.definition,
                source_span=defn.source_span,
                item_type="definition",
                item_id=defn.term,
                model=model,
                threshold=threshold,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Verification failed for definition '%s': %s", defn.term, exc)

    # 3. Flashcards
    for card in output.flashcards:
        try:
            result = verify_claim(
                claim=card.answer,
                source_span=card.source_span,
                item_type="flashcard",
                item_id=str(card.card_id),
                model=model,
                threshold=threshold,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Verification failed for flashcard %s: %s", card.card_id, exc)

    # 4. Quiz questions
    for question in output.quiz:
        try:
            result = verify_claim(
                claim=question.explanation,
                source_span=question.source_span,
                item_type="quiz",
                item_id=str(question.question_id),
                model=model,
                threshold=threshold,
            )
            results.append(result)
        except Exception as exc:
            logger.warning("Verification failed for quiz q%s: %s", question.question_id, exc)

    report = FactualityReport.from_results(results)
    logger.info(
        "Verification complete: %d/%d claims entailed (factuality_rate=%.2f)",
        report.entailed_count,
        report.total_claims,
        report.factuality_rate,
    )
    return report


def filter_verified(
    output: LectureOutput,
    report: FactualityReport,
) -> LectureOutput:
    """Return a copy of LectureOutput with only ENTAILED items preserved.

    Items not present in the report (e.g. failed verification) are kept by default
    to avoid silently dropping content.

    Args:
        output: The original generated LectureOutput.
        report: The FactualityReport from verify_lecture_output.

    Returns:
        A new LectureOutput containing only verified-ENTAILED items.
    """
    from lecturenotes.generation.schemas import SectionNote as _SectionNote
    from lecturenotes.verification.models import VerificationLabel

    entailed_notes: set[str] = set()
    entailed_defns: set[str] = set()
    entailed_cards: set[str] = set()
    entailed_quiz: set[str] = set()

    for r in report.results:
        if r.label == VerificationLabel.ENTAILED:
            if r.item_type == "note":
                entailed_notes.add(r.item_id)
            elif r.item_type == "definition":
                entailed_defns.add(r.item_id)
            elif r.item_type == "flashcard":
                entailed_cards.add(r.item_id)
            elif r.item_type == "quiz":
                entailed_quiz.add(r.item_id)

    filtered_notes: list[SectionNote] = []
    for note in output.section_notes:
        verified_points = [
            p
            for p in note.summary_points
            if str(note.section_id) in entailed_notes or not report.results
        ]
        filtered_notes.append(
            note.model_copy(update={"summary_points": verified_points or note.summary_points})
        )

    filtered_defns = [
        d for d in output.concept_definitions if d.term in entailed_defns or not report.results
    ]
    filtered_cards = [
        c
        for c in output.flashcards
        if str(c.card_id) in entailed_cards or not report.results
    ]
    filtered_quiz = [
        q
        for q in output.quiz
        if str(q.question_id) in entailed_quiz or not report.results
    ]

    return output.model_copy(
        update={
            "section_notes": filtered_notes,
            "concept_definitions": filtered_defns or output.concept_definitions,
            "flashcards": filtered_cards or output.flashcards,
            "quiz": filtered_quiz or output.quiz,
        }
    )
