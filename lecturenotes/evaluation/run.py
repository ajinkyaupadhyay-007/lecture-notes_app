"""Evaluation CLI for LectureNotes AI.

Computes and prints the following metrics:
  • Factuality rate (NLI entailment over generated claims)
  • Concept coverage (fraction of extracted key concepts present in full output)
  • Segmentation quality — Pk and WindowDiff (if gold boundaries available)
  • Distractor quality export  (evaluation_results.csv)

Usage:
    python -m lecturenotes.evaluation.run \\
        --pdf data/sample_lecture.pdf \\
        --provider mock \\
        --verify \\
        --output evaluation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent cp1252 charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lecturenotes.evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_factuality(output, threshold: float | None = None) -> dict:
    """Run NLI verification and return factuality metrics."""
    from lecturenotes.verification.nli_verifier import verify_lecture_output

    report = verify_lecture_output(output, threshold=threshold)
    return {
        "total_claims": report.total_claims,
        "entailed": report.entailed_count,
        "neutral": report.neutral_count,
        "contradiction": report.contradiction_count,
        "factuality_rate": report.factuality_rate,
        "report": report,
    }


def _compute_concept_coverage(all_concepts, output) -> float:
    """Fraction of extracted key concepts that appear somewhere in the output text."""
    if not all_concepts:
        return 0.0

    import re

    # Build a single searchable string from all output text fields
    output_text_parts: list[str] = []
    for note in output.section_notes:
        for pt in note.summary_points:
            output_text_parts.append(pt.summary_sentence)
    for defn in output.concept_definitions:
        output_text_parts.append(defn.definition)
    for card in output.flashcards:
        output_text_parts.append(card.answer)
    for q in output.quiz:
        output_text_parts.append(q.explanation)

    combined = " ".join(output_text_parts).lower()

    found = sum(
        1
        for c in all_concepts
        if re.search(r"\b" + re.escape(c.term.lower()) + r"\b", combined)
    )
    return found / len(all_concepts)


def _compute_segmentation_metrics(sections, gold_boundaries: list[int] | None) -> dict | None:
    """Compute Pk and WindowDiff against gold boundaries if provided.

    gold_boundaries: list of 0-indexed sentence positions where topic breaks occur.
    """
    if gold_boundaries is None:
        return None

    try:
        import numpy as np

        total_sentences = sum(len(s.sentences) for s in sections)
        if total_sentences == 0:
            return None

        # Build predicted binary boundary sequence (1 = break, 0 = continue)
        # A break is placed at the *last* sentence of each section (except the final one)
        pred = [0] * total_sentences
        cumulative = 0
        for sec in sections[:-1]:  # skip last section — no boundary after it
            cumulative += len(sec.sentences)
            if cumulative < total_sentences:
                pred[cumulative - 1] = 1

        # Gold boundary sequence
        gold = [0] * total_sentences
        for idx in gold_boundaries:
            if 0 <= idx < total_sentences:
                gold[idx] = 1

        k = max(1, total_sentences // (2 * max(len(sections), 1)))

        def _pk(ref: list[int], hyp: list[int], k: int) -> float:
            errs = 0
            total = 0
            for i in range(len(ref) - k):
                ref_same = ref[i] == ref[i + k]
                hyp_same = hyp[i] == hyp[i + k]
                if ref_same != hyp_same:
                    errs += 1
                total += 1
            return errs / total if total > 0 else 0.0

        def _windowdiff(ref: list[int], hyp: list[int], k: int) -> float:
            errs = 0.0
            total = 0
            for i in range(len(ref) - k):
                ref_breaks = sum(ref[i : i + k])
                hyp_breaks = sum(hyp[i : i + k])
                errs += abs(ref_breaks - hyp_breaks)
                total += 1
            return errs / total if total > 0 else 0.0

        return {
            "Pk": round(_pk(gold, pred, k), 4),
            "WindowDiff": round(_windowdiff(gold, pred, k), 4),
            "k": k,
            "pred_boundaries": sum(pred),
            "gold_boundaries": sum(gold),
        }
    except Exception as exc:
        logger.warning("Segmentation metrics failed: %s", exc)
        return None


def _export_distractor_csv(output, path: Path) -> None:
    """Export MCQ questions and distractors to CSV for manual quality rating."""
    rows: list[dict] = []
    for q in output.quiz:
        for letter, expl in q.distractor_explanations.items():
            rows.append(
                {
                    "question_id": q.question_id,
                    "question": q.question,
                    "correct_answer": q.correct_answer,
                    "distractor_letter": letter,
                    "distractor_text": q.options.get(letter, ""),
                    "distractor_explanation": expl,
                    "source_span": q.source_span,
                    "manual_rating": "",  # for human rater
                    "notes": "",
                }
            )
    if not rows:
        logger.info("No MCQ distractors to export.")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Distractor CSV written to: %s (%d rows)", path, len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LectureNotes AI — evaluation metric reporter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to a lecture PDF file for end-to-end evaluation.",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="Path to a plain-text lecture file (alternative to --pdf).",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["ollama", "azure", "claude", "gemini", "mock"],
        help="LLM provider to use for generation.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Run NLI factuality verification (requires model download on first use).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.csv"),
        help="Path to write the distractor quality CSV.",
    )
    parser.add_argument(
        "--gold-boundaries",
        nargs="*",
        type=int,
        default=None,
        metavar="IDX",
        help="0-indexed sentence positions of true topic boundaries (for Pk/WindowDiff).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ── Resolve input ─────────────────────────────────────────────────────────
    if args.pdf is None and args.txt is None:
        logger.error("Provide at least one of --pdf or --txt.")
        sys.exit(1)

    from lecturenotes.config import load_config
    from lecturenotes.ingest import extract_pdf, extract_text
    from lecturenotes.segmentation import segment_lecture
    from lecturenotes.concepts.extractor import extract_lecture_concepts, get_all_unique_concepts
    from lecturenotes.generation.generator import generate_lecture_materials
    from lecturenotes.generation.llm_client import LLMClient

    cfg = load_config()
    cfg.llm.provider = args.provider

    # ── Ingest ────────────────────────────────────────────────────────────────
    if args.pdf:
        logger.info("Ingesting PDF: %s", args.pdf)
        segments = extract_pdf(args.pdf, min_chars=cfg.ingest.pdf_min_chars_per_page)
        source_name = args.pdf.stem
    else:
        logger.info("Ingesting text file: %s", args.txt)
        segments = extract_text(args.txt)
        source_name = args.txt.stem

    # ── Segment ───────────────────────────────────────────────────────────────
    logger.info("Segmenting lecture…")
    sections = segment_lecture(segments)
    logger.info("Found %d topic sections", len(sections))

    # ── Concepts ──────────────────────────────────────────────────────────────
    logger.info("Extracting key concepts…")
    section_concepts = extract_lecture_concepts(sections)
    all_concepts = get_all_unique_concepts(section_concepts)
    logger.info("Extracted %d unique concepts", len(all_concepts))

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info("Generating lecture materials (provider=%s)…", args.provider)
    llm = LLMClient(config=cfg)
    output = generate_lecture_materials(
        sections=sections,
        concepts=all_concepts,
        llm=llm,
        lecture_title=source_name.replace("_", " ").replace("-", " ").title(),
    )
    logger.info(
        "Generated: %d notes | %d definitions | %d flashcards | %d MCQs",
        len(output.section_notes),
        len(output.concept_definitions),
        len(output.flashcards),
        len(output.quiz),
    )

    # ── Print results ─────────────────────────────────────────────────────────
    divider = "─" * 60

    print(f"\n{divider}")
    print("  LectureNotes AI — Evaluation Report")
    print(f"  Source: {source_name}")
    print(f"  Provider: {args.provider}")
    print(divider)

    # ── Concept coverage ──────────────────────────────────────────────────────
    coverage = _compute_concept_coverage(all_concepts, output)
    print(f"\n📌 Concept Coverage:    {coverage:.1%}  ({len(all_concepts)} total concepts)")

    # ── Segmentation ──────────────────────────────────────────────────────────
    seg_metrics = _compute_segmentation_metrics(sections, args.gold_boundaries)
    if seg_metrics:
        print(
            f"✂️  Segmentation (k={seg_metrics['k']}):  "
            f"Pk={seg_metrics['Pk']:.4f}  |  WindowDiff={seg_metrics['WindowDiff']:.4f}  "
            f"(pred={seg_metrics['pred_boundaries']} boundaries, "
            f"gold={seg_metrics['gold_boundaries']})"
        )
    else:
        print("✂️  Segmentation:        No gold boundaries provided — skipping Pk/WindowDiff.")

    # ── Factuality ────────────────────────────────────────────────────────────
    if args.verify:
        logger.info("Running NLI factuality verification…")
        fact = _compute_factuality(output)
        print(
            f"\n🔬 Factuality Rate:     {fact['factuality_rate']:.1%}  "
            f"({fact['entailed']}/{fact['total_claims']} entailed, "
            f"{fact['neutral']} neutral, {fact['contradiction']} contradiction)"
        )
    else:
        print("\n🔬 Factuality Rate:     (skipped — use --verify to enable)")

    # ── Generation summary ────────────────────────────────────────────────────
    total_notes = sum(len(n.summary_points) for n in output.section_notes)
    print(f"\n📝 Summary points:      {total_notes} across {len(output.section_notes)} sections")
    print(f"🃏 Flashcards:          {len(output.flashcards)}")
    print(f"📋 MCQ questions:       {len(output.quiz)}")
    print(f"📖 Definitions:         {len(output.concept_definitions)}")

    print(f"\n{divider}\n")

    # ── Export CSV ────────────────────────────────────────────────────────────
    _export_distractor_csv(output, args.output)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
