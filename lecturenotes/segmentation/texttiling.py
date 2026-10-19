"""TextTiling-style topic segmentation using sentence embeddings and cosine similarity drops.

Calculates cosine similarity between adjacent sentence windows using sentence-transformers
embeddings and a custom numpy implementation of Hearst's TextTiling depth-score algorithm.
Strictly deterministic NLP — NO LLM call.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import numpy as np

from lecturenotes.config import load_config
from lecturenotes.ingest.models import Segment
from lecturenotes.segmentation.models import SentenceSpan, TopicSection

logger = logging.getLogger(__name__)

_SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")
_model_cache: dict[str, Any] = {}


from pathlib import Path


def resolve_model_path(model_name: str) -> str:
    """Resolve a model name to its local snapshot directory if cached."""
    p = Path(model_name)
    if p.exists():
        return str(p)

    repo_id = model_name
    if "/" not in repo_id and not repo_id.startswith("models--"):
        repo_folder = f"models--sentence-transformers--{repo_id}"
    else:
        repo_folder = f"models--{repo_id.replace('/', '--')}"

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / repo_folder
    snapshots_dir = hf_cache / "snapshots"
    if snapshots_dir.exists():
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if snapshots:
            logger.info("Using cached HuggingFace snapshot for %s: %s", model_name, snapshots[0])
            return str(snapshots[0])

    return model_name


def get_embedding_model(model_name: str | None = None) -> Any:
    """Load or retrieve the cached SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    if not model_name:
        model_name = load_config().embedding.model
    if model_name not in _model_cache:
        target_path = resolve_model_path(model_name)
        logger.info("Loading sentence-transformers model from: %s", target_path)
        _model_cache[model_name] = SentenceTransformer(target_path)
    return _model_cache[model_name]


def segment_into_sentences(segments: Sequence[Segment]) -> list[SentenceSpan]:
    """Extract individual sentences from input segments while preserving source metadata."""
    sentences: list[SentenceSpan] = []
    sentence_idx = 0

    for seg in segments:
        raw_text = seg.text.strip()
        if not raw_text:
            continue

        page: int | None = getattr(seg, "page", None)
        start: float | None = getattr(seg, "start", None)
        end: float | None = getattr(seg, "end", None)
        source_type = seg.source_type

        # Split text into sentences across lines
        blocks = [b.strip() for b in raw_text.splitlines() if b.strip()]
        for block in blocks:
            raw_sents = _SENTENCE_REGEX.split(block)
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s) < 5:
                    continue
                sentences.append(
                    SentenceSpan(
                        text=clean_s,
                        sentence_idx=sentence_idx,
                        page=page,
                        start=start,
                        end=end,
                        source_type=source_type,
                    )
                )
                sentence_idx += 1

    return sentences


def compute_window_similarities(
    embeddings: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Compute cosine similarity between adjacent sentence windows across all gaps.

    Gap i is the transition between sentence i and sentence i+1.
    For gap i, left window is sentences [i - window_size + 1 .. i],
    and right window is sentences [i + 1 .. i + window_size].

    Returns:
        Array of cosine similarities of length (N - 1), where index i corresponds
        to gap i.
    """
    n_sentences = embeddings.shape[0]
    if n_sentences < 2:
        return np.array([], dtype=float)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    norm_embeddings = embeddings / norms

    similarities = np.ones(n_sentences - 1, dtype=float)

    for i in range(n_sentences - 1):
        left_start = max(0, i - window_size + 1)
        left_end = i + 1
        right_start = i + 1
        right_end = min(n_sentences, i + 1 + window_size)

        left_window = norm_embeddings[left_start:left_end].mean(axis=0)
        right_window = norm_embeddings[right_start:right_end].mean(axis=0)

        left_norm = np.linalg.norm(left_window)
        right_norm = np.linalg.norm(right_window)

        if left_norm > 1e-9 and right_norm > 1e-9:
            sim = float(np.dot(left_window / left_norm, right_window / right_norm))
            similarities[i] = max(-1.0, min(1.0, sim))

    return similarities


def compute_depth_scores(similarities: np.ndarray) -> np.ndarray:
    """Calculate TextTiling depth scores from similarity values.

    Depth score at gap i measures the drop from the left peak and right peak:
    depth[i] = (peak_left - sim[i]) + (peak_right - sim[i])
    """
    n_gaps = len(similarities)
    if n_gaps == 0:
        return np.array([], dtype=float)

    depth_scores = np.zeros(n_gaps, dtype=float)

    for i in range(n_gaps):
        # Scan left for peak
        peak_left = similarities[i]
        for l in range(i - 1, -1, -1):
            if similarities[l] >= peak_left:
                peak_left = similarities[l]
            else:
                break

        # Scan right for peak
        peak_right = similarities[i]
        for r in range(i + 1, n_gaps):
            if similarities[r] >= peak_right:
                peak_right = similarities[r]
            else:
                break

        depth_scores[i] = (peak_left - similarities[i]) + (peak_right - similarities[i])

    return depth_scores


def find_boundaries(
    depth_scores: np.ndarray,
    threshold: float = 0.25,
    min_gap_distance: int = 2,
) -> list[int]:
    """Identify topic boundary gap indices exceeding the depth threshold.

    Enforces a minimum sentence gap distance to prevent over-segmentation.
    """
    candidates = [i for i, score in enumerate(depth_scores) if score >= threshold]
    if not candidates:
        return []

    boundaries: list[int] = []
    for c in candidates:
        is_local_max = True
        if c > 0 and depth_scores[c] < depth_scores[c - 1]:
            is_local_max = False
        if c < len(depth_scores) - 1 and depth_scores[c] < depth_scores[c + 1]:
            is_local_max = False

        if is_local_max:
            if not boundaries or (c - boundaries[-1] >= min_gap_distance):
                boundaries.append(c)
            elif depth_scores[c] > depth_scores[boundaries[-1]]:
                boundaries[-1] = c

    return boundaries


def generate_section_title(sentences: list[SentenceSpan], section_idx: int) -> str:
    """Derive a heuristic topic title from the section's initial sentences without using an LLM."""
    if not sentences:
        return f"Section {section_idx}"

    first_text = sentences[0].text.strip()
    clean = re.sub(
        r"^(welcome to|in this (course|lecture),? we will (study|discuss|cover)|today's lecture focuses on)\s+",
        "",
        first_text,
        flags=re.IGNORECASE,
    )
    parts = clean.split(":")
    if len(parts) > 1 and len(parts[0].split()) <= 6:
        candidate = parts[0].strip()
    else:
        words = clean.split()
        candidate = " ".join(words[:6]).strip(",:;-")

    if not candidate or len(candidate) < 3:
        return f"Section {section_idx}"
    return f"Section {section_idx}: {candidate.capitalize()}"


def segment_lecture(
    segments: Sequence[Segment],
    threshold: float | None = None,
    window_size: int | None = None,
    model_name: str | None = None,
) -> list[TopicSection]:
    """Execute complete TextTiling segmentation on extracted lecture segments."""
    cfg = load_config()
    threshold = threshold if threshold is not None else cfg.segmentation.threshold
    window_size = window_size if window_size is not None else cfg.segmentation.window_size
    model_name = model_name if model_name is not None else cfg.embedding.model

    sentences = segment_into_sentences(segments)
    if not sentences:
        return []

    if len(sentences) <= window_size * 2:
        return [
            TopicSection(
                section_id=1,
                title="Section 1: Overview",
                text=" ".join(s.text for s in sentences),
                sentences=sentences,
                page_start=sentences[0].page,
                page_end=sentences[-1].page,
                time_start=sentences[0].start,
                time_end=sentences[-1].end,
                source_type=sentences[0].source_type,
            )
        ]

    model = get_embedding_model(model_name)
    sentence_texts = [s.text for s in sentences]
    embeddings = np.array(model.encode(sentence_texts, show_progress_bar=False), dtype=float)

    sims = compute_window_similarities(embeddings, window_size=window_size)
    depths = compute_depth_scores(sims)
    boundary_gaps = find_boundaries(depths, threshold=threshold, min_gap_distance=window_size)

    sections: list[TopicSection] = []
    start_idx = 0
    section_id = 1

    boundary_cutoffs = list(boundary_gaps) + [len(sentences) - 1]

    for end_gap in boundary_cutoffs:
        section_sentences = sentences[start_idx : end_gap + 1]
        if not section_sentences:
            continue

        pages = [s.page for s in section_sentences if s.page is not None]
        starts = [s.start for s in section_sentences if s.start is not None]
        ends = [s.end for s in section_sentences if s.end is not None]

        title = generate_section_title(section_sentences, section_id)
        combined_text = " ".join(s.text for s in section_sentences)

        sections.append(
            TopicSection(
                section_id=section_id,
                title=title,
                text=combined_text,
                sentences=section_sentences,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                time_start=min(starts) if starts else None,
                time_end=max(ends) if ends else None,
                source_type=section_sentences[0].source_type,
            )
        )
        section_id += 1
        start_idx = end_gap + 1

    logger.info(
        "Segmented %d sentences into %d topic sections (threshold=%.2f, window=%d)",
        len(sentences),
        len(sections),
        threshold,
        window_size,
    )
    return sections
