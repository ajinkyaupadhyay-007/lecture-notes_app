"""Segmentation package — public API."""

from lecturenotes.segmentation.models import SentenceSpan, TopicSection
from lecturenotes.segmentation.texttiling import (
    compute_depth_scores,
    compute_window_similarities,
    find_boundaries,
    segment_into_sentences,
    segment_lecture,
)

__all__ = [
    "SentenceSpan",
    "TopicSection",
    "compute_depth_scores",
    "compute_window_similarities",
    "find_boundaries",
    "segment_into_sentences",
    "segment_lecture",
]
