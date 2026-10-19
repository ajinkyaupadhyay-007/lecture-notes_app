"""Unit tests for the topic segmentation module (Milestone 2)."""

from pathlib import Path

import numpy as np
import pytest

from lecturenotes.ingest.models import PDFSegment
from lecturenotes.ingest.text_extractor import extract_text
from lecturenotes.segmentation.models import SentenceSpan, TopicSection
from lecturenotes.segmentation.texttiling import (
    compute_depth_scores,
    compute_window_similarities,
    find_boundaries,
    segment_into_sentences,
    segment_lecture,
)

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
SAMPLE_TXT_1 = SAMPLES_DIR / "sample_lecture_1.txt"
SAMPLE_TXT_2 = SAMPLES_DIR / "sample_lecture_2.txt"


class TestSentenceExtraction:
    """Tests for breaking segments into SentenceSpan objects."""

    def test_segment_into_sentences_preserves_metadata(self) -> None:
        segments = [
            PDFSegment(text="First sentence. Second sentence.", page=1),
            PDFSegment(text="Third sentence here.", page=2),
        ]
        sentences = segment_into_sentences(segments)
        assert len(sentences) == 3
        assert [s.sentence_idx for s in sentences] == [0, 1, 2]
        assert sentences[0].page == 1
        assert sentences[1].page == 1
        assert sentences[2].page == 2
        assert sentences[0].text == "First sentence."
        assert sentences[1].text == "Second sentence."
        assert sentences[2].text == "Third sentence here."


class TestWindowSimilaritiesAndDepthScores:
    """Tests for the custom numpy TextTiling implementation."""

    def test_window_similarities_identical(self) -> None:
        # 6 identical sentence vectors -> cosine sim should be 1.0 everywhere
        vec = np.array([1.0, 0.0, 0.0])
        embeddings = np.tile(vec, (6, 1))
        sims = compute_window_similarities(embeddings, window_size=2)
        assert len(sims) == 5
        assert np.allclose(sims, 1.0)

    def test_window_similarities_orthogonal_shift(self) -> None:
        # 3 sentences on topic A, then 3 sentences on topic B
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        embeddings = np.vstack([np.tile(vec_a, (3, 1)), np.tile(vec_b, (3, 1))])
        sims = compute_window_similarities(embeddings, window_size=2)
        assert len(sims) == 5
        # The transition gap between sentence 2 and 3 should have a significant drop
        assert sims[2] < sims[0]
        assert sims[2] < sims[4]

    def test_depth_scores_flat_vs_valley(self) -> None:
        # Flat similarities -> depth should be zero everywhere
        flat_sims = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        flat_depths = compute_depth_scores(flat_sims)
        assert np.allclose(flat_depths, 0.0)

        # A clear valley at index 2
        valley_sims = np.array([0.9, 0.8, 0.2, 0.8, 0.9])
        depths = compute_depth_scores(valley_sims)
        # Depth at gap 2 is (0.9 - 0.2) + (0.9 - 0.2) = 1.4
        assert depths[2] > depths[1]
        assert depths[2] > depths[3]
        assert depths[2] == pytest.approx(1.4)

    def test_find_boundaries_threshold(self) -> None:
        depth_scores = np.array([0.0, 0.1, 0.6, 0.1, 0.0])
        # Threshold 0.25 should catch gap 2
        boundaries = find_boundaries(depth_scores, threshold=0.25)
        assert boundaries == [2]

        # Threshold 0.70 should find no boundaries
        assert find_boundaries(depth_scores, threshold=0.70) == []


class TestLectureSegmentationEndToEnd:
    """Integration tests for segmenting sample lecture texts."""

    def test_segment_sample_lecture_1(self) -> None:
        segments = extract_text(SAMPLE_TXT_1)
        sections = segment_lecture(segments, threshold=0.20, window_size=2)

        assert len(sections) >= 3, f"Expected at least 3 sections, got {len(sections)}"
        assert all(isinstance(s, TopicSection) for s in sections)
        assert all(len(s.sentences) > 0 for s in sections)

        # Check section IDs and titles
        for i, sec in enumerate(sections, start=1):
            assert sec.section_id == i
            assert sec.title.startswith(f"Section {i}")
            assert sec.page_start is not None and sec.page_end is not None
            assert sec.page_start <= sec.page_end

        # Check no sentences lost
        total_orig_sentences = len(segment_into_sentences(segments))
        total_segmented_sentences = sum(len(s.sentences) for s in sections)
        assert total_orig_sentences == total_segmented_sentences

    def test_configurable_threshold_sensitivity(self) -> None:
        segments = extract_text(SAMPLE_TXT_2)
        # Low threshold -> more topic boundaries
        sections_sensitive = segment_lecture(segments, threshold=0.15, window_size=2)
        # High threshold -> fewer topic boundaries
        sections_coarse = segment_lecture(segments, threshold=0.45, window_size=2)

        assert len(sections_sensitive) >= len(sections_coarse)
