"""Educational video and resource recommendation engine for LectureNotes AI."""

from lecturenotes.recommendations.videos import (
    VideoRecommendation,
    get_concept_videos,
    recommend_videos_for_lecture,
)

__all__ = ["VideoRecommendation", "get_concept_videos", "recommend_videos_for_lecture"]
