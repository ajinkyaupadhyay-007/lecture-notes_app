"""Unit tests for educational YouTube video recommendations."""

from lecturenotes.recommendations.videos import (
    VideoRecommendation,
    get_concept_videos,
    recommend_videos_for_lecture,
)
from lecturenotes.concepts.models import KeyConcept


class DummyLLM:
    """Mock LLM returning structured video recommendation JSON."""

    def generate_json(self, prompt: str) -> dict:
        return {
            "videos": [
                {
                    "title": "Machine Learning Overview by 3Blue1Brown",
                    "channel": "3Blue1Brown",
                    "duration": "22:10",
                    "difficulty": "Intermediate",
                    "description": "Visual explanation of neural networks and gradient descent.",
                    "query": "3Blue1Brown neural networks",
                    "key_takeaways": ["Network architecture", "Weight optimization"],
                    "tags": ["Machine Learning", "Neural Networks"],
                }
            ]
        }


def test_video_recommendation_model() -> None:
    v = VideoRecommendation(
        title="Gradient Descent",
        channel="StatQuest",
        duration="15:00",
        difficulty="Beginner",
        description="Intuitive explanation",
        youtube_url="https://youtube.com/results?search_query=gradient+descent",
        key_takeaways=["Step size", "Learning rate"],
        tags=["Optimization"],
    )
    assert v.title == "Gradient Descent"
    assert v.channel == "StatQuest"
    assert "youtube.com" in v.youtube_url


def test_get_concept_videos_curated_en() -> None:
    videos = get_concept_videos("linear regression", language="English", max_results=2)
    assert len(videos) == 2
    assert any("StatQuest" in v.channel or "MIT" in v.channel for v in videos)
    assert all("youtube.com" in v.youtube_url for v in videos)
    assert all(len(v.key_takeaways) > 0 for v in videos)


def test_get_concept_videos_curated_ja() -> None:
    videos = get_concept_videos("linear regression", language="Japanese", max_results=2)
    assert len(videos) >= 1
    assert any("キカガク" in v.channel or "ヨビノリ" in v.channel for v in videos)
    assert all("youtube.com" in v.youtube_url for v in videos)


def test_get_concept_videos_fallback_synthesis() -> None:
    # Non-curated concept
    videos = get_concept_videos("quantum superposition", language="English", max_results=2)
    assert len(videos) == 2
    assert "quantum superposition" in videos[0].title.lower() or "quantum superposition" in videos[0].description.lower()
    assert all(v.difficulty in ("Beginner", "Intermediate", "Advanced") for v in videos)


def test_recommend_videos_for_lecture_with_llm() -> None:
    concepts = [
        KeyConcept(section_id=1, term="gradient descent", score=0.9, source_sentence="Gradients guide updates."),
        KeyConcept(section_id=1, term="backpropagation", score=0.8, source_sentence="Backprop uses chain rule."),
    ]
    videos = recommend_videos_for_lecture(
        concepts=concepts,
        lecture_title="Intro to Neural Networks",
        llm=DummyLLM(),
        language="English",
        max_videos=2,
    )
    assert len(videos) >= 1
    assert videos[0].channel == "3Blue1Brown"
    assert "youtube.com" in videos[0].youtube_url


def test_recommend_videos_for_lecture_fallback() -> None:
    concepts = [
        KeyConcept(section_id=1, term="linear regression", score=0.9, source_sentence="Linear regression fits line."),
        KeyConcept(section_id=1, term="regularisation", score=0.8, source_sentence="Regularization prevents overfitting."),
    ]
    videos = recommend_videos_for_lecture(
        concepts=concepts,
        lecture_title="Regression Analysis",
        llm=None,
        language="English",
        max_videos=3,
    )
    assert len(videos) >= 2
    assert all("youtube.com" in v.youtube_url for v in videos)
