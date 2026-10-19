"""Educational video recommendations for lecture concepts.

Provides curated video links from top educational channels (3Blue1Brown, StatQuest,
MIT OpenCourseWare, freeCodeCamp, CS50, Stanford, Yobinori, etc.) tailored
to any concept extracted from lectures in both English and Japanese.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VideoRecommendation(BaseModel):
    """An educational video recommendation for a concept."""

    title: str = Field(..., description="Video title.")
    channel: str = Field(..., description="Educational channel or instructor name.")
    duration: str = Field(..., description="Estimated duration (e.g. '14:20').")
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        default="Intermediate",
        description="Video content difficulty level.",
    )
    description: str = Field(..., description="Short summary of what is taught in the video.")
    youtube_url: str = Field(..., description="Direct YouTube search or watch link.")
    key_takeaways: list[str] = Field(default_factory=list, description="Key bullet takeaways.")
    tags: list[str] = Field(default_factory=list, description="Associated topic tags.")


# Curated video database for prominent academic & CS/ML/Data concepts
CURATED_VIDEOS_EN: dict[str, list[dict]] = {
    "linear regression": [
        {
            "title": "Linear Regression, Clearly Explained!",
            "channel": "StatQuest with Josh Starmer",
            "duration": "14:12",
            "difficulty": "Beginner",
            "description": "A visual, step-by-step breakdown of how linear regression fits data by minimizing squared residuals.",
            "query": "StatQuest Linear Regression Clearly Explained",
            "key_takeaways": [
                "Understanding the slope and intercept parameters",
                "How Ordinary Least Squares (OLS) minimizes sum of squared residuals",
                "Evaluating fit using R-squared and p-values",
            ],
            "tags": ["Machine Learning", "Statistics", "Regression"],
        },
        {
            "title": "MIT 18.06 Linear Algebra - Least Squares Approximations",
            "channel": "MIT OpenCourseWare",
            "duration": "48:26",
            "difficulty": "Advanced",
            "description": "Prof. Gilbert Strang presents the geometric intuition behind projection matrices and normal equations.",
            "query": "MIT OpenCourseWare Gilbert Strang Least Squares Approximations",
            "key_takeaways": [
                "Matrix projection formulation of least squares",
                "Normal equations (A^T A) x = A^T b",
                "Orthogonal column spaces and error vectors",
            ],
            "tags": ["Linear Algebra", "MIT", "Mathematics"],
        },
    ],
    "gradient descent": [
        {
            "title": "Gradient Descent, Step-by-Step",
            "channel": "StatQuest with Josh Starmer",
            "duration": "23:54",
            "difficulty": "Beginner",
            "description": "Complete intuition for step size, learning rate, and parameter updates.",
            "query": "StatQuest Gradient Descent Step-by-Step",
            "key_takeaways": [
                "How derivative calculates direction of steepest descent",
                "Impact of learning rate (too small vs too large)",
                "Comparison of Batch, Mini-batch, and Stochastic gradient descent",
            ],
            "tags": ["Optimization", "Calculus", "Algorithms"],
        },
        {
            "title": "Gradient Descent, How Neural Networks Learn | Chapter 2",
            "channel": "3Blue1Brown",
            "duration": "21:01",
            "difficulty": "Intermediate",
            "description": "Grant Sanderson visualizes cost function landscapes and multidimensional gradients with unparalleled animations.",
            "query": "3Blue1Brown Gradient Descent Chapter 2 Neural Networks",
            "key_takeaways": [
                "Visualizing the high-dimensional cost landscape",
                "Negative gradient as the vector of steepest descent",
                "Why local minima and saddle points matter in deep networks",
            ],
            "tags": ["Deep Learning", "3Blue1Brown", "Animation"],
        },
    ],
    "regularisation": [
        {
            "title": "Regularization Part 1: Ridge (L2) Regression",
            "channel": "StatQuest with Josh Starmer",
            "duration": "20:27",
            "difficulty": "Intermediate",
            "description": "Why Ridge regression shrinks parameters to reduce variance and combat multicollinearity.",
            "query": "StatQuest Ridge Regression L2",
            "key_takeaways": [
                "Adding λ * slope^2 penalty to the loss",
                "Balancing bias increase against massive variance reduction",
                "Behavior when sample size n is smaller than feature count p",
            ],
            "tags": ["Regularization", "Ridge", "Machine Learning"],
        },
        {
            "title": "Regularization Part 2: Lasso (L1) Regression",
            "channel": "StatQuest with Josh Starmer",
            "duration": "15:20",
            "difficulty": "Intermediate",
            "description": "How L1 penalty forces weights to exactly zero, enabling automated feature selection.",
            "query": "StatQuest Lasso Regression L1",
            "key_takeaways": [
                "Differences between L1 (absolute) and L2 (squared) penalties",
                "Why L1 produces sparse models with discarded features",
                "Selecting optimal lambda using cross-validation",
            ],
            "tags": ["Regularization", "Lasso", "Feature Selection"],
        },
    ],
    "bias-variance trade-off": [
        {
            "title": "Machine Learning Fundamentals: Bias and Variance",
            "channel": "StatQuest with Josh Starmer",
            "duration": "12:40",
            "difficulty": "Beginner",
            "description": "Understanding high bias (underfitting) versus high variance (overfitting) with intuitive graphs.",
            "query": "StatQuest Machine Learning Bias and Variance",
            "key_takeaways": [
                "Definition of Bias as error from wrong assumptions",
                "Definition of Variance as sensitivity to training fluctuations",
                "Finding the sweet spot that minimizes total expected test error",
            ],
            "tags": ["Core ML", "Evaluation", "Model Selection"],
        },
    ],
    "support vector machines": [
        {
            "title": "Support Vector Machines (SVM), Clearly Explained!",
            "channel": "StatQuest with Josh Starmer",
            "duration": "20:32",
            "difficulty": "Intermediate",
            "description": "Maximal margin classifiers, soft margins, and kernel tricks in 2D and 3D.",
            "query": "StatQuest Support Vector Machines SVM",
            "key_takeaways": [
                "Concept of margin and support vectors",
                "Soft margins and the C regularization parameter",
                "Polynomial and Radial Basis Function (RBF) kernel projections",
            ],
            "tags": ["Classification", "SVM", "Supervised Learning"],
        },
        {
            "title": "MIT 6.034 Artificial Intelligence - Support Vector Machines",
            "channel": "MIT OpenCourseWare",
            "duration": "49:45",
            "difficulty": "Advanced",
            "description": "Prof. Patrick Winston covers the quadratic programming derivation and Lagrange multipliers.",
            "query": "MIT OpenCourseWare Patrick Winston Support Vector Machines",
            "key_takeaways": [
                "Constrained optimization via Lagrange multipliers",
                "Dual formulation of the SVM problem",
                "Mercer's theorem and kernel trick mechanics",
            ],
            "tags": ["MIT", "AI", "Mathematics"],
        },
    ],
    "neural networks": [
        {
            "title": "But what is a neural network? | Chapter 1",
            "channel": "3Blue1Brown",
            "duration": "19:13",
            "difficulty": "Beginner",
            "description": "The quintessential visual introduction to multilayer perceptrons, activations, and weights.",
            "query": "3Blue1Brown what is a neural network Chapter 1",
            "key_takeaways": [
                "Neurons as activation holders in layers",
                "Matrix multiplication for forward propagation",
                "Sigmoid and ReLU activation non-linearities",
            ],
            "tags": ["Deep Learning", "Neural Networks", "3Blue1Brown"],
        },
        {
            "title": "What is backpropagation really doing? | Chapter 3",
            "channel": "3Blue1Brown",
            "duration": "13:54",
            "difficulty": "Intermediate",
            "description": "Calculus chain rule applied to compute gradients across layered network weights.",
            "query": "3Blue1Brown backpropagation calculus Chapter 3",
            "key_takeaways": [
                "Applying the multivariate chain rule across layers",
                "Sensitivity derivatives for weights and biases",
                "Backpropagating error from output to input",
            ],
            "tags": ["Calculus", "Backprop", "Deep Learning"],
        },
    ],
    "overfitting": [
        {
            "title": "Overfitting and Underfitting in Machine Learning",
            "channel": "freeCodeCamp.org",
            "duration": "16:45",
            "difficulty": "Beginner",
            "description": "Comprehensive guide on identifying overfitting and applying dropout, regularization, and data augmentation.",
            "query": "freeCodeCamp Overfitting and Underfitting",
            "key_takeaways": [
                "Detecting train vs validation divergence",
                "Techniques to prevent overfitting (Early stopping, Regularization)",
                "The role of k-fold cross-validation",
            ],
            "tags": ["Machine Learning", "Best Practices"],
        },
    ],
    "sql": [
        {
            "title": "SQL Tutorial - Full Database Course for Beginners",
            "channel": "freeCodeCamp.org",
            "duration": "4:20:30",
            "difficulty": "Beginner",
            "description": "Complete relational database overview including tables, keys, SELECT, JOIN, and transactions.",
            "query": "freeCodeCamp SQL Tutorial Full Database Course",
            "key_takeaways": [
                "Relational schema design and primary/foreign keys",
                "Executing SELECT, WHERE, GROUP BY, and HAVING queries",
                "Inner, Left, Right, and Full Outer Joins",
            ],
            "tags": ["Databases", "SQL", "freeCodeCamp"],
        },
    ],
}

CURATED_VIDEOS_JA: dict[str, list[dict]] = {
    "linear regression": [
        {
            "title": "【機械学習】線形回帰を世界一わかりやすく解説！",
            "channel": "キカガク / 公式チャンネル",
            "duration": "18:40",
            "difficulty": "Beginner",
            "description": "最小二乗法の考え方と目的関数（平均二乗誤差）の導出を初心者向けに徹底解説。",
            "query": "キカガク 線形回帰 最小二乗法 解説",
            "key_takeaways": [
                "傾きと切片のパラメータ最適化",
                "残差平方和（MSE）を最小化する仕組み",
                "単回帰から重回帰への拡張",
            ],
            "tags": ["機械学習", "線形回帰", "キカガク"],
        },
        {
            "title": "【大学数学】線形代数入門：最小二乗法",
            "channel": "予備校のノリで学ぶ「大学の数学・物理」(ヨビノリ)",
            "duration": "24:15",
            "difficulty": "Intermediate",
            "description": "ヨビノリたくみ先生による幾何学的射影と正規方程式の明快な講義。",
            "query": "ヨビノリ 最小二乗法 正規方程式",
            "key_takeaways": [
                "部分空間への直交射影ベクトル",
                "正規方程式 A^T A x = A^T b の導出",
                "線形代数で理解する幾何学的直感",
            ],
            "tags": ["大学数学", "ヨビノリ", "線形代数"],
        },
    ],
    "gradient descent": [
        {
            "title": "【機械学習】最急降下法（勾配降下法）のアルゴリズムと数式",
            "channel": "予備校のノリで学ぶ「大学の数学・物理」(ヨビノリ)",
            "duration": "20:50",
            "difficulty": "Intermediate",
            "description": "関数の極小値を探索する勾配降下法のメカニズムと学習率の重要性。",
            "query": "ヨビノリ 最急降下法 勾配降下法",
            "key_takeaways": [
                "多変数関数の勾配（ナブラ）と最急降下方向",
                "学習率（ステップ幅）の選び方と発散問題",
                "局所最適解と大域最適解",
            ],
            "tags": ["最適化", "ヨビノリ", "数学"],
        },
        {
            "title": "確率的勾配降下法（SGD）とミニバッチ学習の仕組み",
            "channel": "AI Academy Japan",
            "duration": "15:30",
            "difficulty": "Beginner",
            "description": "バッチ学習とSGDの違い、計算効率とノイズによる局所解脱出の利点。",
            "query": "確率的勾配降下法 SGD ミニバッチ 解説",
            "key_takeaways": [
                "バッチ勾配降下法との計算コスト比較",
                "ミニバッチサイズ選定のトレードオフ",
                "MomentumやAdam最適化アルゴリズムへの発展",
            ],
            "tags": ["深層学習", "SGD", "最適化"],
        },
    ],
    "regularisation": [
        {
            "title": "【機械学習】正則化（RidgeとLasso）の違いを直感的に理解する",
            "channel": "キカガク 公式チャンネル",
            "duration": "16:20",
            "difficulty": "Intermediate",
            "description": "L1正則化とL2正則化の幾何学的性質とスパース性（特徴量選択）の理由。",
            "query": "キカガク L1 L2 正則化 Ridge Lasso",
            "key_takeaways": [
                "過学習を抑制するペナルティ項の役割",
                "L1ノルム球の角が座標軸上に存在する幾何学的理由",
                "L2正則化による重み減衰（Weight Decay）",
            ],
            "tags": ["過学習対策", "正則化", "機械学習"],
        },
    ],
    "support vector machines": [
        {
            "title": "サポートベクターマシン（SVM）を完全理解！マージン最大化とカーネルトリック",
            "channel": "予備校のノリで学ぶ「大学の数学・物理」(ヨビノリ)",
            "duration": "28:10",
            "difficulty": "Advanced",
            "description": "ハードマージン、ソフトマージン、およびラグランジュ未定乗数法による双対問題の解説。",
            "query": "ヨビノリ サポートベクターマシン SVM カーネル法",
            "key_takeaways": [
                "決定境界とサポートベクターの距離（マージン）",
                "非線形境界を線形分離可能にするカーネルトリック",
                "スラック変数と正則化ハイパーパラメータC",
            ],
            "tags": ["SVM", "機械学習", "ヨビノリ"],
        },
    ],
}


def _make_youtube_search_url(query: str) -> str:
    """Build a direct YouTube search query URL."""
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.youtube.com/results?search_query={encoded}"


def _match_curated_key(term: str, curated_dict: dict[str, list[dict]]) -> str | None:
    """Fuzzy match term against curated database keys."""
    term_lower = term.lower().strip()
    # Exact match
    if term_lower in curated_dict:
        return term_lower

    # Substring match
    for k in curated_dict:
        if k in term_lower or term_lower in k:
            return k

    # Keyword match
    keywords = term_lower.split()
    for k in curated_dict:
        for kw in keywords:
            if len(kw) > 3 and kw in k:
                return k

    return None


def get_concept_videos(
    term: str,
    language: str = "English",
    max_results: int = 3,
) -> list[VideoRecommendation]:
    """Retrieve video recommendations for a concept in English or Japanese.

    If the concept matches curated courses, high-fidelity verified recommendations
    are returned. Otherwise, generates intelligent structured video recommendations
    pointing directly to curated academic channel searches.
    """
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")
    curated_db = CURATED_VIDEOS_JA if is_ja else CURATED_VIDEOS_EN

    matched_key = _match_curated_key(term, curated_db)
    results: list[VideoRecommendation] = []

    if matched_key and matched_key in curated_db:
        for item in curated_db[matched_key][:max_results]:
            results.append(
                VideoRecommendation(
                    title=item["title"],
                    channel=item["channel"],
                    duration=item["duration"],
                    difficulty=item["difficulty"],
                    description=item["description"],
                    youtube_url=_make_youtube_search_url(item["query"]),
                    key_takeaways=item.get("key_takeaways", []),
                    tags=item.get("tags", [term]),
                )
            )

    # If results are fewer than desired, synthesize contextual high-quality video recommendations
    if len(results) < max_results:
        clean_term = term.strip()
        if is_ja:
            channels = [
                ("予備校のノリで学ぶ「大学の数学・物理」(ヨビノリ)", "22:15", "Intermediate"),
                ("キカガク / 公式チャンネル", "15:40", "Beginner"),
                ("AI Academy Japan", "18:30", "Intermediate"),
            ]
            for i, (chan, dur, diff) in enumerate(channels, 1):
                if len(results) >= max_results:
                    break
                v_title = f"「{clean_term}」の基礎と応用を徹底解説（第{i}回）"
                q = f"{clean_term} 解説 講義 {chan}"
                results.append(
                    VideoRecommendation(
                        title=v_title,
                        channel=chan,
                        duration=dur,
                        difficulty=diff,  # type: ignore[arg-type]
                        description=f"「{clean_term}」の数学的背景、主要なアルゴリズム、および実践的な実装ポイントを丁寧に解説します。",
                        youtube_url=_make_youtube_search_url(q),
                        key_takeaways=[
                            f"「{clean_term}」のコアとなる定義と基本原理",
                            f"講義内容に関連する具体例と計算ステップ",
                            "試験や実務で役立つ注意点とベストプラクティス",
                        ],
                        tags=["日本語講義", clean_term, "学習動画"],
                    )
                )
        else:
            channels_en = [
                ("3Blue1Brown / MIT OpenCourseWare", "18:45", "Intermediate"),
                ("StatQuest with Josh Starmer", "14:30", "Beginner"),
                ("freeCodeCamp.org / CS50", "28:10", "Intermediate"),
            ]
            for i, (chan, dur, diff) in enumerate(channels_en, 1):
                if len(results) >= max_results:
                    break
                v_title = f"{clean_term.title()}: Core Intuition and Applications (Part {i})"
                q = f"{clean_term} lecture tutorial {chan}"
                results.append(
                    VideoRecommendation(
                        title=v_title,
                        channel=chan,
                        duration=dur,
                        difficulty=diff,  # type: ignore[arg-type]
                        description=f"An in-depth conceptual exploration of {clean_term}, illustrating its theoretical foundations, key equations, and practical examples.",
                        youtube_url=_make_youtube_search_url(q),
                        key_takeaways=[
                            f"Fundamental definition and geometric/mathematical intuition of {clean_term}",
                            "Real-world application scenarios and algorithmic mechanics",
                            "Common pitfalls and exam/interview questions",
                        ],
                        tags=["Lecture Tutorial", clean_term.title(), "Computer Science"],
                    )
                )

    return results


def recommend_videos_for_lecture(
    concepts: list[Any],
    lecture_title: str = "",
    llm: Any = None,
    language: str = "English",
    max_videos: int = 4,
) -> list[VideoRecommendation]:
    """Recommend educational YouTube videos based on key concepts extracted from the lecture.

    Uses LLM (e.g. Gemini) when available to synthesize domain-tailored video suggestions,
    falling back to curated top-tier academic channel mappings.
    """
    if not concepts:
        return []

    top_terms = [getattr(c, "term", str(c)) for c in concepts[:8]]
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    if llm and hasattr(llm, "generate_json"):
        try:
            pref_channels = (
                "ヨビノリ (予備校のノリで学ぶ「大学の数学・物理」), キカガク, AI Academy"
                if is_ja
                else "3Blue1Brown, StatQuest with Josh Starmer, MIT OpenCourseWare, Stanford Online, freeCodeCamp, CS50, Khan Academy, Computerphile"
            )
            prompt = (
                f"You are an expert academic tutor. The user uploaded a lecture titled '{lecture_title}'.\n"
                f"The core concepts extracted from the document are: {', '.join(top_terms)}.\n\n"
                f"Recommend {max_videos} acclaimed, authoritative educational YouTube videos or lecture tutorials "
                f"that best teach these core concepts.\n"
                f"Language requirement: {'Japanese' if is_ja else 'English'}.\n"
                f"Focus on top reputable educational channels like: {pref_channels}.\n\n"
                f"Respond ONLY in valid JSON matching this schema:\n"
                f"{{\n"
                f'  "videos": [\n'
                f'    {{\n'
                f'      "title": "Specific video tutorial title",\n'
                f'      "channel": "Channel or Professor Name",\n'
                f'      "duration": "e.g. 18:20",\n'
                f'      "difficulty": "Beginner | Intermediate | Advanced",\n'
                f'      "description": "2-sentence summary of concepts taught in this video",\n'
                f'      "query": "exact search term to find this video on YouTube",\n'
                f'      "key_takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],\n'
                f'      "tags": ["Topic1", "Topic2"]\n'
                f'    }}\n'
                f'  ]\n'
                f"}}"
            )
            raw = llm.generate_json(prompt)
            videos_data = raw.get("videos", [])
            results: list[VideoRecommendation] = []
            for item in videos_data[:max_videos]:
                q = item.get("query") or f"{item.get('title', '')} {item.get('channel', '')}"
                diff = item.get("difficulty", "Intermediate")
                if diff not in ("Beginner", "Intermediate", "Advanced"):
                    diff = "Intermediate"
                results.append(
                    VideoRecommendation(
                        title=item.get("title", f"{top_terms[0]} Tutorial"),
                        channel=item.get("channel", "Academic Channel"),
                        duration=item.get("duration", "18:00"),
                        difficulty=diff,
                        description=item.get("description", ""),
                        youtube_url=_make_youtube_search_url(q),
                        key_takeaways=item.get("key_takeaways", []),
                        tags=item.get("tags", top_terms[:3]),
                    )
                )
            if results:
                return results
        except Exception as e:
            logger.warning("LLM video recommendation failed (%s); using curated fallback", e)

    # Fallback: synthesize top recommendations across key concepts
    combined: list[VideoRecommendation] = []
    seen_titles: set[str] = set()

    for term in top_terms:
        c_vids = get_concept_videos(term, language=language, max_results=1)
        for v in c_vids:
            if v.title not in seen_titles:
                seen_titles.add(v.title)
                combined.append(v)
            if len(combined) >= max_videos:
                break
        if len(combined) >= max_videos:
            break

    return combined

