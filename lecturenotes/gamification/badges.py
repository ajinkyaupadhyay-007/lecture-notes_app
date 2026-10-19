"""Badges, user progress tracking, and challenge mechanics for LectureNotes AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from pydantic import BaseModel, Field


class Badge(BaseModel):
    """An achievement badge that can be unlocked by the user."""

    id: str = Field(..., description="Unique identifier for badge.")
    icon: str = Field(..., description="Emoji icon representation.")
    title_en: str = Field(..., description="English title.")
    title_ja: str = Field(..., description="Japanese title.")
    description_en: str = Field(..., description="English description.")
    description_ja: str = Field(..., description="Japanese description.")
    xp_reward: int = Field(default=50, description="XP awarded upon unlock.")
    category: str = Field(default="general", description="Badge category.")

    def get_title(self, language: str = "English") -> str:
        return self.title_ja if language.lower() in ("japanese", "ja", "nihongo", "日本語") else self.title_en

    def get_description(self, language: str = "English") -> str:
        return self.description_ja if language.lower() in ("japanese", "ja", "nihongo", "日本語") else self.description_en


class Challenge(BaseModel):
    """An interactive learning challenge."""

    id: str = Field(..., description="Unique challenge ID.")
    icon: str = Field(..., description="Challenge emoji icon.")
    title_en: str = Field(..., description="English challenge title.")
    title_ja: str = Field(..., description="Japanese challenge title.")
    description_en: str = Field(..., description="English description.")
    description_ja: str = Field(..., description="Japanese description.")
    xp_bonus: int = Field(default=100, description="Bonus XP awarded.")
    challenge_type: str = Field(..., description="Type: speed_run, survival, hard_gauntlet, sprint.")

    def get_title(self, language: str = "English") -> str:
        return self.title_ja if language.lower() in ("japanese", "ja", "nihongo", "日本語") else self.title_en

    def get_description(self, language: str = "English") -> str:
        return self.description_ja if language.lower() in ("japanese", "ja", "nihongo", "日本語") else self.description_en


# Master badges catalog
ALL_BADGES: list[Badge] = [
    Badge(
        id="first_concept",
        icon="🐣",
        title_en="First Spark",
        title_ja="初めの一歩",
        description_en="Master your very first lecture concept.",
        description_ja="最初の講義概念をマスターする。",
        xp_reward=50,
        category="concept",
    ),
    Badge(
        id="three_concepts",
        icon="🧠",
        title_en="Concept Scholar",
        title_ja="概念の探求者",
        description_en="Master 3 key lecture concepts.",
        description_ja="3つの重要概念をマスターする。",
        xp_reward=100,
        category="concept",
    ),
    Badge(
        id="master_all_concepts",
        icon="👑",
        title_en="Domain Master",
        title_ja="講義の完全制覇",
        description_en="Master all extracted concepts in this lecture.",
        description_ja="講義内のすべての抽出概念をマスターする。",
        xp_reward=250,
        category="concept",
    ),
    Badge(
        id="first_quiz_correct",
        icon="🎯",
        title_en="Bullseye",
        title_ja="初的中",
        description_en="Answer a multiple choice quiz question correctly.",
        description_ja="クイズ問題に正解する。",
        xp_reward=30,
        category="quiz",
    ),
    Badge(
        id="streak_3",
        icon="🔥",
        title_en="On Fire!",
        title_ja="連勝ストリーク",
        description_en="Answer 3 quiz questions in a row correctly without errors.",
        description_ja="クイズで3問連続正解を達成する。",
        xp_reward=100,
        category="quiz",
    ),
    Badge(
        id="hard_mode",
        icon="🧗",
        title_en="Hardcore Thinker",
        title_ja="難問ブレイカー",
        description_en="Correctly solve a Hard difficulty question.",
        description_ja="上級（Hard）レベルの難問を正解する。",
        xp_reward=120,
        category="quiz",
    ),
    Badge(
        id="perfect_quiz",
        icon="🏆",
        title_en="Quiz Perfectionist",
        title_ja="満点クイズ王",
        description_en="Complete at least 5 questions with 100% accuracy.",
        description_ja="5問以上のクイズに全問正解（精度100%）する。",
        xp_reward=200,
        category="quiz",
    ),
    Badge(
        id="flashcard_five",
        icon="🃏",
        title_en="Memory Maestro",
        title_ja="暗記マスター",
        description_en="Review and flip at least 5 flashcards.",
        description_ja="5枚以上のフラッシュカードを復習する。",
        xp_reward=80,
        category="flashcard",
    ),
    Badge(
        id="video_explorer",
        icon="📺",
        title_en="Video Explorer",
        title_ja="動画探求者",
        description_en="Explore educational video recommendations for 2 concepts.",
        description_ja="2つ以上の概念についておすすめ動画をチェックする。",
        xp_reward=60,
        category="resource",
    ),
    Badge(
        id="polyglot",
        icon="🇯🇵",
        title_en="Bilingual Prodigy",
        title_ja="バイリンガル学習者",
        description_en="Study lecture materials or quizzes in Japanese.",
        description_ja="日本語で講義ノートやクイズを学習する。",
        xp_reward=150,
        category="language",
    ),
    Badge(
        id="speed_runner",
        icon="⚡",
        title_en="Speed Demon",
        title_ja="スピードスター",
        description_en="Complete the Speed Run Challenge.",
        description_ja="スピードランチャレンジをクリアする。",
        xp_reward=150,
        category="challenge",
    ),
    Badge(
        id="survivor",
        icon="🛡️",
        title_en="Survivalist",
        title_ja="サバイバー",
        description_en="Survive 5 questions in Survival sudden-death mode.",
        description_ja="サバイバルモードで5問連続正解して生き残る。",
        xp_reward=200,
        category="challenge",
    ),
    Badge(
        id="chatbot_curious",
        icon="🤖",
        title_en="Socratic Inquirer",
        title_ja="AI対話の探求者",
        description_en="Engage with the AI Lecture Tutor and ask 3 thoughtful questions.",
        description_ja="AI講義チューターに3つ以上の質問をして理解を深める。",
        xp_reward=80,
        category="tutor",
    ),
    Badge(
        id="concept_deep_dive",
        icon="🌊",
        title_en="Deep Diver",
        title_ja="概念深層探求者",
        description_en="Explore expanded in-depth breakdowns on 3+ flashcards.",
        description_ja="フラッシュカードで3つ以上の概念深層解説を学習する。",
        xp_reward=75,
        category="flashcard",
    ),
    Badge(
        id="streak_master",
        icon="⚡",
        title_en="Daily Streak Titan",
        title_ja="ストリークの達人",
        description_en="Achieve and maintain a 3+ day continuous study streak.",
        description_ja="3日以上の連続学習ストリークを達成する。",
        xp_reward=150,
        category="streak",
    ),
    Badge(
        id="visual_learner",
        icon="🎨",
        title_en="Visual Architect",
        title_ja="視覚的モデラー",
        description_en="Inspect and learn from visual diagrams for 3+ key concepts.",
        description_ja="3つ以上の重要概念について視覚的図解アーキテクチャを確認する。",
        xp_reward=70,
        category="concept",
    ),
    Badge(
        id="arena_conqueror",
        icon="👑",
        title_en="Arena Conqueror",
        title_ja="アリーナ完全覇者",
        description_en="Complete 4 or more interactive challenges in the Study Arena.",
        description_ja="学習アリーナで4つ以上の挑戦チャレンジをクリアする。",
        xp_reward=250,
        category="challenge",
    ),
    Badge(
        id="exam_ace",
        icon="🎖️",
        title_en="Exam Ace",
        title_ja="試験の覇者",
        description_en="Score 80% or higher on a lecture practice exam.",
        description_ja="講義の実践試験で80%以上の高得点を獲得する。",
        xp_reward=150,
        category="exam",
    ),
    Badge(
        id="exam_scholar",
        icon="🎓",
        title_en="Exam Scholar",
        title_ja="試験マイスター",
        description_en="Complete 3 or more practice exams.",
        description_ja="3回以上の模擬試験を完了する。",
        xp_reward=200,
        category="exam",
    ),
]


ALL_CHALLENGES: list[Challenge] = [
    Challenge(
        id="speed_run",
        icon="⚡",
        title_en="Speed Run Challenge",
        title_ja="スピードランチャレンジ",
        description_en="Answer 5 questions as fast as possible! Test your instant recall and accuracy.",
        description_ja="5問のクイズに素早く回答！瞬発力と記憶の定着度をテストします。",
        xp_bonus=150,
        challenge_type="speed_run",
    ),
    Challenge(
        id="survival",
        icon="💀",
        title_en="Survival Mode (Sudden Death)",
        title_ja="サバイバルモード（1ミスで即終了）",
        description_en="One mistake and the run is over. How many questions can you conquer in a row?",
        description_ja="1つのミスも許されない！連続何問正解できるか限界に挑戦。",
        xp_bonus=200,
        challenge_type="survival",
    ),
    Challenge(
        id="hard_gauntlet",
        icon="🧗",
        title_en="Hard Mode Gauntlet",
        title_ja="上級ボスラッシュ",
        description_en="Take on only Hard difficulty questions covering nuanced edge-cases and tradeoffs.",
        description_ja="深層理解とトレードオフが問われる上級難問のみに挑戦。",
        xp_bonus=180,
        challenge_type="hard_gauntlet",
    ),
    Challenge(
        id="concept_sprint",
        icon="🏃",
        title_en="Concept Mastery Sprint",
        title_ja="概念マスタリースプリント",
        description_en="Master 3 new concepts in this session to boost your overall completion score.",
        description_ja="このセッションで3つの新概念を完全マスターしてボーナスXPを獲得。",
        xp_bonus=120,
        challenge_type="sprint",
    ),
    Challenge(
        id="tutor_dialogue",
        icon="🤖",
        title_en="Tutor Socratic Dialogue",
        title_ja="AIチューター対話クエスト",
        description_en="Ask the AI Tutor 3 conceptual questions to test nuance, analogies, and edge cases.",
        description_ja="AIチューターと3回対話して、概念の直観的理解や応用例を確認する。",
        xp_bonus=120,
        challenge_type="tutor",
    ),
    Challenge(
        id="visual_mastery_quest",
        icon="🎨",
        title_en="Visual Concept Discovery",
        title_ja="図解概念ディスカバリー",
        description_en="Inspect concept diagrams and discover their architectural mechanisms.",
        description_ja="概念の視覚的アーキテクチャ図解を確認し、構造的な理解を深める。",
        xp_bonus=100,
        challenge_type="visual",
    ),
    Challenge(
        id="deep_dive_sprint",
        icon="🌊",
        title_en="Deep Dive Sprint",
        title_ja="概念深層スプリント",
        description_en="Explore 3 in-depth concept flashcard breakdowns to unlock comprehensive recall.",
        description_ja="フラッシュカードの深層解説を3つ読んで、詳細な知識をインプットする。",
        xp_bonus=110,
        challenge_type="sprint",
    ),
    Challenge(
        id="visual_diagram_challenge",
        icon="🔬",
        title_en="Visual Schematic Diagnosis",
        title_ja="視覚的アーキテクチャ診断",
        description_en="Inspect concept diagrams and diagnose their internal dataflow, boundary states, and constraints.",
        description_ja="概念の視覚的アーキテクチャ図解を読み解き、入力・処理フロー・構造的制約を正しく診断する。",
        xp_bonus=160,
        challenge_type="visual_interactive",
    ),
    Challenge(
        id="concept_mindmap_challenge",
        icon="🧩",
        title_en="Concept Pipeline & Hierarchy Mapping",
        title_ja="概念パイプライン＆階層マッピング",
        description_en="Connect core lecture concepts into their correct architectural and computational sequence.",
        description_ja="講義の重要概念を論理的・計算的な処理順序に合わせて正しく接続・マッピングする。",
        xp_bonus=140,
        challenge_type="mindmap",
    ),
]


class Reward(BaseModel):
    """An unlockable reward or study perk earned by reaching XP milestones."""

    id: str = Field(..., description="Unique reward ID.")
    icon: str = Field(..., description="Reward icon.")
    title_en: str = Field(..., description="English reward title.")
    title_ja: str = Field(..., description="Japanese reward title.")
    description_en: str = Field(..., description="English description.")
    description_ja: str = Field(..., description="Japanese description.")
    required_xp: int = Field(..., description="XP threshold required to unlock.")
    category: str = Field(default="perk", description="Type: perk, theme, certificate, tool.")

    def get_title(self, language: str = "English") -> str:
        return self.title_ja if language.lower() in ("japanese", "ja", "nihongo", "日本語") else self.title_en

    def get_description(self, language: str = "English") -> str:
        return self.description_ja if language.lower() in ("japanese", "ja", "nihongo", "日本語") else self.description_en


ALL_REWARDS: list[Reward] = [
    Reward(
        id="reward_hint_5050",
        icon="💡",
        title_en="50/50 Quiz Distractor Eliminator",
        title_ja="50/50 クイズ誤答消去ヒント",
        description_en="Unlocks the interactive 50/50 hint tool in quizzes to eliminate two wrong distractor choices.",
        description_ja="クイズ内で2つの誤答選択肢を消去できる「50/50ヒント」機能がアンロックされます。",
        required_xp=100,
        category="perk",
    ),
    Reward(
        id="reward_themes",
        icon="🎨",
        title_en="Executive & Cyberpunk Study Themes",
        title_ja="エグゼクティブ＆サイバーパンク学習テーマ",
        description_en="Unlocks bespoke aesthetic visual study themes (Cyberpunk Violet, Emerald Scholar, and Solar Amber).",
        description_ja="サイバーパンク・エメラルド・ソーラーアンバーの美麗なカスタム学習テーマがアンロックされます。",
        required_xp=250,
        category="theme",
    ),
    Reward(
        id="reward_deep_mind_tutor",
        icon="🧠",
        title_en="Deep Reasoning AI Tutor Mode",
        title_ja="深層数理AIチューターモード",
        description_en="Unlocks mathematical derivation steps, proof outlines, and production Python code in AI Tutor responses.",
        description_ja="AIチューターが数式展開・証明の骨子・実践的なPython実装コードを深層解説する特別モード。",
        required_xp=450,
        category="tool",
    ),
    Reward(
        id="reward_audio_podcast",
        icon="🎙️",
        title_en="Lecture Podcast Script Synthesis",
        title_ja="講義ポッドキャスト対話スクリプト生成",
        description_en="Generates a captivating 2-speaker audio discussion script explaining the entire lecture intuitively.",
        description_ja="講義の全重要論点を2人の対話形式で楽しく解説するオーディオポッドキャスト台本を生成。",
        required_xp=700,
        category="tool",
    ),
    Reward(
        id="reward_mastery_certificate",
        icon="📜",
        title_en="Verified Certificate of Lecture Mastery",
        title_ja="公式検証済み 講義修了証書",
        description_en="Generates an official verified completion certificate with cryptographic factuality seal, ready to download.",
        description_ja="ファクト検証率と学習到達度が刻印された、公式認定のダウンロード可能な修了証書。",
        required_xp=950,
        category="certificate",
    ),
    Reward(
        id="reward_olympiad_engine",
        icon="⚡",
        title_en="Doctoral Gauntlet Mode (Level 6)",
        title_ja="博士課程級・超難問ガントレット (Lv.6)",
        description_en="Unlocks graduate-level edge-case stress testing and automated algorithmic proofs.",
        description_ja="大学院レベルのエッジケースストレステストと数理証明自動生成がアンロックされます。",
        required_xp=1500,
        category="tool",
    ),
    Reward(
        id="reward_socratic_examiner",
        icon="🏛️",
        title_en="Socratic Oral Defense Simulator (Level 7)",
        title_ja="ソクラテス式口頭試問シミュレーター (Lv.7)",
        description_en="Unlocks an adversarial AI examiner that challenges every assumption in real-time oral defense.",
        description_ja="リアルタイムで解答の前提や論理の穴を厳しく突いてくる対話型試問シミュレーター。",
        required_xp=2300,
        category="tool",
    ),
    Reward(
        id="reward_quantum_synthesis",
        icon="🌌",
        title_en="Cross-Domain Synoptic Synthesis (Level 8)",
        title_ja="異分野横断シナプシス統合 (Lv.8)",
        description_en="Unlocks automated synthesis connecting this lecture with external research disciplines.",
        description_ja="本講義の内容を隣接する研究分野や最先端論文と自動的に架橋・統合する高度機能。",
        required_xp=3300,
        category="perk",
    ),
    Reward(
        id="reward_professor_workbench",
        icon="🎓",
        title_en="Master Curriculum Designer Suite (Level 9)",
        title_ja="大学教授カリキュラム設計スイート (Lv.9)",
        description_en="Unlocks exportable course syllabi, lecture slides outline, and exam blueprint generation.",
        description_ja="大学院シラバス・講義スライド骨子・試験ブループリントの自動設計スイート。",
        required_xp=4600,
        category="tool",
    ),
    Reward(
        id="reward_omniscient_hall_of_fame",
        icon="👑",
        title_en="Hall of Fame: Omniscient Laureate (Level 10)",
        title_ja="殿堂入り: 全知の碩学 (Lv.10)",
        description_en="Maximum honor: Golden immortalized learner profile and unlocked infinite difficulty sandbox.",
        description_ja="最高峰の名誉: 黄金の殿堂入りプロフィールと無限難易度サンドボックスの全解放。",
        required_xp=6200,
        category="certificate",
    ),
]


def get_all_badges() -> list[Badge]:
    return list(ALL_BADGES)


def get_all_challenges() -> list[Challenge]:
    return list(ALL_CHALLENGES)


def get_all_rewards() -> list[Reward]:
    return list(ALL_REWARDS)


class UserProgress:
    """Manages active user state, XP, completed concepts, streak, and unlocked badges."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        raw = data or {}
        self.completed_concepts: set[str] = set(raw.get("completed_concepts", []))
        self.quiz_answers: list[dict[str, Any]] = list(raw.get("quiz_answers", []))
        self.reviewed_flashcards: set[int] = set(raw.get("reviewed_flashcards", []))
        self.explored_videos: set[str] = set(raw.get("explored_videos", []))
        self.xp: int = int(raw.get("xp", 0))
        self.daily_streak: int = int(raw.get("daily_streak", raw.get("streak", 1)))
        self.best_daily_streak: int = int(raw.get("best_daily_streak", raw.get("best_streak", self.daily_streak)))
        self.streak: int = self.daily_streak
        self.best_streak: int = self.best_daily_streak
        self.quiz_streak: int = int(raw.get("quiz_streak", 0))
        self.best_quiz_streak: int = int(raw.get("best_quiz_streak", 0))
        self.unlocked_badge_ids: set[str] = set(raw.get("unlocked_badge_ids", []))
        self.completed_challenges: set[str] = set(raw.get("completed_challenges", []))
        self.studied_languages: set[str] = set(raw.get("studied_languages", ["English"]))
        self.chat_queries_count: int = int(raw.get("chat_queries_count", 0))
        self.deep_dives_viewed: set[str] = set(raw.get("deep_dives_viewed", []))
        self.visuals_viewed: set[str] = set(raw.get("visuals_viewed", []))
        self.claimed_rewards: set[str] = set(raw.get("claimed_rewards", []))
        # Exam history: list of {score_pct, correct, total, difficulty, elapsed_secs, xp_earned, lecture_title, ts}
        self.exam_history: list[dict[str, Any]] = list(raw.get("exam_history", []))
        self.last_login_date: str = str(raw.get("last_login_date", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_concepts": list(self.completed_concepts),
            "quiz_answers": self.quiz_answers,
            "reviewed_flashcards": list(self.reviewed_flashcards),
            "explored_videos": list(self.explored_videos),
            "xp": self.xp,
            "streak": self.daily_streak,
            "best_streak": self.best_daily_streak,
            "daily_streak": self.daily_streak,
            "best_daily_streak": self.best_daily_streak,
            "quiz_streak": self.quiz_streak,
            "best_quiz_streak": self.best_quiz_streak,
            "unlocked_badge_ids": list(self.unlocked_badge_ids),
            "completed_challenges": list(self.completed_challenges),
            "studied_languages": list(self.studied_languages),
            "chat_queries_count": self.chat_queries_count,
            "deep_dives_viewed": list(self.deep_dives_viewed),
            "visuals_viewed": list(self.visuals_viewed),
            "claimed_rewards": list(self.claimed_rewards),
            "exam_history": self.exam_history,
            "last_login_date": self.last_login_date,
        }

    def reset_progress(self) -> None:
        """Reset progress back to Level 1, 0 XP, Day 1 Streak."""
        import datetime as _dt
        self.completed_concepts.clear()
        self.quiz_answers.clear()
        self.reviewed_flashcards.clear()
        self.explored_videos.clear()
        self.xp = 0
        self.daily_streak = 1
        self.best_daily_streak = 1
        self.streak = 1
        self.best_streak = 1
        self.quiz_streak = 0
        self.best_quiz_streak = 0
        self.unlocked_badge_ids.clear()
        self.completed_challenges.clear()
        self.chat_queries_count = 0
        self.deep_dives_viewed.clear()
        self.visuals_viewed.clear()
        self.claimed_rewards.clear()
        self.exam_history.clear()
        self.last_login_date = _dt.date.today().isoformat()


    def is_reward_unlocked(self, reward_id: str) -> bool:
        """Check if user has reached the XP required for a specific reward."""
        r = next((x for x in ALL_REWARDS if x.id == reward_id), None)
        return self.xp >= r.required_xp if r else False

    def claim_reward(self, reward_id: str) -> bool:
        """Mark a reward as claimed by the user."""
        if self.is_reward_unlocked(reward_id):
            self.claimed_rewards.add(reward_id)
            return True
        return False

    def is_challenge_eligible(
        self,
        challenge_id: str,
        session_cleared_flags: dict[str, bool] | None = None,
    ) -> tuple[bool, str, float]:
        """Check whether a challenge is legitimately completed and eligible for claiming XP."""
        flags = session_cleared_flags or {}
        if challenge_id == "concept_sprint":
            done = len(self.completed_concepts)
            return done >= 3, f"{min(3, done)} / 3 concepts", min(1.0, done / 3.0)
        elif challenge_id == "tutor_dialogue":
            done = self.chat_queries_count
            return done >= 3, f"{min(3, done)} / 3 queries", min(1.0, done / 3.0)
        elif challenge_id == "visual_mastery_quest":
            done = len(self.visuals_viewed)
            return done >= 3, f"{min(3, done)} / 3 diagrams", min(1.0, done / 3.0)
        elif challenge_id == "deep_dive_sprint":
            done = len(self.deep_dives_viewed)
            return done >= 3, f"{min(3, done)} / 3 breakdowns", min(1.0, done / 3.0)
        elif challenge_id in (
            "speed_run",
            "survival",
            "hard_gauntlet",
            "visual_diagram_challenge",
            "concept_mindmap_challenge",
        ):
            cleared = bool(flags.get(f"cleared_{challenge_id}", False))
            status_txt = "Challenge Cleared in Arena! Ready to Claim" if cleared else "Play and pass in the Arena below"
            return cleared, status_txt, 1.0 if cleared else 0.0
        return False, "Not yet completed", 0.0

    def save_to_db(self, user_id: int) -> None:
        """Persist user progress to SQLite database."""
        try:
            from lecturenotes.db.database import save_user_progress_db
            save_user_progress_db(user_id, self.to_dict())
        except Exception:
            pass

    @classmethod
    def load_from_db(cls, user_id: int) -> UserProgress:
        """Load user progress from SQLite database for specified user_id."""
        try:
            from lecturenotes.db.database import load_user_progress_db
            data = load_user_progress_db(user_id)
            return cls(data)
        except Exception:
            return cls()

    def save_to_disk(self, filepath: Path | str | None = None) -> None:
        """Persist user progress to local cache file."""
        path = Path(filepath) if filepath else Path(".cache") / "user_profile.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def load_from_disk(cls, filepath: Path | str | None = None) -> UserProgress:
        """Load user progress from disk if saved."""
        path = Path(filepath) if filepath else Path(".cache") / "user_profile.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(data)
            except Exception:
                pass
        return cls()

    # ── Leveling computation ──
    def get_level_info(self) -> dict[str, Any]:
        """Compute user level, title, and progress percentage."""
        thresholds = [
            (1, "Novice (見習い)", 0, 100),
            (2, "Apprentice (修習生)", 100, 250),
            (3, "Scholar (研究員)", 250, 500),
            (4, "Expert (専門家)", 500, 900),
            (5, "Specialist (専門特化)", 900, 1500),
            (6, "Master (達人)", 1500, 2300),
            (7, "Grandmaster (大達人)", 2300, 3300),
            (8, "Luminary (碩学)", 3300, 4600),
            (9, "Mythic Sage (神話の賢者)", 4600, 6200),
            (10, "Omniscient Legend (全知の領域)", 6200, 8500),
        ]
        curr_lvl = 1
        title = "Novice"
        low = 0
        high = 100

        for lvl, name, l_bound, h_bound in thresholds:
            if self.xp >= l_bound:
                curr_lvl = lvl
                title = name
                low = l_bound
                high = h_bound

        if self.xp >= 8500:
            curr_lvl = 10
            title = "Omniscient Legend (全知の領域)"
            pct = 1.0
            next_xp = 0
        else:
            pct = max(0.0, min(1.0, (self.xp - low) / (high - low)))
            next_xp = high - self.xp

        return {
            "level": curr_lvl,
            "title": title,
            "xp": self.xp,
            "pct": pct,
            "next_level_xp": next_xp,
        }

    # ── Actions & Badge Unlocks ──
    def mark_concept_completed(self, term: str, total_concepts_count: int = 1) -> list[Badge]:
        """Mark a concept as completed, award XP, and check for unlocks."""
        term_clean = term.strip().lower()
        if term_clean in self.completed_concepts:
            return []

        self.completed_concepts.add(term_clean)
        self.xp += 50
        return self._evaluate_badges(total_concepts_count=total_concepts_count)

    def unmark_concept(self, term: str) -> None:
        """Unmark a concept."""
        term_clean = term.strip().lower()
        self.completed_concepts.discard(term_clean)

    def record_quiz_answer(
        self,
        question_id: int,
        is_correct: bool,
        difficulty: str = "Medium",
        total_quiz_count: int = 5,
    ) -> list[Badge]:
        """Record a quiz attempt, update streak/XP, and return newly unlocked badges."""
        diff_weights = {"Easy": 10, "Medium": 20, "Hard": 35}
        earned_xp = diff_weights.get(difficulty.capitalize(), 20) if is_correct else 2

        self.quiz_answers.append(
            {
                "question_id": question_id,
                "is_correct": is_correct,
                "difficulty": difficulty.capitalize(),
            }
        )

        if is_correct:
            self.quiz_streak += 1
            if self.quiz_streak > self.best_quiz_streak:
                self.best_quiz_streak = self.quiz_streak
            self.xp += earned_xp
        else:
            self.quiz_streak = 0
            self.xp += earned_xp

        return self._evaluate_badges(total_quiz_count=total_quiz_count)

    def record_flashcard_review(self, card_id: int) -> list[Badge]:
        """Record reviewing a flashcard."""
        if card_id not in self.reviewed_flashcards:
            self.reviewed_flashcards.add(card_id)
            self.xp += 10
        return self._evaluate_badges()

    def record_video_explored(self, term: str) -> list[Badge]:
        """Record video suggestion exploration."""
        term_clean = term.strip().lower()
        if term_clean not in self.explored_videos:
            self.explored_videos.add(term_clean)
            self.xp += 15
        return self._evaluate_badges()

    def record_language(self, language: str) -> list[Badge]:
        """Record studying in a specific language."""
        self.studied_languages.add(language)
        if language.lower() in ("japanese", "ja", "nihongo", "日本語"):
            return self._evaluate_badges()
        return []

    def record_challenge_completed(self, challenge_id: str) -> list[Badge]:
        """Record completing an interactive challenge."""
        if challenge_id not in self.completed_challenges:
            self.completed_challenges.add(challenge_id)
            ch = next((c for c in ALL_CHALLENGES if c.id == challenge_id), None)
            if ch:
                self.xp += ch.xp_bonus
        return self._evaluate_badges()

    def _evaluate_badges(self, total_concepts_count: int = 1, total_quiz_count: int = 5) -> list[Badge]:
        """Check all badge rules and unlock qualifying badges."""
        newly_unlocked: list[Badge] = []
        badge_map = {b.id: b for b in ALL_BADGES}

        # 1. First concept
        if "first_concept" not in self.unlocked_badge_ids and len(self.completed_concepts) >= 1:
            self._unlock("first_concept", newly_unlocked, badge_map)

        # 2. Three concepts
        if "three_concepts" not in self.unlocked_badge_ids and len(self.completed_concepts) >= 3:
            self._unlock("three_concepts", newly_unlocked, badge_map)

        # 3. Master all concepts
        if (
            "master_all_concepts" not in self.unlocked_badge_ids
            and total_concepts_count > 0
            and len(self.completed_concepts) >= total_concepts_count
        ):
            self._unlock("master_all_concepts", newly_unlocked, badge_map)

        # 4. First quiz correct
        if "first_quiz_correct" not in self.unlocked_badge_ids:
            if any(a.get("is_correct") for a in self.quiz_answers):
                self._unlock("first_quiz_correct", newly_unlocked, badge_map)

        # 5. Streak of 3
        if "streak_3" not in self.unlocked_badge_ids and (self.best_quiz_streak >= 3 or self.best_daily_streak >= 3):
            self._unlock("streak_3", newly_unlocked, badge_map)

        # 6. Hard mode correct
        if "hard_mode" not in self.unlocked_badge_ids:
            if any(a.get("is_correct") and a.get("difficulty") == "Hard" for a in self.quiz_answers):
                self._unlock("hard_mode", newly_unlocked, badge_map)

        # 7. Perfect quiz (5+ answered and all correct)
        if "perfect_quiz" not in self.unlocked_badge_ids and len(self.quiz_answers) >= min(5, total_quiz_count):
            recent = self.quiz_answers[-min(5, total_quiz_count):]
            if len(recent) >= 5 and all(a.get("is_correct") for a in recent):
                self._unlock("perfect_quiz", newly_unlocked, badge_map)

        # 8. Flashcards
        if "flashcard_five" not in self.unlocked_badge_ids and len(self.reviewed_flashcards) >= 5:
            self._unlock("flashcard_five", newly_unlocked, badge_map)

        # 9. Video Explorer
        if "video_explorer" not in self.unlocked_badge_ids and len(self.explored_videos) >= 2:
            self._unlock("video_explorer", newly_unlocked, badge_map)

        # 10. Polyglot
        if "polyglot" not in self.unlocked_badge_ids:
            if any(lang.lower() in ("japanese", "ja", "nihongo", "日本語") for lang in self.studied_languages):
                self._unlock("polyglot", newly_unlocked, badge_map)

        # 11. Speed runner challenge
        if "speed_runner" not in self.unlocked_badge_ids and "speed_run" in self.completed_challenges:
            self._unlock("speed_runner", newly_unlocked, badge_map)

        # 12. Survivor challenge
        if "survivor" not in self.unlocked_badge_ids and "survival" in self.completed_challenges:
            self._unlock("survivor", newly_unlocked, badge_map)

        # 13. Chatbot Socratic Inquirer
        if "chatbot_curious" not in self.unlocked_badge_ids and self.chat_queries_count >= 3:
            self._unlock("chatbot_curious", newly_unlocked, badge_map)

        # 14. Concept Deep Diver
        if "concept_deep_dive" not in self.unlocked_badge_ids and len(self.deep_dives_viewed) >= 3:
            self._unlock("concept_deep_dive", newly_unlocked, badge_map)

        # 15. Daily Streak Titan
        _daily_st = max(self.daily_streak, getattr(self, "streak", 0), self.best_daily_streak)
        if "streak_master" not in self.unlocked_badge_ids and (_daily_st >= 3 or self.best_quiz_streak >= 5):
            self._unlock("streak_master", newly_unlocked, badge_map)

        # 16. Visual Architect
        if "visual_learner" not in self.unlocked_badge_ids and len(self.visuals_viewed) >= 3:
            self._unlock("visual_learner", newly_unlocked, badge_map)

        # 17. Arena Conqueror (4 or more challenges completed)
        if "arena_conqueror" not in self.unlocked_badge_ids and len(self.completed_challenges) >= 4:
            self._unlock("arena_conqueror", newly_unlocked, badge_map)

        # 18. Exam Ace (>= 80% on an exam)
        if "exam_ace" not in self.unlocked_badge_ids:
            if any(e.get("score_pct", 0) >= 80 for e in self.exam_history):
                self._unlock("exam_ace", newly_unlocked, badge_map)

        # 19. Exam Scholar (>= 3 exams taken)
        if "exam_scholar" not in self.unlocked_badge_ids and len(self.exam_history) >= 3:
            self._unlock("exam_scholar", newly_unlocked, badge_map)

        return newly_unlocked

    def record_exam_result(
        self,
        score_pct: int,
        correct: int,
        total: int,
        difficulty: str,
        elapsed_secs: int,
        xp_earned: int,
        lecture_title: str = "",
    ) -> list[Badge]:
        """Record a completed exam attempt in persistent history and award XP."""
        import datetime as _dt
        self.exam_history.append({
            "score_pct": score_pct,
            "correct": correct,
            "total": total,
            "difficulty": difficulty,
            "elapsed_secs": elapsed_secs,
            "xp_earned": xp_earned,
            "lecture_title": lecture_title,
            "ts": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        self.xp += xp_earned
        return self._evaluate_badges()

    def record_chat_query(self) -> list[Badge]:
        """Record a chat query with the AI Tutor and award XP."""
        self.chat_queries_count += 1
        self.xp += 10
        if self.chat_queries_count >= 3 and "tutor_dialogue" not in self.completed_challenges:
            self.completed_challenges.add("tutor_dialogue")
        return self._evaluate_badges()

    def record_deep_dive(self, term: str) -> list[Badge]:
        """Record expanding a concept deep dive."""
        term_clean = (term or "").strip().lower()
        if term_clean and term_clean not in self.deep_dives_viewed:
            self.deep_dives_viewed.add(term_clean)
            self.xp += 15
        if len(self.deep_dives_viewed) >= 3 and "deep_dive_sprint" not in self.completed_challenges:
            self.completed_challenges.add("deep_dive_sprint")
        return self._evaluate_badges()

    def record_visual_viewed(self, term: str) -> list[Badge]:
        """Record inspecting a concept visual diagram."""
        term_clean = (term or "").strip().lower()
        if term_clean and term_clean not in self.visuals_viewed:
            self.visuals_viewed.add(term_clean)
            self.xp += 15
        if len(self.visuals_viewed) >= 3 and "visual_mastery_quest" not in self.completed_challenges:
            self.completed_challenges.add("visual_mastery_quest")
        return self._evaluate_badges()

    def _unlock(self, badge_id: str, accumulator: list[Badge], badge_map: dict[str, Badge]) -> None:
        self.unlocked_badge_ids.add(badge_id)
        b = badge_map.get(badge_id)
        if b:
            self.xp += b.xp_reward
            accumulator.append(b)
