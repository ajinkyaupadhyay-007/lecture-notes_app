"""Provider-agnostic LLM wrapper class.

Supports Azure OpenAI, Claude API (Anthropic), and Ollama (local llama3.1:8b) fallback.
The active provider is selected via config.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from lecturenotes.config import AppConfig, load_config
from lecturenotes.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified, provider-agnostic interface for LLM JSON generation."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.llm_cfg = self.config.llm
        self.provider = self.llm_cfg.provider.lower()
        logger.info("Initialized LLMClient with provider: %s", self.provider)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        """Send prompt to the configured LLM provider and return parsed JSON."""
        logger.debug("Generating JSON using provider=%s", self.provider)

        if self.provider == "gemini":
            raw_response = self._call_gemini(prompt, system_prompt)
        elif self.provider in ("openai", "chatgpt"):
            raw_response = self._call_openai(prompt, system_prompt)
        elif self.provider == "azure":
            raw_response = self._call_azure(prompt, system_prompt)
        elif self.provider == "claude":
            raw_response = self._call_claude(prompt, system_prompt)
        elif self.provider == "ollama":
            raw_response = self._call_ollama(prompt, system_prompt)
        elif self.provider == "mock":
            raw_response = self._call_mock(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: '{self.provider}'. Expected 'gemini', 'openai', 'azure', 'claude', 'ollama', or 'mock'.")

        return self._clean_and_parse_json(raw_response)

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful pedagogical AI tutor.",
    ) -> str:
        """Send prompt to the configured LLM provider and return plain text response."""
        logger.debug("Generating text using provider=%s", self.provider)

        if self.provider == "gemini":
            return self._call_gemini_text(prompt, system_prompt)
        elif self.provider in ("openai", "chatgpt"):
            return self._call_openai_text(prompt, system_prompt)
        elif self.provider == "azure":
            return self._call_azure_text(prompt, system_prompt)
        elif self.provider == "claude":
            return self._call_claude(prompt, system_prompt)
        elif self.provider == "ollama":
            return self._call_ollama_text(prompt, system_prompt)
        elif self.provider == "mock":
            return "Based on this lecture, the core principles revolve around understanding the foundational concepts, empirical evaluation, and iterative refinement."
        else:
            raise ValueError(f"Unsupported LLM provider: '{self.provider}'. Expected 'gemini', 'openai', 'azure', 'claude', 'ollama', or 'mock'.")

    @classmethod
    def reset_gemini_quota(cls) -> None:
        """Reset the quota exhaustion flag when a new key or provider is set."""
        cls._gemini_quota_exhausted = False

    def _call_gemini_text(self, prompt: str, system_prompt: str) -> str:
        """Call Gemini REST API for free-form conversational text."""
        if LLMClient._gemini_quota_exhausted:
            raise RuntimeError("Gemini quota exhausted.")
        api_key = self.llm_cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing.")

        pref_model = self.llm_cfg.gemini_model or "gemini-2.5-flash"
        candidate_models = ["gemini-2.5-flash", pref_model, "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest"]
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.llm_cfg.temperature,
                "maxOutputTokens": self.llm_cfg.max_tokens,
            },
        }

        for model in dict.fromkeys(candidate_models):
            model_path = model if model.startswith("models/") else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
            try:
                with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
                    res = client.post(url, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                    if res.status_code in (401, 403):
                        LLMClient._gemini_quota_exhausted = True
                        raise RuntimeError(f"Gemini API authentication error ({res.status_code}): {res.text}")
                    # If 429 or 404, continue to next candidate model
            except Exception as e:
                logger.debug("Gemini text candidate %s failed: %s", model, e)
        LLMClient._gemini_quota_exhausted = True
        raise RuntimeError("Gemini text call failed.")

    def _call_azure_text(self, prompt: str, system_prompt: str) -> str:
        """Call Azure OpenAI for free-form conversational text."""
        from openai import AzureOpenAI
        endpoint = self.llm_cfg.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = self.llm_cfg.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self.llm_cfg.azure_api_version,
        )
        response = client.chat.completions.create(
            model=self.llm_cfg.azure_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_ollama_text(self, prompt: str, system_prompt: str) -> str:
        """Call local Ollama for conversational text."""
        base_url = self.llm_cfg.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": self.llm_cfg.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": self.llm_cfg.temperature, "num_predict": self.llm_cfg.max_tokens},
        }
        with httpx.Client(timeout=httpx.Timeout(45.0, connect=2.0)) as client:
            res = client.post(url, json=payload)
            res.raise_for_status()
            return res.json().get("message", {}).get("content", "")

    _gemini_quota_exhausted: bool = False

    def _call_gemini(self, prompt: str, system_prompt: str) -> str:
        """Execute request using Google Gemini REST API with native JSON response mode."""
        if LLMClient._gemini_quota_exhausted:
            raise RuntimeError("Gemini API key is invalid or unauthorized; fast-failing to extractive fallback.")

        api_key = self.llm_cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini provider selected but GEMINI_API_KEY (or GOOGLE_API_KEY) is missing in config or environment."
            )

        pref_model = self.llm_cfg.gemini_model or "gemini-2.5-flash"
        raw_candidates = [pref_model, "gemini-2.5-flash", "gemini-3-flash-preview"]
        candidate_models = [
            m for m in dict.fromkeys(raw_candidates)
            if m not in ("gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.5-flash-lite", "gemini-flash-lite-latest")
        ]
        last_error = None

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.llm_cfg.temperature,
                "maxOutputTokens": self.llm_cfg.max_tokens,
                "responseMimeType": "application/json",
            },
        }

        import time

        for model in candidate_models:
            model_path = model if model.startswith("models/") else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"

            for attempt in range(2):
                try:
                    with httpx.Client(timeout=httpx.Timeout(90.0, connect=8.0)) as client:
                        res = client.post(url, json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    return parts[0].get("text", "{}")

                        err_msg = res.text
                        try:
                            err_json = res.json()
                            err_msg = err_json.get("error", {}).get("message", res.text)
                        except Exception:
                            pass
                        last_error = RuntimeError(f"Gemini API ({model}) failed ({res.status_code}): {err_msg}")
                        if res.status_code in (401, 403):
                            LLMClient._gemini_quota_exhausted = True
                            raise last_error

                        if res.status_code in (429, 503):
                            time.sleep(2.0 * (attempt + 1))
                            continue
                        break
                except httpx.RequestError as err:
                    last_error = ConnectionError(f"Network error connecting to Gemini API: {err}")
                    time.sleep(1.0)
                    continue

        if last_error:
            raise last_error
        raise RuntimeError("Gemini API call failed with unknown error.")

    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Execute request using standard OpenAI API via openai SDK."""
        from openai import OpenAI

        api_key = self.llm_cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI selected but OPENAI_API_KEY is missing in config or environment.")

        client = OpenAI(api_key=api_key)
        model = getattr(self.llm_cfg, "openai_model", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
        )
        return response.choices[0].message.content or "{}"

    def _call_openai_text(self, prompt: str, system_prompt: str) -> str:
        """Call standard OpenAI API for conversational text."""
        from openai import OpenAI

        api_key = self.llm_cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI selected but OPENAI_API_KEY is missing in config or environment.")

        client = OpenAI(api_key=api_key)
        model = getattr(self.llm_cfg, "openai_model", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_azure(self, prompt: str, system_prompt: str) -> str:
        """Execute request using Azure OpenAI via the openai SDK."""
        from openai import AzureOpenAI

        endpoint = self.llm_cfg.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_key = self.llm_cfg.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")

        if not endpoint or not api_key:
            raise ValueError("Azure OpenAI selected but AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is missing.")

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self.llm_cfg.azure_api_version,
        )

        response = client.chat.completions.create(
            model=self.llm_cfg.azure_deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
        )
        content = response.choices[0].message.content or "{}"
        return content

    def _call_claude(self, prompt: str, system_prompt: str) -> str:
        """Execute request using Anthropic Claude API."""
        import anthropic

        api_key = self.llm_cfg.claude_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Claude provider selected but ANTHROPIC_API_KEY is missing in config or environment.")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.llm_cfg.claude_model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.llm_cfg.temperature,
            max_tokens=self.llm_cfg.max_tokens,
        )
        block = response.content[0]
        return getattr(block, "text", "")

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """Execute request using local Ollama REST API (free fallback)."""
        base_url = self.llm_cfg.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": self.llm_cfg.ollama_model,
            "format": "json",
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self.llm_cfg.temperature,
                "num_predict": self.llm_cfg.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=2.0)) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                return data.get("message", {}).get("content", "{}")
        except (httpx.ConnectError, httpx.ConnectTimeout) as err:
            raise ConnectionError(
                f"Could not connect to Ollama at {url}. "
                "Ensure Ollama is running ('ollama serve') and model 'llama3.1:8b' is pulled ('ollama pull llama3.1:8b')."
            ) from err

    def _call_mock(self, prompt: str) -> str:
        """Synthetic mock generator for fast, deterministic, offline unit testing."""
        # Detect language and difficulty
        is_ja = "japanese" in prompt.lower() or "日本語" in prompt
        diff_match = re.search(r"DIFFICULTY LEVEL:\s*(EASY|MEDIUM|HARD)", prompt, re.IGNORECASE)
        difficulty = diff_match.group(1).capitalize() if diff_match else "Medium"

        # Find section_id if in prompt
        sec_match = re.search(r"SECTION ID:\s*(\d+)", prompt)
        sec_id = int(sec_match.group(1)) if sec_match else 1

        title_match = re.search(r"SECTION TITLE:\s*([^\n]+)", prompt)
        sec_title = title_match.group(1).strip() if title_match else "Overview"

        # Check prompt type
        if "flashcards" in prompt.lower():
            cards = []
            for i in range(1, 13):
                if is_ja:
                    cards.append(
                        {
                            "card_id": i,
                            "question": f"復習カード {i}: 教師あり学習における訓練データの役割は何ですか？",
                            "answer": "教師あり学習では、入力xから出力yへのマッピング関数f(x)を近似学習するためにラベル付き訓練データを使用します。",
                            "source_span": "In supervised learning, we are given a training set of labelled examples {(x_1, y_1), ..., (x_n, y_n)}.",
                            "concept": "教師あり学習",
                        }
                    )
                else:
                    cards.append(
                        {
                            "card_id": i,
                            "question": f"Review Question {i}: What is the role of training data in supervised learning?",
                            "answer": "In supervised learning, training examples are used to learn a mapping function f such that f(x) approximates y.",
                            "source_span": "In supervised learning, we are given a training set of labelled examples {(x_1, y_1), ..., (x_n, y_n)}.",
                            "concept": "supervised learning",
                        }
                    )
            return json.dumps({"flashcards": cards})

        elif "quiz" in prompt.lower() or "mcq" in prompt.lower():
            quiz_items = []
            count_match = re.search(r"Generate exactly (\d+)", prompt)
            q_count = int(count_match.group(1)) if count_match else 5

            for i in range(1, q_count + 1):
                # Cycle difficulty if not specifically set or enforce requested difficulty
                curr_diff = difficulty if difficulty in ("Easy", "Hard") else ("Easy" if i == 1 else ("Hard" if i >= 4 else "Medium"))

                if is_ja:
                    if curr_diff == "Easy":
                        q_text = f"設問 {i} (初級): ラベル付きデータから未知のデータの出力を予測する機械学習の手法はどれですか？"
                        opts = {
                            "A": "教師あり学習 (Supervised Learning)",
                            "B": "クラスタリング (Clustering)",
                            "C": "次元削減 (Dimensionality Reduction)",
                            "D": "主成分分析 (PCA)",
                        }
                        correct = "A"
                        expl = "教師あり学習は、入力と正解ラベルのペアから構成される訓練データを用いて学習を行います。"
                        dist_expl = {
                            "B": "クラスタリングは教師なし学習の代表例であり、正解ラベルを用いません。",
                            "C": "次元削減は特徴量の次元を圧縮する教師なし手法です。",
                            "D": "主成分分析はデータの分散を最大化する教師なし射影手法です。",
                        }
                        span = "In supervised learning, we are given a training set of labelled examples {(x_1, y_1), ..., (x_n, y_n)}."
                    elif curr_diff == "Hard":
                        hard_pool_ja = [
                            (
                                f"設問 {i} (上級): L1正則化（Lasso）がL2正則化（Ridge）と異なり特徴量選択（スパース性）を促進する幾何学的理由として正しいものはどれですか？",
                                {
                                    "A": "L1ノルム球の角（特異点）が座標軸上にあり、損失関数の等高線が軸上で接しやすいため",
                                    "B": "L1正則化は目的関数のヘッセ行列を厳密にゼロにするため",
                                    "C": "L1正則化は凸関数ではなく非線形カーネルでのみ局所解を持つため",
                                    "D": "L1正則化項はすべての重みを一律に同じ割合で減衰させるため",
                                },
                                "A",
                                "L1ノルムの等高線はダイヤモンド型（角ばった形状）をしており、損失関数の楕円等高線が軸上の頂点で接する確率が高いため、係数が厳密にゼロになります。",
                                {
                                    "B": "ヘッセ行列をゼロにするわけではありません。",
                                    "C": "L1正則化項は凸関数です。",
                                    "D": "一律の割合で減衰させるのはL2重み減衰（Weight Decay）の特性です。",
                                },
                                "L1 regularisation (Lasso) adds λ||w||_1, which can drive some weights to exactly zero, producing sparse models.",
                            ),
                            (
                                f"設問 {i} (上級): 平均二乗誤差（MSE）のバイアス・バリアンス分解において、「削減不能誤差（σ^2）」が生じる本質的要因は何ですか？",
                                {
                                    "A": "観測不能な潜在変数や真のデータ生成過程に内在する確率的ノイズ",
                                    "B": "モデルの表現力不足による系統的な近似バイアス",
                                    "C": "有限の訓練標本サンプリングに起因する推定量の統計的ばらつき",
                                    "D": "非凸最適化における局所最適解（ローカルミニマ）への捕捉",
                                },
                                "A",
                                "削減不能誤差（不可避誤差）は、目的変数自体の内在的ランダムノイズに由来し、モデルの構造や学習量をいくら改善してもゼロにはできません。",
                                {
                                    "B": "表現力不足はバイアスであり、モデルの複雑度を上げることで削減可能です。",
                                    "C": "標本抽出のばらつきはバリアンスであり、データ量の増加で削減可能です。",
                                    "D": "最適化の局所解はアルゴリズム側の課題であり、削減不能ノイズとは区別されます。",
                                },
                                "The expected test error decomposes into (Bias)^2 + Variance + Irreducible noise.",
                            ),
                            (
                                f"設問 {i} (上級): 再生核ヒルベルト空間（RKHS）において、ある関数が有効なマーサーカーネル（Mercer Kernel）となるための必要十分条件は何ですか？",
                                {
                                    "A": "任意の有限標本に対して構築されるグラム行列（カーネル行列）が対称かつ半正定値であること",
                                    "B": "カーネル関数のリプシッツ定数が全域で1未満であること",
                                    "C": "双対最適化問題における主実行可能領域が有界であること",
                                    "D": "特徴写像関数が可微分な全単射同相写像であること",
                                },
                                "A",
                                "マーサーの定理により、カーネル関数が内積を暗黙的に表現できるための条件は、カーネル行列が対称半正定値行列（Symmetric Positive Semi-Definite）であることです。",
                                {
                                    "B": "リプシッツ連続性は勾配安定性の性質であり、マーサー条件とは直接関係ありません。",
                                    "C": "有界性のみではカーネルの有効性を保証できません。",
                                    "D": "特徴空間への写像は全単射である必要はなく、無限次元への写像も許容されます。",
                                },
                                "By Mercer's theorem, any continuous, symmetric, positive semi-definite kernel computes an inner product in a transformed feature space.",
                            ),
                            (
                                f"設問 {i} (上級): 高次元非凸損失曲面において、確率的勾配降下法（SGD）が鞍点（Saddle Point）を脱出できる数理的メカニズムは何ですか？",
                                {
                                    "A": "ミニバッチサンプリングによる勾配ノイズが、負の曲率方向への脱出を促す等方的な揺らぎを与えるため",
                                    "B": "SGDが停滞点においてヘッセ行列の逆行列を解析的に計算するため",
                                    "C": "学習率スケジュールが目的関数を大域的強凸関数へと変換するため",
                                    "D": "ミニバッチ化によりパラメータ空間のゼロ固有値空間が完全に消滅するため",
                                },
                                "A",
                                "SGDのミニバッチ勾配が持つ確率的ノイズ（Gradient Noise）は、負の固有値を持つ鞍点において摂動として働き、不安定な鞍点から脱出するダイナミクスを形成します。",
                                {
                                    "B": "SGDは一次勾配法であり、ヘッセ行列の計算は行いません。",
                                    "C": "学習率は更新幅を制御するもので、損失関数の曲率自体を変えることはできません。",
                                    "D": "ミニバッチ化しても固有値空間が消えるわけではありません。",
                                },
                                "Stochastic gradient noise acts as an implicit regularizer that helps escape strict saddle points and find flatter minima.",
                            ),
                            (
                                f"設問 {i} (上級): サポートベクターマシン（SVM）におけるマージン最大化問題が、L2正則化ヒンジ損失最小化と等価になる条件は何ですか？",
                                {
                                    "A": "ソフトマージンヒンジ損失関数に対して法線ベクトルwのL2正則化項を付加して定式化した場合",
                                    "B": "決定境界が必ず訓練サンプルの幾何学的中心（重心）を通過する制約を置いた場合",
                                    "C": "スラック変数ξ_iに対してL-infinityノルムペナルティを課した場合",
                                    "D": "正則化ハイパーパラメータCを厳密にゼロに漸近させた場合",
                                },
                                "A",
                                "ソフトマージンSVMの最適化は、各サンプルのヒンジ損失の総和と (1/2)||w||^2 の和を最小化する正則化経験損失最小化（ERM）と完全に等価です。",
                                {
                                    "B": "SVMの境界はサポートベクターによって決定され、全体の重心を通るとは限りません。",
                                    "C": "標準的なSVMではスラック変数にはL1ペナルティ（線形損失）が課されます。",
                                    "D": "Cが0に近づくと正則化項のみが重視され、マージン最大化の意味が失われます。",
                                },
                                "Soft-margin SVMs optimize the trade-off between maximizing the geometric margin (1/||w||) and penalizing margin violations via hinge loss.",
                            ),
                        ]
                        pick = hard_pool_ja[(i - 1) % len(hard_pool_ja)]
                        q_text, opts, correct, expl, dist_expl, span = pick
                    else:
                        q_text = f"設問 {i} (中級): 単一のランダムサンプルまたはミニバッチを使用してパラメータを更新する最適化アルゴリズムはどれですか？"
                        opts = {
                            "A": "確率的勾配降下法 (SGD)",
                            "B": "正規方程式 (Normal Equations)",
                            "C": "サポートベクターマージン",
                            "D": "ジニ不純度基準",
                        }
                        correct = "A"
                        expl = "確率的勾配降下法 (SGD) は、単一のサンプルまたはミニバッチで勾配を近似し、効率的に重みを更新します。"
                        dist_expl = {
                            "B": "正規方程式は解析的な厳密解を求める手法です。",
                            "C": "サポートベクターマージンはSVMの幾何学的境界です。",
                            "D": "ジニ不純度は決定木の分割指標です。",
                        }
                        span = "Stochastic gradient descent (SGD) approximates the full gradient using a single random sample or a mini-batch."
                else:
                    if curr_diff == "Easy":
                        q_text = f"Question {i} (Easy): Which learning paradigm relies on labelled training pairs (x, y)?"
                        opts = {
                            "A": "Supervised learning",
                            "B": "Unsupervised clustering",
                            "C": "Principal component analysis",
                            "D": "Self-organizing maps",
                        }
                        correct = "A"
                        expl = "Supervised learning requires explicit labelled input-target pairs to learn predictive mappings."
                        dist_expl = {
                            "B": "Clustering does not use labelled targets.",
                            "C": "PCA is an unsupervised linear dimensionality reduction method.",
                            "D": "Self-organizing maps are unsupervised neural representations.",
                        }
                        span = "In supervised learning, we are given a training set of labelled examples {(x_1, y_1), ..., (x_n, y_n)}."
                    elif curr_diff == "Hard":
                        hard_pool_en = [
                            (
                                f"Question {i} (Hard): Why does L1 regularisation yield sparse weights while L2 regularisation produces smooth shrinkage?",
                                {
                                    "A": "The diamond geometry of the L1 ball intersects loss contours at axes where coordinates equal zero",
                                    "B": "L1 regularisation eliminates second-order curvature in the Hessian",
                                    "C": "L1 loss formulation restricts gradients exclusively to non-convex manifolds",
                                    "D": "L2 regularization forces weight magnitudes to drop strictly discontinuously",
                                },
                                "A",
                                "The sharp corners of the L1 ball lie on parameter axes, making tangential intersection with elliptical loss contours occur at coordinate zeros.",
                                {
                                    "B": "L1 regularization does not nullify the loss function's Hessian curvature.",
                                    "C": "The L1 objective penalty remains convex throughout.",
                                    "D": "L2 weight decay decays parameters smoothly towards zero rather than discontinuously.",
                                },
                                "L1 regularisation (Lasso) adds λ||w||_1, which can drive some weights to exactly zero, producing sparse models.",
                            ),
                            (
                                f"Question {i} (Hard): In the bias-variance decomposition of mean squared error, what accounts for irreducible error (σ^2)?",
                                {
                                    "A": "Inherent stochastic noise and unobserved latent variables within the true data-generating process",
                                    "B": "Model under-parameterization failing to capture non-linear relationships",
                                    "C": "Sensitivity of estimated parameters to finite training sample fluctuations",
                                    "D": "Non-convex optimization getting trapped in sub-optimal local minima",
                                },
                                "A",
                                "Irreducible error σ^2 stems from fundamental variance and noise in the target distribution that cannot be eliminated by any estimator.",
                                {
                                    "B": "Under-parameterization causes bias, which is reducible by increasing model capacity.",
                                    "C": "Sensitivity to training fluctuations is variance, which is reducible with more data or regularization.",
                                    "D": "Optimization pathology is an algorithmic limitation, not data-generating irreducible noise.",
                                },
                                "The expected test error decomposes into (Bias)^2 + Variance + Irreducible noise.",
                            ),
                            (
                                f"Question {i} (Hard): What mathematical property ensures that Mercer kernels implicitly compute inner products in a Reproducing Kernel Hilbert Space (RKHS)?",
                                {
                                    "A": "The kernel matrix is symmetric and positive semi-definite for all finite sample collections",
                                    "B": "The kernel function exhibits uniform Lipschitz continuity with constant L < 1",
                                    "C": "The dual optimization problem has an empty primal constraint manifold",
                                    "D": "The feature mapping function is an injective homeomorphism between metric spaces",
                                },
                                "A",
                                "Mercer's theorem requires that the continuous kernel function produces a symmetric, positive semi-definite Gram matrix, ensuring an inner product space exists.",
                                {
                                    "B": "Lipschitz continuity is helpful for gradient bounds, not kernel space validity.",
                                    "C": "The dual problem has active constraints (e.g. box constraints in soft-margin SVMs).",
                                    "D": "Feature maps need not be injective or homeomorphisms; they can map into infinite-dimensional Hilbert spaces.",
                                },
                                "By Mercer's theorem, any continuous, symmetric, positive semi-definite kernel computes an inner product in a transformed feature space.",
                            ),
                            (
                                f"Question {i} (Hard): How does Stochastic Gradient Descent (SGD) escape saddle points in high-dimensional non-convex loss landscapes?",
                                {
                                    "A": "Stochastic mini-batch gradient noise provides persistent isotropic perturbation along directions of negative curvature",
                                    "B": "SGD computes the exact inverse Hessian matrix at stationary points",
                                    "C": "The learning rate schedule automatically forces the objective to become strictly convex",
                                    "D": "Mini-batch averaging eliminates all zero-eigenvalue subspaces from the parameter space",
                                },
                                "A",
                                "The gradient covariance noise in mini-batch SGD acts as an isotropic exploration mechanism that destabilizes unstable saddle points with negative eigenvalues.",
                                {
                                    "B": "SGD is a first-order method and never computes the inverse Hessian.",
                                    "C": "Learning rates scale step size; they cannot alter the underlying curvature or convexity of the objective.",
                                    "D": "Mini-batch sampling does not change parameter space dimensionality or null spaces.",
                                },
                                "Stochastic gradient noise acts as an implicit regularizer that helps escape strict saddle points and find flatter minima.",
                            ),
                            (
                                f"Question {i} (Hard): Under what operational condition does Support Vector Machine (SVM) margin maximization become equivalent to L2-regularized empirical risk minimization?",
                                {
                                    "A": "When minimizing the soft-margin hinge loss with an explicit L2 penalty on the normal vector w",
                                    "B": "When the classification boundary is strictly constrained to pass through the empirical mean vector",
                                    "C": "When Slack variables ξ_i are penalized under an infinite-norm L_infinity constraint",
                                    "D": "When the data is linearly separable and the regularization parameter C approaches zero",
                                },
                                "A",
                                "Soft-margin SVM is mathematically equivalent to minimizing empirical hinge loss plus λ||w||_2^2, where C = 1 / (2λ).",
                                {
                                    "B": "SVM hyperplanes are determined by support vectors, not empirical cluster means.",
                                    "C": "Slack variables are typically regularized with an L1 penalty in the standard soft-margin formulation.",
                                    "D": "As C -> 0, regularization dominates and trivializes empirical loss rather than maximizing the separating margin.",
                                },
                                "Soft-margin SVMs optimize the trade-off between maximizing the geometric margin (1/||w||) and penalizing margin violations via hinge loss.",
                            ),
                        ]
                        pick = hard_pool_en[(i - 1) % len(hard_pool_en)]
                        q_text, opts, correct, expl, dist_expl, span = pick
                    else:
                        q_text = f"Question {i} (Medium): Which optimization algorithm updates weights using a single random sample or mini-batch?"
                        opts = {
                            "A": "Stochastic gradient descent (SGD)",
                            "B": "Normal equations",
                            "C": "Support vector margin",
                            "D": "Gini impurity criterion",
                        }
                        correct = "A"
                        expl = "Stochastic gradient descent (SGD) approximates the full gradient using a single random sample or a mini-batch."
                        dist_expl = {
                            "B": "Normal equations provide a closed-form analytical solution, not an iterative mini-batch update.",
                            "C": "Support vector margin is the boundary optimization distance for SVMs.",
                            "D": "Gini impurity is a criterion for splitting decision tree nodes.",
                        }
                        span = "Stochastic gradient descent (SGD) approximates the full gradient using a single random sample or a mini-batch."

                quiz_items.append(
                    {
                        "question_id": i,
                        "question": q_text,
                        "options": opts,
                        "correct_answer": correct,
                        "explanation": expl,
                        "distractor_explanations": dist_expl,
                        "source_span": span,
                        "difficulty": curr_diff,
                    }
                )
            return json.dumps({"quiz": quiz_items})

        elif "TERMS TO DEFINE:" in prompt or '"definitions":' in prompt:
            if is_ja:
                return json.dumps(
                    {
                        "definitions": [
                            {
                                "term": "バイアス・バリアンスのトレードオフ",
                                "definition": "モデルの単純化の仮定によるバイアス誤差と、訓練データの変動に対する感度であるバリアンス誤差の間の相反関係。",
                                "source_span": "The bias-variance trade-off is a fundamental concept in machine learning.",
                            },
                            {
                                "term": "線形回帰",
                                "definition": "平均二乗誤差（MSE）損失を最小化することで特徴量と目的変数の線形関係を学習する教師あり学習アルゴリズム。",
                                "source_span": "Linear regression is one of the simplest supervised learning algorithms.",
                            },
                        ]
                    }
                )
            return json.dumps(
                {
                    "definitions": [
                        {
                            "term": "bias-variance trade-off",
                            "definition": "The trade-off between bias error from simplified assumptions and variance error from training sensitivity.",
                            "source_span": "The bias-variance trade-off is a fundamental concept in machine learning.",
                        },
                        {
                            "term": "linear regression",
                            "definition": "A supervised learning algorithm that fits a linear relationship by minimising mean squared error.",
                            "source_span": "Linear regression is one of the simplest supervised learning algorithms.",
                        },
                    ]
                }
            )

        elif "SECTION ID:" in prompt or '"summary_points":' in prompt or "section_id" in prompt:
            if is_ja:
                return json.dumps(
                    {
                        "section_id": sec_id,
                        "title": sec_title,
                        "summary_points": [
                            {
                                "summary_sentence": f"{sec_title} に関する講義内容の主要な要点です。",
                                "source_span": "In this course we will study how computers learn from data to make predictions and decisions.",
                            },
                            {
                                "summary_sentence": "教師あり学習は、ラベル付き訓練データを用いて入力から目的変数への写像を学習します。",
                                "source_span": "In supervised learning, we are given a training set of labelled examples.",
                            },
                        ],
                    }
                )
            return json.dumps(
                {
                    "section_id": sec_id,
                    "title": sec_title,
                    "summary_points": [
                        {
                            "summary_sentence": f"Key finding for {sec_title} grounded in text.",
                            "source_span": "In this course we will study how computers learn from data to make predictions and decisions.",
                        },
                        {
                            "summary_sentence": "Supervised learning maps inputs to outputs using labelled training data.",
                            "source_span": "In supervised learning, we are given a training set of labelled examples.",
                        },
                    ],
                }
            )

        return "{}"

    @staticmethod
    def _clean_and_parse_json(raw_text: str) -> dict[str, Any]:
        """Strip markdown fences, isolate JSON substring, and parse into dict."""
        cleaned = raw_text.strip()
        # Remove markdown code fences ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Find outer JSON object braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt light syntactic repairs for common LLM JSON glitches
            repaired = re.sub(r':\s*(\d+)\.\s*([,\n\]\}])', r': \1\2', cleaned)
            repaired = re.sub(r',\s*([\}\]])', r'\1', repaired)
            repaired = re.sub(r'//.*?\n', '\n', repaired)
            # Fix unescaped backslashes (e.g. from LaTeX formulas like \gamma, \alpha, \frac, \sum)
            repaired = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                # Also try strict=False in case of raw control characters
                try:
                    return json.loads(repaired, strict=False)
                except json.JSONDecodeError as err:
                    logger.error("Failed to parse JSON from LLM response: %s", raw_text)
                    raise ValueError(f"Invalid JSON returned by LLM: {err}") from err
