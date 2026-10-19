"""Concept Visualizer module.

Provides pedagogical visual schematics and SVG diagrams for key technical concepts,
enhancing conceptual comprehension and visual learning.
"""

from __future__ import annotations

import html
from typing import Any


def get_concept_visual(term: str, definition: str = "", language: str = "English") -> dict[str, Any] | None:
    """Return an educational dark-mode SVG diagram card tailored to the concept."""
    t_clean = (term or "").strip()
    t_lower = t_clean.lower()
    is_ja = language.lower() in ("japanese", "ja", "nihongo", "日本語")

    # 1. Neural Network / Deep Learning
    if any(k in t_lower for k in ["neural", "deep learning", "perceptron", "mlp", "layer", "backprop", "ニューラル", "深層学習"]):
        title = "人工ニューラルネットワーク構造 (Neural Network Architecture)" if is_ja else "Neural Network Architecture"
        caption = (
            "入力層の特徴量が重み行列とバイアスを通じて隠れ層に伝搬し、活性化関数を経て出力を算出します。"
            if is_ja
            else "Input features propagate through weighted connections and activation functions across hidden layers to produce predictions."
        )
        svg = """
<svg viewBox="0 0 520 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px;height:auto;display:block;margin:auto;">
  <defs>
    <linearGradient id="glowLine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#7c4dff" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#60a5fa" stop-opacity="0.7"/>
    </linearGradient>
    <filter id="nodeGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <!-- Connection Lines -->
  <!-- In to Hidden -->
  <line x1="80" y1="50" x2="260" y2="40" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="50" x2="260" y2="110" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="50" x2="260" y2="180" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="110" x2="260" y2="40" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="110" x2="260" y2="110" stroke="url(#glowLine)" stroke-width="2.2"/>
  <line x1="80" y1="110" x2="260" y2="180" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="170" x2="260" y2="40" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="170" x2="260" y2="110" stroke="url(#glowLine)" stroke-width="1.8"/>
  <line x1="80" y1="170" x2="260" y2="180" stroke="url(#glowLine)" stroke-width="1.8"/>
  <!-- Hidden to Out -->
  <line x1="260" y1="40" x2="440" y2="110" stroke="#f472b6" stroke-width="2" stroke-opacity="0.8"/>
  <line x1="260" y1="110" x2="440" y2="110" stroke="#f472b6" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="260" y1="180" x2="440" y2="110" stroke="#f472b6" stroke-width="2" stroke-opacity="0.8"/>

  <!-- Input Nodes -->
  <circle cx="80" cy="50" r="18" fill="#1e1b4b" stroke="#818cf8" stroke-width="2" filter="url(#nodeGlow)"/>
  <text x="80" y="55" fill="#e0e7ff" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">x₁</text>
  <circle cx="80" cy="110" r="18" fill="#1e1b4b" stroke="#818cf8" stroke-width="2" filter="url(#nodeGlow)"/>
  <text x="80" y="115" fill="#e0e7ff" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">x₂</text>
  <circle cx="80" cy="170" r="18" fill="#1e1b4b" stroke="#818cf8" stroke-width="2" filter="url(#nodeGlow)"/>
  <text x="80" y="175" fill="#e0e7ff" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">x₃</text>
  <text x="80" y="205" fill="#94a3b8" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">Input Layer</text>

  <!-- Hidden Nodes -->
  <circle cx="260" cy="40" r="20" fill="#172554" stroke="#60a5fa" stroke-width="2" filter="url(#nodeGlow)"/>
  <text x="260" y="45" fill="#eff6ff" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">h₁</text>
  <circle cx="260" cy="110" r="20" fill="#172554" stroke="#60a5fa" stroke-width="2" filter="url(#nodeGlow)"/>
  <text x="260" y="115" fill="#eff6ff" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">σ(W·x)</text>
  <circle cx="260" cy="180" r="20" fill="#172554" stroke="#60a5fa" stroke-width="2" filter="url(#nodeGlow)"/>
  <text x="260" y="185" fill="#eff6ff" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">h₃</text>
  <text x="260" y="215" fill="#94a3b8" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">Hidden Layer (ReLU/σ)</text>

  <!-- Output Node -->
  <circle cx="440" cy="110" r="22" fill="#500724" stroke="#f472b6" stroke-width="2.5" filter="url(#nodeGlow)"/>
  <text x="440" y="115" fill="#fff1f2" font-size="13" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">ŷ</text>
  <text x="440" y="150" fill="#f472b6" font-size="11" font-family="Inter,sans-serif" text-anchor="middle">Output / Loss</text>
</svg>
"""
        return {"title": title, "caption": caption, "svg_html": svg}

    # 2. Gradient Descent / Optimization / Loss
    if any(k in t_lower for k in ["gradient", "descent", "optimi", "loss", "cost", "勾配", "損失"]):
        title = "勾配降下法と最適化曲線 (Gradient Descent Optimization)" if is_ja else "Gradient Descent & Loss Minimization"
        caption = (
            "損失関数の傾き（勾配 ∇L）の逆方向に学習率 η でステップを進め、大域的最小値を目指します。"
            if is_ja
            else "Iteratively updates parameters in the direction of steepest descent (-∇L) scaled by learning rate η to find the minimum loss."
        )
        svg = """
<svg viewBox="0 0 520 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px;height:auto;display:block;margin:auto;">
  <defs>
    <linearGradient id="curveGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ef4444"/>
      <stop offset="50%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
  <!-- Axes -->
  <line x1="50" y1="180" x2="480" y2="180" stroke="#475569" stroke-width="1.5"/>
  <line x1="50" y1="20" x2="50" y2="180" stroke="#475569" stroke-width="1.5"/>
  <text x="480" y="198" fill="#94a3b8" font-size="11" font-family="Inter,sans-serif" text-anchor="end">Parameter θ</text>
  <text x="45" y="18" fill="#94a3b8" font-size="11" font-family="Inter,sans-serif" text-anchor="end">Loss L(θ)</text>

  <!-- Convex Loss Curve -->
  <path d="M 70 40 Q 250 200 450 50" fill="none" stroke="url(#curveGrad)" stroke-width="3.5" stroke-linecap="round"/>

  <!-- Descent Steps -->
  <circle cx="100" cy="62" r="7" fill="#ef4444" stroke="#fee2e2" stroke-width="2"/>
  <text x="115" y="55" fill="#fca5a5" font-size="11" font-family="Inter,sans-serif">θ₀ (Initial High Loss)</text>

  <path d="M 105 67 L 155 105" stroke="#fbbf24" stroke-width="2" stroke-dasharray="3,3" marker-end="url(#arrow)"/>
  <circle cx="160" cy="110" r="6" fill="#fbbf24"/>
  <text x="175" y="105" fill="#fde68a" font-size="10" font-family="Inter,sans-serif">Step: θ₁ = θ₀ - η∇L</text>

  <path d="M 165 115 L 210 138" stroke="#fbbf24" stroke-width="2" stroke-dasharray="3,3"/>
  <circle cx="215" cy="142" r="5" fill="#fbbf24"/>

  <!-- Minimum Target -->
  <circle cx="255" cy="152" r="9" fill="#10b981" stroke="#d1fae5" stroke-width="2.5"/>
  <text x="255" y="174" fill="#34d399" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">★ Minimum L(θ*)</text>
</svg>
"""
        return {"title": title, "caption": caption, "svg_html": svg}

    # 3. Overfitting / Regularization / Bias-Variance
    if any(k in t_lower for k in ["overfit", "underfit", "regulariz", "bias", "variance", "過学習", "バイアス"]):
        title = "過学習と正則化の直観 (Overfitting vs Balanced Fit)" if is_ja else "Underfitting vs Balanced vs Overfitting"
        caption = (
            "単純すぎるモデル（高バイアス）と、ノイズまで学習する過学習（高バリアンス）の間に、最適な汎化性能が存在します。"
            if is_ja
            else "Balancing model complexity: underfitting lacks capacity (high bias), while overfitting memorizes noise (high variance)."
        )
        svg = """
<svg viewBox="0 0 520 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px;height:auto;display:block;margin:auto;">
  <!-- Panel 1: Underfit -->
  <rect x="20" y="20" width="145" height="130" rx="10" fill="#1e1b4b" stroke="#4338ca" stroke-width="1.5"/>
  <text x="92" y="42" fill="#c7d2fe" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">Underfitting</text>
  <text x="92" y="58" fill="#818cf8" font-size="10" font-family="Inter,sans-serif" text-anchor="middle">High Bias (Too simple)</text>
  <line x1="35" y1="130" x2="150" y2="70" stroke="#ef4444" stroke-width="2.5"/>
  <circle cx="50" cy="110" r="3" fill="#cbd5e1"/>
  <circle cx="75" cy="90" r="3" fill="#cbd5e1"/>
  <circle cx="100" cy="115" r="3" fill="#cbd5e1"/>
  <circle cx="125" cy="80" r="3" fill="#cbd5e1"/>

  <!-- Panel 2: Balanced / Sweet Spot -->
  <rect x="187" y="20" width="145" height="130" rx="10" fill="#064e3b" stroke="#059669" stroke-width="2"/>
  <text x="260" y="42" fill="#a7f3d0" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">Optimal Generalization</text>
  <text x="260" y="58" fill="#34d399" font-size="10" font-family="Inter,sans-serif" text-anchor="middle">Balanced Tradeoff ★</text>
  <path d="M 200 125 Q 260 65 315 95" fill="none" stroke="#34d399" stroke-width="2.5"/>
  <circle cx="215" cy="110" r="3" fill="#cbd5e1"/>
  <circle cx="240" cy="90" r="3" fill="#cbd5e1"/>
  <circle cx="270" cy="85" r="3" fill="#cbd5e1"/>
  <circle cx="300" cy="95" r="3" fill="#cbd5e1"/>

  <!-- Panel 3: Overfitting -->
  <rect x="355" y="20" width="145" height="130" rx="10" fill="#450a0a" stroke="#b91c1c" stroke-width="1.5"/>
  <text x="427" y="42" fill="#fecaca" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">Overfitting</text>
  <text x="427" y="58" fill="#f87171" font-size="10" font-family="Inter,sans-serif" text-anchor="middle">High Variance (Noise fit)</text>
  <path d="M 370 120 Q 385 70 405 110 T 440 75 T 485 100" fill="none" stroke="#f87171" stroke-width="2.5"/>
  <circle cx="385" cy="85" r="3" fill="#cbd5e1"/>
  <circle cx="410" cy="105" r="3" fill="#cbd5e1"/>
  <circle cx="440" cy="78" r="3" fill="#cbd5e1"/>
  <circle cx="470" cy="98" r="3" fill="#cbd5e1"/>
</svg>
"""
        return {"title": title, "caption": caption, "svg_html": svg}

    # 4. Supervised vs Unsupervised Learning / Classification
    if any(k in t_lower for k in ["supervised", "unsupervised", "classif", "regression", "教師あり", "分類", "回帰"]):
        title = "教師あり学習 vs 決定境界 (Supervised Decision Boundary)" if is_ja else "Classification & Decision Boundary Separation"
        caption = (
            "特徴空間において、正例と負例を分離する決定超平面（境界）を学習します。"
            if is_ja
            else "Supervised classifier determines optimal decision boundary separating distinct target classes in feature space."
        )
        svg = """
<svg viewBox="0 0 520 190" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px;height:auto;display:block;margin:auto;">
  <!-- Feature Grid -->
  <rect x="40" y="20" width="440" height="150" rx="10" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
  <!-- Axes -->
  <line x1="60" y1="150" x2="460" y2="150" stroke="#475569" stroke-width="1"/>
  <line x1="60" y1="30" x2="60" y2="150" stroke="#475569" stroke-width="1"/>
  <text x="460" y="165" fill="#94a3b8" font-size="10" font-family="Inter,sans-serif" text-anchor="end">Feature x₁</text>
  <text x="55" y="30" fill="#94a3b8" font-size="10" font-family="Inter,sans-serif" text-anchor="end">Feature x₂</text>

  <!-- Decision Boundary Line -->
  <line x1="120" y1="145" x2="400" y2="35" stroke="#a78bfa" stroke-width="3" stroke-dasharray="6,4"/>
  <text x="405" y="38" fill="#c4b8fa" font-size="11" font-family="Inter,sans-serif" font-weight="600">Decision Boundary: wᵀx + b = 0</text>

  <!-- Class A (Blue Circles) -->
  <circle cx="130" cy="65" r="6" fill="#3b82f6"/>
  <circle cx="160" cy="50" r="6" fill="#3b82f6"/>
  <circle cx="190" cy="75" r="6" fill="#3b82f6"/>
  <circle cx="210" cy="45" r="6" fill="#3b82f6"/>
  <circle cx="230" cy="70" r="6" fill="#3b82f6"/>
  <text x="170" y="100" fill="#93c5fd" font-size="11" font-family="Inter,sans-serif" font-weight="600">● Class A (y = +1)</text>

  <!-- Class B (Red Triangles / Squares) -->
  <rect x="290" y="120" width="11" height="11" fill="#ef4444"/>
  <rect x="330" y="105" width="11" height="11" fill="#ef4444"/>
  <rect x="360" y="130" width="11" height="11" fill="#ef4444"/>
  <rect x="390" y="95" width="11" height="11" fill="#ef4444"/>
  <rect x="420" y="125" width="11" height="11" fill="#ef4444"/>
  <text x="350" y="80" fill="#fca5a5" font-size="11" font-family="Inter,sans-serif" font-weight="600">■ Class B (y = -1)</text>
</svg>
"""
        return {"title": title, "caption": caption, "svg_html": svg}

    # 5. Generic Concept Architecture Blueprint (For any other concept)
    title = f"概念アーキテクチャ図解: {t_clean}" if is_ja else f"Conceptual Architecture: {t_clean}"
    caption = (
        f"「{t_clean}」は、入力データを評価・変換し、明確な推論結果や目標値を導出する中核メカニズムです。"
        if is_ja
        else f"'{t_clean}' acts as the critical transformational engine converting input signals into verifiable lecture takeaways."
    )
    disp_term = html.escape(t_clean)
    svg = f"""
<svg viewBox="0 0 520 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:520px;height:auto;display:block;margin:auto;">
  <defs>
    <linearGradient id="coreGlow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4c1d95"/>
      <stop offset="100%" stop-color="#1e1b4b"/>
    </linearGradient>
  </defs>
  <!-- Step 1: Input -->
  <rect x="20" y="35" width="125" height="90" rx="10" fill="#0f172a" stroke="#38bdf8" stroke-width="1.8"/>
  <text x="82" y="68" fill="#38bdf8" font-size="11" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">INPUT DATA</text>
  <text x="82" y="90" fill="#94a3b8" font-size="10" font-family="Inter,sans-serif" text-anchor="middle">Raw Context &amp; Signals</text>

  <!-- Arrow 1 -->
  <line x1="145" y1="80" x2="185" y2="80" stroke="#7c4dff" stroke-width="2.5" stroke-linecap="round"/>
  <polygon points="185,75 195,80 185,85" fill="#7c4dff"/>

  <!-- Step 2: Core Concept Engine -->
  <rect x="195" y="25" width="150" height="110" rx="12" fill="url(#coreGlow)" stroke="#a78bfa" stroke-width="2.5"/>
  <text x="270" y="60" fill="#fbbf24" font-size="11" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">⭐ CORE MECHANISM</text>
  <text x="270" y="85" fill="#f1f5f9" font-size="12" font-family="Inter,sans-serif" text-anchor="middle" font-weight="700">{disp_term[:22]}</text>
  <text x="270" y="105" fill="#cbd5e1" font-size="10" font-family="Inter,sans-serif" text-anchor="middle">Grounded Logic / Engine</text>

  <!-- Arrow 2 -->
  <line x1="345" y1="80" x2="385" y2="80" stroke="#34d399" stroke-width="2.5" stroke-linecap="round"/>
  <polygon points="385,75 395,80 385,85" fill="#34d399"/>

  <!-- Step 3: Verified Output -->
  <rect x="395" y="35" width="110" height="90" rx="10" fill="#064e3b" stroke="#34d399" stroke-width="1.8"/>
  <text x="450" y="68" fill="#34d399" font-size="11" font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">OUTPUT</text>
  <text x="450" y="90" fill="#a7f3d0" font-size="10" font-family="Inter,sans-serif" text-anchor="middle">Prediction / Mastery</text>
</svg>
"""
    return {"title": title, "caption": caption, "svg_html": svg}
