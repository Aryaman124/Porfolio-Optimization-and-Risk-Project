# src/agents/ai_explainer.py
"""
AI Explainer Agent (Gemini-powered)

This module takes numeric risk metrics (from risk_agent.run_risk, optimizer_agent.run_optimization)
and returns a **plain-language explanation** using Gemini.

Later, we can wrap `explain_risk` as a google-adk FunctionTool so it becomes
a proper ADK agent in a multi-agent system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import os
import json

import google.genai as genai


# =========================
# Config + client bootstrap
# =========================

@dataclass
class ExplainConfig:
    """
    Configuration for asking Gemini to explain a portfolio's risk.

    metrics: dict of risk metrics, e.g.
        {
          "annual_return": 0.122,
          "annual_volatility": 0.195,
          "sharpe": 0.85,
          "var_95": -0.031,
          "cvar_95": -0.045,
          "max_drawdown": -0.22
        }

    weights: optional dict of asset weights, e.g.
        {"AAPL": 0.2, "MSFT": 0.3, ...}

    style: how the explanation should sound:
        - "simple"  -> explain like to a smart beginner
        - "technical" -> more quant/finance jargon
    """
    metrics: Dict[str, float]
    weights: Optional[Dict[str, float]] = None
    style: str = "simple"   # or "technical"
    horizon: str = "1 year" # purely descriptive, for the prompt


_MODEL = None  # lazy-initialized GenerativeModel


def _get_model():
    """
    Lazily configure and return the Gemini model client.
    Uses env vars:
        GOOGLE_API_KEY
        GEMINI_MODEL  (defaults to "gemini-2.5-pro")
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. "
            "Make sure it's defined in your .env or environment."
        )

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    _MODEL = genai.GenerativeModel(model_name)
    return _MODEL


# =========================
# Prompt construction
# =========================

def _build_prompt(cfg: ExplainConfig) -> str:
    """
    Build a clear prompt for Gemini based on portfolio metrics and weights.
    """
    # Pretty-print metrics as JSON so the model can parse them easily
    metrics_json = json.dumps(cfg.metrics, indent=2)

    if cfg.weights:
        weights_json = json.dumps(cfg.weights, indent=2)
    else:
        weights_json = "null"

    style_instruction = (
        "Explain in simple, clear language, like you are talking to a smart beginner "
        "who knows basic investing concepts but not advanced math."
        if cfg.style == "simple"
        else
        "Explain as if you are talking to a junior quantitative analyst. "
        "You can use technical finance terms, but still be clear and structured."
    )

    prompt = f"""
You are a portfolio risk explainer AI.

You are given:
1. A set of **portfolio risk metrics** (annualized).
2. Optionally, a set of **asset weights** in the portfolio.

Your job:
- Explain what these risk metrics mean.
- Interpret whether the risk level is low, moderate, or high.
- Comment on trade-off of risk vs return.
- If weights are provided, comment on concentration (e.g. if one asset is very large).
- Give practical, intuitive interpretation (not just definitions).

Important:
- Do NOT invent numbers. Only interpret the numbers that are given.
- Keep the explanation to 2–4 short paragraphs plus bullet points if helpful.
- Avoid equations. Use percentages and plain English instead.

{style_instruction}

---
Risk metrics (annualized, machine-readable JSON):
{metrics_json}

Portfolio weights (if not null):
{weights_json}

The risk horizon is approximately: {cfg.horizon}.

Now give a clear explanation of this portfolio's risk profile.
"""
    return prompt.strip()


# =========================
# Public API
# =========================

def explain_risk(cfg: ExplainConfig) -> str:
    """
    Call Gemini to explain portfolio risk metrics.

    Returns:
        A plain-language explanation (string).
    Raises:
        RuntimeError on missing API key or model errors.
    """
    model = _get_model()
    prompt = _build_prompt(cfg)

    try:
        response = model.generate_content(prompt)
    except Exception as e:
        raise RuntimeError(f"Gemini risk explainer call failed: {e}") from e

    # google-genai responses usually have `.text`
    text = getattr(response, "text", None)
    if not text:
        # Fallback: try to stringify the whole response
        text = str(response)

    return text


# Convenience wrapper for direct dict inputs (so Streamlit can call easily)
def explain_risk_from_dict(
    metrics: Dict[str, Any],
    weights: Optional[Dict[str, Any]] = None,
    style: str = "simple",
    horizon: str = "1 year",
) -> str:
    """
    Helper to avoid constructing ExplainConfig manually.

    Example:
        explanation = explain_risk_from_dict(
            metrics=risk["metrics"],
            weights=risk.get("weights"),
            style="simple"
        )
    """
    # Coerce numeric values to float where possible
    m_float: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            m_float[k] = float(v)
        except Exception:
            # if it can't be cast, drop it
            continue

    w_float: Optional[Dict[str, float]] = None
    if weights is not None:
        w_float = {}
        for k, v in weights.items():
            try:
                w_float[k] = float(v)
            except Exception:
                continue

    cfg = ExplainConfig(metrics=m_float, weights=w_float, style=style, horizon=horizon)
    return explain_risk(cfg)


__all__ = ["ExplainConfig", "explain_risk", "explain_risk_from_dict"]
