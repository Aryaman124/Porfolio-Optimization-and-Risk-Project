# src/agents/ai_explainer.py
from __future__ import annotations

import os
from typing import Dict, Optional

from google import genai

# Use GEMINI_MODEL from .env or fall back
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def explain_metrics(
    metrics: Dict[str, float],
    horizon: str = "1-year",
    agent_name: str | None = None,
    user_id: str | None = None,
) -> str:
    """
    Core helper that turns raw risk metrics into a natural-language explanation
    using Gemini. Extra args (agent_name, user_id) are accepted but ignored,
    so ADK can pass them without breaking.
    """
    annual_return = metrics.get("annual_return")
    annual_vol = metrics.get("annual_volatility")
    sharpe = metrics.get("sharpe")
    var_95 = metrics.get("var_95")
    cvar_95 = metrics.get("cvar_95")
    max_dd = metrics.get("max_drawdown")

    sharpe_str = "n/a" if sharpe is None else f"{sharpe:.2f}"
    var_str = "n/a" if var_95 is None else f"{var_95 * 100:.2f}%"
    cvar_str = "n/a" if cvar_95 is None else f"{cvar_95 * 100:.2f}%"
    maxdd_str = "n/a" if max_dd is None else f"{max_dd * 100:.2f}%"

    bullet_summary = f"""
    Portfolio risk snapshot over a {horizon} horizon:

    - Annual return: {_pct(annual_return)}
    - Annual volatility: {_pct(annual_vol)}
    - Sharpe ratio: {sharpe_str}
    - 95% VaR: {var_str}
    - 95% CVaR: {cvar_str}
    - Max drawdown: {maxdd_str}
    """

    prompt = f"""
    You are a portfolio risk analyst.

    The user has these portfolio risk metrics:

    {bullet_summary}

    Please:
    1. Explain what each metric means in simple language.
    2. Say whether this looks conservative, moderate, or aggressive.
    3. Point out the main risks they should care about.
    4. Keep it under 4 short paragraphs.
    """

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    return resp.text or "I could not generate an explanation."


def explain_risk_from_dict(
    metrics: Dict[str, float],
    horizon: str = "1-year",
    context: str | None = None,
) -> str:
    """
    Convenience wrapper used by the ADK tools. Ignores extra context for now.
    """
    return explain_metrics(metrics, horizon=horizon)
