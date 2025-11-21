# src/agents/ai_explainer.py

from __future__ import annotations
from typing import Dict, Any
import os

from dotenv import load_dotenv
from google import genai

# Load env vars like GOOGLE_API_KEY, GEMINI_MODEL, etc.
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# Create a reusable Gemini client
_client = genai.Client(api_key=API_KEY) if API_KEY else None


def explain_metrics(portfolio_metrics: Dict[str, Any], style: str = "simple") -> str:
    """
    Use Gemini to explain portfolio risk/return metrics in plain English.

    This is the function your ADK agent (multi_tool_agent/agent.py)
    imports as a TOOL.
    """

    # Safety: if no API key, just return a non-AI explanation
    if _client is None:
        return (
            "AI explanation is unavailable because GOOGLE_API_KEY is not set.\n\n"
            f"Raw metrics I received:\n{portfolio_metrics}"
        )

    # Build a nice prompt for Gemini
    # Example portfolio_metrics dict might look like:
    # {
    #   "annual_return": 0.12,
    #   "annual_volatility": 0.20,
    #   "sharpe": 0.6,
    #   "var_95": -0.035,
    #   "cvar_95": -0.045,
    #   "max_drawdown": -0.28
    # }
    metric_lines = []
    for k, v in portfolio_metrics.items():
        metric_lines.append(f"- {k}: {v}")
    metrics_block = "\n".join(metric_lines)

    if style == "simple":
        tone = (
            "Explain this as if you're talking to a smart beginner in finance. "
            "Use short sentences and avoid heavy jargon."
        )
    elif style == "detailed":
        tone = (
            "Give a detailed, professional explanation suitable for an investment memo. "
            "You can use technical terms like VaR, drawdown, Sharpe, but still stay clear."
        )
    else:
        tone = "Explain clearly in natural language."

    prompt = f"""
You are a portfolio risk and optimization assistant.

The user has a portfolio with the following metrics:

{metrics_block}

1. Explain what each metric means (annual_return, annual_volatility, Sharpe, VaR, CVaR, max_drawdown, etc.).
2. Interpret whether this portfolio seems conservative, moderate, or aggressive.
3. Mention what kind of investor might be OK with this level of risk.
4. Keep it focused on *this* portfolio, not generic textbook definitions.

Tone guideline: {tone}
"""

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    # Different versions of google-genai expose text slightly differently;
    # this usually works in 1.5x:
    try:
        return response.text
    except AttributeError:
        # Fallback: join parts if needed
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            return "".join(getattr(p, "text", "") for p in parts)
        return "I could not generate an explanation. Raw response:\n" + repr(response)
