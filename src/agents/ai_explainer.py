# src/agents/ai_explainer.py

from __future__ import annotations

import os
from typing import Dict, Optional

from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# Load environment + create Gemini client
# --------------------------------------------------

# Load .env once (safe to call multiple times)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION")

if API_KEY:
    client = genai.Client(api_key=API_KEY)
elif VERTEX_PROJECT and VERTEX_LOCATION:
    # Optional Vertex AI path – only if you actually use it
    client = genai.Client(
        vertexai={
            "project": VERTEX_PROJECT,
            "location": VERTEX_LOCATION,
        }
    )
else:
    raise RuntimeError(
        "Gemini credentials not found. Set GOOGLE_API_KEY in your .env "
        "or VERTEX_PROJECT and VERTEX_LOCATION for Vertex AI."
    )

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# --------------------------------------------------
# Core explainer helper
# --------------------------------------------------

def _format_metrics_text(metrics: Dict[str, float] | None) -> str:
    """Turn metrics dict into a readable block of text."""
    if not metrics:
        return "No numerical portfolio metrics were provided."

    lines = []
    ar = metrics.get("annual_return")
    av = metrics.get("annual_volatility")
    sh = metrics.get("sharpe")
    var_ = metrics.get("var_95")
    cvar_ = metrics.get("cvar_95")
    mdd = metrics.get("max_drawdown")

    if ar is not None:
        lines.append(f"- Annual return: {ar:.4f} ({ar*100:.2f}%)")
    if av is not None:
        lines.append(f"- Annual volatility: {av:.4f} ({av*100:.2f}%)")
    if sh is not None:
        lines.append(f"- Sharpe ratio: {sh:.4f}")
    if var_ is not None:
        lines.append(f"- 95% daily VaR: {var_:.4f} ({var_*100:.2f}%)")
    if cvar_ is not None:
        lines.append(f"- 95% daily CVaR: {cvar_:.4f} ({cvar_*100:.2f}%)")
    if mdd is not None:
        lines.append(f"- Max drawdown: {mdd:.4f} ({mdd*100:.2f}%)")

    return "\n".join(lines)


def _format_weights_text(weights: Dict[str, float] | None) -> str:
    """Turn weights dict into text."""
    if not weights:
        return "No weights were provided."

    parts = []
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        parts.append(f"{ticker}: {w:.4f} ({w*100:.2f}%)")

    return "\n".join(parts)


def explain_risk_from_dict(
    metrics: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    question: Optional[str] = None,
) -> str:
    """
    Main entrypoint used by Streamlit.

    - If metrics/weights are provided, the model acts as a portfolio / risk explainer.
    - If they are missing or the user is just small-talking, it responds like a
      friendly portfolio assistant.
    """

    metrics_text = _format_metrics_text(metrics)
    weights_text = _format_weights_text(weights)

    user_question = question or "Explain this portfolio in clear, simple language."

    prompt = f"""
You are a friendly quantitative portfolio assistant.

You may receive:
- Numeric portfolio risk metrics (annual return, volatility, Sharpe, VaR, CVaR, max drawdown).
- A list of asset weights.
- A user question, which can be about the portfolio or just normal conversation.

If the user is just greeting you or asking a general question, respond normally
as a helpful assistant and you do NOT need to force a risk explanation.

If the user asks anything about the portfolio, risk, performance, diversification,
or optimisation, then:
- Use the metrics and weights below (if present),
- Explain them in plain English,
- Comment on risk/return trade-off,
- Mention any red flags (very high volatility, very negative drawdown, etc.),
- Keep it concise and not too technical.

Portfolio metrics:
{metrics_text}

Portfolio weights:
{weights_text}

User question:
\"\"\"{user_question}\"\"\"
"""

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )

    # google-genai 1.x puts the main text in resp.text
    return (resp.text or "").strip()
