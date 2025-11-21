# multi_tool_agent/agent.py
from __future__ import annotations

import os
from typing import Dict, Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part

# Import your project tools
from src.agents.risk_agent import RiskConfig, run_risk
from src.agents.ai_explainer import explain_metrics

AGENT_NAME = "portfolio_quant_agent"
USER_ID = "user"


# ---------- TOOLS ----------

def compute_risk_tool(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK tool: compute risk metrics using your existing RiskAgent.
    """
    rcfg = RiskConfig(**config)
    result = run_risk(rcfg)
    return {
        "metrics": result["metrics"],
        "weights": result["weights"],
    }


def explain_risk_metrics(
    metrics: Dict[str, float],
    horizon: str = "1-year",
) -> Dict[str, str]:
    """
    ADK tool: call Gemini to explain risk metrics in plain English.
    """
    text = explain_metrics(metrics, horizon=horizon)
    return {"explanation": text}


# ---------- BUILD ADK AGENT ----------

def build_agent() -> LlmAgent:
    return (
        LlmAgent.builder()
        .name(AGENT_NAME)
        .model(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
        .description(
            "Quant assistant that can compute portfolio risk metrics "
            "and explain them in plain English."
        )
        .instruction(
            """
            You are a helpful quantitative portfolio assistant.

            You have two tools:

            1. compute_risk_tool(config): computes portfolio risk metrics.
            2. explain_risk_metrics(metrics, horizon): explains those metrics.

            If the user gives you tickers and dates, first call compute_risk_tool.
            If the user directly gives you metrics, call explain_risk_metrics.

            Always answer clearly and concisely, and avoid heavy jargon.
            """
        )
        .tools(
            FunctionTool.create(__name__, "compute_risk_tool"),
            FunctionTool.create(__name__, "explain_risk_metrics"),
        )
        .build()
    )


# ---------- SIMPLE CLI RUNNER ----------

def main() -> None:
    agent = build_agent()
    runner = InMemoryRunner(agent)

    session: Session = runner.sessionService().create_session(
        AGENT_NAME, USER_ID
    ).blockingGet()

    print("💬 portfolio_quant_agent is running. Type messages below.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You > ")
        if user_input.lower() in ("quit", "exit"):
            break

        msg = Content.from_parts(Part.from_text(user_input))
        events = runner.runAsync(USER_ID, session.id(), msg)

        print("\nAgent > ", end="")
        events.blockingForEach(lambda e: print(e.stringifyContent(), end=""))
        print("\n")


if __name__ == "__main__":
    main()
