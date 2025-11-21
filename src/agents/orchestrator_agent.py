# src/agents/orchestrator_agent.py

from __future__ import annotations

import os
from typing import Dict, Any

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai.types import Content, Part

# Your existing tools
from src.agents.optimizer_agent import run_optimization, OptConfig
from src.agents.risk_agent import run_risk, RiskConfig
from src.agents.ai_explainer import explain_risk_from_dict


# ---------------------
#  TOOL WRAPPERS
# ---------------------

def tool_optimize_portfolio(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK tool wrapper for run_optimization.

    config should be a dict that can be unpacked into OptConfig(**config),
    e.g. same fields you pass from Streamlit.
    """
    opt_cfg = OptConfig(**config)
    return run_optimization(opt_cfg)


def tool_risk_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK tool wrapper for run_risk.
    """
    r_cfg = RiskConfig(**config)
    return run_risk(r_cfg)


def tool_explain_risk(
    metrics: Dict[str, float],
    weights: Dict[str, float] | None = None,
    horizon: str = "1-year",
) -> Dict[str, str]:
    """
    ADK tool wrapper for risk explanation.

    Right now we ignore `weights` and just explain the metrics themselves.
    """
    explanation = explain_risk_from_dict(metrics, horizon=horizon)
    return {"explanation": explanation}


# ---------------------
#  BUILD THE AGENT
# ---------------------

def build_orchestrator() -> LlmAgent:
    """
    Build an LlmAgent that can choose between optimization, risk analysis,
    and explanation tools.
    """
    return (
        LlmAgent
        .builder()
        .name("portfolio_orchestrator_agent")
        .model(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
        .description(
            "AI Orchestrator that can optimize portfolios, run risk analysis, "
            "and explain risk metrics in plain English."
        )
        .instruction(
            """
            You are an AI portfolio advisor.

            You have access to three tools:

            1. tool_optimize_portfolio(config)
            2. tool_risk_analysis(config)
            3. tool_explain_risk(metrics, weights, horizon)

            Decide which tool to call based on the user's question.

            Examples:
            - "Optimize my portfolio" -> call tool_optimize_portfolio
            - "Show me the risk of this allocation" -> call tool_risk_analysis
            - "Explain why my VaR is high" -> call tool_explain_risk

            Ask clarifying questions when needed and keep answers concise.
            """
        )
        .tools(
            FunctionTool.create(__name__, "tool_optimize_portfolio"),
            FunctionTool.create(__name__, "tool_risk_analysis"),
            FunctionTool.create(__name__, "tool_explain_risk"),
        )
        .build()
    )


# ---------------------
#  RUN IN TERMINAL
# ---------------------

def start_cli_chat() -> None:
    """
    Run the orchestrator agent in the terminal:
      python -m src.agents.orchestrator_agent
    (from the project root)
    """
    agent = build_orchestrator()
    runner = InMemoryRunner(agent)

    session: Session = runner.sessionService().create_session(
        "portfolio_orchestrator_agent", "user"
    ).blockingGet()

    print("💬 Portfolio Orchestrator Agent is running. Type messages below.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You > ")
        if user_input.lower() in ("quit", "exit"):
            break

        msg = Content.from_parts(Part.from_text(user_input))
        events = runner.runAsync("user", session.id(), msg)

        print("\nAgent > ", end="")
        events.blockingForEach(lambda e: print(e.stringifyContent(), end=""))
        print("\n")


if __name__ == "__main__":
    start_cli_chat()
