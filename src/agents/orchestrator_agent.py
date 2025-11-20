# src/agents/orchestrator_agent.py

from __future__ import annotations
import os
from typing import Dict, Any

# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.tools.FunctionTool import create as create_tool
from google.adk.runner import InMemoryRunner
from google.adk.sessions import Session
from google.genai.types import Content, Part

# Your tools
from src.agents.optimizer_agent import run_optimization, OptConfig
from src.agents.risk_agent import run_risk, RiskConfig
from src.agents.ai_explainer import explain_risk_from_dict


# ---------------------
#  TOOLS (Python funcs)
# ---------------------

def tool_optimize_portfolio(config: Dict[str, Any]) -> Dict[str, Any]:
    """ADK tool wrapper for run_optimization."""
    opt_cfg = OptConfig(**config)
    return run_optimization(opt_cfg)


def tool_risk_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """ADK tool wrapper for run_risk."""
    r_cfg = RiskConfig(**config)
    return run_risk(r_cfg)


def tool_explain_risk(metrics: Dict[str, float], weights: Dict[str, float] = None) -> Dict[str, str]:
    """ADK tool wrapper for risk explanation."""
    explanation = explain_risk_from_dict(metrics, weights)
    return {"explanation": explanation}


# ---------------------
#  BUILD THE AGENT
# ---------------------

def build_orchestrator():
    return (
        LlmAgent
        .builder()
        .name("portfolio_orchestrator_agent")
        .model(os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
        .description("AI Orchestrator: chooses between optimization, risk, and explanation tools.")
        .instruction(
            """
            You are an AI portfolio advisor.

            You have access to three tools:

            1. tool_optimize_portfolio(config)
            2. tool_risk_analysis(config)
            3. tool_explain_risk(metrics, weights)

            Decide which tool to call based on the user's question.

            Examples:
            - "Optimize my portfolio" -> call optimize
            - "Show me the risk" -> call risk_analysis
            - "Explain why my VaR is high" -> call explain_risk

            Be smart. Ask clarifying questions when needed.
            """
        )
        .tools(
            create_tool(__name__, "tool_optimize_portfolio"),
            create_tool(__name__, "tool_risk_analysis"),
            create_tool(__name__, "tool_explain_risk"),
        )
        .build()
    )


# ---------------------
#  RUN IN TERMINAL
# ---------------------

def start_cli_chat():
    """Run the ADK agent in terminal."""
    agent = build_orchestrator()
    runner = InMemoryRunner(agent)

    session: Session = runner.sessionService().create_session(
        "portfolio_orchestrator_agent", "user"
    ).blockingGet()

    print("💬 Portfolio Orchestrator Agent is running. Type messages below.\n")

    while True:
        user_input = input("You > ")
        if user_input.lower() in ("quit", "exit"):
            break

        msg = Content.from_parts(Part.from_text(user_input))
        events = runner.runAsync("user", session.id(), msg)

        print("\nAgent > ", end="")
        events.blockingForEach(lambda e: print(e.stringifyContent(), end=""))

        print("\n")


# If you want: python orchestrator_agent.py to run
if __name__ == "__main__":
    start_cli_chat()
