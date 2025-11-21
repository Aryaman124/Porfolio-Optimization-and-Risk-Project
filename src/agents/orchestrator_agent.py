# src/agents/orchestrator_agent.py

from __future__ import annotations

import os
from typing import Dict, Any

# ADK imports (Python API)
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

# Your tools
from src.agents.optimizer_agent import run_optimization, OptConfig
from src.agents.risk_agent import run_risk, RiskConfig
from src.agents.ai_explainer import explain_risk_from_dict


# ---------------------
#  TOOL WRAPPERS
# ---------------------

def tool_optimize_portfolio(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK tool wrapper for run_optimization.

    Expects a dict that can be unpacked into OptConfig, e.g.:

    {
        "tickers": ["AAPL", "MSFT"],
        "start": "2023-01-01",
        "end": "2024-01-01",
        "objective": "max_sharpe",
        "max_weight": 0.4,
        "long_only": True,
        "risk_free_rate": 0.01,
        "capital": 100000
    }
    """
    opt_cfg = OptConfig(**config)
    return run_optimization(opt_cfg)


def tool_risk_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ADK tool wrapper for run_risk.

    Expects a dict that can be unpacked into RiskConfig, e.g.:

    {
        "tickers": ["AAPL", "MSFT"],
        "start": "2023-01-01",
        "end": "2024-01-01",
        "risk_free_rate": 0.01
    }
    """
    r_cfg = RiskConfig(**config)
    return run_risk(r_cfg)


def tool_explain_risk(
    metrics: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> Dict[str, str]:
    """
    ADK tool wrapper around your natural-language risk explainer.

    metrics: dict like:
      {
        "annual_return": 0.11,
        "annual_volatility": 0.19,
        "sharpe": 0.58,
        "var_95": -0.03,
        "cvar_95": -0.05,
        "max_drawdown": -0.25
      }

    weights: optional dict of weights for extra context.
    """
    explanation = explain_risk_from_dict(metrics, weights)
    return {"explanation": explanation}


# ---------------------
#  BUILD ORCHESTRATOR AGENT
# ---------------------

def build_orchestrator_agent() -> Agent:
    """Create the portfolio orchestrator agent using the Python ADK API."""

    # Wrap your Python functions as FunctionTool instances
    optimize_tool = FunctionTool(func=tool_optimize_portfolio)
    optimize_tool.name = "tool_optimize_portfolio"
    optimize_tool.description = "Optimize a portfolio given an OptConfig-style config dict."

    risk_tool = FunctionTool(func=tool_risk_analysis)
    risk_tool.name = "tool_risk_analysis"
    risk_tool.description = "Run risk analysis for a given RiskConfig-style config dict."

    explain_tool = FunctionTool(func=tool_explain_risk)
    explain_tool.name = "tool_explain_risk"
    explain_tool.description = "Explain risk metrics (VaR, CVaR, Sharpe, drawdown) in plain English."

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    orchestrator = Agent(
        name="portfolio_orchestrator_agent",
        model=model_name,
        instruction=(
            "You are an AI portfolio advisor.\n\n"
            "You have access to three tools:\n"
            "1. tool_optimize_portfolio(config)\n"
            "2. tool_risk_analysis(config)\n"
            "3. tool_explain_risk(metrics, weights)\n\n"
            "Decide which tool to call based on the user's question.\n\n"
            "Examples:\n"
            '- \"Optimize my portfolio\" -> call tool_optimize_portfolio\n'
            '- \"Show me the risk\" -> call tool_risk_analysis\n'
            '- \"Explain why my VaR is high\" -> call tool_explain_risk\n\n'
            "Ask clarifying questions when you are missing required inputs."
        ),
        tools=[optimize_tool, risk_tool, explain_tool],
    )

    return orchestrator


# ---------------------
#  SIMPLE CLI RUNNER
# ---------------------

def start_cli_chat() -> None:
    """Run the orchestrator agent in the terminal using InMemoryRunner."""
    agent = build_orchestrator_agent()

    # app_name can be anything; just keep it consistent
    runner = InMemoryRunner(agent=agent, app_name="PortfolioQuantApp")

    session_id = "cli_session"
    user_id = "user"

    # Create a session (no keyword args – Python ADK expects positional)
    runner.session_service.create_session(session_id, user_id)

    print("💬 Portfolio Orchestrator Agent is running. Type messages below.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You > ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        msg = Content(parts=[Part(text=user_input)], role="user")

        print("Agent > ", end="", flush=True)
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=msg):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        print(part.text, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    start_cli_chat()
