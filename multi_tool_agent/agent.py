# multi_tool_agent/agent.py

import asyncio
import datetime
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

# -------------------------------------------------------------------
#  Make sure we can import your existing src.agents.*
#  (assumes you run this script from the project root)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from agents.optimizer_agent import OptConfig, run_optimization
from agents.risk_agent import RiskConfig, run_risk
from agents.backtest_agent import BacktestConfig, run_backtest
from agents.ai_explainer import explain_metrics

# -------------------------------------------------------------------
#  Convenience: pick model from ENV, default to gemini-2.5-pro
# -------------------------------------------------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

# -------------------------------------------------------------------
#  Tools the agent can call
#  (signatures must be JSON-friendly: str, float, list, dict, etc.)
# -------------------------------------------------------------------

def optimize_portfolio(
    tickers: List[str],
    start: str,
    end: str | None = None,
    objective: str = "max_sharpe",
    max_weight: float = 0.4,
    long_only: bool = True,
    risk_free_rate: float = 0.001,
    capital: float | None = None,
) -> Dict[str, Any]:
    """
    Run your optimizer and return a light summary.
    """
    cfg = OptConfig(
        tickers=tickers,
        start=start,
        end=end or None,
        objective=objective,
        max_weight=max_weight,
        long_only=long_only,
        risk_free_rate=risk_free_rate,
        capital=capital,
    )
    res = run_optimization(cfg)

    # Only return things that are easy for the LLM to reason about
    return {
        "status": "ok",
        "objective": objective,
        "tickers": tickers,
        "weights": res["weights"],
        "expected_return": res["expected_return"],
        "volatility": res["volatility"],
        "sharpe": res["sharpe"],
    }


def analyze_risk(
    tickers: List[str],
    start: str,
    end: str | None = None,
    risk_free_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Run your risk engine (equal-weight for now).
    """
    cfg = RiskConfig(
        tickers=tickers,
        start=start,
        end=end or None,
        risk_free_rate=risk_free_rate,
    )
    res = run_risk(cfg)

    return {
        "status": "ok",
        "tickers": tickers,
        "metrics": res["metrics"],
    }


def backtest_portfolio(
    tickers: List[str],
    start: str,
    end: str | None = None,
    rebalance_freq: str = "M",
    capital: float = 10000.0,
) -> Dict[str, Any]:
    """
    Run your backtest agent.
    """
    cfg = BacktestConfig(
        tickers=tickers,
        start=start,
        end=end or None,
        rebalance_freq=rebalance_freq,
        capital=capital,
    )
    res = run_backtest(cfg)

    return {
        "status": "ok",
        "tickers": tickers,
        "capital": capital,
        "summary": res["summary"],
        # you can add more fields later as needed
    }


def explain_risk_metrics(
    metrics: Dict[str, float],
    horizon: str = "1-year",
) -> Dict[str, str]:
    """
    Call your AI explainer on given metrics.
    """
    text = explain_metrics(metrics, horizon=horizon)
    return {"status": "ok", "explanation": text}


# -------------------------------------------------------------------
#  Define the ADK Agent
# -------------------------------------------------------------------

TODAY = datetime.date.today().isoformat()

portfolio_agent = Agent(
    name="portfolio_quant_agent",
    model=MODEL_NAME,
    description="Helps analyze and optimize investment portfolios.",
    instruction=(
        "You are a helpful quantitative portfolio assistant.\n"
        "You can:\n"
        " - optimize portfolios using the optimize_portfolio tool\n"
        " - analyze risk using the analyze_risk tool\n"
        " - backtest strategies using the backtest_portfolio tool\n"
        " - explain risk metrics using explain_risk_metrics\n\n"
        "Always decide which tool(s) to call based on the user request, "
        "then summarize results in clear plain English.\n"
        f"Today's date is {TODAY}."
    ),
    tools=[
        optimize_portfolio,
        analyze_risk,
        backtest_portfolio,
        explain_risk_metrics,
    ],
)

runner = InMemoryRunner(
    agent=portfolio_agent,
    app_name="portfolioquant_app",
)


# -------------------------------------------------------------------
#  Helper: create a session and query the agent
# -------------------------------------------------------------------

def create_session():
    return asyncio.run(
        runner.session_service.create_session(
            app_name="portfolioquant_app", user_id="user"
        )
    )


def run_agent(session_id: str, message: str):
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    )
    for event in runner.run(
        user_id="user",
        session_id=session_id,
        new_message=content,
    ):
        # Only print text responses
        if event.content and event.content.parts:
            part0 = event.content.parts[0]
            if part0.text:
                print(f"{event.author}: {part0.text}")


# -------------------------------------------------------------------
#  Simple CLI entrypoint
# -------------------------------------------------------------------

if __name__ == "__main__":
    sess = create_session()
    print("PortfolioQuant ADK agent. Type 'quit' to exit.\n")
    while True:
        user_msg = input("You > ")
        if user_msg.strip().lower() in {"quit", "exit"}:
            break
        run_agent(sess.id, user_msg)
        print()
