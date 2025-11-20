# src/agents/backtest_agent.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from src.agents.data_agent import DataConfig, fetch_prices


@dataclass
class BacktestConfig:
    tickers: List[str]
    start: str
    end: Optional[str] = None

    # If None → equal-weight portfolio
    weights: Optional[Dict[str, float]] = None

    # Starting portfolio value
    initial_capital: float = 100_000.0

    # For Sharpe calculation
    risk_free_rate: float = 0.0  # annual


def _align_weights(
    prices: pd.DataFrame, weights: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """
    Align user/optimizer weights to the price columns.
    If weights is None, use equal-weight.
    """
    cols = list(prices.columns)

    if not cols:
        raise ValueError("No price columns available for backtest.")

    if weights is None:
        # Equal-weight across all available tickers
        w = 1.0 / len(cols)
        return {t: w for t in cols}

    # Filter to only tickers present in prices
    filtered = {t: w for t, w in weights.items() if t in cols}

    if not filtered:
        # If nothing overlaps, fall back to equal-weight
        w = 1.0 / len(cols)
        return {t: w for t in cols}

    # Normalize so they sum to 1
    total = float(sum(filtered.values()))
    if total <= 0:
        w = 1.0 / len(cols)
        return {t: w for t in cols}

    return {t: w / total for t, w in filtered.items()}


def _max_drawdown(series: pd.Series) -> float:
    """
    Compute max drawdown of a portfolio value series.
    Returns a negative number (e.g., -0.25 for -25%).
    """
    if series.empty:
        return 0.0

    running_max = series.cummax()
    drawdowns = series / running_max - 1.0
    return float(drawdowns.min())


def run_backtest(cfg: BacktestConfig) -> Dict[str, Any]:
    """
    Simple long-only backtest:

    - Fetch daily adjusted close prices.
    - Build a portfolio with fixed weights at the start (buy-and-hold).
    - Compute portfolio value each day.
    - Return cumulative returns and summary metrics.
    """
    # 1) Fetch prices
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))
    if prices.empty:
        raise ValueError("No price data returned for backtest.")

    # Make sure columns are sorted / stable
    prices = prices.sort_index()
    prices = prices.loc[:, sorted(prices.columns)]

    # 2) Align weights to the actual price columns
    weights = _align_weights(prices, cfg.weights)

    # 3) Compute number of shares bought at start (buy-and-hold)
    first_prices = prices.iloc[0]
    shares: Dict[str, float] = {}
    for t, w in weights.items():
        if t not in first_prices or first_prices[t] <= 0:
            continue
        dollar_alloc = cfg.initial_capital * w
        shares[t] = dollar_alloc / first_prices[t]

    if not shares:
        raise ValueError("No valid shares could be computed for backtest.")

    # 4) Portfolio value over time
    # value_t = sum(shares_i * price_i,t)
    shares_series = pd.Series(shares)
    # Align columns
    prices_for_shares = prices.loc[:, shares_series.index]
    portfolio_values = prices_for_shares.mul(shares_series, axis=1).sum(axis=1)

    # 5) Returns / cumulative returns
    daily_returns = portfolio_values.pct_change().dropna()
    cumulative_returns = (1.0 + daily_returns).cumprod() - 1.0

    # 6) Metrics
    n_days = len(daily_returns)
    if n_days > 0:
        total_return = float(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0)
        annual_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
        annual_vol = float(daily_returns.std() * np.sqrt(252.0))
    else:
        total_return = 0.0
        annual_return = 0.0
        annual_vol = 0.0

    if annual_vol > 0:
        sharpe = (annual_return - cfg.risk_free_rate) / annual_vol
    else:
        sharpe = 0.0

    max_dd = _max_drawdown(portfolio_values)

    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }

    return {
        "config": {
            "tickers": cfg.tickers,
            "start": cfg.start,
            "end": cfg.end,
            "initial_capital": cfg.initial_capital,
            "risk_free_rate": cfg.risk_free_rate,
        },
        "weights": weights,
        "series": {
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "cumulative_returns": cumulative_returns,
        },
        "metrics": metrics,
    }
