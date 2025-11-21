# src/agents/risk_agent.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from src.agents.data_agent import DataConfig, fetch_prices, daily_and_monthly_returns


@dataclass
class RiskConfig:
    tickers: List[str]
    start: str
    end: Optional[str] = None
    risk_free_rate: float = 0.0

    # NEW: optional custom weights (ticker -> weight)
    weights: Optional[Dict[str, float]] = None


def _compute_portfolio_series(
    prices: pd.DataFrame,
    tickers: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns:
      daily_portfolio_returns, cumulative_portfolio_returns
    """
    # 1) Daily returns
    daily_returns, _ = daily_and_monthly_returns(prices)

    # 2) Restrict to the chosen tickers
    cols = [t for t in tickers if t in daily_returns.columns]
    if not cols:
        raise ValueError("No selected tickers found in price data.")

    daily = daily_returns[cols].dropna(how="all")

    # 3) Build weight vector
    n = len(cols)
    if weights:
        w = np.array([weights.get(t, 0.0) for t in cols], dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            # fallback to equal
            w = np.repeat(1.0 / n, n)
        else:
            w = w / w_sum
    else:
        # equal weight
        w = np.repeat(1.0 / n, n)

    # 4) Portfolio daily returns
    port_daily = daily @ w

    # 5) Cumulative
    cum = (1 + port_daily).cumprod() - 1

    return port_daily, cum


def run_risk(cfg: RiskConfig) -> Dict[str, Any]:
    """
    Compute basic portfolio risk metrics.
    If cfg.weights is provided, use those weights (aligned by ticker); otherwise use equal weights.
    """
    # 1) Fetch prices
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))

    # 2) Series
    daily_port, cum_port = _compute_portfolio_series(
        prices, cfg.tickers, cfg.weights
    )

    # 3) Basic stats
    mean_daily = daily_port.mean()
    std_daily = daily_port.std(ddof=1)

    # Annualization (assuming ~252 trading days)
    annual_ret = (1 + mean_daily) ** 252 - 1
    annual_vol = std_daily * (252 ** 0.5)

    # Sharpe
    excess_ret = annual_ret - cfg.risk_free_rate
    sharpe = excess_ret / annual_vol if annual_vol > 0 else 0.0

    # 4) VaR / CVaR (95%)
    alpha = 0.95
    sorted_returns = np.sort(daily_port.values)
    index = int((1 - alpha) * len(sorted_returns))
    var_95 = sorted_returns[index] if len(sorted_returns) > 0 else 0.0

    cvar_95 = sorted_returns[: index + 1].mean() if index >= 0 and len(sorted_returns) > 0 else 0.0

    # 5) Max drawdown
    running_max = (1 + daily_port).cumprod().cummax()
    drawdowns = (1 + daily_port).cumprod() / running_max - 1
    max_dd = drawdowns.min() if not drawdowns.empty else 0.0

    # 6) Final weights actually used (after alignment/renorm)
    cols = [t for t in cfg.tickers if t in prices.columns]
    n = len(cols)
    if cfg.weights:
        w = np.array([cfg.weights.get(t, 0.0) for t in cols], dtype=float)
        w_sum = w.sum()
        if w_sum <= 0:
            w = np.repeat(1.0 / n, n)
        else:
            w = w / w_sum
    else:
        w = np.repeat(1.0 / n, n)

    weights_used = {t: float(wi) for t, wi in zip(cols, w)}

    return {
        "metrics": {
            "annual_return": float(annual_ret),
            "annual_volatility": float(annual_vol),
            "sharpe": float(sharpe),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(max_dd),
        },
        "series": {
            "daily_returns": daily_port,
            "cumulative_returns": cum_port,
        },
        "weights": weights_used,
        "start": cfg.start,
        "end": cfg.end,
    }
