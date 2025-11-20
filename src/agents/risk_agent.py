# src/agents/risk_agent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from src.agents.data_agent import DataConfig, fetch_prices


@dataclass
class RiskConfig:
    tickers: List[str]
    start: str
    end: Optional[str] = None

    # optional weights; if None we use equal-weight
    weights: Optional[Dict[str, float]] = None

    # annual risk-free rate (same idea as in optimizer)
    risk_free_rate: float = 0.0

    # confidence level for VaR / CVaR
    var_level: float = 0.95


def _equal_weights(tickers: List[str]) -> Dict[str, float]:
    n = len(tickers)
    if n == 0:
        return {}
    w = 1.0 / n
    return {t: w for t in tickers}


def _max_drawdown(returns: pd.Series) -> float:
    """
    Simple max drawdown on a series of portfolio returns.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1.0
    return float(drawdown.min())


def run_risk(cfg: RiskConfig) -> Dict[str, Any]:
    """
    Compute basic portfolio risk statistics:

    - Annualized return
    - Annualized volatility
    - Sharpe ratio
    - Historical VaR & CVaR (at cfg.var_level)
    - Max drawdown
    """

    # 1) Get prices using the same data agent you already use
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))

    # 2) Daily returns
    daily = prices.pct_change().dropna()

    if daily.empty:
        raise ValueError("Not enough data to compute risk (no daily returns).")

    tickers = list(daily.columns)

    # 3) Weights (equal weight for now if none provided)
    if cfg.weights is None or len(cfg.weights) == 0:
        weights_dict = _equal_weights(tickers)
    else:
        # keep only tickers we actually have data for
        weights_dict = {t: cfg.weights[t] for t in tickers if t in cfg.weights}

    # Put weights into a numpy array aligned with `tickers`
    w_vec = np.array([weights_dict.get(t, 0.0) for t in tickers], dtype=float)
    if w_vec.sum() <= 0:
        raise ValueError("All weights are zero; cannot compute portfolio risk.")
    w_vec = w_vec / w_vec.sum()

    # 4) Portfolio daily returns
    #    (matrix multiply: each row is a day, each col is a ticker)
    port_returns = daily.values @ w_vec
    port_returns = pd.Series(port_returns, index=daily.index, name="portfolio")

    # 5) Annualization (assuming ~252 trading days)
    mean_daily = port_returns.mean()
    vol_daily = port_returns.std()

    ann_return = (1 + mean_daily) ** 252 - 1
    ann_vol = vol_daily * np.sqrt(252)

    # Sharpe (use cfg.risk_free_rate as annual rate)
    excess_return = ann_return - cfg.risk_free_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else np.nan

    # 6) Historical VaR & CVaR
    alpha = 1.0 - cfg.var_level  # e.g. 0.05 for 95% VaR
    var = np.quantile(port_returns, alpha)  # in return space (negative = loss)
    cvar = port_returns[port_returns <= var].mean() if (port_returns <= var).any() else var

    # 7) Max drawdown
    max_dd = _max_drawdown(port_returns)

    metrics = {
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "VaR": float(var),
        "CVaR": float(cvar),
        "max_drawdown": float(max_dd),
    }

    # Also return series for plotting if needed
    cumulative = (1 + port_returns).cumprod()

    out: Dict[str, Any] = {
        "tickers": tickers,
        "weights": {t: float(w) for t, w in zip(tickers, w_vec)},
        "metrics": metrics,
        "series": {
            "daily_returns": port_returns,
            "cumulative_returns": cumulative,
        },
        "var_level": cfg.var_level,
    }

    return out
