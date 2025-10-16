# OPTIMIZER AGENT

# src/agents/optimizer_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from src.agents.data_agent import DataConfig, fetch_prices, daily_and_monthly_returns


@dataclass
class OptConfig:
    tickers: List[str]
    start: str
    end: Optional[str] = None
    objective: str = "max_sharpe"      # or "min_volatility"
    max_weight: float = 0.40           # cap per asset
    long_only: bool = True
    capital: Optional[float] = None    # if provided, also return share counts via DiscreteAllocation


def _bounds(n: int, max_w: float, long_only: bool):
    """
    Build per-asset (low, high) bounds.
    - long-only: [0, max_w]
    - long/short (simple): [-0.5, min(0.5, max_w)]
    """
    if long_only:
        return tuple((0.0, min(1.0, max_w)) for _ in range(n))
    # allow modest shorting if long_only is False
    return tuple((-0.5, min(0.5, max_w)) for _ in range(n))


def run_optimization(cfg: OptConfig) -> Dict[str, Any]:
    """
    Runs portfolio optimization and returns optimal weights + (optionally) discrete allocation.
    """
    # 1) Fetch prices & returns
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))
    # daily, _ = daily_and_monthly_returns(prices)  # not needed for PyPortfolioOpt below

    # 2) Expected returns (annualized) & covariance (annualized)
    mu = mean_historical_return(prices)                 # uses price history
    S = CovarianceShrinkage(prices).ledoit_wolf()

    # 3) Optimize
    ef = EfficientFrontier(mu, S, weight_bounds=_bounds(len(cfg.tickers), cfg.max_weight, cfg.long_only))
    if cfg.objective == "min_volatility":
        ef.min_volatility()
    else:
        ef.max_sharpe()

    raw_weights = ef.clean_weights()                    # dict {ticker: weight}
    exp_ret, exp_vol, exp_sharpe = ef.portfolio_performance()  # annualized

    # 4) Sample efficient frontier points (for plotting)
    r_min, r_max = float(mu.min()), float(mu.max())
    targets = np.linspace(r_min, r_max, 15)
    frontier_r, frontier_s = [], []
    for tr in targets:
        try:
            ef_t = EfficientFrontier(mu, S, weight_bounds=_bounds(len(cfg.tickers), cfg.max_weight, cfg.long_only))
            ef_t.efficient_return(tr)
            rr, ss, _ = ef_t.portfolio_performance()
            frontier_r.append(float(rr))
            frontier_s.append(float(ss))
        except Exception:
            # infeasible target return → skip
            continue

    out: Dict[str, Any] = {
        "weights": raw_weights,
        "expected_return": float(exp_ret),
        "volatility": float(exp_vol),
        "sharpe": float(exp_sharpe),
        "frontier": {"returns": frontier_r, "risks": frontier_s},
        "universe": cfg.tickers,
        "start": cfg.start,
        "end": cfg.end,
        "objective": cfg.objective,
        "max_weight": cfg.max_weight,
        "long_only": cfg.long_only,
    }

    # 5) Optional: discrete allocation (convert weights → share counts given capital)
    if cfg.capital is not None and cfg.capital > 0:
        # Latest prices aligned to your tickers
        latest = get_latest_prices(prices)
        # Ensure order is consistent between weights vector and tickers list
        weights_vec = np.array([raw_weights.get(t, 0.0) for t in cfg.tickers], dtype=float)
        weights_vec = weights_vec / (weights_vec.sum() if weights_vec.sum() > 0 else 1.0)
        weights_map = {t: w for t, w in zip(cfg.tickers, weights_vec)}

        da = DiscreteAllocation(weights_map, latest_prices=latest, total_portfolio_value=float(cfg.capital))
        allocation, leftover = da.greedy_portfolio()  # or da.lp_portfolio() if you prefer the LP solver

        out["discrete_allocation"] = {
            "capital": float(cfg.capital),
            "shares": allocation,           # {ticker: num_shares}
            "leftover_cash": float(leftover)
        }

    return out
