# src/agents/optimizer_agent.py
# OPTIMIZER AGENT

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from src.agents.data_agent import DataConfig, fetch_prices


@dataclass
class OptConfig:
    """
    Configuration for portfolio optimization.
    """
    tickers: List[str]
    start: str
    end: Optional[str] = None

    # objectives: "max_sharpe", "min_volatility", "black_litterman"
    objective: str = "max_sharpe"

    # common constraints/params
    max_weight: float = 0.40
    long_only: bool = True
    risk_free_rate: float = 0.0

    # optional discrete allocation
    capital: Optional[float] = None  # if provided, return whole-share allocation

    # --- Black–Litterman inputs (for now just metadata; math uses historical) ---
    market_caps: Optional[Dict[str, float]] = None   # ticker -> market cap
    views: Optional[Dict[str, float]] = None         # ticker -> expected return
    bl_tau: float = 0.05                             # blending strength (not used yet)


def _bounds(n: int, max_w: float, long_only: bool):
    """
    Build per-asset (low, high) bounds.
    - long-only: [0, max_w]
    - long/short (simple): [-0.5, min(0.5, max_w)]
    """
    if long_only:
        return tuple((0.0, min(1.0, max_w)) for _ in range(n))
    return tuple((-0.5, min(0.5, max_w)) for _ in range(n))


def _compute_mu_S(prices: pd.DataFrame, cfg: OptConfig) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute expected returns (mu) and covariance (S), annualized.

    For now we always use:
      - mean_historical_return for mu
      - Ledoit–Wolf shrinkage covariance for S

    Even when objective == "black_litterman", we keep the math
    the same as historical mean–variance. The BL fields (market_caps,
    views, bl_tau) are carried as metadata only, so the optimizer
    stays stable and never throws the '.dot' error.
    """
    # Historical annualized mean returns
    mu = mean_historical_return(prices)
    # Ensure ordering matches the columns/tickers
    mu = mu[prices.columns]

    # Shrinkage covariance (annualized)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    return mu, S


def run_optimization(cfg: OptConfig) -> Dict[str, Any]:
    """
    Runs portfolio optimization and returns optimal weights + (optionally) discrete allocation.

    Output dict contains:
      - weights: dict[ticker -> weight]
      - expected_return, volatility, sharpe
      - frontier: { "returns": [...], "risks": [...] }
      - (optional) discrete_allocation: { capital, shares, leftover_cash }
      - (optional) black_litterman metadata block when objective == "black_litterman"
    """
    # 1) Fetch prices (adjusted close)
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))

    # 2) Expected returns (annualized) & covariance (annualized)
    mu, S = _compute_mu_S(prices, cfg)

    # Use the actual tickers present in mu/S (in case some were dropped)
    tickers = list(mu.index)

    # 3) Optimize using EfficientFrontier
    ef = EfficientFrontier(mu, S, weight_bounds=_bounds(len(tickers), cfg.max_weight, cfg.long_only))

    # If BL objective, we already "bake in" its mode by still using max_sharpe;
    # the BL inputs are for later enhancement.
    
    raw_weights = ef.clean_weights()
    exp_ret, exp_vol, exp_sharpe = ef.portfolio_performance(risk_free_rate=cfg.risk_free_rate)

    # 4) Sample efficient frontier points (for plotting)
    r_min, r_max = float(mu.min()), float(mu.max())
    targets = np.linspace(r_min, r_max, 15)
    frontier_r, frontier_s = [], []
    for tr in targets:
        try:
            ef_t = EfficientFrontier(mu, S, weight_bounds=_bounds(len(tickers), cfg.max_weight, cfg.long_only))
            ef_t.efficient_return(tr)
            rr, ss, _ = ef_t.portfolio_performance(risk_free_rate=cfg.risk_free_rate)
            frontier_r.append(float(rr))
            frontier_s.append(float(ss))
        except Exception:
            # infeasible target → skip
            continue

    out: Dict[str, Any] = {
        "weights": raw_weights,
        "expected_return": float(exp_ret),
        "volatility": float(exp_vol),
        "sharpe": float(exp_sharpe),
        "frontier": {"returns": frontier_r, "risks": frontier_s},
        "universe": tickers,
        "start": cfg.start,
        "end": cfg.end,
        "objective": cfg.objective,
        "max_weight": cfg.max_weight,
        "long_only": cfg.long_only,
        "risk_free_rate": cfg.risk_free_rate,
    }

    # 5) Optional: discrete allocation (convert weights → whole shares given capital)
    if cfg.capital is not None and cfg.capital > 0:
        latest = get_latest_prices(prices)

        # Build weight vector in the same order as 'tickers'
        weights_vec = np.array([raw_weights.get(t, 0.0) for t in tickers], dtype=float)
        weights_vec = weights_vec / (weights_vec.sum() if weights_vec.sum() > 0 else 1.0)
        weights_map = {t: w for t, w in zip(tickers, weights_vec)}

        da = DiscreteAllocation(
            weights_map,
            latest_prices=latest,
            total_portfolio_value=float(cfg.capital),
        )
        allocation, leftover = da.greedy_portfolio()  # or da.lp_portfolio()

        out["discrete_allocation"] = {
            "capital": float(cfg.capital),
            "shares": allocation,
            "leftover_cash": float(leftover),
        }

    # 6) If BL mode selected, echo inputs for transparency (even though math is historical)
    if cfg.objective.lower() == "black_litterman":
        out["black_litterman"] = {
            "tau": cfg.bl_tau,
            "views": cfg.views,
            "market_caps_present": bool(cfg.market_caps),
        }

    return out
