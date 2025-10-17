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
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion

from src.agents.data_agent import DataConfig, fetch_prices


@dataclass
class OptConfig:
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

    # --- Black–Litterman inputs (only used if objective == "black_litterman") ---
    # market_caps: map ticker -> market cap (float, e.g., 2.9e12)
    market_caps: Optional[Dict[str, float]] = None
    # views: your absolute expected annual returns per ticker (e.g., {"MSFT": 0.10, "AAPL": 0.08})
    # (You could extend later to support relative views.)
    views: Optional[Dict[str, float]] = None
    # blending parameter (typical 0.02–0.10)
    bl_tau: float = 0.05


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
    Compute expected returns (mu) and covariance (S) depending on objective.
    - For BL, blend market equilibrium with provided views.
    - Otherwise, use historical-mean + Ledoit–Wolf shrinkage.
    Returns (mu, S) annualized.
    """
    if cfg.objective.lower() == "black_litterman":
        # Need a covariance estimate (annualized)
        S = CovarianceShrinkage(prices).ledoit_wolf()

        # Require market_caps and at least one view; otherwise fallback
        if not cfg.market_caps or not cfg.views:
            # Graceful fallback: historical means if BL inputs are missing
            mu_hist = mean_historical_return(prices)
            return mu_hist, S

        # Market risk aversion (optional input to BL; derived from market history)
        try:
            delta = market_implied_risk_aversion(market_prices=prices)
        except Exception:
            delta = None  # BL can still run without explicit delta

        bl = BlackLittermanModel(
            S,
            pi="market",                          # start from market-implied equilibrium
            market_caps=cfg.market_caps,          # dict aligned to tickers
            absolute_views=cfg.views,             # user's/AI's absolute views
            omega="idzorek",                      # view uncertainty scaling
            tau=cfg.bl_tau                        # blend strength
        )

        mu_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        return mu_bl, S_bl

    # Default path: historical returns + shrinkage covariance
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    return mu, S


def run_optimization(cfg: OptConfig) -> Dict[str, Any]:
    """
    Runs portfolio optimization and returns optimal weights + (optionally) discrete allocation.
    """
    # 1) Fetch prices (adjusted close)
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))

    # 2) Expected returns (annualized) & covariance (annualized)
    mu, S = _compute_mu_S(prices, cfg)

    # 3) Optimize
    ef = EfficientFrontier(mu, S, weight_bounds=_bounds(len(cfg.tickers), cfg.max_weight, cfg.long_only))

    # If BL objective, default to max_sharpe on the BL-adjusted mu/S
    if cfg.objective == "min_volatility":
        ef.min_volatility()
    else:
        # For "max_sharpe" and "black_litterman" we call max_sharpe; BL affects mu/S
        ef.max_sharpe(risk_free_rate=cfg.risk_free_rate)

    raw_weights = ef.clean_weights()
    exp_ret, exp_vol, exp_sharpe = ef.portfolio_performance(risk_free_rate=cfg.risk_free_rate)

    # 4) Sample efficient frontier points (for plotting)
    r_min, r_max = float(mu.min()), float(mu.max())
    targets = np.linspace(r_min, r_max, 15)
    frontier_r, frontier_s = [], []
    for tr in targets:
        try:
            ef_t = EfficientFrontier(mu, S, weight_bounds=_bounds(len(cfg.tickers), cfg.max_weight, cfg.long_only))
            ef_t.efficient_return(tr)
            rr, ss, _ = ef_t.portfolio_performance(risk_free_rate=cfg.risk_free_rate)
            frontier_r.append(float(rr))
            frontier_s.append(float(ss))
        except Exception:
            continue  # infeasible target → skip

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
        "risk_free_rate": cfg.risk_free_rate,
    }

    # 5) Optional: discrete allocation (convert weights → whole shares given capital)
    if cfg.capital is not None and cfg.capital > 0:
        latest = get_latest_prices(prices)
        weights_vec = np.array([raw_weights.get(t, 0.0) for t in cfg.tickers], dtype=float)
        weights_vec = weights_vec / (weights_vec.sum() if weights_vec.sum() > 0 else 1.0)
        weights_map = {t: w for t, w in zip(cfg.tickers, weights_vec)}
        da = DiscreteAllocation(weights_map, latest_prices=latest, total_portfolio_value=float(cfg.capital))
        allocation, leftover = da.greedy_portfolio()  # or da.lp_portfolio()

        out["discrete_allocation"] = {
            "capital": float(cfg.capital),
            "shares": allocation,
            "leftover_cash": float(leftover)
        }

    # 6) If BL used, echo inputs for transparency
    if cfg.objective.lower() == "black_litterman":
        out["black_litterman"] = {
            "tau": cfg.bl_tau,
            "views": cfg.views,
            "market_caps_present": bool(cfg.market_caps),
        }

    return out
