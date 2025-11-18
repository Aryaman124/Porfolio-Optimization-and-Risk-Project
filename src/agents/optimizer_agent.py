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
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

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

    Returns (mu, S) annualized, with indices aligned to the price columns.
    """
    # Always start from a shrinkage covariance, aligned with asset columns
    tickers = list(prices.columns)
    S = CovarianceShrinkage(prices).ledoit_wolf()

    # ---------- BLACK–LITTERMAN PATH ----------
    if cfg.objective.lower() == "black_litterman":
        market_caps = cfg.market_caps or {}
        views = cfg.views or {}

        # Build market-cap-weight vector in the same order as 'tickers'
        caps_vec = np.array([market_caps.get(t, 0.0) for t in tickers], dtype=float)

        if caps_vec.sum() > 0:
            mkt_weights = caps_vec / caps_vec.sum()
            # Typical risk aversion parameter; can be tuned later if you want
            delta = 2.5
            pi = market_implied_prior_returns(mkt_weights, S, delta)
        else:
            # No market caps → no equilibrium prior; BL will just lean on views/cov
            pi = None

        # Absolute views: keep only tickers that are actually in our universe
        if views:
            q_series = pd.Series({t: v for t, v in views.items() if t in tickers})
        else:
            q_series = None

        bl = BlackLittermanModel(
            S,
            pi=pi,
            absolute_views=q_series,
            tau=cfg.bl_tau,
        )

        mu_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        # Make sure mu and S are ordered exactly like 'tickers'
        mu_bl = mu_bl[tickers]
        S_bl = S_bl.loc[tickers, tickers]

        return mu_bl, S_bl

    # ---------- DEFAULT (HISTORICAL MEAN-VAR) ----------
    mu = mean_historical_return(prices)
    # Ensure the ordering matches the columns/tickers
    mu = mu[prices.columns]

    return mu, S


def run_optimization(cfg: OptConfig) -> Dict[str, Any]:
    """
    Runs portfolio optimization and returns optimal weights + (optionally) discrete allocation.
    """
    # 1) Fetch prices (adjusted close)
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))

    # 2) Expected returns (annualized) & covariance (annualized)
    mu, S = _compute_mu_S(prices, cfg)

    # Use the actual tickers present in mu/S (in case some were dropped)
    tickers = list(mu.index)

    # 3) Optimize
    ef = EfficientFrontier(mu, S, weight_bounds=_bounds(len(tickers), cfg.max_weight, cfg.long_only))

    # If BL objective, we already baked BL into mu/S, then still choose max_sharpe
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

    # 6) If BL used, echo inputs for transparency
    if cfg.objective.lower() == "black_litterman":
        out["black_litterman"] = {
            "tau": cfg.bl_tau,
            "views": cfg.views,
            "market_caps_present": bool(cfg.market_caps),
        }

    return out
