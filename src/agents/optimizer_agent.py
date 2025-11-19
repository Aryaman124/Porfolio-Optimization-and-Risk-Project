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
from pypfopt.black_litterman import BlackLittermanModel

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
    # absolute views: expected annual returns per ticker, e.g. {"MSFT": 0.10, "AAPL": 0.08}
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


def _hist_mu_S(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Plain historical mean return + Ledoit–Wolf covariance."""
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    # Make sure ordering is consistent
    mu = mu[prices.columns]
    S = S.loc[prices.columns, prices.columns]
    return mu, S


def _compute_mu_S(prices: pd.DataFrame, cfg: OptConfig) -> tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Compute expected returns (mu) and covariance (S) for the optimizer.
    If objective == "black_litterman", try BL first; if it fails, fall back to historical.
    Returns (mu, S, debug_info).
    """
    debug_info: Dict[str, Any] = {}

    # Not a BL run → just historical
    if cfg.objective.lower() != "black_litterman":
        mu, S = _hist_mu_S(prices)
        return mu, S, debug_info

    # BL selected, but inputs missing → fall back
    if not cfg.market_caps or not cfg.views:
        mu, S = _hist_mu_S(prices)
        debug_info["bl_status"] = "fallback_missing_inputs"
        return mu, S, debug_info

    try:
        # Use tickers that we actually have price data for
        tickers = [t for t in cfg.tickers if t in prices.columns]

        if not tickers:
            mu, S = _hist_mu_S(prices)
            debug_info["bl_status"] = "fallback_no_tickers"
            return mu, S, debug_info

        prices = prices[tickers]

        # Covariance matrix (annualized)
        S = CovarianceShrinkage(prices).ledoit_wolf()
        S = S.loc[tickers, tickers]

        # Market caps → Series aligned to tickers
        caps_series = pd.Series(cfg.market_caps, dtype="float64")
        caps_series = caps_series.reindex(tickers)

        # If all caps are NaN after reindex, BL cannot run
        if caps_series.isna().all():
            mu, S_hist = _hist_mu_S(prices)
            debug_info["bl_status"] = "fallback_caps_all_nan"
            return mu, S_hist, debug_info

        # Views → Series aligned to tickers; drop NaNs (only tickers with explicit views matter)
        views_series = pd.Series(cfg.views, dtype="float64")
        views_series = views_series.reindex(tickers)
        views_series = views_series.dropna()

        if views_series.empty:
            mu, S_hist = _hist_mu_S(prices)
            debug_info["bl_status"] = "fallback_no_valid_views"
            return mu, S_hist, debug_info

        # Convert back to dict for absolute_views argument
        absolute_views = views_series.to_dict()

        # Build BL model.
        # IMPORTANT: we do NOT use pi="market" here (that was causing the 'float.dot' issue
        # on some PyPortfolioOpt versions). Instead, we let the library infer market equilibrium
        # from caps + cov.
        bl = BlackLittermanModel(
            S,
            market_caps=caps_series,
            absolute_views=absolute_views,
            tau=cfg.bl_tau,
        )

        mu_bl = bl.bl_returns()
        S_bl = bl.bl_cov()

        # Align to tickers order
        mu_bl = mu_bl.reindex(tickers)
        S_bl = S_bl.loc[tickers, tickers]

        debug_info["bl_status"] = "ok"
        return mu_bl, S_bl, debug_info

    except Exception as e:
        # Any BL failure → fall back, but record error so you can inspect later
        print("Black–Litterman failed, falling back to historical:", repr(e))
        mu, S_hist = _hist_mu_S(prices)
        debug_info["bl_status"] = "fallback_exception"
        debug_info["bl_error"] = str(e)
        return mu, S_hist, debug_info


def run_optimization(cfg: OptConfig) -> Dict[str, Any]:
    """
    Runs portfolio optimization and returns optimal weights + (optionally) discrete allocation.
    """
    # 1) Fetch prices (adjusted close)
    prices = fetch_prices(DataConfig(cfg.tickers, cfg.start, cfg.end))

    # 2) Expected returns & covariance
    mu, S, bl_info = _compute_mu_S(prices, cfg)

    tickers = list(mu.index)

    # 3) Optimize
    ef = EfficientFrontier(mu, S, weight_bounds=_bounds(len(tickers), cfg.max_weight, cfg.long_only))

    if cfg.objective == "min_volatility":
        ef.min_volatility()
    else:
        # "max_sharpe" and "black_litterman" both use max_sharpe; BL affects mu/S
        ef.max_sharpe(risk_free_rate=cfg.risk_free_rate)

    raw_weights = ef.clean_weights()
    exp_ret, exp_vol, exp_sharpe = ef.portfolio_performance(risk_free_rate=cfg.risk_free_rate)

    # 4) Sample efficient frontier points for plotting
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
            continue  # infeasible target → skip

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

        # Build weight vector in same order as tickers
        weights_vec = np.array([raw_weights.get(t, 0.0) for t in tickers], dtype=float)
        weights_vec = weights_vec / (weights_vec.sum() if weights_vec.sum() > 0 else 1.0)
        weights_map = {t: w for t, w in zip(tickers, weights_vec)}

        da = DiscreteAllocation(
            weights_map,
            latest_prices=latest,
            total_portfolio_value=float(cfg.capital),
        )
        allocation, leftover = da.greedy_portfolio()

        out["discrete_allocation"] = {
            "capital": float(cfg.capital),
            "shares": allocation,
            "leftover_cash": float(leftover),
        }

    # 6) If BL used, echo status & inputs for transparency
    if cfg.objective.lower() == "black_litterman":
        out["black_litterman"] = {
            "tau": cfg.bl_tau,
            "views": cfg.views,
            "market_caps_present": bool(cfg.market_caps),
            "status": bl_info.get("bl_status"),
            "error": bl_info.get("bl_error"),
        }

    return out
