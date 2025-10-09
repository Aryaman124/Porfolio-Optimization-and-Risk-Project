# src/agents/data_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------
# Simple parquet cache location
# -----------------------------
CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config
# -----------------------------
@dataclass
class DataConfig:
    tickers: List[str]
    start: str = "2015-01-01"
    end: Optional[str] = None
    auto_adjust: bool = True
    cache: bool = True
    cache_ttl_hours: int = 12  # re-download if cache older than this


# -----------------------------
# Internal helpers (cache + dl)
# -----------------------------
def _cache_path(cfg: DataConfig) -> Path:
    name = "-".join(cfg.tickers).replace("/", "_")
    return CACHE_DIR / f"prices_{name}_{cfg.start}_{cfg.end or 'latest'}.parquet"


def _is_cache_fresh(p: Path, ttl_hours: int) -> bool:
    if not p.exists():
        return False
    age_hours = (time.time() - p.stat().st_mtime) / 3600.0
    return age_hours <= ttl_hours


def _download_yf(cfg: DataConfig, max_retries: int = 3, pause_sec: float = 1.0) -> pd.DataFrame:
    last_err = None
    for i in range(max_retries):
        try:
            df = yf.download(
                cfg.tickers,
                start=cfg.start,
                end=cfg.end,
                auto_adjust=cfg.auto_adjust,
                progress=False,
            )

            # MultiIndex for multiple tickers — take Close
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]
            else:
                # Single ticker: ensure a single 'ticker' column
                if "Close" in df.columns:
                    df = df["Close"].to_frame()

            df = df.dropna(how="all")

            # If single ticker, yfinance may call the col "Close" — rename to ticker
            if df.shape[1] == 1 and df.columns[0] == "Close":
                df.columns = cfg.tickers

            return df
        except Exception as e:
            last_err = e
            time.sleep(pause_sec * (i + 1))
    raise RuntimeError(f"yfinance download failed after {max_retries} retries: {last_err}")


# -----------------------------
# Public API
# -----------------------------
def fetch_prices(cfg: DataConfig) -> pd.DataFrame:
    """
    Returns adjusted close prices in a wide DataFrame [date x tickers].
    Uses simple parquet cache in data/raw/.
    """
    cache_path = _cache_path(cfg)
    if cfg.cache and _is_cache_fresh(cache_path, cfg.cache_ttl_hours):
        prices = pd.read_parquet(cache_path)
    else:
        prices = _download_yf(cfg)
        # Clean index and save
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.sort_index().dropna(how="all")
        if cfg.cache:
            prices.to_parquet(cache_path)

    # Keep only requested tickers we actually have
    cols = [c for c in cfg.tickers if c in prices.columns]
    if not cols:
        raise ValueError("None of the requested tickers returned price data.")
    return prices[cols]


def daily_and_monthly_returns(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily simple returns; monthly returns via compounding daily → month.
    """
    daily = prices.pct_change().dropna(how="all")
    monthly = daily.resample("M").apply(lambda x: (1 + x).prod() - 1).dropna(how="all")
    return daily, monthly


# -----------------------------
# New: summary stats + covariance
# -----------------------------
def summary_stats(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Annualized expected return and volatility per asset from daily returns.
    """
    ann_factor = 252
    mu = daily_returns.mean() * ann_factor
    vol = daily_returns.std(ddof=0) * np.sqrt(ann_factor)
    out = pd.DataFrame({"expected_return": mu, "volatility": vol})
    out.index.name = "ticker"
    return out


def covariance_matrix(daily_returns: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
    """
    Covariance matrix of daily returns.
    method: "sample" | "lw" (Ledoit–Wolf shrinkage; falls back to sample if sklearn unavailable)
    """
    if method == "sample":
        return daily_returns.cov(ddof=0)
    elif method == "lw":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(daily_returns.dropna())
            return pd.DataFrame(lw.covariance_, index=daily_returns.columns, columns=daily_returns.columns)
        except Exception:
            return daily_returns.cov(ddof=0)
    else:
        raise ValueError("Unknown covariance method")
