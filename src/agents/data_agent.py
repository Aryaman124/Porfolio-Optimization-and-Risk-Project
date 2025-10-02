# src/agents/data_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class DataConfig:
    tickers: List[str]
    start: str = "2015-01-01"
    end: Optional[str] = None
    auto_adjust: bool = True
    cache: bool = True
    cache_ttl_hours: int = 12   # re-download if cache older than this

def _cache_path(cfg: DataConfig) -> Path:
    name = "-".join(cfg.tickers)
    name = name.replace("/", "_")
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
            df = yf.download(cfg.tickers, start=cfg.start, end=cfg.end, auto_adjust=cfg.auto_adjust, progress=False)
            # When multiple tickers: columns are MultiIndex; we need "Close"
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]
            else:
                # single ticker returns a Series of Close under "Close"
                if "Close" in df.columns:
                    df = df["Close"].to_frame()
            # Ensure wide format: one column per ticker
            df = df.dropna(how="all")
            # If single ticker, yfinance uses column name as the ticker; fix if needed
            if df.shape[1] == 1 and df.columns[0] == "Close":
                df.columns = cfg.tickers
            return df
        except Exception as e:
            last_err = e
            time.sleep(pause_sec * (i + 1))
    raise RuntimeError(f"yfinance download failed after {max_retries} retries: {last_err}")

def fetch_prices(cfg: DataConfig) -> pd.DataFrame:
    """
    Returns adjusted close prices in a wide DataFrame [date x tickers].
    Uses simple parquet cache in data/raw/.
    """
    cache_path = _cache_path(cfg)
    if cfg.cache and _is_cache_fresh(cache_path, cfg.cache_ttl_hours):
        return pd.read_parquet(cache_path)

    prices = _download_yf(cfg)
    # Clean: enforce DatetimeIndex UTC-normalized, sorted, and drop all-NA rows
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().dropna(how="all")
    # Keep only requested tickers present in data
    cols = [c for c in cfg.tickers if c in prices.columns]
    if not cols:
        raise ValueError("None of the requested tickers returned price data.")
    prices = prices[cols]
    # Cache
    if cfg.cache:
        prices.to_parquet(cache_path)
    return prices

def daily_and_monthly_returns(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily simple returns; monthly returns via compounding daily → month.
    """
    daily = prices.pct_change().dropna(how="all")
    monthly = daily.resample("M").apply(lambda x: (1 + x).prod() - 1).dropna(how="all")
    return daily, monthly

def validate_universe(tickers: List[str]) -> List[str]:
    """
    Minimal sanity checks (symbols uppercase, no duplicates).
    """
    clean = []
    seen = set()
    for t in tickers:
        t2 = t.strip().upper()
        if t2 and t2 not in seen:
            clean.append(t2)
            seen.add(t2)
    return clean
