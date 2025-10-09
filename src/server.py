# src/server.py
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.agents.data_agent import (
    DataConfig, fetch_prices, daily_and_monthly_returns,
    summary_stats, covariance_matrix
)

load_dotenv()

app = FastAPI(title="Portfolio Optimization & Risk API", version="0.1.0")


# ---------- models ----------
class PriceRequest(BaseModel):
    tickers: List[str]
    start: str = "2018-01-01"
    end: Optional[str] = None
    auto_adjust: bool = True
    cache: bool = True

class ReturnsRequest(BaseModel):
    tickers: List[str]
    start: str = "2018-01-01"
    end: Optional[str] = None

class OptimizeRequest(BaseModel):
    tickers: List[str]
    start: str = "2018-01-01"
    end: Optional[str] = None
    objective: str = "max_sharpe"
    long_only: bool = True
    max_weight: float = 0.5

class RiskRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    start: str = "2018-01-01"
    end: Optional[str] = None


# ---------- routes ----------
@app.get("/")
def health():
    return {"status": "ok", "service": "portfolio-api"}


@app.post("/fetch_prices")
def api_fetch_prices(req: PriceRequest):
    try:
        cfg = DataConfig(
            tickers=req.tickers,
            start=req.start,
            end=req.end,
            auto_adjust=req.auto_adjust,
            cache=req.cache,
        )
        prices = fetch_prices(cfg)
        return {
            "columns": list(prices.columns),
            "index": prices.index.astype(str).tolist(),
            "data": prices.fillna(None).values.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/fetch_returns")
def api_fetch_returns(req: ReturnsRequest):
    cfg = DataConfig(tickers=req.tickers, start=req.start, end=req.end)
    prices = fetch_prices(cfg)
    daily, monthly = daily_and_monthly_returns(prices)
    return {
        "daily": {
            "columns": list(daily.columns),
            "index": daily.index.astype(str).tolist(),
            "data": daily.fillna(None).values.tolist(),
        },
        "monthly": {
            "columns": list(monthly.columns),
            "index": monthly.index.astype(str).tolist(),
            "data": monthly.fillna(None).values.tolist(),
        },
    }


@app.post("/summary")
def api_summary(req: ReturnsRequest):
    cfg = DataConfig(tickers=req.tickers, start=req.start, end=req.end)
    prices = fetch_prices(cfg)
    daily, _ = daily_and_monthly_returns(prices)
    stats = summary_stats(daily)
    return {
        "tickers": list(stats.index),
        "expected_return": stats["expected_return"].tolist(),
        "volatility": stats["volatility"].tolist(),
    }


@app.post("/covariance")
def api_covariance(req: ReturnsRequest, method: str = "sample"):
    cfg = DataConfig(tickers=req.tickers, start=req.start, end=req.end)
    prices = fetch_prices(cfg)
    daily, _ = daily_and_monthly_returns(prices)
    cov = covariance_matrix(daily, method=method)
    return {"tickers": list(cov.index), "matrix": cov.values.tolist()}


# --------- stubs to be filled next ---------
@app.post("/optimize")
def api_optimize(req: OptimizeRequest):
    # TODO: replace with PyPortfolioOpt implementation
    n = len(req.tickers)
    if n == 0:
        raise HTTPException(status_code=400, detail="No tickers provided.")
    equal = round(1.0 / n, 6)
    return {"tickers": req.tickers, "weights": [equal] * n, "objective": req.objective}


@app.post("/risk_metrics")
def api_risk_metrics(req: RiskRequest):
    # TODO: implement vol, VaR, CVaR, drawdown
    return {"volatility_ann": None, "var_95": None, "cvar_95": None, "beta": None}
