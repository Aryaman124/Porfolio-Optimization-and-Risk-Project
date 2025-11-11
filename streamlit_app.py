# streamlit_app.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.data_agent import (
    DataConfig, fetch_prices, daily_and_monthly_returns, summary_stats
)
from src.agents.optimizer_agent import OptConfig, run_optimization

st.set_page_config(page_title="PortfolioQuant.ai", page_icon="📈", layout="wide")

# ============================= Helpers =============================

SYMBOLS_PATH = Path("data/symbols/sp500.csv")

def load_universe() -> list[str]:
    """Load ticker universe from data/symbols/sp500.csv or fall back."""
    if SYMBOLS_PATH.exists():
        tickers = [t.strip().upper() for t in SYMBOLS_PATH.read_text().splitlines() if t.strip()]
        return sorted(list(dict.fromkeys(tickers)))
    # small fallback universe for first run
    return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","V","XOM","UNH","AVGO"]

def weights_to_df(weights: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)

def parse_kv(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip().upper()] = float(v.strip())
            except Exception:
                pass
    return out

# -------- TradingView ticker tape (nice animated market strip) --------
# TradingView wants "EXCHANGE:SYMBOL" (e.g., NASDAQ:AAPL). We'll do a simple guess.
NASDAQ_COMMON = {
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","NFLX","AVGO","ADBE","INTC","CSCO","PEP","COST","AMD","QCOM","MU"
}

# Always show these indices first on the tape
FIXED_TAPE = [
    # 📊 Major Indices
    {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
    {"proName": "NASDAQ:NDX", "title": "NASDAQ 100"},
    {"proName": "FOREXCOM:DJI", "title": "Dow Jones"},
    {"proName": "FOREXCOM:US30", "title": "US 30"},
    {"proName": "FOREXCOM:VIX", "title": "VIX (Volatility Index)"},

    # 💵 Currencies
    {"proName": "FX_IDC:EURUSD", "title": "EUR/USD"},
    {"proName": "FX_IDC:USDJPY", "title": "USD/JPY"},

    # 🪙 Commodities
    {"proName": "COMEX:GC1!", "title": "Gold"},
    {"proName": "NYMEX:CL1!", "title": "Crude Oil"},
    {"proName": "TVC:SILVER", "title": "Silver"},
    {"proName": "NYMEX:NG1!", "title": "Natural Gas"},

    # 🏦 Top Tech & Growth Stocks
    {"proName": "NASDAQ:AAPL", "title": "Apple"},
    {"proName": "NASDAQ:MSFT", "title": "Microsoft"},
    {"proName": "NASDAQ:NVDA", "title": "NVIDIA"},
    {"proName": "NASDAQ:META", "title": "Meta"},
    {"proName": "NASDAQ:AMZN", "title": "Amazon"},
    {"proName": "NASDAQ:GOOGL", "title": "Google"},
    {"proName": "NASDAQ:TSLA", "title": "Tesla"},

    # 🏦 Financials & Industrials
    {"proName": "NYSE:JPM", "title": "JPMorgan"},
    {"proName": "NYSE:GS", "title": "Goldman Sachs"},
    {"proName": "NYSE:BRK.B", "title": "Berkshire Hathaway"},
    {"proName": "NYSE:V", "title": "Visa"},

    # 💡 Optional crypto for fun
    {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
    {"proName": "BITSTAMP:ETHUSD", "title": "Ethereum"},
]

def _tv_symbol(t: str) -> str:
    t = t.upper().strip().replace("-", ".")  # BRK-B -> BRK.B
    exch = "NASDAQ" if t in NASDAQ_COMMON else "NYSE"
    return f"{exch}:{t}"

def render_fixed_ticker_tape(height: int = 52, dark: bool = True):
    config = {
        "symbols": FIXED_TAPE,
        "showSymbolLogo": True,
        "colorTheme": "dark" if dark else "light",
        "isTransparent": True,
        "displayMode": "adaptive",
        "locale": "en",
    }
    html = f"""
    <div class="tradingview-widget-container" style="width:100%;">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
        {json.dumps(config)}
      </script>
    </div>
    """
    st.components.v1.html(html, height=height, scrolling=False)


# ============================= Sidebar =============================

st.sidebar.title("⚙️ Settings")

universe = load_universe()
default_focus = [t for t in ["AAPL","MSFT","NVDA"] if t in universe] or universe[:3]

selected = st.sidebar.multiselect("Select tickers", options=universe, default=default_focus)
c1, c2 = st.sidebar.columns(2)
if c1.button("Select All"):
    selected = universe
if c2.button("Clear"):
    selected = []
st.session_state["tickers"] = selected

col_dates = st.sidebar.columns(2)
start = col_dates[0].text_input("Start (YYYY-MM-DD)", "2023-01-01")
end = col_dates[1].text_input("End (YYYY-MM-DD, optional)", "")

objective = st.sidebar.selectbox(
    "Objective",
    ["max_sharpe", "min_volatility", "black_litterman"],
    index=0,
)
max_weight = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 0.40, 0.05)
long_only = st.sidebar.toggle("Long only", value=True)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual, e.g. 0.015 = 1.5%)",
    value=0.0010, step=0.0005, format="%.4f"
)

with st.sidebar.expander("💵 Capital (optional, for share counts)", expanded=False):
    capital = st.number_input("Total portfolio capital ($)", value=0.0, min_value=0.0, step=100.0, format="%.2f")
    capital = capital if capital > 0 else None

bl_inputs = {}
if objective == "black_litterman":
    with st.sidebar.expander("🧠 Black–Litterman Inputs", expanded=True):
        st.caption("Provide market caps and absolute views (annual expected returns). Leave blank to fallback to historical.")
        caps_str = st.text_area("Market Caps (ticker: cap, one per line)",
                                "AAPL: 2900000000000\nMSFT: 3100000000000\nNVDA: 3000000000000")
        views_str = st.text_area("Views (ticker: expected_return, one per line)",
                                 "MSFT: 0.11\nAAPL: 0.08")
        bl_tau = st.number_input("BL tau (blend strength)", value=0.05, step=0.01, format="%.2f")
        market_caps = parse_kv(caps_str) if caps_str.strip() else None
        views = parse_kv(views_str) if views_str.strip() else None
        bl_inputs = dict(market_caps=market_caps, views=views, bl_tau=bl_tau)

# ============================= Header / Tape =============================

st.markdown("### 📈 PortfolioQuant.ai")
render_fixed_ticker_tape(height=52, dark=True)
st.divider()

# ============================= Tabs =============================

tab_data, tab_opt = st.tabs(["📊 Data", "🧮 Optimize"])

# ---------------- DATA TAB ----------------
with tab_data:
    st.header("📊 Market Data Preview")
    if st.button("Fetch Data", type="primary"):
        try:
            if not show:
                st.warning("Please select at least one ticker in the sidebar.")
            else:
                cfg = DataConfig(tickers=show, start=start, end=(end or None))
                prices = fetch_prices(cfg)
                st.success(f"Fetched {prices.shape[0]} rows × {prices.shape[1]} assets")
                st.dataframe(prices.tail().round(2), use_container_width=True)

                daily, monthly = daily_and_monthly_returns(prices)
                st.subheader("Summary Stats (annualized)")
                stats = summary_stats(daily).rename_axis("ticker").reset_index()
                st.dataframe(stats.round(4), use_container_width=True)

                st.subheader("Correlation Heatmap")
                corr = daily.corr()
                fig_corr = px.imshow(
                    corr, text_auto=True, aspect="auto",
                    title="Asset Correlations (daily returns)"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# ---------------- OPTIMIZE TAB ----------------
with tab_opt:
    st.header("🧮 Portfolio Optimization")
    run_it = st.button("Run Optimization", type="primary")
    if run_it:
        try:
            if not show:
                st.warning("Please select at least one ticker in the sidebar.")
            else:
                opt_kwargs = dict(
                    tickers=show,
                    start=start,
                    end=(end or None),
                    objective=objective,
                    max_weight=max_weight,
                    long_only=long_only,
                    risk_free_rate=risk_free_rate,
                    capital=capital,
                )
                if objective == "black_litterman":
                    opt_kwargs.update(bl_inputs)

                res = run_optimization(OptConfig(**opt_kwargs))

                # Weights table + pie
                st.subheader("Allocation")
                wdf = weights_to_df(res["weights"])
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(wdf, use_container_width=True)
                with c2:
                    fig_pie = px.pie(wdf, names="Asset", values="Weight", title="Portfolio Allocation")
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Metrics
                st.subheader("Key Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Expected Return (ann.)", f"{res['expected_return']*100:.2f}%")
                m2.metric("Volatility (ann.)", f"{res['volatility']*100:.2f}%")
                m3.metric("Sharpe", f"{res['sharpe']:.2f}")

                # Efficient frontier
                st.subheader("Efficient Frontier")
                frontier = pd.DataFrame({"Risk": res["frontier"]["risks"], "Return": res["frontier"]["returns"]})
                fig_front = px.line(frontier, x="Risk", y="Return", markers=True, title="Risk vs Return")
                fig_front.add_scatter(
                    x=[res["volatility"]],
                    y=[res["expected_return"]],
                    mode="markers",
                    name="Optimal",
                    marker=dict(size=12)
                )
                st.plotly_chart(fig_front, use_container_width=True)

                # Discrete allocation (if provided)
                if "discrete_allocation" in res:
                    st.subheader("Discrete Allocation (Whole Shares)")
                    da = res["discrete_allocation"]
                    shares_df = pd.DataFrame(list(da["shares"].items()), columns=["Asset", "Shares"])
                    st.dataframe(shares_df, use_container_width=True)
                    st.info(f"Leftover cash: ${da['leftover_cash']:.2f} on capital ${da['capital']:.2f}")

                # BL details (if used)
                if objective == "black_litterman" and "black_litterman" in res:
                    with st.expander("Black–Litterman Details"):
                        st.json(res["black_litterman"])

        except Exception as e:
            st.error(f"Optimization error: {e}")
