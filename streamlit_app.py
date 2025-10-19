# streamlit_app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.data_agent import DataConfig, fetch_prices, daily_and_monthly_returns, summary_stats, covariance_matrix
from src.agents.optimizer_agent import OptConfig, run_optimization

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

# ---- Helpers ---------------------------------------------------------------
def parse_tickers(s: str):
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def weights_to_df(weights: dict):
    df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)

# ---- Sidebar ---------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

tickers_str = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA")
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
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, e.g. 0.015 = 1.5%)", value=0.0, step=0.001, format="%.4f")

with st.sidebar.expander("💵 Capital (optional, for share counts)", expanded=False):
    capital = st.number_input("Total portfolio capital ($)", value=0.0, min_value=0.0, step=100.0, format="%.2f")
    if capital == 0.0:
        capital = None

bl_inputs = {}
if objective == "black_litterman":
    with st.sidebar.expander("🧠 Black–Litterman Inputs", expanded=True):
        st.caption("Provide market caps and absolute views (annual expected returns). Leave blank to fallback to historical.")
        caps_str = st.text_area("Market Caps (ticker: cap, one per line)", "AAPL: 2900000000000\nMSFT: 3100000000000\nNVDA: 3000000000000")
        views_str = st.text_area("Views (ticker: expected_return, one per line)", "MSFT: 0.11\nAAPL: 0.08")
        bl_tau = st.number_input("BL tau (blend strength)", value=0.05, step=0.01, format="%.2f")

        def parse_kv(text, val_type=float):
            out = {}
            for line in text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().upper()
                    try:
                        out[k] = val_type(v.strip())
                    except Exception:
                        pass
            return out

        market_caps = parse_kv(caps_str, float) if caps_str.strip() else None
        views = parse_kv(views_str, float) if views_str.strip() else None
        bl_inputs = dict(market_caps=market_caps, views=views, bl_tau=bl_tau)

# ---- Main tabs -------------------------------------------------------------
tab_data, tab_opt = st.tabs(["📊 Data", "🧮 Optimize"])

# ================= DATA TAB =================
with tab_data:
    st.header("📊 Market Data Preview")
    tickers = parse_tickers(tickers_str)
    if st.button("Fetch Data", type="primary"):
        try:
            cfg = DataConfig(tickers=tickers, start=start, end=(end or None))
            prices = fetch_prices(cfg)
            st.success(f"Fetched {prices.shape[0]} rows × {prices.shape[1]} assets")
            st.dataframe(prices.tail().round(2), use_container_width=True)

            daily, monthly = daily_and_monthly_returns(prices)
            st.subheader("Summary Stats (annualized)")
            stats = summary_stats(daily)
            st.dataframe(stats.round(4), use_container_width=True)

            st.subheader("Correlation Heatmap")
            corr = daily.corr()
            corr_df = corr.copy()
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                aspect="auto",
                title="Asset Correlations (daily returns)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# ================= OPTIMIZE TAB =================
with tab_opt:
    st.header("🧮 Portfolio Optimization")

    run_it = st.button("Run Optimization", type="primary")
    if run_it:
        try:
            tickers = parse_tickers(tickers_str)
            opt_kwargs = dict(
                tickers=tickers,
                start=start,
                end=(end or None),
                objective=objective,
                max_weight=max_weight,
                long_only=long_only,
                risk_free_rate=risk_free_rate,
                capital=capital,
            )
            # Include BL fields only for BL objective
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

            # Echo BL inputs for transparency
            if objective == "black_litterman" and "black_litterman" in res:
                with st.expander("Black–Litterman Details"):
                    st.json(res["black_litterman"])

        except Exception as e:
            st.error(f"Optimization error: {e}")
