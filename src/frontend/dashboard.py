# src/frontend/dashboard.py
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
from pathlib import Path

from src.agents.optimizer_agent import OptConfig, run_optimization
from src.agents.data_agent import DataConfig, fetch_prices, daily_and_monthly_returns, summary_stats

SYMBOLS_PATH = Path("data/symbols/sp500.csv")

def _load_universe():
    if SYMBOLS_PATH.exists():
        tickers = [t.strip().upper() for t in SYMBOLS_PATH.read_text().splitlines() if t.strip()]
        return sorted(list(dict.fromkeys(tickers)))
    return ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA"]

def _live_quote_df(tickers: list[str]) -> pd.DataFrame:
    # Lightweight live fetch: current price & daily change
    # yfinance 'fast' method
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            last = info.get("last_price", None)
            prev = info.get("previous_close", None)
            chg = None
            if last is not None and prev not in (None, 0):
                chg = (last/prev - 1.0) * 100.0
            data.append({"Ticker": t, "Price": last, "Change %": chg})
        except Exception:
            data.append({"Ticker": t, "Price": None, "Change %": None})
    return pd.DataFrame(data)

def _top_bar_live(quotes: pd.DataFrame):
    # Render a slim ticker-tape style bar
    cols = st.columns(len(quotes))
    for i, (_, row) in enumerate(quotes.iterrows()):
        color = "🟢" if (row["Change %"] or 0) >= 0 else "🔴"
        txt = f"**{row['Ticker']}**  \n{row['Price']:.2f}  \n{color} {row['Change %']:.2f}%"
        cols[i].markdown(txt)

def render_dashboard():
    st.session_state.page = "app"  # ensure we stay here on rerun
    st.title("📈 PortfolioQuant.ai")

    # ---------- Live market top bar (auto-refresh) ----------
    st.caption("Live market snapshot (auto-refreshes every 30s)")
    refresh = st.sidebar.checkbox("Auto-refresh top bar (30s)", value=True)
    if refresh:
        st.experimental_autorefresh(interval=30_000, key="livebar")

    default_focus = ["AAPL","MSFT","NVDA","AMZN","GOOGL"]
    universe = _load_universe()

    # ---------- Ticker picker ----------
    with st.sidebar:
        st.subheader("Universe")
        # Multiselect with search over many tickers
        picks = st.multiselect("Select tickers", universe, default=default_focus)
        # “Select all” convenience
        c1, c2 = st.columns(2)
        if c1.button("Select All"):
            picks = universe
        if c2.button("Clear"):
            picks = []
        st.session_state.selected_tickers = picks

    show = st.session_state.get("selected_tickers", default_focus) or default_focus
    quotes = _live_quote_df(show[:8])  # show up to 8 across top bar
    _top_bar_live(quotes)

    st.divider()

    # ---------- Tabs for the app ----------
    tab_data, tab_opt = st.tabs(["📊 Data", "🧮 Optimize"])

    # ----- DATA TAB -----
    with tab_data:
        st.header("📊 Market Data Preview")
        start = st.text_input("Start (YYYY-MM-DD)", "2023-01-01")
        end = st.text_input("End (YYYY-MM-DD, optional)", "")
        if st.button("Fetch Data", type="primary"):
            try:
                cfg = DataConfig(tickers=show, start=start, end=(end or None))
                prices = fetch_prices(cfg)
                st.success(f"Fetched {prices.shape[0]} rows × {prices.shape[1]} assets")
                st.dataframe(prices.tail().round(2), use_container_width=True)

                daily, monthly = daily_and_monthly_returns(prices)
                st.subheader("Summary Stats (annualized)")
                stats = summary_stats(daily)
                st.dataframe(stats.round(4), use_container_width=True)

                st.subheader("Correlation Heatmap")
                corr = daily.corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     title="Asset Correlations (daily returns)")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching data: {e}")

    # ----- OPTIMIZE TAB -----
    with tab_opt:
        st.header("🧮 Portfolio Optimization")
        c1, c2, c3 = st.columns([1,1,1])
        objective = c1.selectbox("Objective", ["max_sharpe","min_volatility","black_litterman"], index=0)
        max_weight = c2.slider("Max weight", 0.05, 1.0, 0.40, 0.05)
        long_only = c3.toggle("Long only", value=True)

        risk_free_rate = st.number_input("Risk-free rate (annual)", value=0.0, step=0.001, format="%.4f")
        capital = st.number_input("Capital (optional for share counts)", value=0.0, step=100.0, format="%.2f")
        capital = capital if capital > 0 else None

        bl_kwargs = {}
        if objective == "black_litterman":
            with st.expander("🧠 Black–Litterman Inputs", expanded=True):
                st.caption("Provide market caps and absolute return views")
                caps_str = st.text_area("Market Caps: ticker: cap", "AAPL: 2900000000000\nMSFT: 3100000000000\nNVDA: 3000000000000")
                views_str = st.text_area("Views: ticker: expected_return", "MSFT: 0.11\nAAPL: 0.08")
                bl_tau = st.number_input("BL tau", value=0.05, step=0.01, format="%.2f")

                def parse_kv(text):
                    out = {}
                    for line in text.splitlines():
                        if ":" in line:
                            k,v = line.split(":",1)
                            try:
                                out[k.strip().upper()] = float(v.strip())
                            except:
                                pass
                    return out
                bl_kwargs = dict(
                    market_caps=parse_kv(caps_str) if caps_str.strip() else None,
                    views=parse_kv(views_str) if views_str.strip() else None,
                    bl_tau=bl_tau
                )

        if st.button("Run Optimization", type="primary"):
            try:
                start = st.text_input("Start date for optimization (YYYY-MM-DD)", "2023-01-01", key="opt_start")
                end = st.text_input("End date for optimization (optional)", "", key="opt_end")
                opt = OptConfig(
                    tickers=show,
                    start=start,
                    end=(end or None),
                    objective=objective,
                    max_weight=max_weight,
                    long_only=long_only,
                    risk_free_rate=risk_free_rate,
                    capital=capital,
                    **bl_kwargs
                )
                res = run_optimization(opt)

                # Allocation
                w = pd.DataFrame(list(res["weights"].items()), columns=["Asset","Weight"])
                w["Weight %"] = (w["Weight"]*100).round(2)
                c1, c2 = st.columns([1,1])
                with c1:
                    st.subheader("Allocation")
                    st.dataframe(w.sort_values("Weight", ascending=False), use_container_width=True)
                with c2:
                    st.subheader("Allocation Pie")
                    st.plotly_chart(px.pie(w, names="Asset", values="Weight"), use_container_width=True)

                # Metrics
                m1,m2,m3 = st.columns(3)
                m1.metric("Expected Return (ann.)", f"{res['expected_return']*100:.2f}%")
                m2.metric("Volatility (ann.)", f"{res['volatility']*100:.2f}%")
                m3.metric("Sharpe", f"{res['sharpe']:.2f}")

                # Frontier
                st.subheader("Efficient Frontier")
                f = pd.DataFrame({"Risk": res["frontier"]["risks"], "Return": res["frontier"]["returns"]})
                fig = px.line(f, x="Risk", y="Return", markers=True, title="Risk vs Return")
                fig.add_scatter(x=[res["volatility"]], y=[res["expected_return"]], mode="markers", name="Optimal", marker=dict(size=12))
                st.plotly_chart(fig, use_container_width=True)

                # Discrete allocation
                if "discrete_allocation" in res:
                    st.subheader("Discrete Allocation (Shares)")
                    da = res["discrete_allocation"]
                    st.dataframe(pd.DataFrame(list(da["shares"].items()), columns=["Asset","Shares"]), use_container_width=True)
                    st.info(f"Leftover cash: ${da['leftover_cash']:.2f} on capital ${da['capital']:.2f}")

                if "black_litterman" in res:
                    with st.expander("Black–Litterman Details"):
                        st.json(res["black_litterman"])

            except Exception as e:
                st.error(f"Optimization error: {e}")
