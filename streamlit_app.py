# streamlit_app.py
from pathlib import Path
import json
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.agents.data_agent import (
    DataConfig,
    fetch_prices,
    daily_and_monthly_returns,
    summary_stats,
)
from src.agents.optimizer_agent import OptConfig, run_optimization
from src.agents.risk_agent import RiskConfig, run_risk
from src.agents.ai_explainer import explain_risk_from_dict
from data.ticker_names import TICKER_NAMES  # your dict of ticker -> name

# Load .env so GOOGLE_API_KEY etc. are available
load_dotenv()

st.set_page_config(page_title="PortfolioQuant.ai", page_icon="📈", layout="wide")

# =============================================================
# Helpers
# =============================================================

SYMBOLS_PATH = Path("data/symbols/sp500.csv")


def load_universe() -> list[str]:
    """Load ticker universe from CSV, fallback if missing."""
    if SYMBOLS_PATH.exists():
        tickers = [
            t.strip().upper()
            for t in SYMBOLS_PATH.read_text().splitlines()
            if t.strip()
        ]
        return sorted(list(dict.fromkeys(tickers)))
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "BRK-B", "JPM", "V", "XOM", "UNH", "AVGO"
    ]


def weights_to_df(weights: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def parse_kv(text: str) -> dict[str, float]:
    """Parse 'TICKER: value' lines into a dict."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip().upper()] = float(v.strip())
            except Exception:
                pass
    return out


def fmt_ticker(t: str) -> str:
    name = TICKER_NAMES.get(t, "")
    return f"{t} — {name}" if name else t


# =============================================================
# TradingView market tape
# =============================================================

FIXED_TAPE = [
    {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
    {"proName": "NASDAQ:NDX", "title": "NASDAQ 100"},
    {"proName": "FOREXCOM:DJI", "title": "Dow Jones"},
    {"proName": "FOREXCOM:US30", "title": "US 30"},
    {"proName": "FOREXCOM:VIX", "title": "VIX"},
    {"proName": "FX_IDC:EURUSD", "title": "EUR/USD"},
    {"proName": "FX_IDC:USDJPY", "title": "USD/JPY"},
    {"proName": "COMEX:GC1!", "title": "Gold"},
    {"proName": "NYMEX:CL1!", "title": "Crude Oil"},
    {"proName": "TVC:SILVER", "title": "Silver"},
    {"proName": "NYMEX:NG1!", "title": "Natural Gas"},
    {"proName": "NASDAQ:AAPL", "title": "Apple"},
    {"proName": "NASDAQ:MSFT", "title": "Microsoft"},
    {"proName": "NASDAQ:NVDA", "title": "NVIDIA"},
    {"proName": "NASDAQ:META", "title": "Meta"},
    {"proName": "NASDAQ:AMZN", "title": "Amazon"},
    {"proName": "NASDAQ:GOOGL", "title": "Google"},
    {"proName": "NASDAQ:TSLA", "title": "Tesla"},
    {"proName": "NYSE:JPM", "title": "JPMorgan"},
    {"proName": "NYSE:GS", "title": "Goldman Sachs"},
    {"proName": "NYSE:BRK.B", "title": "Berkshire Hathaway"},
    {"proName": "NYSE:V", "title": "Visa"},
    {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
    {"proName": "BITSTAMP:ETHUSD", "title": "Ethereum"},
]


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


# =============================================================
# Sidebar
# =============================================================

st.sidebar.title("⚙️ Settings")

universe = load_universe()
default_focus = [t for t in ["AAPL", "MSFT", "NVDA"] if t in universe] or universe[:3]

if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = default_focus

selected = st.sidebar.multiselect(
    "Select tickers",
    options=universe,
    default=st.session_state["selected_tickers"],
    format_func=fmt_ticker,
    key="selected_tickers",
)

c1, c2 = st.sidebar.columns(2)
if c1.button("Select All"):
    st.session_state["selected_tickers"] = universe
if c2.button("Clear"):
    st.session_state["selected_tickers"] = []

selected_tickers = st.session_state["selected_tickers"]

col_dates = st.sidebar.columns(2)
start = col_dates[0].text_input("Start (YYYY-MM-DD)", "2023-01-01")
end = col_dates[1].text_input("End (YYYY-MM-DD)", "")

objective = st.sidebar.selectbox(
    "Objective",
    ["max_sharpe", "min_volatility", "black_litterman"],
    index=0,
)

max_weight = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 0.40, 0.05)
long_only = st.sidebar.toggle("Long only", value=True)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual, e.g. 0.015 = 1.5%)",
    value=0.0010,
    step=0.0005,
    format="%.4f",
)

with st.sidebar.expander("💵 Capital (optional, for share counts)", expanded=False):
    capital = st.number_input(
        "Total portfolio capital ($)",
        value=0.0,
        min_value=0.0,
        step=100.0,
        format="%.2f",
    )
    capital = capital if capital > 0 else None

bl_inputs: dict = {}
if objective == "black_litterman":
    with st.sidebar.expander("🧠 Black–Litterman Inputs", expanded=True):
        st.caption(
            "Provide market caps and absolute views (annual expected returns). "
            "Leave blank to fallback to historical."
        )
        caps_str = st.text_area(
            "Market Caps (ticker: cap, one per line)",
            "AAPL: 2900000000000\nMSFT: 3100000000000\nNVDA: 3000000000000",
        )
        views_str = st.text_area(
            "Views (ticker: expected_return, one per line)",
            "MSFT: 0.11\nAAPL: 0.08",
        )
        bl_tau = st.number_input(
            "BL tau (blend strength)", value=0.05, step=0.01, format="%.2f"
        )
        market_caps = parse_kv(caps_str) if caps_str.strip() else None
        views = parse_kv(views_str) if views_str.strip() else None
        bl_inputs = dict(market_caps=market_caps, views=views, bl_tau=bl_tau)


# =============================================================
# Header
# =============================================================

st.markdown("### 📈 PortfolioQuant.ai")
render_fixed_ticker_tape()
st.divider()

# =============================================================
# Tabs (Data / Optimize / Risk / Chat)
# =============================================================

tab_data, tab_opt, tab_risk, tab_chat = st.tabs(
    ["📊 Data", "🧮 Optimize", "⚠️ Risk", "💬 AI Chat"]
)

# ============================= DATA TAB =============================
with tab_data:
    st.header("📊 Market Data Preview")

    if st.button("Fetch Data", type="primary"):
        try:
            if not selected_tickers:
                st.warning("Please select at least one ticker.")
            else:
                cfg = DataConfig(selected_tickers, start, end or None)
                prices = fetch_prices(cfg)
                st.success(f"Fetched {prices.shape[0]} rows × {prices.shape[1]} assets")
                st.dataframe(prices.tail().round(2), use_container_width=True)

                daily, monthly = daily_and_monthly_returns(prices)
                st.subheader("Summary Stats (annualized)")
                st.dataframe(summary_stats(daily).round(4), use_container_width=True)

                st.subheader("Correlation Heatmap")
                fig = px.imshow(daily.corr(), text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


# ============================= OPTIMIZER TAB =============================
with tab_opt:
    st.header("🧮 Portfolio Optimization")

    run_it = st.button("Run Optimization", type="primary")
    if run_it:
        try:
            if not selected_tickers:
                st.warning("Please select at least one ticker.")
            else:
                opt_kwargs = dict(
                    tickers=selected_tickers,
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

                # Store in session so Risk tab / Chat tab can use it
                st.session_state["last_opt_result"] = res

                st.subheader("Allocation")
                wdf = weights_to_df(res["weights"])
                c1, c2 = st.columns([1, 1])
                c1.dataframe(wdf, use_container_width=True)
                c2.plotly_chart(px.pie(wdf, names="Asset", values="Weight"), use_container_width=True)

                st.subheader("Key Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Expected Return", f"{res['expected_return']*100:.2f}%")
                m2.metric("Volatility", f"{res['volatility']*100:.2f}%")
                m3.metric("Sharpe", f"{res['sharpe']:.2f}")

                st.subheader("Efficient Frontier")
                fdf = pd.DataFrame(
                    {"Risk": res["frontier"]["risks"], "Return": res["frontier"]["returns"]}
                )
                fig = px.line(fdf, x="Risk", y="Return", markers=True)
                fig.add_scatter(
                    x=[res["volatility"]],
                    y=[res["expected_return"]],
                    mode="markers",
                    name="Optimal",
                    marker=dict(size=12),
                )
                st.plotly_chart(fig, use_container_width=True)

                if "discrete_allocation" in res:
                    da = res["discrete_allocation"]
                    st.subheader("Discrete Allocation")
                    shares_df = pd.DataFrame(list(da["shares"].items()), columns=["Asset", "Shares"])
                    st.dataframe(shares_df, use_container_width=True)

                    leftover = da["leftover_cash"]
                    cap_val = da["capital"]
                    st.markdown(
                        f"""
                        <p style="font-size:16px;">
                        <b>Leftover cash:</b> ${leftover:,.2f} <b>Left of</b> ${cap_val:,.2f} Capital
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Optimization error: {e}")


# ============================= RISK TAB =============================
with tab_risk:
    st.header("⚠️ Risk Analysis")

    weight_mode = st.radio(
        "Weights for risk analysis",
        ["Equal weight", "Use last optimized weights"],
        horizontal=True,
    )

    if st.button("Run Risk Analysis", type="primary"):
        try:
            if not selected_tickers:
                st.warning("Please select at least one ticker.")
            else:
                custom_weights: Dict[str, float] | None = None

                if weight_mode == "Use last optimized weights":
                    if "last_opt_result" not in st.session_state:
                        st.warning(
                            "No optimization result found. Run the optimizer first or use Equal weight."
                        )
                    else:
                        opt_res = st.session_state["last_opt_result"]
                        opt_w = opt_res.get("weights", {})
                        # Restrict weights to currently selected tickers and renormalize
                        w_vec = {t: float(opt_w.get(t, 0.0)) for t in selected_tickers}
                        total = sum(w_vec.values())
                        if total > 0:
                            custom_weights = {t: w / total for t, w in w_vec.items()}
                        else:
                            st.warning(
                                "Optimized weights sum to 0 for current selection. Falling back to equal weight."
                            )
                            custom_weights = None

                rcfg = RiskConfig(
                    tickers=selected_tickers,
                    start=start,
                    end=(end or None),
                    risk_free_rate=risk_free_rate,
                    weights=custom_weights,
                )

                risk = run_risk(rcfg)

                # Save risk result so Chat tab can use it
                st.session_state["last_risk_result"] = risk

                # Show metrics (convert to % where it makes sense)
                st.subheader("Risk Metrics (annualized)")
                metrics = risk["metrics"].copy()
                display_metrics = {
                    "annual_return (%)": metrics["annual_return"] * 100,
                    "annual_volatility (%)": metrics["annual_volatility"] * 100,
                    "sharpe": metrics["sharpe"],
                    "VaR 95% (daily, %)": metrics["var_95"] * 100,
                    "CVaR 95% (daily, %)": metrics["cvar_95"] * 100,
                    "max_drawdown (%)": metrics["max_drawdown"] * 100,
                }
                metrics_df = (
                    pd.DataFrame.from_dict(display_metrics, orient="index", columns=["Value"])
                    .round(4)
                )
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Weights Used (Risk)")
                wdf = pd.DataFrame(list(risk["weights"].items()), columns=["Asset", "Weight"])
                wdf["Weight %"] = (wdf["Weight"] * 100).round(2)
                st.dataframe(wdf.sort_values("Weight", ascending=False), use_container_width=True)

                st.subheader("Cumulative Portfolio Return")
                cum = risk["series"]["cumulative_returns"]
                st.line_chart(cum)

                # AI explanation for the risk metrics
                st.subheader("🧠 AI Explanation")
                with st.spinner("Thinking about your risk profile..."):
                    explanation = explain_risk_from_dict(risk["metrics"], risk["weights"])
                st.write(explanation)

        except Exception as e:
            st.error(f"Risk analysis error: {e}")


# ============================= CHAT TAB (GPT-style UI) =============================
with tab_chat:
    st.header("💬 AI Portfolio Chatbot")

    st.caption(
        "Ask questions about your portfolio, optimization results, or risk profile. "
        "If no portfolio is loaded yet, I can still chat normally."
    )

    # Chat history as list of dicts: {"role": "user"|"assistant", "content": str}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Render previous messages as chat bubbles
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input at the bottom (like GPT)
    user_msg = st.chat_input("Ask about your portfolio...")
    if user_msg:
        # 1) Add user message
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_msg}
        )

        # 2) Build context: use last risk or last optimization if present
        metrics_ctx = None
        weights_ctx = None

        if "last_risk_result" in st.session_state:
            r = st.session_state["last_risk_result"]
            metrics_ctx = r.get("metrics")
            weights_ctx = r.get("weights")
        elif "last_opt_result" in st.session_state:
            o = st.session_state["last_opt_result"]
            metrics_ctx = {
                "annual_return": o.get("expected_return", 0.0),
                "annual_volatility": o.get("volatility", 0.0),
                "sharpe": o.get("sharpe", 0.0),
            }
            weights_ctx = o.get("weights", {})

        # 3) Generate answer (works even if metrics_ctx is None)
        with st.spinner("Thinking..."):
            bot_reply = explain_risk_from_dict(
                metrics=metrics_ctx,
                weights=weights_ctx,
                question=user_msg,
            )

        # 4) Add bot message
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": bot_reply}
        )

        # Re-run to show new messages immediately
        st.rerun()
