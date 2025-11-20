# streamlit_app.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.data_agent import (
    DataConfig,
    fetch_prices,
    daily_and_monthly_returns,
    summary_stats,
)
from src.agents.optimizer_agent import OptConfig, run_optimization
from src.agents.risk_agent import RiskConfig, run_risk
from src.agents.backtest_agent import BacktestConfig, run_backtest
from data.ticker_names import TICKER_NAMES  # your ticker -> name dict


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
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "BRK-B",
        "JPM",
        "V",
        "XOM",
        "UNH",
        "AVGO",
    ]


def weights_to_df(weights: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(weights.items()), columns=["Asset", "Weight"])
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def parse_kv(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip().upper()] = float(v.strip())
            except Exception:
                # ignore bad lines
                pass
    return out


def fmt_ticker(t: str) -> str:
    """Show 'TICKER — Company Name' in the UI."""
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

# session state for ticker selections
if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = default_focus

st.sidebar.multiselect(
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
    value=0.0010,
    step=0.0005,
    format="%.4f",
)

with st.sidebar.expander("💵 Capital (optional, for shares / backtest)"):
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
            "Provide market caps and absolute views (annual expected returns).\n"
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
        bl_tau = st.number_input("BL tau (blend strength)", value=0.05, step=0.01)
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
# Tabs: Data / Optimize / Risk / Backtest
# =============================================================

tab_data, tab_opt, tab_risk, tab_bt = st.tabs(
    ["📊 Data", "🧮 Optimize", "⚠️ Risk", "⏪ Backtest"]
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
                st.success(
                    f"Fetched {prices.shape[0]} rows × {prices.shape[1]} assets"
                )
                st.dataframe(prices.tail().round(2), use_container_width=True)

                daily, monthly = daily_and_monthly_returns(prices)
                st.subheader("Summary Stats (annualized)")
                st.dataframe(
                    summary_stats(daily).round(4),
                    use_container_width=True,
                )

                st.subheader("Correlation Heatmap")
                fig = px.imshow(
                    daily.corr(),
                    text_auto=True,
                    aspect="auto",
                    title="Asset Correlations (daily returns)",
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


# ============================= OPTIMIZER TAB =============================
with tab_opt:
    st.header("🧮 Portfolio Optimization")

    if st.button("Run Optimization", type="primary"):
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

                # Save for Backtest tab
                st.session_state["last_opt_result"] = res

                st.subheader("Allocation")
                wdf = weights_to_df(res["weights"])
                c1, c2 = st.columns([1, 1])
                c1.dataframe(wdf, use_container_width=True)
                c2.plotly_chart(
                    px.pie(
                        wdf,
                        names="Asset",
                        values="Weight",
                        title="Portfolio Allocation",
                    ),
                    use_container_width=True,
                )

                st.subheader("Key Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Expected Return (ann.)",
                    f"{res['expected_return']*100:.2f}%",
                )
                m2.metric(
                    "Volatility (ann.)",
                    f"{res['volatility']*100:.2f}%",
                )
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
                    st.subheader("Discrete Allocation (Whole Shares)")
                    shares_df = pd.DataFrame(
                        list(da["shares"].items()), columns=["Asset", "Shares"]
                    )
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
    st.caption("Uses an equal-weight portfolio for now. We can later plug in optimizer weights.")

    if st.button("Run Risk Analysis", type="primary"):
        try:
            if not selected_tickers:
                st.warning("Please select at least one ticker.")
            else:
                rcfg = RiskConfig(
                    tickers=selected_tickers,
                    start=start,
                    end=(end or None),
                    risk_free_rate=risk_free_rate,
                )

                risk = run_risk(rcfg)

                st.subheader("Risk Metrics (annualized)")

                # Format as % where it makes sense
                raw_metrics = risk["metrics"]
                display_rows = []
                for k, v in raw_metrics.items():
                    key_lower = k.lower()
                    if key_lower in [
                        "annual_return",
                        "annual_volatility",
                        "var_95",
                        "cvar_95",
                        "max_drawdown",
                    ]:
                        display_rows.append((k, f"{v * 100:.2f}%"))
                    else:
                        # e.g. Sharpe, beta, etc.
                        display_rows.append((k, f"{v:.4f}"))

                metrics_df = pd.DataFrame(display_rows, columns=["Metric", "Value"])
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Weights Used (Risk)")
                wdf = pd.DataFrame(
                    list(risk["weights"].items()), columns=["Asset", "Weight"]
                )
                wdf["Weight %"] = (wdf["Weight"] * 100).round(2)
                st.dataframe(wdf, use_container_width=True)

                st.subheader("Cumulative Portfolio Return")
                cum = risk["series"]["cumulative_returns"]
                st.line_chart(cum)

        except Exception as e:
            st.error(f"Risk analysis error: {e}")


# ============================= BACKTEST TAB =============================
with tab_bt:
    st.header("⏪ Backtest Portfolio")
    st.caption(
        "Simulate how this portfolio would have performed historically.\n"
        "You can use equal-weight or the last optimized allocation."
    )

    bt_mode = st.radio(
        "Weights to use for backtest",
        ["Equal-weight (selected tickers)", "Use last optimized weights"],
        index=0,
    )

    if st.button("Run Backtest", type="primary"):
        try:
            if bt_mode.startswith("Equal"):
                # Use current sidebar tickers, equal-weight
                if not selected_tickers:
                    st.warning("Please select at least one ticker.")
                else:
                    bt_tickers = selected_tickers
                    bt_weights = None
                    bt_start = start
                    bt_end = end or None
            else:
                # Use last optimization result
                if "last_opt_result" not in st.session_state:
                    st.warning("Run an optimization first in the 'Optimize' tab.")
                    bt_tickers = None  # just to avoid reference error
                else:
                    last = st.session_state["last_opt_result"]
                    bt_tickers = last["universe"]
                    bt_weights = last["weights"]
                    bt_start = last.get("start", start)
                    bt_end = last.get("end", end or None)

            if bt_mode.startswith("Use last") and "last_opt_result" not in st.session_state:
                # no backtest because no optimization yet
                st.stop()

            if not bt_tickers:
                st.warning("No tickers available for backtest.")
                st.stop()

            initial_cap = capital if capital is not None else 100000.0

            cfg_bt = BacktestConfig(
                tickers=bt_tickers,
                start=bt_start,
                end=bt_end,
                weights=bt_weights,
                initial_capital=initial_cap,
                risk_free_rate=risk_free_rate,
            )

            bt = run_backtest(cfg_bt)

            st.subheader("Backtest Metrics")

            m = bt["metrics"]
            rows = []
            for k, v in m.items():
                k_lower = k.lower()
                if k_lower in ["total_return", "annual_return", "annual_volatility", "max_drawdown"]:
                    rows.append((k, f"{v * 100:.2f}%"))
                else:
                    rows.append((k, f"{v:.4f}"))
            mdf = pd.DataFrame(rows, columns=["Metric", "Value"])
            st.dataframe(mdf, use_container_width=True)

            st.subheader("Weights Used in Backtest")
            wdf = pd.DataFrame(list(bt["weights"].items()), columns=["Asset", "Weight"])
            wdf["Weight %"] = (wdf["Weight"] * 100).round(2)
            st.dataframe(wdf, use_container_width=True)

            st.subheader("Cumulative Return")
            cum = bt["series"]["cumulative_returns"]
            st.line_chart(cum)

            st.subheader("Portfolio Value Over Time")
            vals = bt["series"]["portfolio_values"]
            st.line_chart(vals)

        except Exception as e:
            st.error(f"Backtest error: {e}")
