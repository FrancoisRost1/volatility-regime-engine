"""
Streamlit dashboard for volatility-regime-engine.

Bloomberg dark mode, 5 tabs: OVERVIEW, REGIMES, PORTFOLIO, ANALYTICS, ATTRIBUTION.
All charts via Plotly. Loads pre-computed tearsheet from the pipeline run.

Run from project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# --- Page config MUST be the first Streamlit call ---
st.set_page_config(
    page_title="Volatility Regime Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

from style_inject import inject_styles, styled_header, TOKENS
from app.helpers import load_data

inject_styles()

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        f"<h2 style='color: {TOKENS['accent_primary']}; "
        f"font-family: {TOKENS['font_display']};'>"
        "◆ Vol Regime Engine</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color: {TOKENS['text_muted']}; font-size: 0.85rem;'>"
        "Volatility targeting + regime detection.<br>"
        "HMM + composite classifiers, walk-forward backtest.</p>",
        unsafe_allow_html=True,
    )

# --- Load all data (cached) ---
data = load_data()

# --- Header with staleness check ---
import datetime

ts = data["ts"]
date_range = f"{ts.index[0].strftime('%b %Y')} — {ts.index[-1].strftime('%b %Y')}"
run_date = data.get("tearsheet_run_date")
run_str = run_date.strftime("%Y-%m-%d %H:%M") if run_date else "unknown"
styled_header("Volatility Regime Engine",
              f"Walk-forward backtest · {date_range} · Run: {run_str}")

if run_date:
    age_days = (datetime.datetime.now() - run_date).days
    if age_days > 7:
        st.warning(
            f"Tearsheet is {age_days} days old. "
            f"Run `python3 main.py` to refresh with latest data."
        )

# --- Tabs ---
tab_ov, tab_reg, tab_port, tab_ana, tab_attr = st.tabs(
    ["OVERVIEW", "REGIMES", "PORTFOLIO", "ANALYTICS", "ATTRIBUTION"]
)

from app import tab_overview, tab_regimes, tab_portfolio, tab_analytics, tab_attribution

with tab_ov:
    tab_overview.render(data)

with tab_reg:
    tab_regimes.render(data)

with tab_port:
    tab_portfolio.render(data)

with tab_ana:
    tab_analytics.render(data)

with tab_attr:
    tab_attribution.render(data)

# --- Footer ---
_lag = data["config"]["backtest"]["signal_to_execution_lag_days"]
_bps = data["config"]["rebalancing"]["cost_bps"]
st.markdown(
    f"<div style='text-align: center; color: {TOKENS['text_muted']}; "
    f"font-size: 0.75rem; margin-top: 2rem;'>"
    f"Volatility Regime Engine · HMM filtered probabilities · "
    f"Signal at t, trade at t+{_lag} · {_bps:.0f}bps turnover cost"
    "</div>",
    unsafe_allow_html=True,
)
