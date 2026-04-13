"""
PORTFOLIO tab, Asset weights, exposure, realized vs target vol.

Shows how the portfolio composition evolves through regime shifts and
vol-targeting adjustments. The vol ribbon chart visualizes how well
the engine tracks its target.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_section_label, styled_divider, styled_kpi,
    apply_plotly_theme, TOKENS,
)
from app.helpers import REGIME_COLORS, ASSET_COLORS


def render(data: dict) -> None:
    """Render the PORTFOLIO tab."""
    ts = data["ts"]
    alloc_tickers = data["config"]["data"]["allocation_tickers"]
    weight_cols = [f"w_{t}" for t in alloc_tickers]

    # --- Asset weight stacked area ---
    styled_section_label("Asset Allocation Over Time")
    fig_w = go.Figure()
    for ticker in alloc_tickers:
        col = f"w_{ticker}"
        fig_w.add_trace(go.Scatter(
            x=ts.index, y=ts[col], name=ticker, stackgroup="one",
            line=dict(width=0),
            marker=dict(color=ASSET_COLORS.get(ticker, TOKENS["text_secondary"])),
        ))
    fig_w.update_layout(
        title="Portfolio Weights (Stacked)",
        yaxis_title="Weight", height=380,
        legend=dict(orientation="h", y=-0.15),
    )
    apply_plotly_theme(fig_w)
    st.plotly_chart(fig_w, width="stretch")

    styled_divider()

    # --- Two-column: Gross exposure + Vol tracking ---
    col_left, col_right = st.columns(2)

    with col_left:
        styled_section_label("Gross Exposure")
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(
            x=ts.index, y=ts["gross_exposure"], name="Gross Exposure",
            line=dict(color=TOKENS["accent_primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.08)",
        ))
        fig_exp.update_layout(
            title="Daily Gross Exposure",
            yaxis_title="Exposure", yaxis_tickformat=".0%",
            height=340,
        )
        apply_plotly_theme(fig_exp)
        st.plotly_chart(fig_exp, width="stretch")

    with col_right:
        styled_section_label("Realized vs Target Volatility")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=ts.index, y=ts["target_vol"], name="Target Vol",
            line=dict(color=TOKENS["accent_warning"], width=2, dash="dash"),
        ))
        fig_vol.add_trace(go.Scatter(
            x=ts.index, y=ts["realized_port_vol"], name="Realized Vol",
            line=dict(color=TOKENS["accent_primary"], width=1.5),
        ))
        fig_vol.update_layout(
            title="Vol Tracking",
            yaxis_title="Annualized Vol", yaxis_tickformat=".0%",
            height=340,
            legend=dict(orientation="h", y=-0.15),
        )
        apply_plotly_theme(fig_vol)
        st.plotly_chart(fig_vol, width="stretch")

    styled_divider()

    # --- Current allocation snapshot ---
    styled_section_label("Current Portfolio Snapshot")
    cols = st.columns(len(alloc_tickers))
    for i, ticker in enumerate(alloc_tickers):
        with cols[i]:
            w = ts[f"w_{ticker}"].iloc[-1]
            styled_kpi(ticker, f"{w:.1%}",
                       delta_color=ASSET_COLORS.get(ticker, TOKENS["text_muted"]))

    # --- Avg turnover + cost ---
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    ann_factor = data["config"]["vol_estimation"]["annualization_factor"]
    rebal_days = ts[ts["rebalanced"] == True]
    with c1:
        avg_turnover = ts["turnover"].sum() / (len(ts) / ann_factor)
        styled_kpi("Ann. Turnover", f"{avg_turnover:.1%}")
    with c2:
        total_cost = ts["cost"].sum() * ts["nav"].iloc[0]
        styled_kpi("Total Cost Drag", f"{total_cost:.1f} bps NAV")
    with c3:
        n_rebal = ts["rebalanced"].sum()
        styled_kpi("Rebalance Events", f"{int(n_rebal)}")
