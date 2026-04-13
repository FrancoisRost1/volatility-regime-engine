"""
REGIMES tab, HMM probabilities, composite regime, transition matrix.

Visualizes regime dynamics: how the model classifies markets over time,
transition probabilities between states, and agreement between HMM and
the rule-based composite classifier.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_kpi, styled_divider, styled_section_label,
    apply_plotly_theme, TOKENS,
)
from app.helpers import REGIME_COLORS


def render(data: dict) -> None:
    """Render the REGIMES tab."""
    ts = data["ts"]
    regime_cond = data["regime_cond"]
    duration_stats = data["duration_stats"]
    transition = data["transition"]

    # --- HMM state probability area chart ---
    styled_section_label("HMM Regime Probabilities")
    fig_prob = go.Figure()
    # Hex -> rgba for stacked area fill transparency
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for col, regime, color in [
        ("p_risk_on", "RISK_ON", REGIME_COLORS["RISK_ON"]),
        ("p_neutral", "NEUTRAL", REGIME_COLORS["NEUTRAL"]),
        ("p_risk_off", "RISK_OFF", REGIME_COLORS["RISK_OFF"]),
    ]:
        fig_prob.add_trace(go.Scatter(
            x=ts.index, y=ts[col], name=regime, stackgroup="one",
            line=dict(width=0.5, color=color),
            fillcolor=_hex_to_rgba(color, 0.6),
        ))
    fig_prob.update_layout(
        title="Filtered State Probabilities (Forward Algorithm)",
        yaxis_title="Probability", yaxis_range=[0, 1], height=360,
        legend=dict(orientation="h", y=-0.15),
    )
    apply_plotly_theme(fig_prob)
    st.plotly_chart(fig_prob, width="stretch")

    styled_divider()

    # --- Two-column layout: Transition matrix + Time-in-regime ---
    col_left, col_right = st.columns(2)

    with col_left:
        styled_section_label("Transition Matrix P(next | current)")
        labels = ["RISK_ON", "NEUTRAL", "RISK_OFF"]
        fig_tm = go.Figure(data=go.Heatmap(
            z=transition.values,
            x=labels, y=labels,
            colorscale=[
                [0, TOKENS["bg_elevated"]],
                [1, TOKENS["accent_primary"]],
            ],
            text=np.round(transition.values, 3),
            texttemplate="%{text:.1%}",
            textfont=dict(size=13, color=TOKENS["text_primary"]),
            showscale=False,
        ))
        fig_tm.update_layout(
            title="Regime Persistence & Transitions",
            xaxis_title="To", yaxis_title="From",
            yaxis_autorange="reversed", height=340,
        )
        apply_plotly_theme(fig_tm)
        st.plotly_chart(fig_tm, width="stretch")

    with col_right:
        styled_section_label("Time in Each Regime")
        pct = regime_cond["pct_time"]
        fig_pie = go.Figure(data=go.Pie(
            labels=pct.index, values=pct.values,
            marker=dict(colors=[REGIME_COLORS[r] for r in pct.index]),
            hole=0.65,
            textinfo="label+percent",
            textfont=dict(size=12, color=TOKENS["text_primary"]),
        ))
        fig_pie.update_layout(title="Regime Distribution", height=340,
                              showlegend=False)
        apply_plotly_theme(fig_pie)
        st.plotly_chart(fig_pie, width="stretch")

    styled_divider()

    # --- Duration stats + Agreement rate ---
    col_a, col_b, col_c = st.columns(3)
    for i, (regime, row) in enumerate(duration_stats.iterrows()):
        col = [col_a, col_b, col_c][i]
        with col:
            styled_kpi(
                f"{regime} Avg Duration",
                f"{row['avg_duration_days']:.0f} days",
                delta=f"{int(row['n_episodes'])} episodes",
                delta_color=REGIME_COLORS.get(regime, TOKENS["text_muted"]),
            )

    # HMM vs Composite agreement rate
    agreement = (ts["regime_hmm"] == ts["regime_composite"]).mean()
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        styled_kpi("HMM vs Composite Agreement", f"{agreement:.0%}")

    styled_divider()

    # --- Strategic allocation table (from config.yaml) ---
    styled_section_label("Regime Strategic Weights")
    config = data["config"]
    strat_w = config["position_sizing"]["strategic_weights"]
    target_vol = config["position_sizing"]["target_vol"]
    lev_caps = config["position_sizing"]["leverage_caps"]
    alloc_rows = []
    for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
        row = {"Regime": regime}
        for asset in ["SPY", "TLT", "GLD", "PDBC"]:
            row[asset] = f"{strat_w[regime][asset]:.0%}"
        row["Target Vol"] = f"{target_vol[regime]:.0%}"
        row["Lev Cap"] = f"{lev_caps[regime]:.0%}"
        alloc_rows.append(row)
    st.dataframe(pd.DataFrame(alloc_rows), hide_index=True,
                 width="stretch")
