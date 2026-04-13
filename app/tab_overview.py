"""
OVERVIEW tab, NAV chart with regime bands, KPIs, current state.

Shows the strategy's equity curve against all benchmarks with colored
regime overlay bands. Log scale NAV reveals compounding dynamics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import styled_kpi, styled_divider, apply_plotly_theme, TOKENS
from app.helpers import REGIME_COLORS


def render(data: dict) -> None:
    """Render the OVERVIEW tab."""
    ts = data["ts"]
    overall = data["overall"]
    benchmarks = data["benchmarks"]
    config = data["config"]
    ann_factor = config["vol_estimation"]["annualization_factor"]

    # --- KPI row 1: Core metrics ---
    current_regime = ts["regime_hmm"].iloc[-1]
    regime_color = REGIME_COLORS.get(current_regime, TOKENS["text_muted"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        styled_kpi("CAGR", f"{overall['cagr']:.1%}")
    with c2:
        styled_kpi("Sharpe", f"{overall['sharpe']:.2f}")
    with c3:
        styled_kpi("Max Drawdown", f"{overall['max_drawdown']:.1%}",
                   delta_color=TOKENS["accent_danger"])
    with c4:
        styled_kpi("Current Regime", current_regime, delta_color=regime_color)

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    # --- KPI row 2: Exposure + comparison vs SPY ---
    spy_nav = benchmarks.get("spy_buyhold")
    spy_dd_reduction = np.nan
    vol_reduction_pct = np.nan
    if spy_nav is not None:
        spy_aligned = spy_nav.reindex(ts.index).dropna()
        if len(spy_aligned) > 1:
            spy_peak = spy_aligned.cummax()
            spy_max_dd = ((spy_aligned - spy_peak) / spy_peak).min()
            spy_dd_reduction = spy_max_dd - overall["max_drawdown"]
            spy_ret = spy_aligned.pct_change().dropna()
            spy_vol = spy_ret.std() * np.sqrt(ann_factor)
            strat_vol = overall["annualized_vol"]
            if spy_vol > 0:
                vol_reduction_pct = 1.0 - strat_vol / spy_vol

    c5, c6, c7 = st.columns(3)
    with c5:
        styled_kpi("Gross Exposure", f"{ts['gross_exposure'].iloc[-1]:.0%}")
    with c6:
        dd_val = f"{abs(spy_dd_reduction):.1%}" if pd.notna(spy_dd_reduction) else "N/A"
        styled_kpi("DD Reduction vs SPY", dd_val,
                   delta_color=TOKENS["accent_success"])
    with c7:
        vol_val = f"{vol_reduction_pct:.0%}" if pd.notna(vol_reduction_pct) else "N/A"
        styled_kpi("Vol Reduction vs SPY", vol_val,
                   delta_color=TOKENS["accent_success"])

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # --- NAV chart with regime bands ---
    fig = go.Figure()

    # Regime color bands (background rectangles)
    _add_regime_bands(fig, ts)

    # Strategy NAV (log scale)
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts["nav"], name="Strategy",
        line=dict(color=TOKENS["accent_primary"], width=2.5),
    ))

    # Benchmark NAVs
    bench_styles = {
        "spy_buyhold": ("SPY B&H", TOKENS["accent_danger"], 1.5, "dot"),
        "sixty_forty": ("60/40", TOKENS["accent_info"], 1.5, "dash"),
        "static_vol_parity": ("Static Vol-Parity", TOKENS["accent_secondary"], 1.5, "dashdot"),
        "regime_only": ("Regime Only", TOKENS["accent_warning"], 1.5, "dash"),
    }
    for bname, bnav in benchmarks.items():
        aligned = bnav.reindex(ts.index).dropna()
        label, color, width, dash = bench_styles.get(
            bname, (bname, TOKENS["text_secondary"], 1, "solid"),
        )
        fig.add_trace(go.Scatter(
            x=aligned.index, y=aligned.values, name=label,
            line=dict(color=color, width=width, dash=dash),
        ))

    fig.update_layout(
        title="NAV: Strategy vs Benchmarks",
        yaxis_title="NAV (log scale)", yaxis_type="log",
        height=480, legend=dict(orientation="h", y=-0.15),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")

    styled_divider()

    # --- Summary metrics row ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        styled_kpi("Sortino", f"{overall['sortino']:.2f}")
    with c2:
        styled_kpi("Calmar", f"{overall['calmar']:.2f}")
    with c3:
        styled_kpi("Win Rate (Monthly)", f"{overall['win_rate_monthly']:.0%}")
    with c4:
        styled_kpi("SPY Correlation", f"{overall['correlation_spy']:.2f}")


def _add_regime_bands(fig: go.Figure, ts: pd.DataFrame) -> None:
    """Add regime-colored vertical bands to a plotly figure."""
    regimes = ts["regime_hmm"]
    dates = ts.index
    current = regimes.iloc[0]
    start = dates[0]

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current or i == len(regimes) - 1:
            color = REGIME_COLORS.get(current, TOKENS["text_muted"])
            fig.add_vrect(
                x0=start, x1=dates[i],
                fillcolor=color, opacity=0.07, line_width=0, layer="below",
            )
            current = regimes.iloc[i]
            start = dates[i]
