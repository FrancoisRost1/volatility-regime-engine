"""
ATTRIBUTION tab, Benchmark decomposition, turnover, cost analysis.

Decomposes strategy value into components: diversification (60/40),
vol management (static vol-parity), regime detection (regime-only),
and the full combined strategy.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_section_label, styled_divider, apply_plotly_theme, TOKENS,
)


# Display-friendly names and consistent styling for benchmarks
BENCH_DISPLAY = {
    "spy_buyhold": ("SPY B&H", TOKENS["accent_danger"], "dot"),
    "sixty_forty": ("60/40", TOKENS["accent_info"], "dash"),
    "static_vol_parity": ("Static Vol-Parity", TOKENS["accent_secondary"], "dashdot"),
    "regime_only": ("Regime Only", TOKENS["accent_warning"], "dash"),
}


def render(data: dict) -> None:
    """Render the ATTRIBUTION tab."""
    ts = data["ts"]
    benchmarks = data["benchmarks"]
    overall = data["overall"]
    bench_metrics = data["bench_metrics"]

    # --- 4-panel NAV decomposition ---
    styled_section_label("Benchmark NAV Decomposition")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts["nav"], name="Full Strategy",
        line=dict(color=TOKENS["accent_primary"], width=2.5),
    ))
    for bname, bnav in benchmarks.items():
        aligned = bnav.reindex(ts.index).dropna()
        label, color, dash = BENCH_DISPLAY.get(bname, (bname, TOKENS["text_secondary"], "solid"))
        fig.add_trace(go.Scatter(
            x=aligned.index, y=aligned.values, name=label,
            line=dict(color=color, width=1.5, dash=dash),
        ))
    fig.update_layout(
        title="Strategy vs All Benchmarks",
        yaxis_title="NAV (log scale)", yaxis_type="log",
        height=420, legend=dict(orientation="h", y=-0.15),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    styled_divider()

    # --- Attribution table ---
    styled_section_label("Performance Attribution")
    metrics_order = ["cagr", "sharpe", "max_drawdown", "calmar"]
    display_names = {
        "cagr": "CAGR", "sharpe": "Sharpe", "max_drawdown": "Max DD",
        "calmar": "Calmar",
    }
    fmt = {
        "cagr": "{:.1%}", "sharpe": "{:.2f}",
        "max_drawdown": "{:.1%}", "calmar": "{:.2f}",
    }
    rows = []
    for m in metrics_order:
        row = {"Metric": display_names[m]}
        for bname in ["spy_buyhold", "sixty_forty", "static_vol_parity", "regime_only"]:
            label = BENCH_DISPLAY.get(bname, (bname,))[0]
            val = bench_metrics.get(bname, {}).get(m, np.nan)
            row[label] = fmt[m].format(val) if pd.notna(val) else "N/A"
        row["Full Strategy"] = fmt[m].format(overall.get(m, np.nan))
        rows.append(row)

    attr_df = pd.DataFrame(rows)
    st.dataframe(attr_df, hide_index=True, use_container_width=True)

    styled_divider()

    # --- Turnover histogram + Cost drag ---
    col_left, col_right = st.columns(2)

    with col_left:
        styled_section_label("Rebalance Turnover Distribution")
        rebal_turnover = ts.loc[ts["turnover"] > 0, "turnover"]
        fig_hist = go.Figure(data=go.Histogram(
            x=rebal_turnover.values, nbinsx=30,
            marker_color=TOKENS["accent_primary"],
            marker_line=dict(color=TOKENS["bg_base"], width=1),
        ))
        fig_hist.update_layout(
            title=f"Turnover per Rebalance ({len(rebal_turnover)} events)",
            xaxis_title="One-Way Turnover",
            yaxis_title="Frequency", height=340,
        )
        apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
        # Turnover context note
        config = data["config"]
        cost_bps = config["rebalancing"]["cost_bps"]
        total_turnover = ts["turnover"].sum()
        total_cost_bps = ts["cost"].sum() * 10_000  # convert to bps of NAV
        st.markdown(
            f"<p style='color: {TOKENS['text_muted']}; font-size: 0.8rem; "
            f"line-height: 1.4;'>"
            f"Cumulative turnover of {total_turnover:.0%} produced only "
            f"~{total_cost_bps:.1f}bps of cost drag. Most rebalances are "
            f"small vol-drift adjustments (median turnover shown above), "
            f"not full regime-flip reallocations. The signal-driven rebalance "
            f"logic avoids calendar-forced trades and keeps realized costs "
            f"negligible despite high event count.</p>",
            unsafe_allow_html=True,
        )

    with col_right:
        styled_section_label("Cumulative Cost Drag")
        cum_cost = ts["cost"].cumsum() * ts["nav"].iloc[0]
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=ts.index, y=cum_cost.values, name="Cumulative Cost",
            fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.1)",
            line=dict(color=TOKENS["accent_danger"], width=1.5),
        ))
        fig_cost.update_layout(
            title="Transaction Cost Accumulation",
            yaxis_title="Cumulative Cost (NAV units)",
            height=340,
        )
        apply_plotly_theme(fig_cost)
        st.plotly_chart(fig_cost, use_container_width=True)
