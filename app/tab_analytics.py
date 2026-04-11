"""
ANALYTICS tab — Tearsheet, drawdown chart, monthly heatmap, stress tests.

Full institutional-quality analytics: drawdown tracking, monthly return
calendar, regime-conditional Sharpe, and stress period comparisons.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_section_label, styled_divider, styled_kpi,
    apply_plotly_theme, TOKENS,
)
from app.helpers import REGIME_COLORS


def render(data: dict) -> None:
    """Render the ANALYTICS tab."""
    ts = data["ts"]
    overall = data["overall"]
    dd_series = data["dd_series"]
    regime_cond = data["regime_cond"]
    stress = data["stress"]

    # --- Drawdown / Underwater chart ---
    styled_section_label("Underwater Chart")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series.values, name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(color=TOKENS["accent_danger"], width=1.5),
    ))
    fig_dd.update_layout(
        title="Drawdown from Peak",
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
        height=320,
    )
    apply_plotly_theme(fig_dd)
    st.plotly_chart(fig_dd, use_container_width=True)

    styled_divider()

    # --- Monthly return heatmap ---
    styled_section_label("Monthly Returns")
    monthly = (1 + ts["daily_return"]).resample("ME").prod() - 1
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0, TOKENS["accent_danger"]],
            [0.5, TOKENS["bg_elevated"]],
            [1, TOKENS["accent_success"]],
        ],
        text=np.where(
            np.isnan(pivot.values), "",
            np.char.add(
                (pivot.values * 100).round(1).astype(str), "%"
            ),
        ),
        texttemplate="%{text}",
        textfont=dict(size=11, color=TOKENS["text_primary"]),
        showscale=False,
        zmid=0,
    ))
    fig_hm.update_layout(
        title="Monthly Return Heatmap",
        yaxis_autorange="reversed", height=340,
    )
    apply_plotly_theme(fig_hm)
    st.plotly_chart(fig_hm, use_container_width=True)

    styled_divider()

    # --- Regime-conditional Sharpe bars + Stress test table ---
    col_left, col_right = st.columns(2)

    with col_left:
        styled_section_label("Sharpe by Regime")
        regimes = regime_cond.index.tolist()
        sharpes = regime_cond["sharpe"].tolist()
        colors = [REGIME_COLORS.get(r, "#94A3B8") for r in regimes]
        fig_bar = go.Figure(data=go.Bar(
            x=regimes, y=sharpes, marker_color=colors,
            text=[f"{s:.2f}" for s in sharpes], textposition="auto",
            textfont=dict(color=TOKENS["text_primary"]),
        ))
        fig_bar.update_layout(
            title="Regime-Conditional Sharpe Ratio",
            yaxis_title="Sharpe", height=340,
        )
        apply_plotly_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        styled_section_label("Stress Period Performance")
        if not stress.empty:
            display = stress.copy()
            display["return"] = display["return"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""
            )
            display["max_drawdown"] = display["max_drawdown"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""
            )
            display["sharpe"] = display["sharpe"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
            st.dataframe(display, hide_index=True, use_container_width=True)

            # Check if GFC period falls outside data range
            config = data.get("config", {})
            gfc_end = config.get("analytics", {}).get(
                "stress_periods", {},
            ).get("GFC", {}).get("end", "")
            data_start = ts.index[0]
            if gfc_end and pd.Timestamp(gfc_end) < data_start:
                st.markdown(
                    f"<p style='color: {TOKENS['text_muted']}; font-size: 0.8rem;'>"
                    f"<b>Note:</b> GFC period (Sep 2008 - Mar 2009) shows "
                    f"\"\" because PDBC data begins {data_start.strftime('%b %Y')}, "
                    f"after the HMM warmup window. The backtest cannot produce "
                    f"live signals for dates before its first trade. "
                    f"COVID and 2022 Rate Shock are fully covered.</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No stress period data available for this backtest window.")

    styled_divider()

    # --- Full tearsheet metrics ---
    styled_section_label("Full Performance Metrics")
    metrics_display = {
        "Total Return": f"{overall['total_return']:.1%}",
        "CAGR": f"{overall['cagr']:.1%}",
        "Annualized Vol": f"{overall['annualized_vol']:.1%}",
        "Sharpe Ratio": f"{overall['sharpe']:.2f}",
        "Sortino Ratio": f"{overall['sortino']:.2f}",
        "Max Drawdown": f"{overall['max_drawdown']:.1%}",
        "Max DD Duration": f"{overall['max_dd_duration_days']} days",
        "Calmar Ratio": f"{overall['calmar']:.2f}",
        "Monthly Win Rate": f"{overall['win_rate_monthly']:.0%}",
        "Avg Monthly Gain": f"{overall['avg_monthly_gain']:.2%}",
        "Avg Monthly Loss": f"{overall['avg_monthly_loss']:.2%}",
        "SPY Correlation": f"{overall['correlation_spy']:.2f}",
    }
    df_metrics = pd.DataFrame(
        list(metrics_display.items()), columns=["Metric", "Value"],
    )
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
