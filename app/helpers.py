"""
Shared data loading and chart helpers for the Streamlit dashboard.

Loads pre-computed tearsheet and benchmarks so the dashboard starts
instantly without re-running the HMM backtest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so engine/ imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import datetime

from utils.config_loader import load_config
from engine.data_loader import load_prices as _load_prices_yf, compute_log_returns


def _load_prices_or_snapshot(config):
    """
    Prefer the shipped prices snapshot at outputs/prices.csv.
    Falls back to yfinance when the snapshot is absent (local dev only).
    Streamlit Cloud blocks yfinance, so the snapshot must be committed.
    """
    snap = _PROJECT_ROOT / "outputs" / "prices.csv"
    if snap.exists():
        return pd.read_csv(snap, parse_dates=[0], index_col=0)
    return _load_prices_yf(config)
from engine.benchmarks import run_benchmarks, _run_buy_hold
from engine.analytics import compute_overall_metrics, compute_drawdown
from engine.analytics_regime import (
    compute_regime_conditional_metrics,
    compute_regime_duration_stats,
    compute_transition_matrix,
    compute_stress_period_metrics,
)

# Regime color mapping, consistent across all charts
REGIME_COLORS = {
    "RISK_ON": "#10B981",   # Emerald / success
    "NEUTRAL": "#F59E0B",   # Amber / warning
    "RISK_OFF": "#EF4444",  # Red / danger
}

ASSET_COLORS = {
    "SPY": "#6366F1",   # Indigo
    "TLT": "#3B82F6",   # Blue
    "GLD": "#F59E0B",   # Amber
    "PDBC": "#8B5CF6",  # Purple
}


@st.cache_data(ttl=3600)
def load_data():
    """
    Load tearsheet, prices, benchmarks, and compute all analytics.

    Returns a dict with all data needed by every tab. Cached for 1 hour
    so the dashboard reloads instantly on tab switches.
    """
    config = load_config(str(_PROJECT_ROOT / "config.yaml"))

    # Tearsheet from pipeline run
    ts = pd.read_csv(
        _PROJECT_ROOT / "outputs" / "tearsheet.csv",
        parse_dates=["date"], index_col="date",
    )

    # Price data for benchmarks (prefers shipped snapshot, falls back to yfinance locally)
    prices = _load_prices_or_snapshot(config)
    returns = compute_log_returns(prices)

    # Benchmarks
    benchmarks = run_benchmarks(
        returns, ts["regime_hmm"].values, ts.index, config,
    )
    spy_nav = _run_buy_hold(returns, "SPY", config["backtest"]["initial_nav"])

    # Analytics
    spy_ret = returns["SPY"].reindex(ts.index).fillna(0)
    overall = compute_overall_metrics(ts["nav"], ts["daily_return"], spy_ret, config)
    regime_cond = compute_regime_conditional_metrics(
        ts["daily_return"], ts["regime_hmm"], config,
    )
    duration_stats = compute_regime_duration_stats(ts["regime_hmm"])
    transition = compute_transition_matrix(ts["regime_hmm"])
    stress = compute_stress_period_metrics(
        ts["nav"], ts["daily_return"],
        spy_nav.reindex(ts.index), spy_ret, config,
    )
    dd_series, _, _ = compute_drawdown(ts["nav"])

    # Benchmark metrics for attribution table
    bench_metrics = {}
    for bname, bnav in benchmarks.items():
        ba = bnav.reindex(ts.index).dropna()
        if len(ba) < 2:
            continue
        bret = ba.pct_change().dropna()
        bench_metrics[bname] = compute_overall_metrics(ba, bret, spy_ret, config)

    # Tearsheet run date, detect from file modification time
    tearsheet_file = _PROJECT_ROOT / "outputs" / "tearsheet.csv"
    tearsheet_mtime = datetime.datetime.fromtimestamp(
        tearsheet_file.stat().st_mtime
    )

    return {
        "config": config,
        "ts": ts,
        "benchmarks": benchmarks,
        "overall": overall,
        "regime_cond": regime_cond,
        "duration_stats": duration_stats,
        "transition": transition,
        "stress": stress,
        "dd_series": dd_series,
        "bench_metrics": bench_metrics,
        "tearsheet_run_date": tearsheet_mtime,
    }
