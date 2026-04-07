from __future__ import annotations

"""
Benchmark strategies for volatility-regime-engine.

Computes NAV series for 4 benchmarks run in parallel with the main strategy:
  1. SPY buy-and-hold
  2. 60/40 (SPY/TLT), monthly rebalanced
  3. Static vol-parity (inverse-vol, no regime, constant 10% target)
  4. Regime-only (regime tilts, no vol scaling)

All benchmarks use simple returns for NAV compounding and respect the
same t+1 execution lag as the main strategy to ensure fair attribution.

Financial rationale: benchmarks 3 and 4 enable clean attribution:
  - Static -> vol-target: value from vol management alone
  - Vol-target -> full strategy: value from regime detection
"""

import numpy as np
import pandas as pd

from engine.vol_estimator import (
    compute_blended_vol,
    compute_ewma_covariance,
    compute_portfolio_vol,
)
from engine.position_sizer import _inverse_vol_adjust, _get_strategic_weights


def run_benchmarks(returns: pd.DataFrame, regime_series: np.ndarray,
                   regime_index: pd.DatetimeIndex,
                   config: dict) -> dict[str, pd.Series]:
    """
    Run all 4 benchmark strategies and return NAV series.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns for all tickers (used for vol/cov estimation).
    regime_series : np.ndarray
        Filtered regime labels aligned to regime_index.
    regime_index : pd.DatetimeIndex
        Date index for regime labels.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict[str, pd.Series]
        Mapping of benchmark name to NAV series.
    """
    alloc = config["data"]["allocation_tickers"]
    alloc_ret_log = returns[alloc].reindex(regime_index).dropna()
    initial_nav = config["backtest"]["initial_nav"]

    # Simple returns for NAV compounding (log returns wrong for weighted PnL)
    alloc_ret_simple = np.exp(alloc_ret_log) - 1

    results = {}
    results["spy_buyhold"] = _run_buy_hold(returns, "SPY", initial_nav)
    results["sixty_forty"] = _run_sixty_forty(returns, config, initial_nav)
    results["static_vol_parity"] = _run_static_vol_parity(
        alloc_ret_log, alloc_ret_simple, config, initial_nav
    )
    results["regime_only"] = _run_regime_only(
        alloc_ret_simple, regime_series, regime_index, config, initial_nav
    )
    return results


def _run_buy_hold(returns: pd.DataFrame, ticker: str,
                  initial_nav: float) -> pd.Series:
    """
    SPY buy-and-hold benchmark.

    Financial rationale: the simplest equity benchmark. Any strategy
    that doesn't beat SPY over a full cycle needs a strong risk-adjusted
    justification.
    """
    # Convert log returns to simple for compounding
    simple_ret = np.exp(returns[ticker]) - 1
    nav = initial_nav * (1 + simple_ret).cumprod()
    return nav


def _run_sixty_forty(returns: pd.DataFrame, config: dict,
                     initial_nav: float) -> pd.Series:
    """
    60/40 SPY/TLT, rebalanced monthly.

    Financial rationale: the classic institutional benchmark. Tests whether
    active regime detection adds value over a static diversified allocation.
    """
    bench_cfg = config["backtest"]["benchmarks"]["sixty_forty"]
    tickers = bench_cfg["tickers"]
    weights = np.array(bench_cfg["weights"])
    # Use simple returns for NAV compounding
    ret_log = returns[tickers].dropna()
    ret = np.exp(ret_log) - 1

    if len(ret) < 2:
        return pd.Series(dtype=float)

    nav_series = pd.Series(np.nan, index=ret.index)
    nav = initial_nav
    w = weights.copy()

    prev_month = None
    for i, (date, row) in enumerate(ret.iterrows()):
        curr_month = date.month
        if prev_month is not None and curr_month != prev_month:
            w = weights.copy()  # monthly rebalance
        prev_month = curr_month

        daily_ret = float(w @ row.values)
        nav = nav * (1 + daily_ret)
        nav_series.iloc[i] = nav

        # Drift weights by returns
        w = w * (1 + row.values)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

    return nav_series.dropna()


def _run_static_vol_parity(alloc_ret_log: pd.DataFrame,
                           alloc_ret_simple: pd.DataFrame,
                           config: dict,
                           initial_nav: float) -> pd.Series:
    """
    Static vol-parity: inverse-vol weights, no regime, 10% constant target.

    Uses i-1 information for i's weights to match the strategy's
    execution convention (no same-day lookahead).

    Financial rationale: isolates the value of vol management without
    regime overlay. The diff between this and the full strategy = value
    of regime detection.
    """
    vol_est = compute_blended_vol(alloc_ret_log, config)
    cov_dict = compute_ewma_covariance(alloc_ret_log, config)
    target_vol = config["backtest"]["benchmarks"]["static_vol_parity"]["target_vol"]

    if len(alloc_ret_simple) < 2:
        return pd.Series(dtype=float)

    nav_series = pd.Series(np.nan, index=alloc_ret_simple.index)
    nav = initial_nav
    n = alloc_ret_simple.shape[1]
    equal_strat = np.ones(n) / n

    prev_month = None
    w = np.ones(n) / n

    for i in range(len(alloc_ret_simple)):
        date = alloc_ret_simple.index[i]
        curr_month = date.month

        # Rebalance monthly using PREVIOUS day's vol/cov (no lookahead)
        if prev_month is not None and curr_month != prev_month and (i - 1) in cov_dict:
            vols = vol_est.iloc[i - 1].values.astype(float)
            w_ivol = _inverse_vol_adjust(equal_strat, vols)
            cov_mat = cov_dict[i - 1]
            pvol = compute_portfolio_vol(w_ivol, cov_mat)
            if pvol > 0 and not np.isnan(pvol):
                scale = target_vol / pvol
                w = w_ivol * scale
                gross = np.abs(w).sum()
                if gross > 1.0:
                    w = w * (1.0 / gross)
            else:
                w = w_ivol
        prev_month = curr_month

        # Use simple returns for PnL
        daily_ret = float(w @ alloc_ret_simple.iloc[i].values.astype(float))
        nav = nav * (1 + daily_ret)
        nav_series.iloc[i] = nav

    return nav_series.dropna()


def _run_regime_only(alloc_ret_simple: pd.DataFrame,
                     regime_series: np.ndarray,
                     regime_index: pd.DatetimeIndex, config: dict,
                     initial_nav: float) -> pd.Series:
    """
    Regime-only: strategic tilts by regime, no vol scaling.

    Regime signal at t applied to t+1 return (same lag as main strategy).

    Financial rationale: isolates the value of regime detection without
    vol targeting. The diff between this and static = value of regime overlay.
    """
    ps_cfg = config["position_sizing"]
    tickers = config["data"]["allocation_tickers"]

    if len(alloc_ret_simple) < 2:
        return pd.Series(dtype=float)

    nav_series = pd.Series(np.nan, index=alloc_ret_simple.index)
    nav = initial_nav
    n = len(tickers)
    w = np.ones(n) / n

    for i in range(len(alloc_ret_simple)):
        date = alloc_ret_simple.index[i]
        # Use PREVIOUS day's regime to set weights (t-1 signal -> t return)
        if i > 0 and date in regime_index:
            prev_date = alloc_ret_simple.index[i - 1]
            if prev_date in regime_index:
                prev_idx = regime_index.get_loc(prev_date)
                if prev_idx < len(regime_series) and regime_series[prev_idx] != "":
                    w = _get_strategic_weights(
                        regime_series[prev_idx], tickers, ps_cfg
                    )

        daily_ret = float(w @ alloc_ret_simple.iloc[i].values.astype(float))
        nav = nav * (1 + daily_ret)
        nav_series.iloc[i] = nav

    return nav_series.dropna()
