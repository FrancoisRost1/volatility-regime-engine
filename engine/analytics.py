from __future__ import annotations

"""
Core analytics for volatility-regime-engine.

Overall performance metrics and benchmark attribution.

Financial rationale: institutional allocators evaluate strategies on
risk-adjusted returns, drawdown behavior, and return decomposition.
"""

import numpy as np
import pandas as pd


def compute_overall_metrics(nav: pd.Series, daily_returns: pd.Series,
                            spy_returns: pd.Series,
                            config: dict) -> dict:
    """
    Compute full overall performance metrics.

    Includes CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate,
    and correlation to SPY.

    Financial rationale: these are the standard metrics an allocator
    reviews before sizing a strategy. Risk-free rate = 0 (simplifying
    assumption: cash earns nothing in the model).

    Parameters
    ----------
    nav : pd.Series
        Daily NAV series.
    daily_returns : pd.Series
        Daily portfolio returns.
    spy_returns : pd.Series
        Daily SPY log returns (for correlation).
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Dictionary of metric name -> value.
    """
    rf = config["backtest"]["risk_free_rate"]
    ann = config["vol_estimation"]["annualization_factor"]
    n_days = len(nav)

    if n_days < 2:
        return {k: np.nan for k in [
            "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
            "max_dd_duration_days", "calmar", "win_rate_monthly",
            "avg_monthly_gain", "avg_monthly_loss", "correlation_spy",
            "annualized_vol",
        ]}

    n_years = n_days / ann

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1

    excess = daily_returns - rf / ann
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(ann))
        if excess.std() > 0 else np.nan
    )

    # Sortino: annualized mean excess return / annualized downside deviation
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(ann) if len(downside) > 1 else np.nan
    ann_mean_excess = float(excess.mean() * ann)
    sortino = float(ann_mean_excess / downside_std) if (
        downside_std is not None and downside_std > 0
        and not np.isnan(downside_std)
    ) else np.nan

    dd_series, max_dd, max_dd_duration = compute_drawdown(nav)
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else np.nan

    # Monthly stats — compound daily returns, not sum
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    win_rate = (
        float((monthly > 0).sum() / len(monthly))
        if len(monthly) > 0 else np.nan
    )
    avg_pos = float(monthly[monthly > 0].mean()) if (monthly > 0).any() else 0.0
    avg_neg = float(monthly[monthly < 0].mean()) if (monthly < 0).any() else 0.0

    # Correlation to SPY
    aligned = pd.concat([daily_returns, spy_returns], axis=1).dropna()
    corr_spy = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_dd_duration_days": max_dd_duration,
        "calmar": calmar,
        "win_rate_monthly": win_rate,
        "avg_monthly_gain": avg_pos,
        "avg_monthly_loss": avg_neg,
        "correlation_spy": corr_spy,
        "annualized_vol": float(daily_returns.std() * np.sqrt(ann)),
    }


def compute_benchmark_attribution(
    strategy_metrics: dict, benchmark_metrics: dict[str, dict]
) -> pd.DataFrame:
    """
    Build the benchmark attribution table.

    Financial rationale: decomposes strategy value into components —
    diversification, vol management, and regime detection.

    Parameters
    ----------
    strategy_metrics : dict
        Overall metrics for the full strategy.
    benchmark_metrics : dict[str, dict]
        Mapping of benchmark name to its overall metrics dict.

    Returns
    -------
    pd.DataFrame
        Attribution table with rows = metrics, columns = portfolios.
    """
    metrics = ["cagr", "sharpe", "max_drawdown", "calmar"]
    data = {}
    for bname, bmetrics in benchmark_metrics.items():
        data[bname] = {m: bmetrics.get(m, np.nan) for m in metrics}
    data["full_strategy"] = {m: strategy_metrics.get(m, np.nan) for m in metrics}
    return pd.DataFrame(data, index=metrics)


def compute_drawdown(nav: pd.Series) -> tuple[pd.Series, float, int]:
    """
    Compute drawdown series, max drawdown, and max drawdown duration.

    Parameters
    ----------
    nav : pd.Series
        Daily NAV series.

    Returns
    -------
    tuple
        (drawdown_series, max_drawdown_float, max_duration_days)
    """
    peak = nav.cummax()
    dd = nav / peak - 1
    max_dd = float(dd.min())

    underwater = dd < 0
    if not underwater.any():
        return dd, 0.0, 0

    groups = (~underwater).cumsum()
    durations = underwater.groupby(groups).sum()
    max_duration = int(durations.max()) if len(durations) > 0 else 0

    return dd, max_dd, max_duration
