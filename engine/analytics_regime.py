"""
Regime-conditional and stress period analytics for volatility-regime-engine.

Financial rationale: regime-conditional metrics prove the strategy adapts
correctly to each market state. Stress period isolation is the acid test
for drawdown control, the primary value proposition of regime-based allocation.
"""

import numpy as np
import pandas as pd


def compute_regime_conditional_metrics(
    daily_returns: pd.Series, regimes: pd.Series, config: dict
) -> pd.DataFrame:
    """
    Compute Sharpe, avg return, max drawdown, time fraction per regime.

    Financial rationale: strategy should deliver positive risk-adjusted
    returns in all regimes. Risk-off Sharpe is especially important —
    it proves the defensive posture works.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    regimes : pd.Series
        Daily regime labels aligned to daily_returns.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Regime-conditional metrics, one row per regime.
    """
    ann = config["vol_estimation"]["annualization_factor"]
    results = []

    for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
        mask = regimes == regime
        ret_r = daily_returns[mask]
        n = len(ret_r)

        if n < 2:
            results.append({
                "regime": regime, "sharpe": np.nan, "avg_daily_return": np.nan,
                "max_drawdown": np.nan, "pct_time": 0.0, "n_days": n,
            })
            continue

        sharpe = (
            float(ret_r.mean() / ret_r.std() * np.sqrt(ann))
            if ret_r.std() > 0 else np.nan
        )
        nav_r = (1 + ret_r).cumprod()
        peak = nav_r.cummax()
        dd = (nav_r / peak - 1).min()

        results.append({
            "regime": regime,
            "sharpe": sharpe,
            "avg_daily_return": float(ret_r.mean()),
            "max_drawdown": float(dd),
            "pct_time": n / len(daily_returns),
            "n_days": n,
        })

    return pd.DataFrame(results).set_index("regime")


def compute_regime_duration_stats(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute average regime duration per regime.

    Financial rationale: regime persistence informs rebalancing
    frequency expectations and trading cost budgets.

    Parameters
    ----------
    regimes : pd.Series
        Daily regime label series.

    Returns
    -------
    pd.DataFrame
        Average duration and episode count per regime.
    """
    if len(regimes) == 0:
        return pd.DataFrame({
            "avg_duration_days": [0, 0, 0], "n_episodes": [0, 0, 0],
        }, index=["RISK_ON", "NEUTRAL", "RISK_OFF"])

    durations = {"RISK_ON": [], "NEUTRAL": [], "RISK_OFF": []}
    current = regimes.iloc[0]
    count = 1

    for i in range(1, len(regimes)):
        if regimes.iloc[i] == current:
            count += 1
        else:
            durations[current].append(count)
            current = regimes.iloc[i]
            count = 1
    durations[current].append(count)

    stats = {}
    for regime, durs in durations.items():
        stats[regime] = {
            "avg_duration_days": np.mean(durs) if durs else 0,
            "n_episodes": len(durs),
        }
    return pd.DataFrame(stats).T


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute regime transition probability matrix P(regime_{t+1} | regime_t).

    Financial rationale: reveals regime dynamics and persistence.
    High diagonal = sticky regimes; off-diagonal = typical transitions.

    Parameters
    ----------
    regimes : pd.Series
        Daily regime label series.

    Returns
    -------
    pd.DataFrame
        3x3 transition matrix.
    """
    labels = ["RISK_ON", "NEUTRAL", "RISK_OFF"]
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)

    if len(regimes) < 2:
        # Not enough data for transitions, return uniform
        return pd.DataFrame(
            1.0 / len(labels), index=labels, columns=labels,
        )

    for i in range(len(regimes) - 1):
        fr = regimes.iloc[i]
        to = regimes.iloc[i + 1]
        if fr in labels and to in labels:
            matrix.loc[fr, to] += 1

    row_sums = matrix.sum(axis=1)
    for regime in labels:
        if row_sums[regime] == 0:
            # Unobserved regime: uniform transition probability (1/3 each)
            matrix.loc[regime] = 1.0 / len(labels)
        else:
            matrix.loc[regime] = matrix.loc[regime] / row_sums[regime]
    return matrix


def compute_stress_period_metrics(
    nav: pd.Series, daily_returns: pd.Series,
    spy_nav: pd.Series, spy_returns: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """
    Isolate performance during stress periods (GFC, COVID, 2022 rate shock).

    Financial rationale: drawdown control during crises is the primary
    value proposition. These periods are the acid test.

    Parameters
    ----------
    nav : pd.Series
        Strategy NAV.
    daily_returns : pd.Series
        Strategy daily returns.
    spy_nav : pd.Series
        SPY NAV for comparison.
    spy_returns : pd.Series
        SPY daily returns.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Stress period metrics for strategy vs SPY.
    """
    stress_cfg = config["analytics"]["stress_periods"]
    ann = config["vol_estimation"]["annualization_factor"]
    results = []

    for name, period in stress_cfg.items():
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])

        for label, nav_s, ret_s in [
            ("strategy", nav, daily_returns),
            ("SPY", spy_nav, spy_returns),
        ]:
            mask = (ret_s.index >= start) & (ret_s.index <= end)
            ret_p = ret_s[mask]
            nav_p = nav_s.reindex(ret_p.index).dropna()

            if len(nav_p) < 2:
                results.append({
                    "period": name, "portfolio": label,
                    "return": np.nan, "max_drawdown": np.nan,
                    "sharpe": np.nan,
                })
                continue

            total_ret = nav_p.iloc[-1] / nav_p.iloc[0] - 1
            peak = nav_p.cummax()
            max_dd = float((nav_p / peak - 1).min())
            sharpe = (
                float(ret_p.mean() / ret_p.std() * np.sqrt(ann))
                if ret_p.std() > 0 else np.nan
            )

            results.append({
                "period": name, "portfolio": label,
                "return": float(total_ret), "max_drawdown": max_dd,
                "sharpe": sharpe,
            })

    return pd.DataFrame(results)
