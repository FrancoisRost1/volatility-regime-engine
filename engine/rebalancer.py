from __future__ import annotations

"""
Rebalancing logic for volatility-regime-engine.

Signal-driven rebalancing: triggers on regime change or portfolio vol drift.
Transaction costs are proportional to realized turnover.

Financial rationale:
  - Regime change triggers ensure the portfolio reflects the new risk
    environment promptly after a confirmed regime shift.
  - Vol drift triggers catch situations where the portfolio's risk
    has drifted materially from target even within a stable regime
    (e.g., correlation shifts, single-asset vol spikes).
  - Turnover-proportional costs are more realistic than flat fees —
    a small vol-drift rebalance costs less than a full regime flip.

Simplifying assumption: flat 5 bps per unit of turnover. No slippage model.
"""

import numpy as np


def should_rebalance(regime_today: str, regime_last_rebal: str,
                     port_vol_today: float, port_vol_last_rebal: float,
                     config: dict) -> tuple[bool, str]:
    """
    Determine whether to rebalance today.

    Returns True if EITHER condition holds:
      A: regime_today != regime_last_rebal (confirmed regime change)
      B: |port_vol_today / port_vol_last_rebal - 1| > threshold

    Financial rationale:
      Condition A — regime shift changes target vol and strategic tilt.
      Condition B — within a stable regime, vol can drift enough that
      actual portfolio risk deviates materially from target.

    Parameters
    ----------
    regime_today : str
        Today's confirmed regime label.
    regime_last_rebal : str
        Regime at the last rebalance.
    port_vol_today : float
        Current annualized portfolio vol estimate.
    port_vol_last_rebal : float
        Portfolio vol at last rebalance.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    tuple[bool, str]
        (should_rebalance, reason) where reason is 'regime_change',
        'vol_drift', or '' if no rebalance.
    """
    threshold = config["rebalancing"]["vol_deviation_threshold"]

    # Condition A: regime change
    if regime_today != regime_last_rebal:
        return True, "regime_change"

    # Condition B: vol drift
    if port_vol_last_rebal > 0 and not np.isnan(port_vol_last_rebal):
        drift = abs(port_vol_today / port_vol_last_rebal - 1)
        if drift > threshold:
            return True, "vol_drift"

    return False, ""


def compute_transaction_cost(w_new: np.ndarray, w_old: np.ndarray,
                             config: dict) -> float:
    """
    Compute transaction cost as turnover times cost rate.

    cost = (cost_bps / 10_000) * sum(|w_new - w_old|)

    Applied only on rebalance days.

    Financial rationale: cost proportional to actual turnover is more
    realistic than a flat fee. A small vol-drift rebalance costs less
    than a full regime-flip reallocation. Penalizes excessive trading.

    Parameters
    ----------
    w_new : np.ndarray
        New portfolio weights.
    w_old : np.ndarray
        Previous portfolio weights.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    float
        Transaction cost as a fraction of NAV (e.g., 0.0005 = 5 bps).
    """
    cost_bps = config["rebalancing"]["cost_bps"]
    turnover = float(np.abs(w_new - w_old).sum())
    return (cost_bps / 10_000) * turnover


def compute_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """
    Compute one-way turnover: sum(|w_new - w_old|).

    Financial rationale: turnover is the standard measure of trading
    activity in portfolio management. Used for analytics and cost calc.

    Parameters
    ----------
    w_new : np.ndarray
        New portfolio weights.
    w_old : np.ndarray
        Previous portfolio weights.

    Returns
    -------
    float
        One-way turnover (0 = no change, 2.0 = full 180-degree flip).
    """
    return float(np.abs(w_new - w_old).sum())
