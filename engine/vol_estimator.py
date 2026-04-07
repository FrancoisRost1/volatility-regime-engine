from __future__ import annotations

"""
Volatility estimation for volatility-regime-engine.

Computes per-asset volatility (EWMA + realized blend) and an EWMA
covariance matrix for portfolio-level vol calculation.

Financial rationale:
  - EWMA vol (RiskMetrics, lambda=0.94) reacts faster to recent shocks
    than a flat rolling window, making it more suitable for risk management.
  - Blending with realized vol smooths out EWMA overreaction.
  - The EWMA covariance matrix captures time-varying correlations,
    critical for accurate portfolio vol in shifting regimes.

Simplifying assumption: inverse-vol weighting, not full risk parity.
Covariance-based risk parity deferred to Project 8.
"""

import numpy as np
import pandas as pd


def compute_ewma_vol(returns: pd.Series, lam: float,
                     annualization: int) -> pd.Series:
    """
    Compute EWMA volatility using the RiskMetrics recursive formula.

    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r_t^2
    Annualized: sigma_t * sqrt(annualization_factor)

    Financial rationale: exponential weighting gives more influence to
    recent returns, making the estimate responsive to volatility clusters
    (a well-documented stylized fact of financial returns).

    Parameters
    ----------
    returns : pd.Series
        Daily log returns for a single asset.
    lam : float
        Decay factor (0.94 = RiskMetrics standard, ~11-day half-life).
    annualization : int
        Trading days per year (252).

    Returns
    -------
    pd.Series
        Annualized EWMA volatility, same index as returns.
    """
    if len(returns) == 0:
        return pd.Series(dtype=float)

    var_ewma = pd.Series(np.nan, index=returns.index, dtype=float)
    # Seed with first squared return
    var_ewma.iloc[0] = returns.iloc[0] ** 2

    for t in range(1, len(returns)):
        var_ewma.iloc[t] = (
            lam * var_ewma.iloc[t - 1] + (1 - lam) * returns.iloc[t] ** 2
        )

    return np.sqrt(var_ewma) * np.sqrt(annualization)


def compute_realized_vol(returns: pd.Series, window: int,
                         annualization: int) -> pd.Series:
    """
    Compute rolling realized volatility (annualized).

    Financial rationale: simple rolling std is the most transparent
    vol estimator and provides a stable anchor when blended with EWMA.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns for a single asset.
    window : int
        Rolling window in trading days.
    annualization : int
        Trading days per year.

    Returns
    -------
    pd.Series
        Annualized rolling realized vol.
    """
    return returns.rolling(window).std() * np.sqrt(annualization)


def compute_blended_vol(returns: pd.DataFrame,
                        config: dict) -> pd.DataFrame:
    """
    Compute blended (EWMA + realized) volatility for each asset.

    sigma_est = blend * realized + (1 - blend) * ewma

    Financial rationale: the blend stabilizes estimates — EWMA reacts
    quickly to shocks while realized vol prevents overreaction.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns for allocation assets only.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Annualized blended vol per asset, same index/columns as returns.
    """
    vol_cfg = config["vol_estimation"]
    lam = vol_cfg["ewma_lambda"]
    window = vol_cfg["realized_window"]
    ann = vol_cfg["annualization_factor"]
    blend = vol_cfg["ewma_realized_blend"]
    floor = vol_cfg["min_vol_floor"]

    result = pd.DataFrame(index=returns.index, columns=returns.columns,
                          dtype=float)

    for col in returns.columns:
        ewma = compute_ewma_vol(returns[col], lam, ann)
        realized = compute_realized_vol(returns[col], window, ann)
        blended = blend * realized + (1 - blend) * ewma
        # Floor to avoid division by zero in position sizing
        result[col] = blended.clip(lower=floor)

    return result


def compute_ewma_covariance(returns: pd.DataFrame,
                            config: dict) -> dict[int, np.ndarray]:
    """
    Compute daily EWMA covariance matrices (annualized).

    Sigma_ij(t) = lambda * Sigma_ij(t-1) + (1 - lambda) * r_i(t) * r_j(t)
    Annualized by multiplying by 252.

    Financial rationale: captures time-varying cross-asset correlations.
    During stress, correlations spike (diversification breaks down);
    EWMA reflects this faster than a flat rolling window.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns for allocation assets.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from integer row index to annualized covariance matrix.
    """
    vol_cfg = config["vol_estimation"]
    lam = vol_cfg["ewma_lambda"]
    ann = vol_cfg["annualization_factor"]
    reg = vol_cfg["cov_regularization"]

    n_assets = returns.shape[1]
    ret_vals = returns.values

    if len(ret_vals) == 0:
        return {}

    # Seed with outer product of first return vector
    cov_prev = np.outer(ret_vals[0], ret_vals[0])
    cov_dict = {0: cov_prev * ann}

    for t in range(1, len(returns)):
        r_t = ret_vals[t]
        cov_t = lam * cov_prev + (1 - lam) * np.outer(r_t, r_t)
        # Regularize if needed for numerical stability
        cov_ann = cov_t * ann + reg * np.eye(n_assets)
        cov_dict[t] = cov_ann
        cov_prev = cov_t

    return cov_dict


def compute_portfolio_vol(weights: np.ndarray,
                          cov_matrix: np.ndarray) -> float:
    """
    Compute annualized portfolio volatility: sigma_port = sqrt(w' Sigma w).

    Used for: (1) rebalance trigger check, (2) vol-targeting scalar,
    (3) tearsheet realized vol tracking.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights vector (n_assets,).
    cov_matrix : np.ndarray
        Annualized covariance matrix (n_assets x n_assets).

    Returns
    -------
    float
        Annualized portfolio volatility.
    """
    port_var = weights @ cov_matrix @ weights
    # Guard against numerical issues
    if port_var <= 0:
        return np.nan
    return float(np.sqrt(port_var))
