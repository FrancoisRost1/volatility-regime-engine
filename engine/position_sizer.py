"""
Position sizing for volatility-regime-engine.

Implements the 5-step sizing pipeline:
  1. Regime-conditional strategic tilt
  2. Inverse-vol adjustment within regime
  3. Volatility-targeting scalar
  4. Leverage cap enforcement
  5. Cash residual (implicit)

Financial rationale: the regime tilt shifts the portfolio composition
(risk-off = bonds/gold, not just smaller equity). Inverse-vol equalization
prevents high-vol assets from dominating risk. Vol-targeting scales total
exposure to hit a regime-dependent risk budget. Leverage caps prevent
excessive leverage in low-vol environments.

Simplifying assumption: inverse-vol weighting, not full risk parity.
"""

import numpy as np

from engine.vol_estimator import compute_portfolio_vol


def compute_weights(regime: str, vol_estimates: np.ndarray,
                    cov_matrix: np.ndarray,
                    config: dict) -> np.ndarray:
    """
    Full position sizing pipeline: strategic tilt -> inverse-vol ->
    vol-target scalar -> leverage cap.

    Parameters
    ----------
    regime : str
        Current confirmed regime: RISK_ON, NEUTRAL, or RISK_OFF.
    vol_estimates : np.ndarray
        Blended annualized vol per asset (n_assets,).
    cov_matrix : np.ndarray
        Annualized covariance matrix (n_assets x n_assets).
    config : dict
        Full configuration dictionary.

    Returns
    -------
    np.ndarray
        Final portfolio weights (n_assets,). Sum may be < 1.0 (cash).
    """
    ps_cfg = config["position_sizing"]
    alloc_tickers = config["data"]["allocation_tickers"]

    # Step 1: Strategic tilt for current regime
    strategic = _get_strategic_weights(regime, alloc_tickers, ps_cfg)

    # Step 2: Inverse-vol adjustment
    w_ivol = _inverse_vol_adjust(strategic, vol_estimates)

    # Step 3: Vol-targeting scalar
    target_vol = ps_cfg["target_vol"][regime]
    port_vol = compute_portfolio_vol(w_ivol, cov_matrix)

    if np.isnan(port_vol) or port_vol <= 0:
        # Edge case: cannot compute portfolio vol, return strategic weights
        return strategic

    scale = target_vol / port_vol
    w_scaled = w_ivol * scale

    # Step 4: Leverage cap
    w_final = _apply_leverage_cap(w_scaled, regime, ps_cfg)

    return w_final


def _get_strategic_weights(regime: str, tickers: list,
                           ps_cfg: dict) -> np.ndarray:
    """
    Retrieve strategic weight vector for the given regime.

    Financial rationale: regime-conditional tilts ensure risk-off is a
    genuinely defensive posture (overweight bonds/gold), not just a
    scaled-down version of the risk-on portfolio.

    Parameters
    ----------
    regime : str
        Current regime label.
    tickers : list
        Ordered list of allocation tickers.
    ps_cfg : dict
        Position sizing config section.

    Returns
    -------
    np.ndarray
        Strategic weights summing to 1.0.
    """
    regime_weights = ps_cfg["strategic_weights"][regime]
    return np.array([regime_weights[t] for t in tickers])


def _inverse_vol_adjust(strategic: np.ndarray,
                        vol_estimates: np.ndarray) -> np.ndarray:
    """
    Adjust strategic weights by inverse volatility and renormalize.

    w_ivol(i) = strategic(i) / sigma(i)
    w_norm(i) = w_ivol(i) / sum(w_ivol)

    Financial rationale: equalizes per-asset risk contribution without
    requiring covariance inversion. An improvement on equal weighting
    with minimal complexity.

    Edge case: if sigma(i) is NaN or 0, set that asset's weight to 0
    and redistribute proportionally.

    Parameters
    ----------
    strategic : np.ndarray
        Base strategic weights.
    vol_estimates : np.ndarray
        Per-asset annualized vol.

    Returns
    -------
    np.ndarray
        Inverse-vol adjusted weights summing to 1.0.
    """
    # Handle zero or NaN vol, set weight to 0 for those assets
    valid = (vol_estimates > 0) & ~np.isnan(vol_estimates)
    w_ivol = np.zeros_like(strategic, dtype=float)
    w_ivol[valid] = strategic[valid] / vol_estimates[valid]

    total = w_ivol.sum()
    if total <= 0:
        # All vols invalid, fall back to equal weight on valid assets
        return strategic / strategic.sum()

    return w_ivol / total


def _apply_leverage_cap(weights: np.ndarray, regime: str,
                        ps_cfg: dict) -> np.ndarray:
    """
    Enforce regime-dependent leverage cap on gross exposure.

    If sum(|w|) > cap, scale all weights proportionally.
    If gross < 1.0, remainder sits in cash (return = 0).

    Financial rationale: prevents excessive leverage in low-vol
    environments where vol-targeting would otherwise push exposure
    above prudent limits.

    Parameters
    ----------
    weights : np.ndarray
        Pre-cap portfolio weights.
    regime : str
        Current regime label.
    ps_cfg : dict
        Position sizing config section.

    Returns
    -------
    np.ndarray
        Capped portfolio weights.
    """
    cap = ps_cfg["leverage_caps"][regime]
    gross = np.abs(weights).sum()

    if gross > cap:
        weights = weights * (cap / gross)

    return weights
