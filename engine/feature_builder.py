"""
Feature engineering for regime detection.

Constructs an 8-feature vector from prices and returns for use by both
the HMM and composite regime classifiers. Also provides z-scoring with
expanding-window constraints to prevent lookahead bias.

Financial rationale: each feature captures a distinct risk dimension —
vol level, vol term structure, momentum, skewness, trend, credit stress,
and VIX dynamics. Together they give the HMM a rich observable vector
from which to infer latent market regimes.
"""

import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame, returns: pd.DataFrame,
                   config: dict) -> pd.DataFrame:
    """
    Construct the 8-feature matrix from prices and returns.

    Returns raw (unscaled) features. Z-scoring is applied separately
    inside the HMM fitting loop to respect the expanding window constraint.

    Financial rationale: features are computed on raw data first so that
    the composite detector (which doesn't need z-scoring) can consume
    them directly.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices for all tickers including SPY, VIX, LQD, IEF.
    returns : pd.DataFrame
        Log returns for all tickers.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pd.DataFrame
        DataFrame with 8 feature columns, indexed by date.
    """
    feat_cfg = config["features"]
    data_cfg = config["data"]

    spy_ret = returns["SPY"]
    spy_px = prices["SPY"]
    vix_px = prices["^VIX"]
    lqd_ret = returns["LQD"]
    ief_ticker = data_cfg["ief_ticker"]
    ief_ret = returns[ief_ticker]

    short_w = feat_cfg["realized_vol_short_window"]
    long_w = feat_cfg["realized_vol_long_window"]
    ann = np.sqrt(config["vol_estimation"]["annualization_factor"])

    features = pd.DataFrame(index=returns.index)

    # Feature 1: 21-day realized vol (annualized)
    features["realized_vol_21d"] = spy_ret.rolling(short_w).std() * ann

    # Feature 2: 63-day realized vol (annualized)
    features["realized_vol_63d"] = spy_ret.rolling(long_w).std() * ann

    # Feature 3: 21-day cumulative log return (momentum)
    features["return_21d"] = spy_ret.rolling(feat_cfg["return_lookback"]).sum()

    # Feature 4: 63-day skewness of SPY returns
    # Financial rationale: negative skew signals tail risk build-up
    features["skew_63d"] = spy_ret.rolling(feat_cfg["skew_window"]).skew()

    # Feature 5: SPY price vs 200-day SMA (trend signal)
    sma200 = spy_px.rolling(feat_cfg["sma_long_window"]).mean()
    features["spy_vs_sma200"] = spy_px / sma200 - 1

    # Feature 6: Credit stress proxy — LQD 21d return minus IEF 21d return
    # Simplifying assumption: price-based proxy for IG credit spreads,
    # not a true OAS derived from bond cashflows
    cw = feat_cfg["credit_stress_window"]
    features["credit_stress_proxy"] = (
        lqd_ret.rolling(cw).sum() - ief_ret.rolling(cw).sum()
    )

    # Feature 7: VIX level (raw)
    features["vix_level"] = vix_px.reindex(features.index)

    # Feature 8: VIX momentum — VIX / 21-day SMA of VIX - 1
    vix_ma = vix_px.rolling(feat_cfg["vix_ma_window"]).mean()
    features["vix_momentum"] = (vix_px / vix_ma - 1).reindex(features.index)

    return features


def z_score_features(features: pd.DataFrame, fit_end_idx: int) -> pd.DataFrame:
    """
    Z-score each feature using mean/std computed only on rows 0..fit_end_idx.

    Financial rationale: prevents future distributional information from
    leaking into regime classification. On each expanding-window refit date,
    scaling parameters are computed from inception to the refit point only.

    Parameters
    ----------
    features : pd.DataFrame
        Raw (unscaled) feature matrix.
    fit_end_idx : int
        Integer index (inclusive) of the last row used for computing
        mean and std. All subsequent rows are scaled with these parameters.

    Returns
    -------
    pd.DataFrame
        Z-scored feature matrix (same shape as input).
    """
    train_slice = features.iloc[: fit_end_idx + 1]
    mu = train_slice.mean()
    sigma = train_slice.std()

    # Avoid division by zero — if a feature has zero variance, leave unscaled (NaN)
    sigma = sigma.replace(0, np.nan)

    scaled = (features - mu) / sigma
    # Do NOT zero-fill: NaN rows are dropped before HMM training in backtester
    return scaled
