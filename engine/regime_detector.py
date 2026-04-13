from __future__ import annotations

"""
Regime detection for volatility-regime-engine.

Two parallel classifiers:
  1. HMMRegimeDetector, Gaussian HMM with 3 states, filtered probabilities only
  2. CompositeRegimeDetector, transparent rule-based classifier (4 signals)

Both output one of: RISK_ON, NEUTRAL, RISK_OFF.

A persistence filter prevents acting on transient regime flips.

Financial rationale: the HMM captures non-linear regime dynamics that
simple rules miss, while the composite model provides an interpretable
benchmark. Running both enables model comparison and agreement analysis.
"""

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    Gaussian HMM with 3 hidden states trained on the 8-feature vector.

    Uses filtered (forward-algorithm) probabilities only, no smoothing.
    This guarantees regime classification at date t uses only data up to t.

    States are labeled post-hoc by sorting emission means of realized_vol_21d:
        lowest vol mean  -> RISK_ON
        middle vol mean  -> NEUTRAL
        highest vol mean -> RISK_OFF

    Simplifying assumption: state label ordering by vol is assumed stable.
    In pathological conditions (e.g., pre-crash calm), a low-vol state
    could theoretically coincide with stress. Documented limitation.
    """

    def __init__(self, config: dict):
        """
        Initialize HMM detector with parameters from config.

        Parameters
        ----------
        config : dict
            Full configuration dictionary.
        """
        hmm_cfg = config["hmm"]
        self.n_states = hmm_cfg["n_states"]
        self.covariance_type = hmm_cfg["covariance_type"]
        self.n_iter = hmm_cfg["n_iter"]
        self.random_state = hmm_cfg["random_state"]
        self.label_feature = hmm_cfg["label_by_feature"]
        self.state_labels = hmm_cfg["state_labels"]
        self.model: GaussianHMM | None = None
        self._label_map: dict | None = None

    def fit(self, features_scaled: np.ndarray, feature_names: list[str]) -> None:
        """
        Fit GaussianHMM on the scaled feature matrix.

        After fitting, states are labeled by sorting the emission mean
        of realized_vol_21d in ascending order (low vol = RISK_ON).

        Parameters
        ----------
        features_scaled : np.ndarray
            Z-scored feature matrix (n_obs x n_features).
        feature_names : list[str]
            Column names matching features_scaled columns.
        """
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(features_scaled)
        self._build_label_map(feature_names)

    def _build_label_map(self, feature_names: list[str]) -> None:
        """
        Map raw HMM state indices to regime labels by sorting emission
        means of the labeling feature (realized_vol_21d).

        Financial rationale: vol-based labeling is deterministic and
        robust across refits because it ties state identity to an
        observable quantity, not an arbitrary index.
        """
        feat_idx = feature_names.index(self.label_feature)
        vol_means = self.model.means_[:, feat_idx]
        sorted_states = np.argsort(vol_means)
        self._label_map = {
            int(sorted_states[i]): self.state_labels[i]
            for i in range(self.n_states)
        }

    def predict_filtered(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        Return regime labels using ONLY filtered probabilities.

        Implementation: for each time step t, run predict_proba on
        features_scaled[0:t+1] and take the argmax of the last row.
        This simulates live forward-only inference, no future data.

        CAUTION: hmmlearn's .predict() uses Viterbi (smoothed). We must
        NOT use it. Instead we use the forward algorithm via score_samples
        and manual posterior extraction, applied incrementally.

        Parameters
        ----------
        features_scaled : np.ndarray
            Z-scored feature matrix (n_obs x n_features).

        Returns
        -------
        np.ndarray
            Array of regime label strings, length n_obs.
        """
        n_obs = features_scaled.shape[0]
        labels = np.empty(n_obs, dtype=object)

        for t in range(n_obs):
            # Forward algorithm on sequence [0..t], filtered posterior at t
            proba = self.model.predict_proba(features_scaled[: t + 1])
            state_idx = int(np.argmax(proba[-1]))
            labels[t] = self._label_map[state_idx]

        return labels

    def get_state_probabilities(self, features_scaled: np.ndarray) -> pd.DataFrame:
        """
        Return filtered state probabilities for each time step.

        Financial rationale: probability series enable dashboard
        visualization of regime uncertainty and transition dynamics.

        Parameters
        ----------
        features_scaled : np.ndarray
            Z-scored feature matrix.

        Returns
        -------
        pd.DataFrame
            Columns: [p_risk_on, p_neutral, p_risk_off], one row per obs.
        """
        n_obs = features_scaled.shape[0]
        prob_cols = ["p_risk_on", "p_neutral", "p_risk_off"]
        probs = np.zeros((n_obs, self.n_states))

        for t in range(n_obs):
            row_proba = self.model.predict_proba(features_scaled[: t + 1])[-1]
            # Reorder probabilities to match label ordering
            for raw_idx, label in self._label_map.items():
                col_idx = self.state_labels.index(label)
                probs[t, col_idx] = row_proba[raw_idx]

        return pd.DataFrame(probs, columns=prob_cols)


class CompositeRegimeDetector:
    """
    Transparent rule-based regime classifier using 4 interpretable signals.

    Each signal votes +1 (supportive) or -1 (stress). Total score -> regime.
      Signal 1, Trend:         SPY > 200-day SMA -> +1, else -1
      Signal 2, Vol stress:    21d vol < 63d vol -> +1, else -1
      Signal 3, Drawdown:      SPY drawdown < threshold -> +1, else -1
      Signal 4, Credit stress: LQD vs IEF spread > threshold -> +1, else -1

    Financial rationale: each signal proxies a distinct risk dimension
    (trend, vol regime, correction depth, credit conditions). Thresholds
    are transparent and defensible.
    """

    def __init__(self, config: dict):
        """
        Initialize composite detector with thresholds from config.

        Parameters
        ----------
        config : dict
            Full configuration dictionary.
        """
        comp = config["composite"]
        self.dd_window = comp["drawdown_window"]
        self.dd_threshold = comp["drawdown_threshold"]
        self.credit_threshold = comp["credit_stress_threshold"]
        self.risk_on_threshold = comp["risk_on_threshold"]
        self.risk_off_threshold = comp["risk_off_threshold"]

    def predict(self, features: pd.DataFrame,
                prices: pd.DataFrame) -> np.ndarray:
        """
        Classify each day into RISK_ON / NEUTRAL / RISK_OFF.

        Parameters
        ----------
        features : pd.DataFrame
            Raw (unscaled) feature matrix from build_features.
        prices : pd.DataFrame
            Full price DataFrame (needs SPY for drawdown calc).

        Returns
        -------
        np.ndarray
            Array of regime label strings aligned to features.index.
        """
        idx = features.index
        scores = np.zeros(len(idx), dtype=int)

        # Signal 1: Trend, SPY vs 200-day SMA
        trend = features["spy_vs_sma200"].reindex(idx)
        scores += np.where(trend > 0, 1, -1)

        # Signal 2: Vol stress, short vol < long vol means calming
        # Financial rationale: vol mean-reversion; short < long = de-stressing
        vol_short = features["realized_vol_21d"].reindex(idx)
        vol_long = features["realized_vol_63d"].reindex(idx)
        scores += np.where(vol_short < vol_long, 1, -1)

        # Signal 3: Drawdown from 252-day high
        spy_px = prices["SPY"].reindex(idx)
        rolling_high = spy_px.rolling(self.dd_window, min_periods=1).max()
        drawdown = (spy_px / rolling_high) - 1
        scores += np.where(drawdown.abs() < self.dd_threshold, 1, -1)

        # Signal 4: Credit stress, LQD vs IEF spread
        credit = features["credit_stress_proxy"].reindex(idx)
        scores += np.where(credit > self.credit_threshold, 1, -1)

        # Map scores to regimes
        labels = np.where(
            scores >= self.risk_on_threshold, "RISK_ON",
            np.where(scores <= self.risk_off_threshold, "RISK_OFF", "NEUTRAL"),
        )
        return labels


def apply_persistence_filter(regimes: np.ndarray, min_days: int = 3) -> np.ndarray:
    """
    Prevent acting on transient regime flips.

    A regime change is only recognized if the new regime persists for
    >= min_days consecutive days. Until confirmed, carry forward the
    previous regime.

    Financial rationale: short-lived regime flips generate spurious
    turnover and are often noise. AQR and similar vol-targeting funds
    use confirmation filters of 1-5 days.

    Parameters
    ----------
    regimes : np.ndarray
        Raw regime label array (strings).
    min_days : int
        Minimum consecutive days a new regime must hold to be confirmed.

    Returns
    -------
    np.ndarray
        Filtered regime labels with transient flips removed.
    """
    if len(regimes) == 0:
        return regimes.copy()

    filtered = regimes.copy()
    confirmed_regime = regimes[0]
    pending_regime = None
    pending_count = 0

    for i in range(len(regimes)):
        if regimes[i] == confirmed_regime:
            # Still in confirmed regime, reset any pending
            filtered[i] = confirmed_regime
            pending_regime = None
            pending_count = 0
        elif regimes[i] == pending_regime:
            # Candidate regime continues
            pending_count += 1
            if pending_count >= min_days:
                confirmed_regime = pending_regime
                filtered[i] = confirmed_regime
            else:
                filtered[i] = confirmed_regime
        else:
            # New candidate regime
            pending_regime = regimes[i]
            pending_count = 1
            filtered[i] = confirmed_regime

    return filtered
