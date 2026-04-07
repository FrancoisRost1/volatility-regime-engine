"""
Tests for engine/regime_detector.py.

Covers: HMMRegimeDetector fit/predict, CompositeRegimeDetector,
apply_persistence_filter behavior and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from engine.regime_detector import (
    HMMRegimeDetector,
    CompositeRegimeDetector,
    apply_persistence_filter,
)
from engine.feature_builder import z_score_features

VALID_LABELS = {"RISK_ON", "NEUTRAL", "RISK_OFF"}


class TestHMMRegimeDetector:
    """Tests for HMMRegimeDetector."""

    def test_fit_and_predict(self, config, synthetic_features):
        """HMM fits without error and predict returns valid labels."""
        hmm = HMMRegimeDetector(config)
        feat = synthetic_features.dropna()
        scaled = z_score_features(feat, len(feat) - 1)
        data = scaled.values

        hmm.fit(data, list(feat.columns))
        # Predict on a small subset for speed
        labels = hmm.predict_filtered(data[:50])
        assert len(labels) == 50
        assert set(labels).issubset(VALID_LABELS)

    def test_state_probabilities_sum_to_one(self, config, synthetic_features):
        """State probabilities at each time step should sum to ~1.0."""
        hmm = HMMRegimeDetector(config)
        feat = synthetic_features.dropna()
        scaled = z_score_features(feat, len(feat) - 1)
        data = scaled.values[:30]

        hmm.fit(data, list(feat.columns))
        probs = hmm.get_state_probabilities(data)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_label_map_uses_vol_ordering(self, config, synthetic_features):
        """States are labeled by ascending vol mean, not raw index."""
        hmm = HMMRegimeDetector(config)
        feat = synthetic_features.dropna()
        scaled = z_score_features(feat, len(feat) - 1)
        hmm.fit(scaled.values, list(feat.columns))

        # The label map should assign RISK_ON to the state with lowest vol mean
        feat_idx = list(feat.columns).index("realized_vol_21d")
        vol_means = hmm.model.means_[:, feat_idx]
        sorted_states = np.argsort(vol_means)
        assert hmm._label_map[int(sorted_states[0])] == "RISK_ON"
        assert hmm._label_map[int(sorted_states[2])] == "RISK_OFF"


class TestCompositeRegimeDetector:
    """Tests for CompositeRegimeDetector."""

    def test_predict_returns_valid_labels(self, config, synthetic_features,
                                          synthetic_prices):
        """Composite detector returns only valid regime labels."""
        comp = CompositeRegimeDetector(config)
        feat = synthetic_features.dropna()
        labels = comp.predict(feat, synthetic_prices)
        assert len(labels) == len(feat)
        assert set(labels).issubset(VALID_LABELS)

    def test_all_signals_positive_gives_risk_on(self, config):
        """When all 4 signals are +1 (score=4), regime should be RISK_ON."""
        comp = CompositeRegimeDetector(config)
        dates = pd.bdate_range("2020-01-01", periods=5)
        feat = pd.DataFrame({
            "spy_vs_sma200": [0.05] * 5,         # SPY above SMA -> +1
            "realized_vol_21d": [0.10] * 5,       # short < long -> +1
            "realized_vol_63d": [0.15] * 5,
            "credit_stress_proxy": [0.01] * 5,    # above threshold -> +1
            "return_21d": [0.05] * 5,
            "skew_63d": [0.0] * 5,
            "vix_level": [15.0] * 5,
            "vix_momentum": [0.0] * 5,
        }, index=dates)
        prices = pd.DataFrame({
            "SPY": [100, 101, 102, 103, 104.0],   # no drawdown -> +1
        }, index=dates)
        labels = comp.predict(feat, prices)
        assert all(l == "RISK_ON" for l in labels)

    def test_all_signals_negative_gives_risk_off(self, config):
        """When all 4 signals are -1 (score=-4), regime should be RISK_OFF."""
        comp = CompositeRegimeDetector(config)
        dates = pd.bdate_range("2020-01-01", periods=260)
        feat = pd.DataFrame({
            "spy_vs_sma200": [-0.15] * 260,         # below SMA -> -1
            "realized_vol_21d": [0.30] * 260,        # short > long -> -1
            "realized_vol_63d": [0.15] * 260,
            "credit_stress_proxy": [-0.02] * 260,    # below threshold -> -1
            "return_21d": [-0.10] * 260,
            "skew_63d": [-1.0] * 260,
            "vix_level": [30.0] * 260,
            "vix_momentum": [0.5] * 260,
        }, index=dates)
        # Big drawdown -> -1
        spy = np.linspace(100, 60, 260)
        prices = pd.DataFrame({"SPY": spy}, index=dates)
        labels = comp.predict(feat, prices)
        assert all(l == "RISK_OFF" for l in labels)


class TestPersistenceFilter:
    """Tests for apply_persistence_filter."""

    def test_stable_regime_unchanged(self):
        """A stable regime series passes through unmodified."""
        regimes = np.array(["RISK_ON"] * 20)
        filtered = apply_persistence_filter(regimes, min_days=3)
        np.testing.assert_array_equal(filtered, regimes)

    def test_transient_flip_rejected(self):
        """A 2-day flip (< min_days=3) is filtered out."""
        regimes = np.array(
            ["RISK_ON"] * 5 + ["RISK_OFF"] * 2 + ["RISK_ON"] * 5
        )
        filtered = apply_persistence_filter(regimes, min_days=3)
        # The 2-day RISK_OFF should be overwritten to RISK_ON
        assert all(f == "RISK_ON" for f in filtered)

    def test_persistent_flip_accepted(self):
        """A 3-day flip (= min_days=3) IS confirmed."""
        regimes = np.array(
            ["RISK_ON"] * 5 + ["RISK_OFF"] * 4 + ["RISK_ON"] * 3
        )
        filtered = apply_persistence_filter(regimes, min_days=3)
        # RISK_OFF should appear (confirmed after day 3)
        assert "RISK_OFF" in filtered

    def test_single_element(self):
        """Single-element array passes through."""
        regimes = np.array(["NEUTRAL"])
        filtered = apply_persistence_filter(regimes, min_days=3)
        assert filtered[0] == "NEUTRAL"

    def test_filter_preserves_length(self):
        """Output length matches input length."""
        regimes = np.array(["RISK_ON", "RISK_OFF", "NEUTRAL"] * 10)
        filtered = apply_persistence_filter(regimes, min_days=3)
        assert len(filtered) == len(regimes)

    def test_min_days_one_requires_one_continuation(self):
        """With min_days=1, a regime that persists for 1 extra day is confirmed."""
        # RO RO RO OFF OFF OFF RO RO
        regimes = np.array(
            ["RISK_ON"] * 3 + ["RISK_OFF"] * 3 + ["RISK_ON"] * 2
        )
        filtered = apply_persistence_filter(regimes, min_days=1)
        # RISK_OFF should be confirmed (persists >=1 day after first appearance)
        assert "RISK_OFF" in filtered
