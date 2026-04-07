"""
Tests for engine/feature_builder.py.

Covers: 8-feature output, z-scoring no-lookahead invariant, zero-variance edge case.
"""

import numpy as np
import pandas as pd
import pytest

from engine.feature_builder import build_features, z_score_features


class TestBuildFeatures:
    """Tests for build_features."""

    def test_output_has_8_columns(self, synthetic_prices, synthetic_returns, config):
        """Feature matrix must have exactly 8 columns."""
        feat = build_features(synthetic_prices, synthetic_returns, config)
        assert feat.shape[1] == 8

    def test_expected_column_names(self, synthetic_features):
        """Verify all 8 feature names are present."""
        expected = [
            "realized_vol_21d", "realized_vol_63d", "return_21d",
            "skew_63d", "spy_vs_sma200", "credit_stress_proxy",
            "vix_level", "vix_momentum",
        ]
        assert sorted(synthetic_features.columns.tolist()) == sorted(expected)

    def test_index_aligned_to_returns(self, synthetic_features, synthetic_returns):
        """Feature index should match returns index."""
        assert synthetic_features.index[0] == synthetic_returns.index[0]
        assert synthetic_features.index[-1] == synthetic_returns.index[-1]

    def test_vix_level_is_positive(self, synthetic_features):
        """VIX level should always be positive."""
        vix = synthetic_features["vix_level"].dropna()
        assert (vix > 0).all()

    def test_realized_vol_nonnegative(self, synthetic_features):
        """Realized vol is non-negative where defined."""
        for col in ["realized_vol_21d", "realized_vol_63d"]:
            vals = synthetic_features[col].dropna()
            assert (vals >= 0).all()


class TestZScoreFeatures:
    """Tests for z_score_features."""

    def test_no_lookahead(self, synthetic_features):
        """Z-scoring with fit_end_idx=N must use only rows 0..N for stats."""
        fit_idx = 100
        scaled = z_score_features(synthetic_features, fit_idx)

        # Training slice should have mean~0, std~1 per column (approximately)
        train = scaled.iloc[: fit_idx + 1]
        for col in train.columns:
            col_vals = train[col].dropna()
            if len(col_vals) > 5:
                assert abs(col_vals.mean()) < 0.1, f"{col} mean not near 0"

    def test_future_data_not_used(self, synthetic_features):
        """Changing data after fit_end_idx should not affect scaling params."""
        fit_idx = 100
        feat_a = synthetic_features.copy()
        feat_b = synthetic_features.copy()
        # Perturb data after fit_end_idx
        feat_b.iloc[fit_idx + 1:] = feat_b.iloc[fit_idx + 1:] * 10

        scaled_a = z_score_features(feat_a, fit_idx)
        scaled_b = z_score_features(feat_b, fit_idx)

        # Training portion should be identical
        pd.testing.assert_frame_equal(
            scaled_a.iloc[: fit_idx + 1],
            scaled_b.iloc[: fit_idx + 1],
        )

    def test_zero_variance_column_stays_nan(self):
        """A feature with zero variance should remain NaN (not zero-filled).

        Financial rationale: zero-filling creates synthetic observations that
        bias HMM training. The backtester's NaN mask drops these rows instead.
        """
        dates = pd.bdate_range("2020-01-01", periods=50)
        feat = pd.DataFrame({
            "constant": [5.0] * 50,
            "varying": np.random.randn(50),
        }, index=dates)
        scaled = z_score_features(feat, 30)
        assert scaled["constant"].isna().all()
        assert not scaled["varying"].isna().all()

    def test_output_shape_matches_input(self, synthetic_features):
        """Z-scored output has same shape as input."""
        scaled = z_score_features(synthetic_features, 200)
        assert scaled.shape == synthetic_features.shape
