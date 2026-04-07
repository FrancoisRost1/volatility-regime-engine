"""
Tests for engine/position_sizer.py.

Covers: weights sum constraint, leverage caps, inverse-vol logic,
zero-vol edge case, regime-specific tilts.
"""

import numpy as np
import pytest

from engine.position_sizer import (
    compute_weights,
    _get_strategic_weights,
    _inverse_vol_adjust,
    _apply_leverage_cap,
)


class TestGetStrategicWeights:
    """Tests for _get_strategic_weights."""

    def test_weights_sum_to_one(self, config):
        """Strategic weights for every regime sum to 1.0."""
        tickers = config["data"]["allocation_tickers"]
        ps_cfg = config["position_sizing"]
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            w = _get_strategic_weights(regime, tickers, ps_cfg)
            np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_risk_off_has_more_tlt(self, config):
        """RISK_OFF should overweight TLT relative to RISK_ON."""
        tickers = config["data"]["allocation_tickers"]
        ps_cfg = config["position_sizing"]
        w_on = _get_strategic_weights("RISK_ON", tickers, ps_cfg)
        w_off = _get_strategic_weights("RISK_OFF", tickers, ps_cfg)
        tlt_idx = tickers.index("TLT")
        assert w_off[tlt_idx] > w_on[tlt_idx]

    def test_risk_on_has_more_spy(self, config):
        """RISK_ON should overweight SPY relative to RISK_OFF."""
        tickers = config["data"]["allocation_tickers"]
        ps_cfg = config["position_sizing"]
        w_on = _get_strategic_weights("RISK_ON", tickers, ps_cfg)
        w_off = _get_strategic_weights("RISK_OFF", tickers, ps_cfg)
        spy_idx = tickers.index("SPY")
        assert w_on[spy_idx] > w_off[spy_idx]


class TestInverseVolAdjust:
    """Tests for _inverse_vol_adjust."""

    def test_output_sums_to_one(self):
        """Inverse-vol adjusted weights sum to 1.0."""
        strategic = np.array([0.4, 0.3, 0.2, 0.1])
        vols = np.array([0.15, 0.10, 0.12, 0.20])
        w = _inverse_vol_adjust(strategic, vols)
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_lower_vol_gets_higher_weight(self):
        """Asset with lower vol should get relatively more weight."""
        strategic = np.array([0.25, 0.25, 0.25, 0.25])
        vols = np.array([0.05, 0.20, 0.15, 0.10])
        w = _inverse_vol_adjust(strategic, vols)
        # Asset 0 has lowest vol -> should have highest weight
        assert w[0] == w.max()

    def test_zero_vol_asset_gets_zero_weight(self):
        """An asset with zero vol should get 0 weight, others redistributed."""
        strategic = np.array([0.25, 0.25, 0.25, 0.25])
        vols = np.array([0.0, 0.15, 0.10, 0.20])
        w = _inverse_vol_adjust(strategic, vols)
        assert w[0] == 0.0
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_nan_vol_asset_gets_zero_weight(self):
        """An asset with NaN vol should get 0 weight."""
        strategic = np.array([0.25, 0.25, 0.25, 0.25])
        vols = np.array([np.nan, 0.15, 0.10, 0.20])
        w = _inverse_vol_adjust(strategic, vols)
        assert w[0] == 0.0
        np.testing.assert_almost_equal(w.sum(), 1.0)

    def test_all_zero_vols_fallback(self):
        """If all vols are zero, fall back to strategic / sum(strategic)."""
        strategic = np.array([0.4, 0.3, 0.2, 0.1])
        vols = np.zeros(4)
        w = _inverse_vol_adjust(strategic, vols)
        np.testing.assert_almost_equal(w.sum(), 1.0)


class TestApplyLeverageCap:
    """Tests for _apply_leverage_cap."""

    def test_under_cap_unchanged(self, config):
        """Weights under the cap should pass through unchanged."""
        ps_cfg = config["position_sizing"]
        w = np.array([0.2, 0.3, 0.2, 0.1])  # gross = 0.8
        w_out = _apply_leverage_cap(w, "NEUTRAL", ps_cfg)
        np.testing.assert_array_equal(w_out, w)

    def test_over_cap_scaled_down(self, config):
        """Weights exceeding the cap should be scaled to exactly the cap."""
        ps_cfg = config["position_sizing"]
        w = np.array([0.6, 0.4, 0.3, 0.2])  # gross = 1.5
        w_out = _apply_leverage_cap(w, "NEUTRAL", ps_cfg)  # cap = 1.0
        gross = np.abs(w_out).sum()
        np.testing.assert_almost_equal(gross, 1.0)

    def test_risk_on_allows_higher_leverage(self, config):
        """RISK_ON cap is 1.5 — weights up to 1.5 should pass."""
        ps_cfg = config["position_sizing"]
        w = np.array([0.5, 0.3, 0.3, 0.3])  # gross = 1.4
        w_out = _apply_leverage_cap(w, "RISK_ON", ps_cfg)
        np.testing.assert_array_equal(w_out, w)  # under 1.5, unchanged


class TestComputeWeights:
    """Tests for compute_weights (full pipeline)."""

    def test_leverage_cap_respected(self, config):
        """Final weights should never exceed the regime leverage cap."""
        vols = np.array([0.02, 0.02, 0.02, 0.02])  # very low vol -> high scale
        cov = np.eye(4) * 0.0004  # low var
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            w = compute_weights(regime, vols, cov, config)
            cap = config["position_sizing"]["leverage_caps"][regime]
            assert np.abs(w).sum() <= cap + 1e-10

    def test_output_length(self, config):
        """Output has n_assets elements."""
        vols = np.array([0.15, 0.10, 0.12, 0.18])
        cov = np.eye(4) * 0.02
        w = compute_weights("RISK_ON", vols, cov, config)
        assert len(w) == 4

    def test_no_negative_weights(self, config):
        """Long-only strategy should not produce negative weights."""
        vols = np.array([0.15, 0.10, 0.12, 0.18])
        cov = np.eye(4) * 0.02
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            w = compute_weights(regime, vols, cov, config)
            assert (w >= -1e-10).all()
