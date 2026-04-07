"""
Tests for engine/rebalancer.py.

Covers: should_rebalance trigger logic, transaction cost formula,
turnover calculation, edge cases.
"""

import numpy as np
import pytest

from engine.rebalancer import (
    should_rebalance,
    compute_transaction_cost,
    compute_turnover,
)


class TestShouldRebalance:
    """Tests for should_rebalance."""

    def test_regime_change_triggers(self, config):
        """Different regime today vs last rebalance -> rebalance."""
        rebal, reason = should_rebalance(
            "RISK_OFF", "RISK_ON", 0.10, 0.10, config
        )
        assert rebal is True
        assert reason == "regime_change"

    def test_same_regime_no_vol_drift(self, config):
        """Same regime, vol within threshold -> no rebalance."""
        rebal, reason = should_rebalance(
            "RISK_ON", "RISK_ON", 0.10, 0.10, config
        )
        assert rebal is False
        assert reason == ""

    def test_vol_drift_triggers(self, config):
        """Same regime but vol drifted >20% -> rebalance."""
        # 0.13 / 0.10 - 1 = 0.30 > 0.20 threshold
        rebal, reason = should_rebalance(
            "RISK_ON", "RISK_ON", 0.13, 0.10, config
        )
        assert rebal is True
        assert reason == "vol_drift"

    def test_vol_drift_below_threshold(self, config):
        """Vol drift of 15% (< 20%) should NOT trigger rebalance."""
        rebal, reason = should_rebalance(
            "RISK_ON", "RISK_ON", 0.115, 0.10, config
        )
        assert rebal is False

    def test_zero_last_vol_no_crash(self, config):
        """Zero port_vol_last_rebal should not cause division error."""
        rebal, reason = should_rebalance(
            "RISK_ON", "RISK_ON", 0.10, 0.0, config
        )
        assert isinstance(rebal, bool)

    def test_nan_last_vol_no_crash(self, config):
        """NaN port_vol_last_rebal should not crash."""
        rebal, reason = should_rebalance(
            "RISK_ON", "RISK_ON", 0.10, np.nan, config
        )
        assert isinstance(rebal, bool)


class TestComputeTransactionCost:
    """Tests for compute_transaction_cost."""

    def test_no_change_no_cost(self, config):
        """Identical weights -> zero cost."""
        w = np.array([0.3, 0.3, 0.2, 0.2])
        cost = compute_transaction_cost(w, w, config)
        assert cost == 0.0

    def test_cost_formula(self, config):
        """Verify cost = (bps/10000) * sum(|w_new - w_old|)."""
        w_old = np.array([0.3, 0.3, 0.2, 0.2])
        w_new = np.array([0.4, 0.2, 0.3, 0.1])
        cost = compute_transaction_cost(w_new, w_old, config)
        turnover = np.abs(w_new - w_old).sum()  # 0.1+0.1+0.1+0.1 = 0.4
        expected = (config["rebalancing"]["cost_bps"] / 10_000) * turnover
        np.testing.assert_almost_equal(cost, expected)

    def test_cost_nonnegative(self, config):
        """Cost should always be non-negative."""
        rng = np.random.RandomState(99)
        for _ in range(20):
            w_old = rng.dirichlet([1, 1, 1, 1])
            w_new = rng.dirichlet([1, 1, 1, 1])
            cost = compute_transaction_cost(w_new, w_old, config)
            assert cost >= 0


class TestComputeTurnover:
    """Tests for compute_turnover."""

    def test_no_change(self):
        """Identical weights -> zero turnover."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert compute_turnover(w, w) == 0.0

    def test_full_flip(self):
        """Complete reversal: turnover = 2 * sum(old)."""
        w_old = np.array([1.0, 0.0, 0.0, 0.0])
        w_new = np.array([0.0, 1.0, 0.0, 0.0])
        assert compute_turnover(w_new, w_old) == 2.0

    def test_symmetry(self):
        """Turnover(a, b) == turnover(b, a)."""
        w_a = np.array([0.4, 0.3, 0.2, 0.1])
        w_b = np.array([0.1, 0.2, 0.3, 0.4])
        assert compute_turnover(w_a, w_b) == compute_turnover(w_b, w_a)
