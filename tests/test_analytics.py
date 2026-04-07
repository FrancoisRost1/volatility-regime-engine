"""
Tests for engine/analytics.py and engine/analytics_regime.py.

Covers: overall metrics, drawdown calculation, regime-conditional metrics,
transition matrix, stress period metrics, benchmark attribution.
"""

import numpy as np
import pandas as pd
import pytest

from engine.analytics import (
    compute_overall_metrics,
    compute_benchmark_attribution,
    compute_drawdown,
)
from engine.analytics_regime import (
    compute_regime_conditional_metrics,
    compute_regime_duration_stats,
    compute_transition_matrix,
    compute_stress_period_metrics,
)


@pytest.fixture
def sample_nav():
    """Synthetic NAV series — upward trend with a drawdown."""
    dates = pd.bdate_range("2019-01-02", periods=500)
    rng = np.random.RandomState(42)
    daily_ret = 0.0003 + 0.01 * rng.randn(500)
    nav = 100 * np.exp(np.cumsum(daily_ret))
    return pd.Series(nav, index=dates)


@pytest.fixture
def sample_returns(sample_nav):
    """Daily returns derived from NAV."""
    return sample_nav.pct_change().dropna()


@pytest.fixture
def sample_spy_returns(sample_returns):
    """Synthetic SPY returns aligned to sample returns."""
    rng = np.random.RandomState(99)
    return pd.Series(
        0.0004 + 0.012 * rng.randn(len(sample_returns)),
        index=sample_returns.index,
    )


class TestComputeDrawdown:
    """Tests for compute_drawdown."""

    def test_monotonic_nav_no_drawdown(self):
        """Monotonically increasing NAV has zero max drawdown."""
        nav = pd.Series([100, 101, 102, 103, 104.0])
        dd_series, max_dd, max_dur = compute_drawdown(nav)
        assert max_dd == 0.0
        assert max_dur == 0

    def test_known_drawdown(self):
        """Verify drawdown for a known series."""
        nav = pd.Series([100, 110, 105, 108, 115.0])
        dd_series, max_dd, max_dur = compute_drawdown(nav)
        # Peak = 110, trough = 105 -> dd = 105/110 - 1 = -0.04545
        assert max_dd == pytest.approx(-5 / 110, abs=1e-6)

    def test_drawdown_series_nonpositive(self, sample_nav):
        """Drawdown series is always <= 0."""
        dd_series, _, _ = compute_drawdown(sample_nav)
        assert (dd_series <= 1e-10).all()


class TestComputeOverallMetrics:
    """Tests for compute_overall_metrics."""

    def test_returns_dict(self, sample_nav, sample_returns,
                          sample_spy_returns, config):
        """Output is a dict with expected keys."""
        metrics = compute_overall_metrics(
            sample_nav, sample_returns, sample_spy_returns, config
        )
        expected_keys = [
            "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
            "calmar", "win_rate_monthly", "annualized_vol", "correlation_spy",
        ]
        for k in expected_keys:
            assert k in metrics, f"Missing key: {k}"

    def test_sharpe_positive_for_positive_returns(
        self, sample_nav, sample_returns, sample_spy_returns, config
    ):
        """A NAV with positive drift should have positive Sharpe."""
        metrics = compute_overall_metrics(
            sample_nav, sample_returns, sample_spy_returns, config
        )
        assert metrics["sharpe"] > 0

    def test_max_drawdown_negative(self, sample_nav, sample_returns,
                                    sample_spy_returns, config):
        """Max drawdown should be <= 0."""
        metrics = compute_overall_metrics(
            sample_nav, sample_returns, sample_spy_returns, config
        )
        assert metrics["max_drawdown"] <= 0

    def test_win_rate_between_zero_and_one(self, sample_nav, sample_returns,
                                            sample_spy_returns, config):
        """Monthly win rate should be in [0, 1]."""
        metrics = compute_overall_metrics(
            sample_nav, sample_returns, sample_spy_returns, config
        )
        assert 0 <= metrics["win_rate_monthly"] <= 1


class TestBenchmarkAttribution:
    """Tests for compute_benchmark_attribution."""

    def test_output_shape(self):
        """Attribution table has correct shape."""
        strat = {"cagr": 0.10, "sharpe": 1.2, "max_drawdown": -0.15, "calmar": 0.67}
        bench = {
            "spy": {"cagr": 0.08, "sharpe": 0.8, "max_drawdown": -0.30, "calmar": 0.27},
        }
        table = compute_benchmark_attribution(strat, bench)
        assert "spy" in table.columns
        assert "full_strategy" in table.columns
        assert len(table) == 4


class TestRegimeConditionalMetrics:
    """Tests for compute_regime_conditional_metrics."""

    def test_output_has_all_regimes(self, config):
        """Should have one row per regime."""
        dates = pd.bdate_range("2020-01-01", periods=90)
        ret = pd.Series(np.random.randn(90) * 0.01, index=dates)
        regimes = pd.Series(
            ["RISK_ON"] * 30 + ["NEUTRAL"] * 30 + ["RISK_OFF"] * 30,
            index=dates,
        )
        result = compute_regime_conditional_metrics(ret, regimes, config)
        assert set(result.index) == {"RISK_ON", "NEUTRAL", "RISK_OFF"}

    def test_pct_time_sums_to_one(self, config):
        """Percentage of time across regimes should sum to 1.0."""
        dates = pd.bdate_range("2020-01-01", periods=90)
        ret = pd.Series(np.random.randn(90) * 0.01, index=dates)
        regimes = pd.Series(
            ["RISK_ON"] * 30 + ["NEUTRAL"] * 30 + ["RISK_OFF"] * 30,
            index=dates,
        )
        result = compute_regime_conditional_metrics(ret, regimes, config)
        np.testing.assert_almost_equal(result["pct_time"].sum(), 1.0)


class TestRegimeDurationStats:
    """Tests for compute_regime_duration_stats."""

    def test_single_regime(self):
        """Single regime: 1 episode, duration = full length."""
        regimes = pd.Series(["RISK_ON"] * 50)
        stats = compute_regime_duration_stats(regimes)
        assert stats.loc["RISK_ON", "n_episodes"] == 1
        assert stats.loc["RISK_ON", "avg_duration_days"] == 50

    def test_alternating_regimes(self):
        """Known alternating pattern: correct episode counts."""
        regimes = pd.Series(
            ["RISK_ON"] * 10 + ["RISK_OFF"] * 10 + ["RISK_ON"] * 10
        )
        stats = compute_regime_duration_stats(regimes)
        assert stats.loc["RISK_ON", "n_episodes"] == 2
        assert stats.loc["RISK_OFF", "n_episodes"] == 1


class TestTransitionMatrix:
    """Tests for compute_transition_matrix."""

    def test_rows_sum_to_one(self):
        """Each row of the transition matrix should sum to 1.0."""
        regimes = pd.Series(
            ["RISK_ON"] * 30 + ["NEUTRAL"] * 30 + ["RISK_OFF"] * 30
        )
        tm = compute_transition_matrix(regimes)
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            row_sum = tm.loc[regime].sum()
            if row_sum > 0:
                np.testing.assert_almost_equal(row_sum, 1.0)

    def test_stable_regime_high_diagonal(self):
        """A single-regime series should have 1.0 on its diagonal."""
        regimes = pd.Series(["RISK_ON"] * 100)
        tm = compute_transition_matrix(regimes)
        assert tm.loc["RISK_ON", "RISK_ON"] == 1.0

    def test_shape_3x3(self):
        """Transition matrix is 3x3."""
        regimes = pd.Series(["RISK_ON", "NEUTRAL", "RISK_OFF"] * 10)
        tm = compute_transition_matrix(regimes)
        assert tm.shape == (3, 3)


class TestStressPeriodMetrics:
    """Tests for compute_stress_period_metrics."""

    def test_output_columns(self, config):
        """Output should have period, portfolio, return, max_drawdown, sharpe."""
        dates = pd.date_range("2007-01-01", periods=5000, freq="B")
        rng = np.random.RandomState(42)
        nav = pd.Series(100 * np.exp(np.cumsum(rng.randn(5000) * 0.005)), index=dates)
        ret = nav.pct_change().dropna()
        spy_nav = nav * 0.95
        spy_ret = spy_nav.pct_change().dropna()
        result = compute_stress_period_metrics(nav, ret, spy_nav, spy_ret, config)
        assert "period" in result.columns
        assert "portfolio" in result.columns
        assert "return" in result.columns
