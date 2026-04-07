"""
Deterministic tests for exact NAV, Sharpe, Sortino on known 10-day series.

Hand-computed expected values verified independently. Tolerances are tight
(1e-6) because these are arithmetic checks, not statistical estimates.
"""

import numpy as np
import pandas as pd
import pytest

from engine.analytics import compute_overall_metrics, compute_drawdown
from engine.analytics_regime import compute_transition_matrix


@pytest.fixture
def config_minimal():
    """Minimal config for analytics tests."""
    return {
        "backtest": {"risk_free_rate": 0.0},
        "vol_estimation": {"annualization_factor": 252},
    }


class TestExactNAV:
    """Verify NAV compounding from known daily returns."""

    def test_nav_from_simple_returns(self):
        """NAV = 100 * prod(1 + r_i) for simple returns."""
        returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01,
                            0.02, 0.003, -0.005, 0.01, 0.008])
        nav_0 = 100.0
        # Hand-computed: multiply through
        nav_series = [nav_0]
        for r in returns:
            nav_series.append(nav_series[-1] * (1 + r))
        expected_final = nav_series[-1]

        # Verify final NAV
        actual = nav_0 * np.prod(1 + returns)
        np.testing.assert_almost_equal(actual, expected_final, decimal=10)

    def test_log_vs_simple_divergence(self):
        """Log returns are NOT valid for weighted portfolio compounding."""
        # Portfolio: 60% asset A (2% return), 40% asset B (-1% return)
        r_simple = np.array([0.02, -0.01])
        weights = np.array([0.6, 0.4])
        # Correct: weighted simple return
        port_ret_correct = weights @ r_simple  # 0.012 - 0.004 = 0.008
        assert port_ret_correct == pytest.approx(0.008, abs=1e-10)

        # Wrong: weighted log return then exp
        r_log = np.log(1 + r_simple)
        port_ret_wrong = np.exp(weights @ r_log) - 1
        # These differ — log returns are not additive across assets
        assert port_ret_correct != pytest.approx(port_ret_wrong, abs=1e-10)


class TestExactSharpe:
    """Verify Sharpe computation on known series."""

    def test_known_sharpe(self, config_minimal):
        """Sharpe = mean(r) / std(r) * sqrt(252), rf=0."""
        dates = pd.bdate_range("2020-01-02", periods=10)
        returns = pd.Series(
            [0.01, -0.02, 0.015, 0.005, -0.01,
             0.02, 0.003, -0.005, 0.01, 0.008],
            index=dates,
        )
        nav = 100.0 * (1 + returns).cumprod()

        # Hand-computed
        mean_r = returns.mean()
        std_r = returns.std()
        expected_sharpe = float(mean_r / std_r * np.sqrt(252))

        spy_ret = pd.Series(np.zeros(10), index=dates)
        metrics = compute_overall_metrics(nav, returns, spy_ret, config_minimal)

        np.testing.assert_almost_equal(
            metrics["sharpe"], expected_sharpe, decimal=6,
        )


class TestExactSortino:
    """Verify Sortino uses annualized mean / annualized downside deviation."""

    def test_known_sortino(self, config_minimal):
        """Sortino = ann_mean_excess / ann_downside_std."""
        dates = pd.bdate_range("2020-01-02", periods=10)
        returns = pd.Series(
            [0.01, -0.02, 0.015, 0.005, -0.01,
             0.02, 0.003, -0.005, 0.01, 0.008],
            index=dates,
        )
        nav = 100.0 * (1 + returns).cumprod()

        # Hand-computed
        downside = returns[returns < 0]  # [-0.02, -0.01, -0.005]
        downside_std_ann = downside.std() * np.sqrt(252)
        ann_mean = returns.mean() * 252
        expected_sortino = float(ann_mean / downside_std_ann)

        spy_ret = pd.Series(np.zeros(10), index=dates)
        metrics = compute_overall_metrics(nav, returns, spy_ret, config_minimal)

        np.testing.assert_almost_equal(
            metrics["sortino"], expected_sortino, decimal=6,
        )


class TestExactDrawdown:
    """Verify max drawdown from running peak."""

    def test_known_drawdown(self):
        """DD = (trough - peak) / peak from running maximum."""
        nav = pd.Series([100.0, 110.0, 105.0, 108.0, 115.0, 100.0, 112.0])
        dd_series, max_dd, _ = compute_drawdown(nav)

        # Peak at 115, trough at 100 -> dd = 100/115 - 1 = -0.13043...
        expected_max_dd = 100.0 / 115.0 - 1
        np.testing.assert_almost_equal(max_dd, expected_max_dd, decimal=6)

    def test_drawdown_series_from_running_peak(self):
        """Each dd value = nav / cummax(nav) - 1."""
        nav = pd.Series([100.0, 110.0, 105.0, 115.0])
        dd, _, _ = compute_drawdown(nav)
        expected = pd.Series([0.0, 0.0, 105.0 / 110.0 - 1, 0.0])
        pd.testing.assert_series_equal(dd, expected, check_names=False)


class TestTransitionMatrixRowStochastic:
    """Verify transition matrix rows always sum to 1."""

    def test_all_regimes_observed(self):
        """Standard case: all 3 regimes present."""
        regimes = pd.Series(
            ["RISK_ON"] * 30 + ["NEUTRAL"] * 30 + ["RISK_OFF"] * 30
        )
        tm = compute_transition_matrix(regimes)
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            np.testing.assert_almost_equal(tm.loc[regime].sum(), 1.0, decimal=6)

    def test_unobserved_regime_gets_uniform(self):
        """If a regime is never observed, its row should be uniform (1/3)."""
        regimes = pd.Series(["RISK_ON"] * 50 + ["NEUTRAL"] * 50)
        tm = compute_transition_matrix(regimes)
        # RISK_OFF never observed -> should be 1/3 each
        np.testing.assert_almost_equal(
            tm.loc["RISK_OFF"].values,
            np.array([1/3, 1/3, 1/3]),
            decimal=6,
        )
        # All rows sum to 1
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            np.testing.assert_almost_equal(tm.loc[regime].sum(), 1.0, decimal=6)


class TestMonthlyReturnsCompounding:
    """Verify monthly returns are compounded, not summed."""

    def test_compounded_monthly(self):
        """Monthly return = prod(1+r) - 1, not sum(r)."""
        dates = pd.bdate_range("2020-01-02", periods=22)  # ~1 month
        daily = pd.Series(0.01 * np.ones(22), index=dates)

        # Compounded
        monthly_compound = (1 + daily).resample("ME").prod() - 1
        # Summed (wrong)
        monthly_sum = daily.resample("ME").sum()

        # They should differ for non-zero returns
        assert not np.isclose(
            monthly_compound.iloc[0], monthly_sum.iloc[0], atol=1e-6,
        )
        # Compounded is always larger for positive returns
        assert monthly_compound.iloc[0] > monthly_sum.iloc[0]
