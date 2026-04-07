"""
End-to-end integration test for volatility-regime-engine.

Runs the full pipeline on 600 days of synthetic price data and asserts:
  - NAV is a valid float series with no NaN after warmup
  - Output DataFrame has all required columns
  - Regime labels are valid throughout
  - Benchmarks produce valid NAV series

No yfinance calls — all data is synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data_loader import compute_log_returns
from engine.backtester import WalkForwardBacktester
from engine.benchmarks import run_benchmarks
from engine.analytics import compute_overall_metrics, compute_drawdown
from engine.analytics_regime import (
    compute_regime_conditional_metrics,
    compute_transition_matrix,
)


@pytest.fixture
def integration_config(config):
    """Config with shortened warmup for integration test speed."""
    cfg = config.copy()
    cfg["hmm"] = config["hmm"].copy()
    cfg["hmm"]["warmup_days"] = 100
    cfg["hmm"]["refit_every_days"] = 100
    return cfg


@pytest.fixture
def integration_data():
    """600 business days of synthetic multi-asset prices."""
    dates = pd.bdate_range("2018-01-02", periods=600, freq="B")
    rng = np.random.RandomState(123)

    prices = pd.DataFrame(index=dates)
    specs = [
        ("SPY", 250, 0.0003, 0.012), ("TLT", 120, 0.0001, 0.008),
        ("GLD", 160, 0.0002, 0.009), ("PDBC", 18, 0.0001, 0.011),
        ("LQD", 115, 0.00005, 0.004), ("IEF", 105, 0.00008, 0.005),
    ]
    for ticker, start, mu, sigma in specs:
        ret = mu + sigma * rng.randn(len(dates))
        ret[0] = 0.0
        prices[ticker] = start * np.exp(np.cumsum(ret))

    # VIX: synthetic mean-reverting series
    prices["^VIX"] = 18 + 8 * np.abs(np.sin(np.linspace(0, 15, len(dates))))

    returns = compute_log_returns(prices)
    return prices, returns


class TestEndToEndPipeline:
    """Full pipeline integration test."""

    def test_backtest_produces_valid_nav(self, integration_config,
                                         integration_data):
        """NAV series has no NaN and contains only valid floats."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)

        nav = result["nav"]
        assert not nav.isna().any(), "NAV contains NaN values"
        assert np.all(np.isfinite(nav.values)), "NAV contains inf values"
        assert len(nav) > 0, "NAV series is empty"

    def test_nav_stays_positive(self, integration_config, integration_data):
        """NAV should remain positive throughout the backtest."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)
        assert (result["nav"] > 0).all()

    def test_all_required_columns(self, integration_config, integration_data):
        """Result DataFrame has all expected columns."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)

        required = [
            "regime_hmm", "regime_composite", "nav", "daily_return",
            "gross_exposure", "target_vol", "realized_port_vol",
            "rebalanced", "turnover", "cost",
            "w_SPY", "w_TLT", "w_GLD", "w_PDBC",
            "p_risk_on", "p_neutral", "p_risk_off",
        ]
        for col in required:
            assert col in result.columns, f"Missing: {col}"

    def test_regime_labels_valid(self, integration_config, integration_data):
        """All regime labels are one of the 3 valid states."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)
        valid = {"RISK_ON", "NEUTRAL", "RISK_OFF"}
        assert set(result["regime_hmm"].unique()).issubset(valid)
        assert set(result["regime_composite"].unique()).issubset(valid)

    def test_gross_exposure_within_bounds(self, integration_config,
                                          integration_data):
        """Gross exposure should never exceed max leverage cap (1.5)."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)
        max_cap = max(
            integration_config["position_sizing"]["leverage_caps"].values()
        )
        assert (result["gross_exposure"] <= max_cap + 1e-6).all()

    def test_analytics_on_backtest_output(self, integration_config,
                                           integration_data):
        """Analytics functions run without error on backtest output."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)

        nav = result["nav"]
        daily_ret = result["daily_return"]
        spy_ret = returns["SPY"].reindex(result.index).fillna(0)

        metrics = compute_overall_metrics(nav, daily_ret, spy_ret,
                                          integration_config)
        assert "cagr" in metrics
        assert "sharpe" in metrics
        assert np.isfinite(metrics["cagr"])

    def test_regime_conditional_analytics(self, integration_config,
                                           integration_data):
        """Regime-conditional metrics run on backtest output."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)

        regime_metrics = compute_regime_conditional_metrics(
            result["daily_return"], result["regime_hmm"], integration_config
        )
        assert len(regime_metrics) == 3

    def test_transition_matrix_valid(self, integration_config,
                                      integration_data):
        """Transition matrix rows sum to 1.0."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)

        tm = compute_transition_matrix(result["regime_hmm"])
        for regime in ["RISK_ON", "NEUTRAL", "RISK_OFF"]:
            row_sum = tm.loc[regime].sum()
            if row_sum > 0:
                np.testing.assert_almost_equal(row_sum, 1.0)

    def test_benchmarks_produce_valid_nav(self, integration_config,
                                           integration_data):
        """All benchmarks produce non-empty NAV series with no NaN."""
        prices, returns = integration_data
        bt = WalkForwardBacktester(integration_config)
        result = bt.run(prices, returns)

        benchmarks = run_benchmarks(
            returns, result["regime_hmm"].values, result.index,
            integration_config,
        )
        for name, bnav in benchmarks.items():
            assert len(bnav) > 0, f"Benchmark {name} is empty"
            assert not bnav.isna().any(), f"Benchmark {name} has NaN"
