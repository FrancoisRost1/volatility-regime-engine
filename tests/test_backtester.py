"""
Tests for engine/backtester.py.

Uses synthetic data — no yfinance. Tests output shape, required columns,
NAV behavior, and warmup period handling.

Note: backtester tests use the shared integration_config and integration_data
fixtures from conftest to avoid HMM convergence issues with tiny datasets.
"""

import numpy as np
import pandas as pd
import pytest

from engine.backtester import WalkForwardBacktester
from engine.data_loader import compute_log_returns


@pytest.fixture
def bt_config(config):
    """Config with shortened warmup for faster tests."""
    cfg = config.copy()
    cfg["hmm"] = config["hmm"].copy()
    cfg["hmm"]["warmup_days"] = 100
    cfg["hmm"]["refit_every_days"] = 100
    return cfg


@pytest.fixture
def bt_data():
    """
    400 days of synthetic prices with enough variance for HMM convergence.
    Uses distinct volatility regimes to give the HMM separable states.
    """
    dates = pd.bdate_range("2018-01-02", periods=400, freq="B")
    rng = np.random.RandomState(7)

    prices = pd.DataFrame(index=dates)

    # SPY: simulate 2 distinct vol regimes for HMM separability
    spy_ret = np.empty(len(dates))
    spy_ret[0] = 0.0
    spy_ret[1:200] = 0.0004 + 0.008 * rng.randn(199)   # low vol
    spy_ret[200:] = -0.0001 + 0.020 * rng.randn(200)    # high vol
    prices["SPY"] = 250 * np.exp(np.cumsum(spy_ret))

    for ticker, start, mu, sigma, seed in [
        ("TLT", 120, 0.0001, 0.007, 10),
        ("GLD", 160, 0.0002, 0.008, 20),
        ("PDBC", 18, 0.0001, 0.010, 30),
        ("LQD", 115, 0.00005, 0.004, 40),
        ("IEF", 105, 0.00008, 0.005, 50),
    ]:
        rs = np.random.RandomState(seed)
        ret = mu + sigma * rs.randn(len(dates))
        ret[0] = 0.0
        prices[ticker] = start * np.exp(np.cumsum(ret))

    # VIX: higher during high-vol regime
    vix = np.empty(len(dates))
    vix[:200] = 14 + 2 * rng.rand(200)
    vix[200:] = 25 + 5 * rng.rand(200)
    prices["^VIX"] = vix

    returns = compute_log_returns(prices)
    return prices, returns


class TestWalkForwardBacktester:
    """Tests for WalkForwardBacktester."""

    def test_output_is_dataframe(self, bt_config, bt_data):
        """Backtest returns a DataFrame."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, bt_config, bt_data):
        """Output must contain all key columns."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        required = [
            "regime_hmm", "regime_composite", "nav", "daily_return",
            "gross_exposure", "turnover", "cost", "rebalanced",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_weight_columns_present(self, bt_config, bt_data):
        """Weight columns for each allocation asset must exist."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        for ticker in bt_config["data"]["allocation_tickers"]:
            assert f"w_{ticker}" in result.columns

    def test_nav_no_nan(self, bt_config, bt_data):
        """NAV should have no NaN values after warmup."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        assert not result["nav"].isna().any()

    def test_nav_starts_near_initial(self, bt_config, bt_data):
        """First NAV should be close to initial_nav (one day of return)."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        initial = bt_config["backtest"]["initial_nav"]
        assert abs(result["nav"].iloc[0] - initial) / initial < 0.1

    def test_regime_labels_valid(self, bt_config, bt_data):
        """All regime labels should be valid strings."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        valid = {"RISK_ON", "NEUTRAL", "RISK_OFF"}
        assert set(result["regime_hmm"].unique()).issubset(valid)
        assert set(result["regime_composite"].unique()).issubset(valid)

    def test_cost_only_on_rebalance(self, bt_config, bt_data):
        """Transaction cost should be 0 on non-rebalance days."""
        prices, returns = bt_data
        bt = WalkForwardBacktester(bt_config)
        result = bt.run(prices, returns)
        no_turnover = result[result["turnover"] == 0.0]
        if len(no_turnover) > 0:
            assert (no_turnover["cost"] == 0.0).all()
