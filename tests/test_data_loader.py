"""
Tests for engine/data_loader.py.

Uses synthetic data only — no yfinance calls.
Covers: compute_log_returns happy path, NaN propagation, single-column input.
"""

import numpy as np
import pandas as pd
import pytest

from engine.data_loader import compute_log_returns


class TestComputeLogReturns:
    """Tests for compute_log_returns."""

    def test_happy_path(self, synthetic_prices):
        """Log returns have correct shape and no NaN in output."""
        ret = compute_log_returns(synthetic_prices)
        assert len(ret) == len(synthetic_prices) - 1
        assert ret.columns.tolist() == synthetic_prices.columns.tolist()
        # First row was NaN and should be dropped
        assert not ret.iloc[0].isna().any()

    def test_log_return_values(self):
        """Verify log return formula: ln(P_t / P_{t-1})."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"A": [100.0, 110.0, 105.0]}, index=dates)
        ret = compute_log_returns(prices)
        expected_0 = np.log(110 / 100)
        expected_1 = np.log(105 / 110)
        np.testing.assert_almost_equal(ret["A"].iloc[0], expected_0, decimal=10)
        np.testing.assert_almost_equal(ret["A"].iloc[1], expected_1, decimal=10)

    def test_single_column(self):
        """Works with a single-asset DataFrame."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"SPY": [100, 101, 102, 103, 104.0]}, index=dates)
        ret = compute_log_returns(prices)
        assert ret.shape == (4, 1)
        assert (ret > 0).all().all()

    def test_constant_prices_give_zero_returns(self):
        """Constant prices produce zero log returns."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"A": [50.0] * 5}, index=dates)
        ret = compute_log_returns(prices)
        np.testing.assert_array_almost_equal(ret["A"].values, 0.0)

    def test_nan_in_prices_propagates(self):
        """NaN in prices propagates to returns (not silently dropped)."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"A": [100, 101, np.nan, 103, 104.0]}, index=dates)
        ret = compute_log_returns(prices)
        # Row where NaN appeared and the row after should have NaN
        assert ret["A"].isna().sum() >= 1

    def test_output_index_matches(self, synthetic_prices):
        """Return index is prices index minus first row."""
        ret = compute_log_returns(synthetic_prices)
        assert ret.index[0] == synthetic_prices.index[1]
        assert ret.index[-1] == synthetic_prices.index[-1]
