"""
Tests for engine/vol_estimator.py.

Covers: EWMA vol, realized vol, blended vol, covariance matrix,
portfolio vol, edge cases (zero returns, NaN).
"""

import numpy as np
import pandas as pd
import pytest

from engine.vol_estimator import (
    compute_ewma_vol,
    compute_realized_vol,
    compute_blended_vol,
    compute_ewma_covariance,
    compute_portfolio_vol,
)


class TestComputeEwmaVol:
    """Tests for compute_ewma_vol."""

    def test_output_length(self):
        """EWMA vol series has same length as input."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        ret = pd.Series(np.random.randn(100) * 0.01, index=dates)
        vol = compute_ewma_vol(ret, lam=0.94, annualization=252)
        assert len(vol) == 100

    def test_nonnegative(self):
        """Volatility is always non-negative."""
        dates = pd.bdate_range("2020-01-01", periods=100)
        ret = pd.Series(np.random.randn(100) * 0.01, index=dates)
        vol = compute_ewma_vol(ret, lam=0.94, annualization=252)
        assert (vol.dropna() >= 0).all()

    def test_zero_returns_give_zero_vol(self):
        """All-zero returns produce zero EWMA vol."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        ret = pd.Series(0.0, index=dates)
        vol = compute_ewma_vol(ret, lam=0.94, annualization=252)
        np.testing.assert_array_almost_equal(vol.values, 0.0)

    def test_higher_vol_input_gives_higher_estimate(self):
        """Higher return variance should produce higher EWMA vol."""
        dates = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.RandomState(42)
        low_vol = pd.Series(rng.randn(200) * 0.005, index=dates)
        high_vol = pd.Series(rng.randn(200) * 0.03, index=dates)
        vol_low = compute_ewma_vol(low_vol, 0.94, 252)
        vol_high = compute_ewma_vol(high_vol, 0.94, 252)
        assert vol_high.iloc[-1] > vol_low.iloc[-1]


class TestComputeRealizedVol:
    """Tests for compute_realized_vol."""

    def test_output_length(self):
        """Output has same length as input (with NaNs for warmup)."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        ret = pd.Series(np.random.randn(50) * 0.01, index=dates)
        vol = compute_realized_vol(ret, window=21, annualization=252)
        assert len(vol) == 50

    def test_warmup_nans(self):
        """First (window-1) values should be NaN."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        ret = pd.Series(np.random.randn(50) * 0.01, index=dates)
        vol = compute_realized_vol(ret, window=21, annualization=252)
        assert vol.iloc[:20].isna().all()
        assert vol.iloc[20:].notna().all()


class TestComputeBlendedVol:
    """Tests for compute_blended_vol."""

    def test_output_columns_match_input(self, synthetic_alloc_returns, config):
        """Blended vol has same columns as input returns."""
        vol = compute_blended_vol(synthetic_alloc_returns, config)
        assert vol.columns.tolist() == synthetic_alloc_returns.columns.tolist()

    def test_vol_floor_enforced(self, config):
        """Blended vol never drops below min_vol_floor."""
        floor = config["vol_estimation"]["min_vol_floor"]
        dates = pd.bdate_range("2020-01-01", periods=100)
        # Near-zero returns
        ret = pd.DataFrame({"SPY": np.zeros(100), "TLT": np.zeros(100),
                            "GLD": np.zeros(100), "PDBC": np.zeros(100)},
                           index=dates)
        vol = compute_blended_vol(ret, config)
        assert (vol.dropna() >= floor - 1e-10).all().all()


class TestComputeEwmaCovariance:
    """Tests for compute_ewma_covariance."""

    def test_output_is_dict(self, synthetic_alloc_returns, config):
        """Returns a dict mapping row index to covariance matrix."""
        cov = compute_ewma_covariance(synthetic_alloc_returns.iloc[:50], config)
        assert isinstance(cov, dict)
        assert 0 in cov

    def test_matrix_shape(self, synthetic_alloc_returns, config):
        """Each cov matrix is n_assets x n_assets."""
        n = synthetic_alloc_returns.shape[1]
        cov = compute_ewma_covariance(synthetic_alloc_returns.iloc[:50], config)
        assert cov[0].shape == (n, n)

    def test_matrix_symmetric(self, synthetic_alloc_returns, config):
        """Covariance matrices should be symmetric."""
        cov = compute_ewma_covariance(synthetic_alloc_returns.iloc[:50], config)
        for t, mat in cov.items():
            np.testing.assert_array_almost_equal(mat, mat.T)

    def test_diagonal_nonnegative(self, synthetic_alloc_returns, config):
        """Diagonal (variances) should be non-negative."""
        cov = compute_ewma_covariance(synthetic_alloc_returns.iloc[:50], config)
        for t, mat in cov.items():
            assert (np.diag(mat) >= 0).all()


class TestComputePortfolioVol:
    """Tests for compute_portfolio_vol."""

    def test_equal_weights_identity_cov(self):
        """Equal weights with identity cov = sqrt(sum(w^2))."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        cov = np.eye(4)
        pvol = compute_portfolio_vol(w, cov)
        expected = np.sqrt(w @ cov @ w)
        np.testing.assert_almost_equal(pvol, expected)

    def test_zero_weights_give_zero_vol(self):
        """All-zero weights should give NaN (0 variance, guarded)."""
        w = np.zeros(4)
        cov = np.eye(4)
        pvol = compute_portfolio_vol(w, cov)
        # 0 weights -> 0 variance -> 0 vol (sqrt(0) = 0, but guarded as nan)
        # Implementation returns nan for port_var <= 0
        assert pvol == 0.0 or np.isnan(pvol)

    def test_single_asset_full_weight(self):
        """100% in one asset: portfolio vol = that asset's vol."""
        w = np.array([1.0, 0.0, 0.0, 0.0])
        cov = np.diag([0.04, 0.01, 0.02, 0.03])  # variances
        pvol = compute_portfolio_vol(w, cov)
        np.testing.assert_almost_equal(pvol, 0.2)  # sqrt(0.04)
