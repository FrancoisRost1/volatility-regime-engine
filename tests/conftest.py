"""
Shared fixtures for volatility-regime-engine tests.

All data is synthetic — no yfinance calls. Fixtures provide realistic-looking
price, return, and feature DataFrames that respect the structure expected by
each engine module.
"""

import numpy as np
import pandas as pd
import pytest

from utils.config_loader import load_config


@pytest.fixture
def config():
    """Load the real config.yaml so tests use production parameters."""
    return load_config()


@pytest.fixture
def synthetic_dates():
    """600 business days starting 2018-01-02."""
    return pd.bdate_range("2018-01-02", periods=600, freq="B")


def _gbm_prices(dates, start=100.0, mu=0.0003, sigma=0.01, seed=42):
    """Generate a synthetic GBM price series."""
    rng = np.random.RandomState(seed)
    log_ret = mu + sigma * rng.randn(len(dates))
    log_ret[0] = 0.0
    return pd.Series(start * np.exp(np.cumsum(log_ret)), index=dates)


@pytest.fixture
def synthetic_prices(synthetic_dates):
    """
    DataFrame of synthetic prices for the full ticker universe:
    SPY, TLT, GLD, PDBC, LQD, ^VIX, IEF.
    """
    dates = synthetic_dates
    prices = pd.DataFrame(index=dates)
    prices["SPY"] = _gbm_prices(dates, 250, 0.0003, 0.012, seed=1)
    prices["TLT"] = _gbm_prices(dates, 120, 0.0001, 0.008, seed=2)
    prices["GLD"] = _gbm_prices(dates, 160, 0.0002, 0.009, seed=3)
    prices["PDBC"] = _gbm_prices(dates, 18, 0.0001, 0.011, seed=4)
    prices["LQD"] = _gbm_prices(dates, 115, 0.00005, 0.004, seed=5)
    prices["^VIX"] = 15 + 5 * np.abs(np.sin(np.linspace(0, 12, len(dates))))
    prices["IEF"] = _gbm_prices(dates, 105, 0.00008, 0.005, seed=6)
    return prices


@pytest.fixture
def synthetic_returns(synthetic_prices):
    """Log returns from synthetic prices (first row dropped)."""
    from engine.data_loader import compute_log_returns
    return compute_log_returns(synthetic_prices)


@pytest.fixture
def synthetic_alloc_returns(synthetic_returns):
    """Returns for allocation assets only: SPY, TLT, GLD, PDBC."""
    return synthetic_returns[["SPY", "TLT", "GLD", "PDBC"]]


@pytest.fixture
def synthetic_features(synthetic_prices, synthetic_returns, config):
    """Raw feature matrix built from synthetic data."""
    from engine.feature_builder import build_features
    return build_features(synthetic_prices, synthetic_returns, config)
