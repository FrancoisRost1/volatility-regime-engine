"""
Data loader for volatility-regime-engine.

Fetches adjusted close prices from yfinance, computes log returns,
and validates data quality. All downstream modules consume the
clean DataFrames produced here.

Financial rationale: using adjusted close accounts for dividends and
splits. Log returns are additive across time, making them natural for
rolling statistics and HMM feature construction.

Simplifying assumption: VIX has no adjusted close, regular close is used.
PDBC history starts ~2014; engine auto-adjusts start date if needed.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_prices(config: dict) -> pd.DataFrame:
    """
    Fetch adjusted close prices for all tickers from yfinance.

    Downloads allocation tickers, signal tickers, and IEF (credit proxy).
    Forward-fills up to max_ffill_days for minor gaps in illiquid assets.
    Raises ValueError if any ticker exceeds the max_missing_fraction threshold.

    Parameters
    ----------
    config : dict
        Full configuration dictionary from config_loader.

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices, indexed by date, columns = tickers.
    """
    data_cfg = config["data"]
    all_tickers = data_cfg["all_tickers"] + [data_cfg["ief_ticker"]]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"]

    logger.info("Fetching prices for %s from %s to %s", all_tickers, start, end)

    raw = yf.download(
        tickers=all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty DataFrame for {all_tickers}")

    # Extract close prices, yfinance returns MultiIndex columns for multi-ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    # VIX has no adjusted close, already handled by using Close above
    # Drop rows where ALL prices are NaN
    prices = prices.dropna(how="all")

    # Forward-fill minor gaps (illiquid assets like GLD, PDBC)
    prices = prices.ffill(limit=data_cfg["max_ffill_days"])

    # Auto-adjust start date: trim to first date where ALL tickers have data.
    # Simplifying assumption: PDBC launched ~2014; rather than excluding it,
    # we shorten the history so the full asset universe is available throughout.
    first_valid = prices.apply(lambda col: col.first_valid_index())
    common_start = first_valid.max()
    if common_start > prices.index[0]:
        logger.info(
            "Trimming start to %s (latest first-valid across tickers)",
            common_start.strftime("%Y-%m-%d"),
        )
        prices = prices.loc[common_start:]

    # Validate missing data threshold
    max_missing = data_cfg["max_missing_fraction"]
    for ticker in prices.columns:
        missing_frac = prices[ticker].isna().sum() / len(prices)
        if missing_frac > max_missing:
            raise ValueError(
                f"{ticker} has {missing_frac:.1%} missing rows "
                f"(threshold: {max_missing:.1%}). Check data availability."
            )

    # Drop any remaining rows with NaN after validation
    prices = prices.dropna()

    logger.info("Loaded %d rows × %d tickers", len(prices), len(prices.columns))
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: ln(P_t / P_{t-1}).

    First row is NaN and is dropped before returning.

    Financial rationale: log returns are time-additive, making cumulative
    return calculation and rolling statistics mathematically cleaner than
    simple returns. Used for features and vol estimation only, NAV
    compounding uses simple returns via compute_simple_returns().

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Daily log returns, same columns as prices, first row dropped.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError("Need at least 2 price rows to compute returns.")
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.iloc[1:]


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple (arithmetic) returns: P_t / P_{t-1} - 1.

    First row is NaN and is dropped before returning.

    Financial rationale: simple returns are the correct basis for
    portfolio PnL compounding, NAV(t+1) = NAV(t) * (1 + w'r_simple).
    Log returns are NOT additive across assets in a weighted portfolio.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Daily simple returns, same columns as prices, first row dropped.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError("Need at least 2 price rows to compute returns.")
    simple_ret = prices / prices.shift(1) - 1
    return simple_ret.iloc[1:]
