"""
Main orchestrator for volatility-regime-engine.

No logic here — all computation lives in engine/ modules.
This file wires together the pipeline: load config, fetch data,
run backtest, compute analytics, save outputs.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config_loader import load_config
from engine.data_loader import load_prices, compute_log_returns
from engine.backtester import WalkForwardBacktester
from engine.benchmarks import run_benchmarks, _run_buy_hold
from engine.analytics import (
    compute_overall_metrics,
    compute_benchmark_attribution,
)
from engine.analytics_regime import (
    compute_regime_conditional_metrics,
    compute_regime_duration_stats,
    compute_transition_matrix,
    compute_stress_period_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Run the full volatility-regime-engine pipeline.

    Steps:
      1. Load configuration
      2. Fetch and validate price data
      3. Run walk-forward backtest (HMM + composite + vol targeting)
      4. Run benchmark strategies
      5. Compute analytics (overall, regime-conditional, stress, attribution)
      6. Save outputs to data/processed/ and outputs/
    """
    # --- Step 1: Config ---
    config = load_config()
    logger.info("Config loaded.")

    # --- Step 2: Data ---
    prices = load_prices(config)
    returns = compute_log_returns(prices)
    logger.info("Data loaded: %d days, %d tickers", len(returns), len(returns.columns))

    # --- Step 3: Backtest ---
    backtester = WalkForwardBacktester(config)
    results = backtester.run(prices, returns)
    logger.info("Backtest complete: %d trading days", len(results))

    # --- Step 4: Benchmarks ---
    regime_series = results["regime_hmm"].values
    regime_index = results.index
    benchmarks = run_benchmarks(returns, regime_series, regime_index, config)
    logger.info("Benchmarks computed: %s", list(benchmarks.keys()))

    # --- Step 5: Analytics ---
    nav = results["nav"]
    daily_ret = results["daily_return"]
    spy_ret = returns["SPY"].reindex(results.index).fillna(0)
    initial_nav = config["backtest"]["initial_nav"]

    # SPY NAV for stress comparison
    spy_nav = _run_buy_hold(returns, "SPY", initial_nav).reindex(results.index)

    overall = compute_overall_metrics(nav, daily_ret, spy_ret, config)
    logger.info("Overall: CAGR=%.2f%%, Sharpe=%.2f, MaxDD=%.1f%%",
                overall["cagr"] * 100, overall["sharpe"],
                overall["max_drawdown"] * 100)

    regime_cond = compute_regime_conditional_metrics(
        daily_ret, results["regime_hmm"], config
    )
    logger.info("Regime-conditional metrics:\n%s", regime_cond)

    duration_stats = compute_regime_duration_stats(results["regime_hmm"])
    transition = compute_transition_matrix(results["regime_hmm"])

    stress = compute_stress_period_metrics(nav, daily_ret, spy_nav, spy_ret, config)

    # Benchmark overall metrics
    bench_metrics = {}
    for bname, bnav in benchmarks.items():
        bnav_aligned = bnav.reindex(results.index).dropna()
        if len(bnav_aligned) < 2:
            continue
        bret = bnav_aligned.pct_change().dropna()
        bench_metrics[bname] = compute_overall_metrics(
            bnav_aligned, bret, spy_ret, config
        )

    attribution = compute_benchmark_attribution(overall, bench_metrics)
    logger.info("Attribution table:\n%s", attribution)

    # --- Step 6: Save outputs ---
    _ensure_dirs(config)

    results.to_csv(config["outputs"]["tearsheet_path"])
    results[["regime_hmm", "regime_composite"]].to_csv(
        config["outputs"]["regime_labels_path"]
    )
    results[["nav"]].to_csv(config["outputs"]["nav_series_path"])
    weight_cols = [c for c in results.columns if c.startswith("w_")]
    results[weight_cols].to_csv(config["outputs"]["weights_path"])

    logger.info("Outputs saved. Pipeline complete.")

    return results, overall, regime_cond, stress, attribution


def _ensure_dirs(config: dict) -> None:
    """
    Create output directories if they don't exist.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    """
    for key in ["tearsheet_path", "regime_labels_path", "nav_series_path",
                "weights_path"]:
        path = Path(config["outputs"][key])
        path.parent.mkdir(parents=True, exist_ok=True)

    processed = Path(config["data"]["processed_path"])
    processed.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
