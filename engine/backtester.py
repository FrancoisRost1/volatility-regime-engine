"""
Walk-forward backtesting engine for volatility-regime-engine.

Implements a strict expanding-window walk-forward backtest with:
  - Signal at t, trade at t+1, return credited t+1->t+2 (no lookahead)
  - Turnover-based transaction costs on rebalance days only
  - HMM warmup period (504 days) before first trade
  - Annual HMM refits on expanding window
  - Persistence-filtered regime labels

Financial rationale: the walk-forward design with expanding windows
and one-day execution lag mirrors how an institutional allocator would
deploy this strategy live. Every design choice guards against
lookahead bias.
"""

import logging

import numpy as np
import pandas as pd

from engine.feature_builder import build_features, z_score_features
from engine.regime_detector import (
    HMMRegimeDetector,
    CompositeRegimeDetector,
    apply_persistence_filter,
)
from engine.vol_estimator import (
    compute_blended_vol,
    compute_ewma_covariance,
    compute_portfolio_vol,
)
from engine.position_sizer import compute_weights
from engine.rebalancer import (
    should_rebalance,
    compute_transaction_cost,
    compute_turnover,
)

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Expanding-window walk-forward backtester.

    Timeline:
      t=0 to t=warmup-1:  Warmup — HMM not yet fit, no trades.
      t=warmup:            First HMM fit. First signal generated.
      Every refit_every:   HMM refit on expanding window [0..t].
      Every day:           Features, regime, rebalance check, NAV update.

    Trade execution: signal at close of t, trade at close of t+1,
    return credited from close t+1 to close t+2.
    """

    def __init__(self, config: dict):
        """
        Initialize backtester with full configuration.

        Parameters
        ----------
        config : dict
            Full configuration dictionary.
        """
        self.config = config
        self.hmm_cfg = config["hmm"]
        self.warmup = self.hmm_cfg["warmup_days"]
        self.refit_every = self.hmm_cfg["refit_every_days"]
        self.persist_days = config["persistence"]["min_confirmation_days"]
        self.initial_nav = config["backtest"]["initial_nav"]
        self.alloc_tickers = config["data"]["allocation_tickers"]
        self.n_assets = len(self.alloc_tickers)

    def run(self, prices: pd.DataFrame,
            returns: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices for all tickers.
        returns : pd.DataFrame
            Log returns for all tickers (used for features/vol only).

        Returns
        -------
        pd.DataFrame
            Daily output with regime labels, weights, NAV, costs, etc.
        """
        if len(returns) < self.warmup + 3:
            raise ValueError(
                f"Need at least {self.warmup + 3} return rows, got {len(returns)}."
            )

        # Simple returns for NAV compounding (log returns are wrong for weighted PnL)
        simple_ret = prices / prices.shift(1) - 1
        simple_ret = simple_ret.iloc[1:]  # drop first NaN row

        features_raw = build_features(prices, returns, self.config)
        alloc_ret_log = returns[self.alloc_tickers].reindex(features_raw.index)
        alloc_ret_simple = simple_ret[self.alloc_tickers].reindex(features_raw.index)

        # Pre-compute vol estimates and covariance matrices (use log returns — correct for risk)
        vol_est = compute_blended_vol(alloc_ret_log, self.config)
        cov_dict = compute_ewma_covariance(alloc_ret_log, self.config)

        # Build valid-feature mask: only rows where ALL features are non-NaN
        # Critical: do NOT zero-fill NaN features before HMM training
        feature_valid_mask = features_raw.notna().all(axis=1)

        # Initialize detectors
        hmm = HMMRegimeDetector(self.config)
        composite = CompositeRegimeDetector(self.config)

        # Composite regimes (no warmup needed)
        raw_composite = composite.predict(features_raw, prices)
        filtered_composite = apply_persistence_filter(
            raw_composite, self.persist_days
        )

        # Walk-forward HMM
        n_days = len(features_raw)
        feature_names = list(features_raw.columns)
        hmm_regimes_raw = np.full(n_days, "", dtype=object)
        hmm_probs = pd.DataFrame(
            np.nan, index=features_raw.index,
            columns=["p_risk_on", "p_neutral", "p_risk_off"],
        )

        refit_dates = list(range(self.warmup, n_days, self.refit_every))
        if len(refit_dates) == 0:
            raise ValueError(
                f"No refit dates generated. warmup={self.warmup}, n_days={n_days}."
            )
        if refit_dates[0] != self.warmup:
            refit_dates.insert(0, self.warmup)

        for r_idx, refit_start in enumerate(refit_dates):
            # Determine prediction range: refit_start .. next_refit-1 or end
            pred_end = (
                refit_dates[r_idx + 1] if r_idx + 1 < len(refit_dates)
                else n_days
            )

            # Z-score features using data up to refit point only
            scaled = z_score_features(features_raw, refit_start)

            # Train only on rows where all features are valid (no NaN/zero-fill)
            train_slice = scaled.iloc[: refit_start + 1]
            valid_train = train_slice[feature_valid_mask.iloc[: refit_start + 1]]
            if len(valid_train) < 50:
                logger.warning(
                    "Only %d valid training rows at refit idx %d, skipping refit",
                    len(valid_train), refit_start,
                )
                continue
            hmm.fit(valid_train.values, feature_names)

            # Predict filtered for the full sequence up to pred_end
            # Use only valid rows for prediction sequence
            full_seq_scaled = scaled.iloc[: pred_end]
            valid_pred_mask = feature_valid_mask.iloc[: pred_end]
            valid_indices = np.where(valid_pred_mask.values)[0]

            if len(valid_indices) == 0:
                continue

            valid_seq = full_seq_scaled.iloc[valid_indices].values
            labels = hmm.predict_filtered(valid_seq)
            probs = hmm.get_state_probabilities(valid_seq)

            # Map predictions back to original indices
            for j, orig_idx in enumerate(valid_indices):
                if refit_start <= orig_idx < pred_end:
                    hmm_regimes_raw[orig_idx] = labels[j]
                    hmm_probs.iloc[orig_idx] = probs.iloc[j].values

            logger.info(
                "HMM refit at idx %d, predicting %d-%d",
                refit_start, refit_start, pred_end - 1,
            )

        # Apply persistence filter to HMM regimes (warmup onwards)
        hmm_filtered = hmm_regimes_raw.copy()
        hmm_filtered[self.warmup:] = apply_persistence_filter(
            hmm_regimes_raw[self.warmup:], self.persist_days
        )

        # --- NAV simulation with t+1 execution, t+2 return credit ---
        # Signal at close t -> trade executed at close t+1 -> return from t+1 to t+2
        dates = features_raw.index
        records = []
        nav = self.initial_nav
        w_current = np.zeros(self.n_assets)
        regime_last_rebal = ""
        vol_last_rebal = np.nan
        lag = self.config["backtest"]["signal_to_execution_lag_days"]

        for t in range(self.warmup, n_days - lag - 1):
            # t = signal day (close), t+lag = execution day, t+lag+1 = return day
            t_exec = t + lag
            t_return = t_exec + 1  # return credited from close t+lag to close t+lag+1

            if t_return >= n_days:
                break

            regime_t = hmm_filtered[t]
            if regime_t == "":
                continue  # no valid regime prediction

            vol_vec = vol_est.iloc[t].values.astype(float)
            cov_mat = cov_dict.get(t)

            if cov_mat is None:
                continue

            # Compute portfolio vol with CURRENT weights (pre-trade)
            port_vol_pre = compute_portfolio_vol(w_current, cov_mat)

            # Check rebalance (uses persistence-filtered regime)
            rebal, reason = should_rebalance(
                regime_t, regime_last_rebal, port_vol_pre, vol_last_rebal,
                self.config,
            )

            turnover = 0.0
            cost = 0.0
            did_rebalance = False

            if rebal or regime_last_rebal == "":
                w_new = compute_weights(regime_t, vol_vec, cov_mat, self.config)
                turnover = compute_turnover(w_new, w_current)
                cost = compute_transaction_cost(w_new, w_current, self.config)
                w_current = w_new
                regime_last_rebal = regime_t
                vol_last_rebal = compute_portfolio_vol(w_current, cov_mat)
                did_rebalance = True

            # Use SIMPLE returns for PnL (not log returns)
            day_ret = alloc_ret_simple.iloc[t_return].values.astype(float)
            port_ret = float(w_current @ day_ret)
            nav = nav * (1 + port_ret) - cost * nav

            # Record post-trade portfolio vol for analytics
            port_vol_post = compute_portfolio_vol(w_current, cov_mat)

            records.append({
                "date": dates[t_return],
                "regime_hmm": hmm_filtered[t],
                "regime_composite": filtered_composite[t],
                **{f"w_{tk}": w_current[i]
                   for i, tk in enumerate(self.alloc_tickers)},
                "gross_exposure": float(np.abs(w_current).sum()),
                "target_vol": self.config["position_sizing"]["target_vol"].get(
                    regime_t, np.nan
                ),
                "realized_port_vol": port_vol_post,
                "nav": nav,
                "daily_return": port_ret,
                "rebalanced": did_rebalance,
                "turnover": turnover,
                "cost": cost,
                "p_risk_on": hmm_probs.iloc[t]["p_risk_on"],
                "p_neutral": hmm_probs.iloc[t]["p_neutral"],
                "p_risk_off": hmm_probs.iloc[t]["p_risk_off"],
            })

        if not records:
            raise ValueError("Backtest produced no records. Check warmup vs data length.")

        return pd.DataFrame(records).set_index("date")
