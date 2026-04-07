# CLAUDE.md — volatility-regime-engine

> Local source of truth for this project. Read this fully before writing any code.
> Master CLAUDE.md lives in CODE/ root. This file contains full detail for this project only.

---

## Project overview

A two-layer systematic allocation engine:

**Layer 1 — Regime Detection:** Two parallel regime classifiers run simultaneously:
- `HMMRegimeDetector`: Gaussian Hidden Markov Model (3 states, full covariance) trained on a multi-signal feature vector. Uses **filtered probabilities only** (forward algorithm) — never the Viterbi smoother — to guarantee no future information enters the signal.
- `CompositeRegimeDetector`: Transparent rule-based classifier built from interpretable signals (trend, vol stress, drawdown). Serves as both a sanity check and a benchmark for the HMM.

Both classifiers output one of 3 states: **RISK_ON / NEUTRAL / RISK_OFF**.

**Layer 2 — Volatility Targeting:** Given today's regime and a vol estimate (EWMA + realized blend), the engine computes inverse-vol weights per asset, scales portfolio exposure to hit the target volatility, and applies regime-dependent leverage caps.

**Rebalancing:** Signal-driven — only rebalances when regime changes (with a 3-day persistence filter) or when portfolio vol drifts >20% from the last rebalance level.

**Backtesting:** Walk-forward with expanding window. Signal formed at t, trade executed at t+1. Transaction costs based on realized turnover.

---

## Repo structure

```
volatility-regime-engine/
├── main.py                          # Orchestrator only — no logic here
├── config.yaml                      # All parameters — never hardcode numbers
├── CLAUDE.md                        # This file
│
├── engine/
│   ├── data_loader.py               # yfinance fetch, adj close, log returns, validation
│   ├── feature_builder.py           # 8-feature vector + z-scoring (expanding window)
│   ├── regime_detector.py           # HMMRegimeDetector + CompositeRegimeDetector
│   ├── vol_estimator.py             # EWMA vol per asset + EWMA covariance matrix
│   ├── position_sizer.py            # Inverse-vol weights + leverage cap + regime allocation
│   ├── rebalancer.py                # Trigger logic (regime change + vol drift) + cost calc
│   ├── backtester.py                # Walk-forward loop, NAV, P&L, trade log
│   └── analytics.py                 # Full tearsheet: overall + regime-conditional + attribution
│
├── app/
│   └── streamlit_app.py             # Bloomberg dark mode, 5 tabs
│
├── data/
│   ├── raw/                         # Raw OHLCV from yfinance (cached)
│   └── processed/                   # Feature matrix, regime labels, NAV series
│
├── outputs/
│   └── tearsheet.csv                # Full daily output
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_builder.py
│   ├── test_regime_detector.py
│   ├── test_vol_estimator.py
│   ├── test_position_sizer.py
│   ├── test_rebalancer.py
│   ├── test_backtester.py
│   └── test_analytics.py
│
├── docs/
│   └── analysis.md
└── README.md
```

---

## Asset universe

| Asset | Ticker | Role |
|---|---|---|
| US Equities | SPY | Primary risk asset |
| Long Bonds | TLT | Safe haven / duration |
| Gold | GLD | Inflation hedge / crisis asset |
| Commodities | PDBC | Commodity cycle exposure |
| Credit stress proxy | LQD | Input to regime signal only (not allocated) |

**Note on LQD:** Used only as a regime feature input. It is NOT included in the allocation portfolio. It serves as a credit stress proxy — specifically, LQD's rolling return relative to IEF (intermediate treasuries) captures IG credit risk appetite. This is explicitly a *proxy*, not a true OAS credit spread. Documented as a simplifying assumption.

**VIX:** Fetched via `^VIX` from yfinance. Used as a regime feature input only. Not allocatable.

**Allocation assets:** SPY, TLT, GLD, PDBC (4 assets).

---

## Data pipeline — `data_loader.py`

```python
def load_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches adjusted close prices from yfinance for all tickers.
    Returns a DataFrame indexed by date, columns = tickers.
    Drops rows where ALL prices are NaN. Forward-fills up to 2 days for
    illiquid assets (GLD, PDBC) to handle minor gaps.
    Raises ValueError if any ticker has >5% missing rows after fill.
    """

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns: ln(P_t / P_{t-1}).
    First row will be NaN — drop before use.
    """
```

**Edge cases:**
- VIX has no "adjusted close" — use regular close column.
- PDBC may have limited history (launched 2014). Start date automatically adjusted if data is unavailable before that.
- If yfinance returns an empty DataFrame, raise a clear error with the ticker name.

---

## Feature engineering — `feature_builder.py`

8 features computed daily, used as inputs to regime classifiers.

```
Feature 1:  realized_vol_21d     = rolling 21-day std of SPY log returns × √252
Feature 2:  realized_vol_63d     = rolling 63-day std of SPY log returns × √252
Feature 3:  return_21d           = SPY cumulative log return over past 21 days
Feature 4:  skew_63d             = rolling 63-day skewness of SPY log returns
Feature 5:  spy_vs_sma200        = SPY price / SPY 200-day SMA − 1
Feature 6:  credit_stress_proxy  = LQD 21d return − IEF 21d return
                                   (captures IG credit vs rates divergence)
Feature 7:  vix_level            = raw VIX close
Feature 8:  vix_momentum         = VIX / VIX 21-day SMA − 1
```

**Z-scoring (critical for HMM):**
On each expanding window refit date, compute mean and std of each feature from t=0 to t_refit. Apply that scaling to all subsequent observations until the next refit. Never use future data to compute scaling parameters.

```python
def build_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the 8-feature matrix from prices and returns.
    Returns raw (unscaled) features. Z-scoring is applied separately
    inside the HMM fitting loop to respect the expanding window constraint.
    """

def z_score_features(features: pd.DataFrame, fit_end_idx: int) -> pd.DataFrame:
    """
    Z-scores each feature column using mean/std computed only on rows
    0..fit_end_idx (inclusive). Returns scaled DataFrame for full history.
    Financial rationale: prevents future distributional info from leaking
    into regime classification.
    """
```

---

## Regime detection — `regime_detector.py`

### Class 1: HMMRegimeDetector

```python
class HMMRegimeDetector:
    """
    Gaussian HMM with 3 hidden states trained on the 8-feature vector.
    Uses filtered (forward-algorithm) probabilities only — no smoothing.
    This guarantees regime classification at date t uses only data up to t.

    States are labeled post-hoc by sorting emission means of realized_vol_21d:
        lowest vol mean  → RISK_ON
        middle vol mean  → NEUTRAL
        highest vol mean → RISK_OFF

    This labeling is deterministic and robust across refits because it ties
    state identity to an observable financial quantity (volatility), not to
    an arbitrary state index that can flip between fits.

    Simplifying assumption: state label ordering by vol is assumed stable.
    In pathological market conditions, a low-vol state could theoretically
    coincide with stress (e.g., pre-crash calm). This is documented.
    """

    def fit(self, features_scaled: np.ndarray) -> None:
        """Fits GaussianHMM(n_components=3, covariance_type='full', n_iter=100)."""

    def predict_filtered(self, features_scaled: np.ndarray) -> np.ndarray:
        """
        Returns regime label array using ONLY filtered probabilities.
        Implementation: run score_samples() incrementally or use predict()
        which internally uses the Viterbi algorithm — CAUTION.
        Correct approach: use model.predict_proba() on expanding prefix
        to simulate live filtered inference.
        Returns: array of strings ['RISK_ON', 'NEUTRAL', 'RISK_OFF']
        """

    def get_state_probabilities(self, features_scaled: np.ndarray) -> pd.DataFrame:
        """Returns DataFrame with columns [p_risk_on, p_neutral, p_risk_off] per date."""
```

**HMM refit schedule:**
- Warmup: first 504 trading days (~2 years). No predictions made during warmup.
- Refit: every 252 trading days thereafter, using ALL data from t=0 to t_refit (expanding).
- After each refit, re-label states using vol mean ordering.

**Filtered vs smoothed (critical implementation note):**
`hmmlearn`'s `.predict()` uses Viterbi — this is a smoothed estimate and uses future observations. To get filtered probabilities, use `.predict_proba()` applied to incrementally growing sequences, or implement the forward algorithm manually. The backtester must call filtered inference only.

---

### Class 2: CompositeRegimeDetector

```python
class CompositeRegimeDetector:
    """
    Transparent rule-based regime classifier. Uses 4 interpretable signals.
    Each signal votes +1 (supportive) or -1 (stress). Total score → regime.

    Signal 1 — Trend:         SPY > 200-day SMA → +1, else −1
    Signal 2 — Vol stress:    21d realized vol < 63d realized vol → +1, else −1
                              (vol mean-reversion: short < long = calming)
    Signal 3 — Drawdown:      SPY drawdown from 252d high < 10% → +1, else −1
    Signal 4 — Credit stress: credit_stress_proxy (LQD vs IEF) > −0.5% → +1, else −1

    Score = sum of 4 signals ∈ {−4, −2, 0, +2, +4}
    RISK_ON:   score ≥ 2
    NEUTRAL:   score == 0
    RISK_OFF:  score ≤ −2

    Financial rationale: each signal proxies a distinct risk dimension
    (price trend, vol regime, depth of correction, credit conditions).
    Thresholds are transparent and defensible in an interview.
    """

    def predict(self, features: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
        """Returns array of regime labels ['RISK_ON', 'NEUTRAL', 'RISK_OFF']."""
```

---

### Regime persistence filter (applied to BOTH classifiers)

```python
def apply_persistence_filter(regimes: np.ndarray, min_days: int = 3) -> np.ndarray:
    """
    Prevents acting on transient regime flips.
    A regime change is only recognized if the new regime persists for
    >= min_days consecutive days.
    Until confirmed, carry forward the previous regime.

    Financial rationale: short-lived regime flips generate spurious turnover
    and are often noise rather than genuine state transitions. AQR and similar
    vol-targeting funds use confirmation filters of 1–5 days.
    """
```

---

## Volatility estimation — `vol_estimator.py`

### Per-asset vol estimate

```python
# Realized vol
σ_realized(i,t) = rolling_std(log_returns[i], window=21) × √252

# EWMA vol (RiskMetrics, λ=0.94)
# Recursive formula:
σ²_ewma(i,t) = λ × σ²_ewma(i,t-1) + (1−λ) × r(i,t)²
σ_ewma(i,t)  = √(σ²_ewma(i,t)) × √252

# Blend
σ_est(i,t) = 0.5 × σ_realized(i,t) + 0.5 × σ_ewma(i,t)
```

### EWMA covariance matrix (for portfolio vol calculation)

```python
# EWMA covariance between assets i and j:
Σ_ij(t) = λ × Σ_ij(t-1) + (1−λ) × r_i(t) × r_j(t)
# Full matrix Σ(t) is n_assets × n_assets (4×4 for SPY, TLT, GLD, PDBC)
# Annualized by × 252

def compute_portfolio_vol(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    σ_port = √(wᵀ Σ w)
    Used for: (1) rebalance trigger check, (2) tearsheet realized vol tracking.
    """
```

**Edge cases:**
- If σ_est(i,t) == 0 or NaN → set weight for asset i to 0, redistribute proportionally.
- If covariance matrix is not positive semi-definite (numerical issue) → add small diagonal regularization: Σ + 1e-8 × I.

---

## Position sizing — `position_sizer.py`

### Step 1: Regime-conditional strategic tilt

Before vol-targeting, apply a base strategic allocation that shifts with regime.
This ensures risk-off is not just "smaller same portfolio" but a genuinely defensive posture.

```python
STRATEGIC_WEIGHTS = {
    'RISK_ON':   {'SPY': 0.50, 'TLT': 0.20, 'GLD': 0.15, 'PDBC': 0.15},
    'NEUTRAL':   {'SPY': 0.35, 'TLT': 0.35, 'GLD': 0.20, 'PDBC': 0.10},
    'RISK_OFF':  {'SPY': 0.15, 'TLT': 0.50, 'GLD': 0.30, 'PDBC': 0.05},
}
```

These are the *direction* weights (sum to 1.0). The vol-targeting scalar then scales this vector up or down.

### Step 2: Inverse-vol adjustment within regime

Adjust the strategic weights by inverse vol to equalize per-asset risk contribution:

```python
w_ivol(i) = strategic_weight(i) / σ_est(i,t)
w_ivol_norm(i) = w_ivol(i) / Σ w_ivol(j)   # renormalize to sum = 1
```

**Note:** This is inverse-vol weighting, not full risk parity (which would use the covariance matrix). This is a deliberate simplification: it improves on equal weighting without requiring covariance inversion, which is deferred to Project 8 (portfolio-optimization-engine).

### Step 3: Volatility-targeting scalar

```python
TARGET_VOL = {
    'RISK_ON':  0.15,   # 15% annualized
    'NEUTRAL':  0.10,   # 10% annualized
    'RISK_OFF': 0.06,   # 6% annualized
}

σ_port = √(w_ivol_normᵀ × Σ × w_ivol_norm)   # portfolio vol with EWMA cov
scale  = TARGET_VOL[regime] / σ_port
w_final(i) = w_ivol_norm(i) × scale
```

### Step 4: Leverage cap by regime

```python
LEVERAGE_CAPS = {
    'RISK_ON':  1.50,   # up to 150% gross exposure
    'NEUTRAL':  1.00,   # fully invested, no leverage
    'RISK_OFF': 1.00,   # fully invested, defensive
}

gross_exposure = Σ|w_final(i)|
if gross_exposure > LEVERAGE_CAPS[regime]:
    w_final = w_final × (LEVERAGE_CAPS[regime] / gross_exposure)
```

### Step 5: Cash residual

If gross exposure < 1.0 (vol is high, engine scales down), remainder sits in cash (return = 0). This is the natural risk-off mechanism.

---

## Rebalancing logic — `rebalancer.py`

```python
def should_rebalance(
    regime_today: str,
    regime_last_rebal: str,
    σ_port_today: float,
    σ_port_last_rebal: float,
    vol_deviation_threshold: float = 0.20
) -> bool:
    """
    Returns True if EITHER condition holds:
      A: regime_today != regime_last_rebal  (confirmed regime change)
      B: |σ_port_today / σ_port_last_rebal − 1| > vol_deviation_threshold

    Financial rationale:
      Condition A: regime shift changes the target vol and strategic tilt.
      Condition B: even within a stable regime, vol can drift enough that
                   actual portfolio risk deviates materially from target.
                   20% drift threshold = if targeting 10% vol, rebalance
                   if realized port vol has moved to <8% or >12%.
    """

def compute_transaction_cost(
    w_new: np.ndarray,
    w_old: np.ndarray,
    cost_bps: float = 5.0
) -> float:
    """
    cost = (cost_bps / 10000) × Σ|w_new(i) − w_old(i)|
    Applied only on rebalance days.
    Financial rationale: cost is proportional to actual turnover,
    not a flat fee. A small vol-drift rebalance costs less than a
    full regime-flip reallocation.
    """
```

---

## Backtesting engine — `backtester.py`

```python
class WalkForwardBacktester:
    """
    Implements a strict expanding-window walk-forward backtest.

    Timeline:
      t=0 to t=503:     Warmup period. HMM is not yet fit. No trades made.
                        Composite regime still runs (no warmup needed).
      t=504:            First HMM fit. First trades placed.
      Every 252 days:   HMM refit on all data up to t_refit.
      Every day:        Features computed, regime classified (filtered),
                        persistence filter applied, rebalance check run.

    Trade execution rule (strict):
      Signal formed at close of day t using data available at t.
      Trade executed at open of day t+1 (approximated as close of t+1).
      Returns credited from t+1 close to t+2 close.
      This prevents any same-day lookahead.

    NAV calculation:
      NAV(t+1) = NAV(t) × (1 + Σ w_final(i,t) × r(i,t+1)) − cost(t)
      where cost(t) > 0 only on rebalance days.

    Outputs per day:
      - date
      - regime_hmm
      - regime_composite
      - weights (per asset)
      - gross_exposure
      - target_vol
      - realized_port_vol
      - nav
      - daily_return
      - rebalanced (bool)
      - turnover
      - cost
    """
```

**Benchmarks computed in parallel:**
1. SPY buy-and-hold
2. 60/40 (SPY 60%, TLT 40%), rebalanced monthly
3. Static vol-parity (inverse-vol weights, no regime, 12% constant target vol)
4. Regime overlay only (no vol scaling, just regime-conditional strategic weights)

Benchmarks 3 and 4 enable clean attribution:
- Static → Vol-target: value from vol management alone
- Vol-target → Full strategy: value from regime detection

---

## Analytics — `analytics.py`

### Overall performance
- CAGR, Total Return
- Sharpe ratio (annualized, risk-free = 0)
- Sortino ratio (downside deviation denominator)
- Max drawdown, Max drawdown duration
- Calmar ratio = CAGR / |Max Drawdown|
- Win rate (% of months positive)
- Avg monthly return (positive months), avg monthly loss (negative months)
- Correlation to SPY

### Regime-conditional analytics
- Sharpe per regime (RISK_ON / NEUTRAL / RISK_OFF)
- Avg daily return per regime
- Max drawdown per regime
- % of time spent in each regime
- Avg regime duration (days)
- Transition matrix: P(regime_t+1 | regime_t) — 3×3 matrix

### Portfolio behavior
- Avg gross exposure over full history
- Avg annualized turnover (Σ|Δw| × 252)
- Realized vol vs target vol (daily series, for dashboard)
- HMM regime probability series (3 columns: p_risk_on, p_neutral, p_risk_off)

### Stress period isolation
- GFC: 2008-09-01 to 2009-03-31
- COVID: 2020-02-19 to 2020-03-23
- 2022 Rate Shock: 2022-01-01 to 2022-12-31
- Report max drawdown, Sharpe, return for strategy vs SPY in each period

### Benchmark attribution table
| Metric | SPY | 60/40 | Static Vol-Parity | Regime Only | Full Strategy |
|---|---|---|---|---|---|
| CAGR | | | | | |
| Sharpe | | | | | |
| Max DD | | | | | |
| Calmar | | | | | |

---

## Streamlit dashboard — `streamlit_app.py`

Use Bloomberg dark mode from `style_inject.py`. 5 tabs:

| Tab | Key visuals |
|---|---|
| **OVERVIEW** | NAV chart vs all benchmarks (log scale), regime color bands overlay, current regime badge, current exposure, target vs realized vol |
| **REGIMES** | HMM state probability area chart, composite regime bar chart, transition matrix heatmap, time-in-regime pie, avg duration table, HMM vs Composite agreement rate |
| **PORTFOLIO** | Asset weight time series (stacked area), daily gross exposure line, realized vol vs target vol ribbon chart, per-asset contribution to portfolio vol |
| **ANALYTICS** | Full tearsheet table, underwater/drawdown chart, monthly return heatmap (calendar), regime-conditional Sharpe bar chart, stress test comparison table |
| **ATTRIBUTION** | 4-panel benchmark decomposition (static → vol-only → regime-only → full), turnover histogram, cost drag analysis |

---

## Key financial formulas (summary)

```
# Vol estimation
σ²_ewma(i,t)    = 0.94 × σ²_ewma(i,t-1) + 0.06 × r(i,t)²
σ_est(i,t)      = 0.5 × σ_realized_21d + 0.5 × σ_ewma  [annualized]

# Inverse-vol weights (within regime strategic tilt)
w_ivol(i)       = strategic_weight(i) / σ_est(i,t)
w_norm(i)       = w_ivol(i) / Σ w_ivol(j)

# Portfolio vol
σ_port(t)       = √(w_normᵀ × Σ_ewma × w_norm)  [annualized]

# Vol-targeting scalar
scale(t)        = σ_target[regime] / σ_port(t)
w_final(i,t)    = w_norm(i) × scale(t)  [pre-cap]

# Post-leverage cap
if Σ|w_final| > cap[regime]: w_final × = cap / Σ|w_final|

# Transaction cost
cost(t)         = (5 / 10000) × Σ|w_final(i,t) − w_final(i,t-1)|  [on rebal days only]

# NAV
NAV(t+1)        = NAV(t) × (1 + Σ w_final(i,t) × r(i,t+1)) − cost(t)
```

---

## Simplifying assumptions (all documented in code)

1. **Tax and margin costs ignored.** Real leveraged portfolios have financing costs (Fed Funds + spread). Not modeled.
2. **No slippage model.** Transaction cost = flat turnover × 5bps. Real impact would be larger for PDBC and GLD.
3. **Weights applied to close prices.** Open execution approximated as previous close. In practice, would use VWAP.
4. **LQD used as credit stress proxy, not true OAS.** Price-based relative return, not a spread derived from bond cashflows.
5. **PDBC history limited.** Engine falls back to GLD-only commodities if PDBC unavailable before 2014.
6. **HMM state labels assumed monotone in vol.** In extreme market conditions, this ordering may break.
7. **Inverse-vol weighting, not full risk parity.** Covariance-matrix-based risk parity deferred to Project 8.
8. **Risk-free rate = 0** in Sharpe calculation. Cash held during de-risking earns nothing in the model.

---

## Design decisions (rationale for code review)

| Decision | Rationale |
|---|---|
| HMM with filtered probabilities only | Prevents forward-looking bias. Uses forward algorithm output, not Viterbi. |
| Annual HMM refit on expanding window | Adapts to structural breaks (post-GFC, post-COVID) without survivorship bias |
| States labeled by vol mean, not index | Prevents regime label flipping across refits |
| Composite model runs in parallel | Provides interpretable benchmark; allows model comparison in dashboard |
| 3-day persistence filter | Eliminates noise-driven regime flips and spurious turnover |
| Strategic tilt shifts with regime | Risk-off = genuinely defensive allocation (bonds/gold), not just smaller equity |
| Target vol 15/10/6 | Realistic for long-only multi-asset; avoids "juiced" backtest appearance |
| Signal at t, trade at t+1 | Strict no-lookahead rule. Non-negotiable. |
| Turnover-proportional transaction costs | More realistic than flat fee; penalizes large reallocation events appropriately |

---

## Universal coding rules (from master CLAUDE.md)

- `main.py` orchestrates only. All logic in `engine/`.
- All weights, thresholds, assumptions in `config.yaml`. Never hardcode.
- Docstring on every class and method: financial rationale, not just mechanics.
- Handle edge cases: division by zero → `np.nan`, missing values, negative returns.
- No file longer than ~150 lines. Split if needed.
- `pandas` for data. `numpy` only where needed.
- Config loaded once via `utils/config_loader.py`, passed as dict.
- All simplifying assumptions documented in inline comments.

---

*Project CLAUDE.md — volatility-regime-engine*
*Created: 2026-04-06*
*Status: READY FOR SCAFFOLD*
