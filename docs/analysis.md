# Analysis — Volatility Regime Engine

## Strategy Thesis

Markets alternate between distinct regimes — risk-on environments where equities trend upward with low volatility, and risk-off environments where correlations spike, volatility expands, and traditional diversification fails. A static allocation ignores this entirely: a 60/40 portfolio holds the same equity weight on the eve of a crash as it does during a recovery. The thesis behind this engine is that regime detection, combined with volatility-targeted position sizing, can deliver equity-like returns with meaningfully lower drawdowns by scaling risk exposure to match the current environment.

The strategy does not try to predict returns. It identifies which regime the market is in *now* — using only backward-looking data — and adjusts how much risk to take, not where to take it. In risk-on, the portfolio tilts toward equities at 15% target vol. In risk-off, it rotates into bonds and gold at 6% target vol. The insight is not that regime detection adds return; it is that it reduces the cost of earning it.

---

## How Regime Detection Works

### Why two classifiers

No single regime model is trustworthy on its own. The engine runs two classifiers in parallel:

**HMM (Hidden Markov Model).** A 3-state Gaussian HMM trained on an 8-feature vector that includes realized volatility (21d and 63d), short-term equity momentum, return skewness, trend (price vs 200-day SMA), credit stress (LQD vs IEF), VIX level, and VIX momentum. The model is fit on an expanding window (first fit after 504 days, refitted annually) and produces filtered probabilities only — the forward algorithm, never the Viterbi smoother. This guarantees the regime label at date *t* uses only data available up to *t*.

States are labeled post-hoc by sorting each state's emission mean for realized volatility: the lowest-vol state becomes RISK_ON, the highest becomes RISK_OFF. This anchors state identity to an observable financial quantity and prevents label flipping across refits.

The HMM's strength is that it finds latent structure in the data without hand-tuned rules. Its weakness is opacity — when the HMM calls a regime change, it is difficult to explain exactly *why*.

**Composite (rule-based).** Four interpretable signals — trend (SPY vs 200-day SMA), vol stress (short vol vs long vol), drawdown depth (from 252-day high), and credit stress (LQD vs IEF) — each vote +1 or -1. The sum maps to RISK_ON (>= 2), NEUTRAL (0), or RISK_OFF (<= -2). Every threshold is transparent and defensible.

The composite exists to keep the HMM honest. When both classifiers agree, conviction is high. When they disagree, it flags periods where the HMM may be reacting to statistical noise rather than genuine regime shifts. The dashboard tracks their agreement rate over time.

Both classifiers pass through a 3-day persistence filter before any trades are placed. A regime change must hold for 3 consecutive days to be acted on. This eliminates the single largest source of unnecessary turnover: one-day regime flips driven by noise.

---

## What Vol Targeting Does

Regime detection answers *what state are we in*. Vol targeting answers *how much risk should we take given that state*.

For each regime, the engine:

1. **Sets a strategic tilt.** RISK_ON allocates 50% SPY / 20% TLT / 15% GLD / 15% PDBC. RISK_OFF flips to 15% SPY / 50% TLT / 30% GLD / 5% PDBC. This ensures risk-off is a genuinely defensive posture, not just a smaller version of the same portfolio.

2. **Adjusts by inverse volatility.** Within the strategic tilt, assets with higher estimated vol receive lower weight, equalizing per-asset risk contribution. Vol is estimated as a 50/50 blend of 21-day realized and EWMA (lambda = 0.94).

3. **Scales to target vol.** Portfolio vol is computed using an EWMA covariance matrix. The scaling factor = target vol / portfolio vol. In RISK_ON, the target is 15% (with up to 1.5x leverage). In RISK_OFF, the target is 6% (no leverage, and any shortfall is held in cash).

The result: during calm markets, the engine is close to fully invested with a modest equity tilt. During stress, it de-risks aggressively — not by selling equities entirely, but by shrinking gross exposure and rotating into bonds and gold.

---

## Key Results

| Metric | Full Strategy | SPY B&H | 60/40 | Static Vol-Parity | Regime Only |
|--------|:------------:|:-------:|:-----:|:-----------------:|:-----------:|
| CAGR | 10.9% | 10.2% | 7.8% | 6.5% | 8.7% |
| Sharpe | 1.03 | 0.58 | 0.62 | 0.71 | 0.74 |
| Max DD | -21.9% | -33.7% | -25.1% | -18.4% | -28.3% |
| Calmar | 0.50 | 0.30 | 0.31 | 0.35 | 0.31 |

### Interpretation

**Drawdown reduction is the headline.** The strategy's max drawdown of -21.9% is 35% less severe than SPY's -33.7%. For an allocator, this is the number that matters most — it determines how much capital can be committed before hitting a loss tolerance. A fund that can accept -25% drawdown can size this strategy at roughly 1.0x, but would have to size SPY B&H at ~0.7x, resulting in lower dollar returns despite SPY's similar CAGR.

**Sharpe of 1.03 is strong for a multi-asset strategy.** It nearly doubles SPY's 0.58 and exceeds both 60/40 (0.62) and the static vol-parity benchmark (0.71). The improvement over static vol-parity (1.03 vs 0.71) is the marginal Sharpe from regime detection — evidence that the HMM is adding signal beyond simple volatility management.

**CAGR of 10.9% modestly beats SPY.** The strategy is not designed to maximize return. It is designed to maximize risk-adjusted return. The similar CAGR with much lower drawdown is exactly the intended profile.

**Attribution across benchmarks.** Moving from SPY to 60/40 adds diversification. From 60/40 to static vol-parity adds volatility management. From static vol-parity to regime-only adds regime-conditional tilting. From regime-only to the full strategy adds vol-targeting on top of regime detection. Each layer contributes independently.

---

## Stress Test Findings

### COVID crash (Feb 19 – Mar 23, 2020)

The COVID drawdown was the fastest 30%+ equity decline in history. SPY fell approximately -34% peak-to-trough. The strategy's response demonstrates the regime engine working as designed: the composite detector's drawdown signal fired first (SPY dropped >10% from its 252-day high), followed by the HMM transitioning to RISK_OFF as the 8-feature vector shifted. After the 3-day persistence filter confirmed the regime change, the engine reduced equity exposure, rotated into TLT and GLD, and cut target vol to 6%.

The strategy's drawdown during this period was materially less severe than SPY's. Critically, it also re-risked relatively quickly once regime signals stabilized, capturing a meaningful portion of the recovery.

### 2022 rate shock (Jan – Dec 2022)

2022 was the worst year for 60/40 in decades. Both equities and bonds sold off simultaneously as the Fed raised rates aggressively. SPY fell approximately -25% peak-to-trough; TLT fell approximately -30%. This is the environment that breaks naive diversification.

The engine navigated this period through a combination of reduced gross exposure (vol was elevated, so the vol-targeting scalar shrank positions) and the regime detector shifting to NEUTRAL or RISK_OFF for extended periods. The strategy still drew down, but less severely than SPY or 60/40. The key insight: vol targeting provided more protection than regime detection in this period, because the crisis was slow-moving rather than acute — the vol estimate had time to adapt.

---

## Limitations and Simplifying Assumptions

**No financing costs.** In RISK_ON, the strategy can lever up to 1.5x gross. In practice, the long-only cash needed to fund that leverage would cost Fed Funds + spread. This is not modeled, which overstates RISK_ON returns by roughly 50 bps/year in a normal rate environment.

**Transaction costs are simplified.** Costs are modeled as 5 bps x turnover. Real execution involves bid-ask spreads (wider for GLD and PDBC), market impact, and timing uncertainty. The flat 5 bps is conservative for SPY and TLT but optimistic for less liquid assets.

**LQD as credit stress proxy.** The credit stress feature uses LQD's rolling return relative to IEF, not actual OAS or credit spreads. This is a price-based proxy that conflates credit risk with rate risk. A production system would use ICE BofA OAS indices or CDS spreads.

**HMM state labels assumed monotone in volatility.** States are labeled by sorting emission means of 21-day realized vol. In extreme conditions (e.g., the calm before a crash), a low-vol state could coincide with rising systemic risk. The composite detector partially hedges this failure mode, but it is a known limitation.

**Inverse-vol weighting, not risk parity.** The position sizer equalizes risk using per-asset vol but ignores correlations. During regime transitions, correlations shift rapidly (e.g., equity-bond correlation can flip from negative to positive), and per-asset vol weighting does not capture this. Full covariance-based risk parity is deferred to Project 8.

**Risk-free rate = 0%.** Cash held during de-risking earns nothing. In a positive rate environment, this understates strategy returns by the risk-free rate times the average cash allocation (typically 5–20% of portfolio).

**Survivorship-free but history-limited.** PDBC (commodity ETF) has data only from 2014. The backtest starts in 2007 for SPY/TLT/GLD but the full 4-asset engine activates only once PDBC data becomes available.

---

## What Project 8 Adds

Project 8 (portfolio-optimization-engine) upgrades the position sizing layer from inverse-vol weighting to full covariance-based optimization:

- **Covariance matrix cleaning** via Ledoit-Wolf shrinkage and Random Matrix Theory, replacing the raw EWMA covariance that is noisy in small-sample regimes.
- **True risk parity** using the covariance matrix to equalize marginal risk contributions, not just per-asset vol.
- **Hierarchical Risk Parity (HRP)** as a tree-based alternative that avoids matrix inversion entirely — more robust when asset correlations are unstable (exactly the regime-transition periods where this engine needs it most).
- **Black-Litterman** for incorporating regime-derived views into the optimization, rather than applying regime tilts as a separate pre-step.

The vol-regime engine provides the signal layer (which regime, what vol target). Project 8 provides a better answer to the question: "given this signal, what is the optimal way to allocate across assets?" The two projects are designed to compose — the regime engine's output feeds directly into Project 8's optimizer as a constraint set.

---

*Francois Rostaing — [GitHub](https://github.com/FrancoisRost1)*
