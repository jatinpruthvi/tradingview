# The profitability backlog

Twenty things that could plausibly improve this project's *honest* results, ranked,
with the evidence each rests on and the way each would have to be validated.

## Read this first

Two uncomfortable facts govern everything below.

**1. A NOT VALIDATED strategy cannot be made profitable by adding features.**
The crypto study now scores 3/8, the Indian one 2/6, and both lose money at
*zero* cost (crypto: -1.74% at `fee=0, slip=0`). When a system loses before it
pays the broker, the problem is the hypothesis, not the execution model. No
indicator added on top of a hypothesis that does not work will fix that; it will
only multiply the search space until something looks good by luck.

**2. Every item below is a hypothesis, and the project's own machinery exists to
reject hypotheses.** The DSR is computed over the number of trials actually run.
Testing 20 variants and keeping the best one is 20 trials; that is exactly the
multiple-testing penalty the study already charges. So the rule here is: **one
item at a time, measured on a walk-forward, kept only if it survives the
out-of-sample half** — never a batch.

That is why this file is a backlog and not a commit. Landing 20 changes at once
would make every number in `output/` unattributable, and the honest reading of
"it got better" would become impossible.

## What has been measured so far

**Gap-aware stop fills** (done, and it made results *worse*): -949.80 → -1,251.09
on crypto, -3,077 across all ten symbols. Correct, and it cost 24% of the
reported loss. Realism always costs something; that is how you know it worked.

**Trailing stop and breakeven** (implemented this pass, opt-in, measured):

| variant | net P&L | return | PF | Sharpe | trades | win % | avg R |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (no trail) | -1,251.09 | -1.25 % | 0.513 | -1.224 | 31 | 29.0 | -0.373 |
| `trail_atr_mult 2.0` | -1,642.28 | -1.64 % | 0.379 | -1.670 | 32 | 25.0 | -0.469 |
| `trail_atr_mult 3.0` | -1,251.09 | -1.25 % | 0.513 | -1.224 | 31 | 29.0 | -0.373 |
| `breakeven_at_r 1.0` | -1,533.95 | -1.53 % | 0.388 | -1.542 | 32 | 21.9 | -0.478 |
| both | -1,633.14 | -1.63 % | 0.354 | -1.648 | 33 | 21.2 | -0.486 |

Both rules hurt. The 3.0-ATR trail is *bit-identical* to the baseline (it sits
outside the initial 2.05-ATR stop and never binds), which is a useful control: the
machinery only changes anything when it actually triggers. Ranked worst first:
the 2.0-ATR trail (-1,642.28), then the combined rule (-1,633.14), then breakeven
alone (-1,533.95). Breakeven converts would-be winners into
scratches and takes the win rate from 29.0% to 21.9%; the trail instead clips the
right tail the fixed 2.5R target was already collecting. On a strategy whose stop
is 2.05 ATR, a 1R breakeven cut sits exactly where normal pullbacks live.

This is the shape of the whole exercise: **the intuitive improvement was negative,
and the measurement said so in one command.**

---

## Tier 1 — fixes to the measurement itself (do before anything else)

These cannot make a strategy profitable. They stop it from reporting profit it
does not have, which is the precondition for trusting anything in Tier 2.

| # | Item | Why | How to verify |
|--:|:-----|:----|:--------------|
| 1 | **Real cost model per market** | India delivery round-trip is ~25-30 bp (brokerage + STT + stamp + exchange), and the config uses 3+5 bp per side. Crypto uses 10+5 taker and ignores funding, which a 4h strategy holding days actually pays. | Cost sweep ×1/×2/×4 must keep its sign and roughly its slope; a strategy that dies at realistic cost was never alive. |
| 2 | **Funding rate in crypto P&L** | Perpetual positions held for days pay or receive funding; ignoring it flatters longs in a bull tape. | Replay: total funding charged should be comparable to the fee line. |
| 3 | **Sample size** | India: 16 trades directionally, 8 is the minimum the study will score. Nothing can clear 8 checks on 16 trades. | Trade count ≥ 100 per market before believing a PF. |
| 4 | **Cross-market validation** | A rule that only works on synthetic crypto is a coincidence generator. | The same config on 2+ markets, same sign. |
| 5 | **Partial fills / liquidity cap** | Sizing ignores `min_quote_volume` at trade time; a 25%-of-equity position in a thin alt is not achievable. | Cap units at a fraction of the bar's volume and see if P&L survives. |

## Tier 2 — the hypothesis, not the plumbing (each measured in isolation)

| # | Item | Why it might work | Risk |
|--:|:-----|:------------------|:-----|
| 6 | **Regime gate** | `tvlab nse breadth` and the dashboard already compute breadth, and **no strategy reads it**. Longs only when breadth > threshold, shorts only below. This is the single largest unused signal in the repo. | Fewer trades → the 8-trade minimum bites. |
| 7 | **Relative-strength ranking** | Rank the universe by N-bar return and only take setups in the top tercile; trend-following works best on the strongest names. | Ranking on the same bars you trade is a lookahead trap — must be computed from data available at the signal bar. |
| 8 | **Shorter time exit** | `max_bars_in_trade: 120` (India) is 6 months; a 4h crypto trade at 90 bars is 15 days. Trades that neither hit stop nor target are dead money. Shorten and re-measure. | None; it is a `--set`. |
| 9 | **Volatility-targeted sizing** | Fixed 0.5% risk per trade gives the same notional to a 1% and a 20% vol name; crypto and equities then cannot be compared. | Sizing changes concentration, so correlation caps must be re-checked. |
| 10 | **Stop distance vs implied move** | 8 of 15 measured Indian candidates had stops inside 1σ of the 28-day option-implied move — those stops are inside the noise. Widen or skip. | Wider stop = smaller size = same risk, but fewer target hits. |
| 11 | **Entry timing filter (avoid extension)** | Entering at the close of a confirmation bar is late; the stop often sits inside the retracement that follows. | Adds a parameter; must not become a curve-fit. |
| 12 | **Correlation-aware portfolio construction** | BTC+ETH is one bet at 25% each (correlation cap caught it, yet both were funded). Weight by cluster, not by symbol. | Needs a covariance window; unstable in short samples. |
| 13 | **Time-of-day / session filter** | Indian intraday edge concentrates in the first and last hour; 4h crypto bars are not uniform in activity. | More splits = more trials. |
| 14 | **Asymmetric R** | Fixed 2.5R target is arbitrary. Test 1.5/2.5/4 R as three *separate* trials and let DSR charge for the search. | Explicitly a 3-trial search. |

## Tier 3 — structural improvements to the research loop

| # | Item | Why |
|--:|:-----|:----|
| 15 | **Variant runner (`tvlab variants`)** | Runs N named configs over the same data, scores each, and feeds the trial count to the DSR. This is what makes items 6-14 safe to test. Build before testing them. |
| 16 | **Hold-out set that is never looked at** | Walk-forward re-fits on the same history a human keeps peeking at. A dated, locked hold-out (e.g. 2024+) is the only defence against researcher overfitting. |
| 17 | **Cost/liquidity-aware shortlisting** | The screener already rejects on cost (`rejected on cost` in the briefing) — extend it to reject when expected R does not clear `k × round-trip cost`. |
| 18 | **Regime-conditional reporting** | Report P&L split by trend/breadth regime, so "works in bull, dies in chop" is visible instead of averaged away. |
| 19 | **Execution realism for entries** | Fill at the next open is good; add a limit-order variant (fill only if the bar trades through the level) for setups that are not market orders. |
| 20 | **Kill criteria, written down** | Before each hypothesis: what result would make you abandon it. Without this, every negative result becomes "try a different parameter". |

---

## Suggested order

1. **#15** (variant runner) — it is the tool for everything else and touches no
   strategy logic.
2. **#1, #2, #3** — costs, funding, sample. Until costs are right, every ranking
   below is noise.
3. **#6** (breadth regime gate) and **#7** (relative strength) — the two with the
   most unused signal in the repo.
4. **#16** (hold-out) before trusting any winner produced by 1-3.
5. Everything else, one at a time, each with its walk-forward and its DSR.

## What is implemented today

- **Gap-aware stop fills** — `backtest/engine.py`, `screener/review.py`; 6 tests;
  measured at -3,077 across the crypto universe.
- **Trailing stop and breakeven move** — `backtest.trail_atr_mult`,
  `backtest.breakeven_at_r`, both `null` by default so no published study moves;
  4 tests, including the intrabar-order guarantee and the R-denominator trap.
  Measured above: both negative on this strategy.

Nothing in Tier 2 or Tier 3 is implemented. That is deliberate, and the
measurement of the two exit rules is the argument for it.
