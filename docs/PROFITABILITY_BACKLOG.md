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

**VWAP as a stock picker** (measured, negative — no code changed): the repo has
`indicators.technical.vwap` and nothing reads it, so before wiring it into
ranking the question was asked directly — does "price above VWAP" pick today's
best stock? Measured on live NSE bars: 60 names chosen by traded value, 60
sessions of 5m bars (3,600 name-days) and 496 sessions of daily bars (29,150
name-days), cross-sectional Spearman IC with days as the observations, each day
demeaned so a market-wide move cannot manufacture a ranking.

| signal | predicts | IC | t | top-minus-bottom quintile |
|---|---|---:|---:|---:|
| 09:40 close vs session VWAP (5m) | rest of the same day | +0.010 | 0.6 | +7.7 bp (t=1.4) |
| 09:40 close vs session VWAP (5m) | next session | -0.143 | -5.7 | -43.2 bp (t=-4.05) |
| close-in-range, daily | next day | -0.033 | -4.5 | 12.5 bp |
| close-in-range, daily | next 5 days | -0.016 | -2.2 | — |
| distance from EMA20, daily (control) | next day | +0.003 | 0.3 | — |

**Same day: nothing.** +0.010 with t=0.6 over 3,600 name-days is
indistinguishable from zero, and the quintile ladder is noise (-9.0 / +3.3 /
-3.4 / +3.1 / +5.9 bp): no ordering, and the middle three buckets change sign. A
ranking signal has to sort names; this one does not.

**Next day: real and negative.** Names trading above VWAP underperform the next
session by -0.143 IC intraday and -0.033 daily (t=-4.5 over 493 days), and the
daily quintiles do sort monotonically (6.5 / 1.5 / 0.9 / -2.8 / -6.0 bp).
Extension is a chase, not a trend. It survives the obvious objection: strip the
day's own return out of the signal and the residual still predicts -0.110
(t=-5.71). But it is mostly the day's move (correlation 0.73 with it, 0.23 with
EMA20 distance), so it is not independent information.

**Tradeable: no, at this size.** The daily spread is 12.5 bp against ~16 bp of
round-trip cost in the Indian config (3+5 bp per side), and the intraday
version's 43 bp sits on the leg `risk.allow_short: false` forbids. That makes it
a veto ("do not chase a name 2σ above VWAP today") and a cost check — not a way
to choose the best stock of the day.

Caveats, because they bound the result: names are picked by *today's* traded
value (survivorship and liquidity bias), the intraday sample is one 60-session
window (Yahoo's cap), and the ICs are cost-free.

**Opening-range breakout (ORB)** (measured, and now registered as `orb`): the only
gross edge found in this project that is not noise. Rule: take the high and low of
the first 15 minutes, trade the first break of that range, stop at the other side,
exit at the session close. Measured on 60 live NSE names x 60 sessions of 5m bars
(3,510 breakouts), one 16 bp round trip per day:

| rule | gross/day | net/day | t | green days | longest red run |
|---|---:|---:|---:|---:|---:|
| ORB, all breakouts, long+short | +13.4 bp | -2.6 bp | 3.23 | 40.0 % | 8 |
| ORB + relative-volume filter (causal) | +10.8 bp | -5.2 bp | 1.63 | 46.0 % | — |
| gap fade | +5.2 bp | -10.8 bp | 0.55 | 36.7 % | 7 |
| VWAP fade at 10:00 | -32.4 bp | -48.4 bp | -3.46 | 27.1 % | 14 |
| equal-weight hold (benchmark) | +4.5 bp | -11.5 bp | 0.39 | 45.0 % | 7 |

The fill does not flatter it: entering at the next bar's open instead of the
breaking bar's close gives the same +13.4 bp, because 5m bars are contiguous.

**The published "stocks in play" refinement did not survive a causal test.** A
relative-volume filter computed over the first 40 minutes showed +9.6 bp net
(t=2.71); restricted to the opening range alone - the only volume knowable before
a 09:30 breakout - it is -5.2 bp. The first version was lookahead, and it is the
most useful negative result in this file: a filter looks like the edge when it is
really reading tape the entry has not seen yet.

Run through the walk-forward harness (`study --strategy orb`, RELIANCE/TCS/INFY,
5m, 6-combo grid, 30 trials): 40 trades, 25.0 % win rate, profit factor 1.08,
expectancy +16.33, deflated Sharpe 0.260 (**fail**), 2/5 out-of-sample windows
profitable, OOS return -0.17 %. The harness reaches the same verdict the study
did: not validated. A 25 % win rate with a 3.24 payoff is the shape of the thing
- many small losses, a few large wins - which is exactly why "profitable every
day" is the wrong goal.

**Next-session reversal is a veto, not a trade** (measured, no ranking code
changed): close-in-range predicts the next day at IC -0.033 (t=-4.48 over 493
sessions, Q1-Q5 12.5 bp). That is close-to-close, and yesterday's close is not a
price you can trade. Re-measured with an executable fill - buy today's open, sell
today's close - on 60 names x 496 sessions:

| rule | gross/day | net/day | t |
|---|---:|---:|---:|
| long bottom decile by prior-day return | -2.5 bp | -18.5 bp | -0.47 |
| long bottom - short top decile, prior return | +12.0 bp | -20.0 bp | 2.67 |
| long bottom - short top decile, close-in-range | +6.5 bp | -25.5 bp | 1.69 |

Most of the reversal is paid in the overnight gap: yesterday's biggest winners
gap **+16.9 bp** and then give back **-12.4 bp** open-to-close. By 09:15 the move
has happened. Holding longer does not rescue it either - non-overlapping 3/5/10/20
session tranches produce no net-positive, statistically meaningful cell, and from
5 sessions out the excess versus the equal-weight universe is negative, i.e. beta
rather than alpha. The usable form is a zero-cost veto on a swing entry.

**"Profitable every day" is not a strategy that exists in this data.** ORB's best
case is green on 40 % of days with 8 consecutive losses. Even the hindsight ceiling
is low: the average name is up open-to-close on only 56.7 % of days. Externally,
SEBI's FY2022-23 study finds 71 % of individual intraday traders lose money, rising
to **80 %** for those doing more than 500 trades a year; the NBER study of SENSEX
day traders finds them losing in every one of seven years, at -3.8 bp per trade. A
positive expectancy with a 40 % daily hit rate is the realistic shape; a rule that
is green every day is not.

**NSE delivery percentage** (measured; `data/delivery.py` added and no strategy
reads it): the ORB relative-volume filter failed as lookahead, and delivery % is
the replacement that cannot be peeked at - NSE publishes it in the evening
bhavcopy, so for session D it is knowable before D+1 opens and never during D.
60 most-liquid NSE names x 245 sessions (14,629 symbol-days, 100 % join coverage),
Yahoo's adjusted prices with only NSE's two columns joined on:

| question | IC | t | verdict |
|---|---:|---:|---|
| yesterday's delivery % -> today open-to-close | +0.045 | +3.77 | not tradeable |
| yesterday's delivery % -> today's opening gap | -0.117 | -10.04 | not tradeable |
| yesterday's delivery % -> today's high-low range | **-0.375** | **-39.14** | gating / sizing |
| same-day delivery % -> same-day range (control) | -0.465 | -52.56 | not tradeable |

**Direction: no.** The quintile ladder by yesterday's delivery is -4.6 / -8.4 /
-0.5 / +3.4 / +1.3 bp open-to-close: no ordering, and the best bucket is not the
top one. Q5-Q1 is +5.8 bp gross (t=0.95) and -26.2 bp net of two round trips; the
long-only top quintile is +1.3 bp gross and -14.8 bp net (t=0.26). Up-rates are
48.0 % top versus 46.1 % bottom against a 47.4 % universe. The statistically
detectable +0.045 IC is carried by the middle of the distribution, not by a
tradeable extreme. The split-half spread is stable (+5.5 then +6.2 bp), which
rules out instability but not unprofitability.

**Range: yes, and it is not just volatility clustering.** High delivery yesterday
means a *quieter* session today, monotonically: 357 / 284 / 254 / 232 / 214 bp
from the bottom to the top quintile, a -143 bp spread at t=-31. The obvious
objection is that delivery % merely proxies yesterday's range, and volatility
clusters (yesterday's range onto today's: IC +0.41, t=46). A double sort removes
that explanation: inside each yesterday's-range quintile, splitting on delivery
still separates today's range by -20 / -33 / -42 / -48 / -72 bp (t -5.7 to -9.5),
and the effect *grows* in the wider-range buckets rather than flattening. The
same-session control says what the variable is: delivery accumulates on quiet,
accumulative days and is thin on churny ones.

**What it is good for.** Not an entry signal and not a way to be profitable
daily. It is a forecast of the session's expected range, which is the missing
input to two decisions this project makes badly today: whether a range-dependent
rule has enough range to pay its 16 bp (ORB's opening range has to be wide enough
to clear it), and how large the position should be. That measurement has now been run on
the 5m sample already in the repo - see **Does it rescue ORB?** below - and it
does not clear the bar.

**Caveats.** One year, one regime; the universe is *today's* most-liquid 60 names
(survivorship and liquidity bias); prices are Yahoo-adjusted while delivery comes
from NSE, so the ratio is trustworthy but the two series are never compared
bar-for-bar; "range" is the full-session high-low and includes the opening range
ORB actually trades; and 100 % coverage is a property of these 60 liquid names,
not of NSE.

**Does it rescue ORB?** (measured on the existing 5m sample; `src/` unchanged).
60 names x 60 sessions, 3,510 breakouts, 98 % of them matched to a previous
session's delivery row. The trade list is imported from the ORB study rather than
re-derived, and the baseline row reproduces the documented one exactly, which is
the self-check that it is the same 3,510 trades:

| variant | trades | gross bp/day | net bp/day | t | total net |
|---|---:|---:|---:|---:|---:|
| all breakouts (baseline) | 3,510 | +13.4 | -2.6 | 3.24 | -1.60 % |
| gate: bottom 40 % by delivery % | 1,372 | +16.6 | **+0.6** | 2.55 | +0.30 % |
| gate: top 40 % by delivery % | 1,387 | +13.5 | -2.5 | 2.64 | -1.53 % |
| size by forecast range | 3,510 | +13.7 | -1.6 | 2.91 | -1.02 % |

Quintile ladder (net bp/day): Q1 +1.2, Q2 +0.6, Q3 -7.4, Q4 -4.0, Q5 -1.2.

**The mechanism is real and it replicates.** Inside the 5m sample, yesterday's
delivery quintile sorts the *opening* range monotonically - 173 / 136 / 113 /
100 / 93 bp, IC -0.41 - so the gate does reach the variable it claims to. ORB
also only pays when that range is wide: the widest realised opening-range
quartile is +2.8 bp per trade and the other three are -5.0 / -2.9 / -5.3. Wide
opening ranges are 25 % of trades.

**The translation to P&L does not hold.** The best gate cell is +0.6 bp/day
against -2.6 for everything, i.e. a 3.2 bp/day improvement that buys break-even
and nothing more. Three things say it is not a real rescue. The middle quintiles
are non-monotone (Q3 is the worst cell in the table). The *time-series* form of
the same idea - is this name's delivery low for itself, over its last month? -
points the opposite way: +1.0 bp/day when it is high (t=3.06) against -1.9 when
low. And sizing by the forecast moves -2.6 to -1.6 bp while *worsening* the
drawdown (-3.17 % to -3.55 %), which is the wrong trade to make for 1 bp.

**Resolution, not conviction, is the binding constraint.** 60 sessions resolve
8.2 bp/day at t=2 on the full sample and 13.1 bp/day on the gated subset. The
entire effect under test is 3 bp. So this is not "delivery is useless" - it is
"this sample cannot see it", which is a different sentence and needs a different
response. Five years (~1,250 sessions) would resolve about 1.8 bp/day at the same
confidence, which is the data purchase in Tier 1 item 3 rather than a new filter.

Caveats on top: eight variants were tried (the DSR penalty applies to all of
them), the cross-sectional and time-series forms disagree, and 60 % of the
trading days in the gated subset are the same 60 days as the baseline, so the
comparison is paired rather than independent.

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

Item 1 has since been measured for the intraday case: ORB is +13.4 bp gross and
-2.6 bp net at the config's 16 bp, so it only turns positive below ~13 bp of round
trip. At the 25-30 bp real delivery cost in the same row it is clearly negative.

## Tier 2 — the hypothesis, not the plumbing (each measured in isolation)

| # | Item | Why it might work | Risk |
|--:|:-----|:------------------|:-----|
| 6 | **Regime gate** | `tvlab nse breadth` and the dashboard already compute breadth, and **no strategy reads it**. Longs only when breadth > threshold, shorts only below. This is the single largest unused signal in the repo. | Fewer trades → the 8-trade minimum bites. |
| 7 | **Relative-strength ranking** | Rank the universe by N-bar return and only take setups in the top tercile; trend-following works best on the strongest names. | Ranking on the same bars you trade is a lookahead trap — must be computed from data available at the signal bar. |
| 8 | **Shorter time exit** | `max_bars_in_trade: 120` (India) is 6 months; a 4h crypto trade at 90 bars is 15 days. Trades that neither hit stop nor target are dead money. Shorten and re-measure. | None; it is a `--set`. |
| 9 | **Volatility-targeted sizing** | Fixed 0.5% risk per trade gives the same notional to a 1% and a 20% vol name; crypto and equities then cannot be compared. | Sizing changes concentration, so correlation caps must be re-checked. |
| 10 | **Stop distance vs implied move** | 8 of 15 measured Indian candidates had stops inside 1σ of the 28-day option-implied move — those stops are inside the noise. Widen or skip. | Wider stop = smaller size = same risk, but fewer target hits. |
| 11 | **Entry timing filter (avoid extension)** | Entering at the close of a confirmation bar is late; the stop often sits inside the retracement that follows. VWAP extension was measured first (above): -0.033 next-day IC, below costs. | Adds a parameter; must not become a curve-fit. |
| 12 | **Correlation-aware portfolio construction** | BTC+ETH is one bet at 25% each (correlation cap caught it, yet both were funded). Weight by cluster, not by symbol. | Needs a covariance window; unstable in short samples. |
| 13 | **Time-of-day / session filter** | Indian intraday edge concentrates in the first and last hour; 4h crypto bars are not uniform in activity. | More splits = more trials. |
| 14 | **Asymmetric R** | Fixed 2.5R target is arbitrary. Test 1.5/2.5/4 R as three *separate* trials and let DSR charge for the search. | Explicitly a 3-trial search. |

Item 13 has since been measured: no clock-based rule validated. A 09:30 entry is
better than 09:45 only on execution (entry-bar range 0.35 % vs 0.44 %, 4.7 bp of
drift given up, t=1.56), and the equal-weight basket loses to cost at both clocks
(-3.6 / -8.4 bp net). The only non-noise intraday edge found is price-triggered
(ORB), not time-triggered.

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
- **VWAP** — `indicators.technical.vwap` exists and is lookahead-audited, and is
  read by no strategy, screener or ranking step. The measurement above is why:
  it has no same-day ranking power. Nothing was changed to act on it.
- **Opening-range breakout** — `strategy/setups.py::OpeningRangeBreakout`,
  registered as `orb`: one break per session, stop at the other side of the
  range, no target, flat by the session close. 6 tests; the lookahead audit picks
  it up automatically through `available()`. Measured above: the only non-noise
  gross edge, and still rejected by the walk-forward harness.
- **Session-close exit** — an optional `exit` column in the signal frame
  (`backtest/engine.py`). Additive and off unless a strategy sets it; stops and
  targets still take precedence, because they are intrabar events that happen
  before the closing print.

- **NSE delivery source** - `data/delivery.py`, registered as
  `--data-source nse_delivery` behind an optional `delivery` extra
  (`jugaad-data`). Joins `_delivery_pct` and `_avg_trade_value` onto the
  existing adjusted price series by session date, never forward-fills a missing
  bhavcopy, and reports the matched share in `attrs`. 17 tests, including the
  18:30-UTC date stamp and the duplicate `T0` rows that a naive join reads as
  two sessions per day. Nothing in `strategy/` reads it yet, and the measurement above says it
  should stay that way: on this 60-session sample the gate reaches break-even,
  which is below what the sample can resolve. See the range finding above for
  what would have to be built first.

Nothing in Tier 2 or Tier 3 is implemented. That is deliberate, and the
measurement of the two exit rules is the argument for it.
