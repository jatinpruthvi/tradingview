# tvlab — a trading research workbench that tries to prove you wrong

Multi-market, config-driven backtesting, validation, screening and a live
lightweight-charts dashboard. One engine, 4 markets × 4 trading styles = 16
selectable combinations, conservative risk mandate enforced in code.

---

## Read this first: the honest result

**The default strategy is NOT validated, and on the bundled data it should not
be.**

Running the full study today, with no overrides - this is what the committed
config produces out of the box:

```bash
tvlab study --market crypto --style swing --recovery --xval
```

```
VERDICT: NOT VALIDATED                          3 of 8 checks passed

 1  Enough trades to measure            PASS    31 trades
 2  Deflated Sharpe (multiple testing)  FAIL    DSR=0.000 over 320 trials
 3  Drawdown within mandate             PASS    max DD 0.97% vs limit 10.0%
 4  Profit factor > 1 after costs       FAIL    PF=0.58
 5  Walk-forward windows profitable     FAIL    1/5 OOS windows profitable (20%)
 6  OOS retains IS performance          FAIL    not measurable - see below
 7  Beats shuffled-return null          FAIL    p=0.533 (beat only 47% of 60 permutations)
 8  Monte Carlo DD-breach risk          PASS    P(max DD >= 10%) = 0.0% of 2000 sims

 Portfolio: -0.95% return, 0.97% max DD, 31 trades, PF 0.58, Sharpe -1.00,
            fees 1,366 on 100,000 of capital
 Walk-forward OOS: +0.43% return, Sharpe 0.32, PF 1.58 - over 3 trades total
```

Those are the numbers the command produces with the committed config
(`data.synthetic_bars: 4000`, `walk_forward.n_splits: 5`, `monte_carlo.n_sims:
2000`, `max_trials: 64`, which the walk-forward multiplies across its 5 windows to
reach 320).

**A shorter sample looks better, and that is why it is not the headline.** This
README previously quoted `--set data.synthetic_bars=2500`, which produces 13 trades,
PF 1.09 and a +0.08% return - the only configuration tested where the strategy is
net positive. It was carried over from an older run rather than chosen, but the
effect is the same either way: it flatters the result. At the default 4000 bars the
same strategy loses money. Documenting the config a user actually gets is the only
defensible choice, so the override is gone. (Before that it quoted a 4-split run
with 500 simulations, whose portfolio line matched but whose checks did not. A
number quoted without the flags that produced it is not reproducible, which is how
documentation drifts.)

Why this is the correct answer rather than a failure of the project:

- **The default data is synthetic.** No exchange or broker is reachable from this
  environment, so `tvlab` ships a generator (Markov regime chain + GARCH(1,1)
  volatility + Student-t tails). Synthetic prices contain regime drift and
  volatility clustering, but **no exploitable microstructure**. A cost-bearing
  strategy on such data should converge to roughly zero minus costs, and it does:
  −0.95% over 4,000 bars, of which 1,366 is fees on 100,000 of capital.
- **31 trades clears check 1's floor and still supports no conclusion.** The floor
  is 30 because below it a win rate has enormous confidence intervals; clearing it
  is the minimum for the ratio to mean anything, not evidence that it does. The
  29.0% win rate here is a description of one sample.
- **Check 6 is unmeasurable, and the report now says exactly why.** It compares
  each walk-forward window's out-of-sample objective against its in-sample one. The
  in-sample fits are fine - 8 to 11 trades each against a floor of 8. It is the
  *out-of-sample* segments that fail: 5 rolling splits over 4,000 bars leave each
  test segment about 320 bars long, which produces **0 or 1 trades** against the
  same floor, so every OOS objective is degenerate and the ratio has no numerator.
  The check reports `not measurable` with those counts rather than a bare `n/a`,
  because `n/a` reads like an unimplemented feature when it is actually a finding
  about the sample. The overfitting question - do parameters chosen in-sample keep
  working out of sample? - **cannot be answered on this data in either direction.**
  The fixes are fewer splits (longer test segments) or a longer history; changing
  the validation config to make a check measurable is not something this repo does
  silently.
- **The most flattering number in the report is the least meaningful.** The stitched
  walk-forward equity shows +0.43%, Sharpe 0.32 and PF 1.58 - better than the
  in-sample result. It rests on **3 trades in total** across all five out-of-sample
  segments. This is the trap the whole validity stack exists to catch, appearing in
  the output of the stack itself.
- **The checks that did run all point the same way.** DSR 0.000 across 320 trials
  says the Sharpe is indistinguishable from the best of 320 random tries. The
  shuffled-return null gives p=0.533: the strategy beat only 47% of permutations of
  its own returns, i.e. it did *worse* than its own noise. Neither is a close call.

What would change the verdict is **real data**, not more tuning:

```bash
pip install -r requirements-optional.txt          # ccxt / yfinance
tvlab data  --market crypto --style swing --data-source ccxt --export data/live
tvlab study --market crypto --style swing --data-source csv
```

Tuning parameters until the checks pass would make the report green and the
account red. Every number above came from a run where the parameter search,
the walk-forward splits and the null tests were all live. That is the point of
the tool.

---

## Quickstart

```bash
./scripts/setup.sh                    # venv + deps + clone the 18 vendored repos
. .venv/bin/activate

tvlab markets                         # what is configured
tvlab daily    --market us_equities --style swing     # today's shortlist, journaled
tvlab suggest  --market crypto --style swing          # the best idea, after costs
tvlab scan     --market crypto --style swing          # who is in a setup right now
tvlab backtest --market crypto --style swing          # single pass, portfolio roll-up
tvlab study    --market crypto --style swing          # the full validation run
tvlab serve    --market crypto --style swing          # dashboard on :8000
pytest                                                # 625 tests
```

`tvlab daily` is the one to schedule. It composes the other three stages and writes
a record - see [The daily run](#the-daily-run).

### `tvlab suggest` ranks by expectancy after costs, not by R:R

Reward:risk is a ratio of two distances, so it can be made arbitrarily large by
shrinking the stop. Sorting a shortlist on it therefore promotes the *tightest*
stops - the ones ordinary noise hits - and reports the best-looking numbers for
the worst ideas.

`tvlab suggest` ranks on the R:R **you actually get**. Costs are charged on
notional while risk is measured in R, so the conversion divides by the stop
distance:

```
round-trip cost = 2 * (fee_bps + slippage_bps) / 10000
cost_in_R       = round-trip cost / stop_distance          # what you pay to enter and exit
breakeven       = (1 + cost_in_R) / (1 + R:R)              # win rate needed to not lose money
```

At the crypto default of 30 bp round trip, a 0.20% stop costs **1.5R to place the
trade**, so even a 5:1 idea needs a 41.7% win rate - against 16.7% if costs were
zero. The same idea at a 1.50% stop needs 20.0%:

| stop distance | cost in R | breakeven at R:R 5 | at R:R 3 | at R:R 2 |
|---------------|-----------|--------------------|----------|----------|
| 0.20%         | 1.50      | **41.7%**          | 62.5%    | 83.3%    |
| 0.50%         | 0.60      | 26.7%              | 40.0%    | 53.3%    |
| 1.00%         | 0.30      | 21.7%              | 32.5%    | 43.3%    |
| 1.50%         | 0.20      | 20.0%              | 30.0%    | 40.0%    |
| 2.50%         | 0.12      | 18.7%              | 28.0%    | 37.3%    |

The breakeven win rate is **arithmetic on the fee schedule and the stop distance,
not a forecast** - it needs no view about the market, which is why it is the
threshold the tool presents rather than an expected return.

`suggest` then compares it against the win rate the strategy *actually achieved* in
your most recent `tvlab study`, states the sample size, **and reports the verdict
that win rate came from**:

```
$ tvlab suggest --market crypto --style swing
  round-trip cost 30.0 bp  ·  scanned 10 symbols  ·  0 candidate(s), 0 rejected  ·  max breakeven 60%
  measured on the last study: win rate 29.0% over 31 trades (thin sample), profit factor 0.58
  that study was NOT VALIDATED (3/8 checks passed): this win rate describes one historical sample, not an edge.
  failed: Deflated Sharpe (multiple testing), Profit factor > 1 after costs, Walk-forward windows profitable …and 2 more
```

A win rate quoted on its own reads as an endorsement. The same number next to the
study's verdict does not, which is the difference between a threshold comparison and
a reason to trade. Reports written before the verdict was serialised to JSON render
as *unknown* rather than as a pass - absence of evidence is not printed as evidence
of absence. The dashboard shows the same three cases.

Anything whose breakeven exceeds `--max-breakeven` (default 60%) is **rejected**,
not ranked lower: a lower rank implies a worse version of a good idea, and an idea
that cannot be taken at any win rate is not that. A stop below the risk mandate's
`min_stop_distance_pct` is rejected by that floor too, and the two gates are
independent - loosening one does not resurrect the other.

The score combines five weighted components (reward:risk 25%, cost drag 25%, stop
quality 20%, trend alignment 15%, location in a zone 15%). The weights are **not
fitted** - fitting them on 31 trades is noise mining, so they are fixed and stated
in `src/tvlab/screener/quality.py`.

`GET /api/suggest` serves the same structure to the dashboard, through the same
code path.

> A ranked shortlist is not a recommendation. Nothing in this repository has been
> validated as profitable - `tvlab study` is the authority on that, and its current
> verdict is **NOT VALIDATED**.

`tvlab study` writes `output/<strategy>_<market>_<tf>/` containing a markdown
report, a machine-readable JSON, the trade log, the equity curve and six charts.

Any config value can be overridden on the command line without editing YAML:

```bash
tvlab study --market forex --style intraday \
            --set risk.risk_per_trade=0.0025 \
            --set walk_forward.n_splits=6 \
            --allocation inverse_vol
```

---

## The daily run

`tvlab daily` is the command to schedule every morning. It composes three stages
that already existed and adds a record:

```
scan (is there a setup?)  ->  rank (can it be taken at these costs?)  ->  gate (is it worth owning?)
screener/scanner.py            screener/quality.py                        screener/fundamental.py
                                          |
                                          v
                                   output/daily/journal.jsonl
```

```bash
tvlab daily --market us_equities --style swing            # today's shortlist
tvlab daily --market indian_equities --style swing        # NSE large caps
tvlab daily --market us_equities --no-fundamentals        # technical only, no network needed
tvlab daily --market us_equities --set fundamentals.max_pe_ttm=25 --set fundamentals.min_roe=0.15
```

### Fundamentals: gates, not weights

For equities the technical half answers *"is there a setup right now?"* and cannot
answer *"is this a company worth owning while the setup plays out?"* The second
question is not optional: a textbook liquidity sweep into a name reporting earnings
tomorrow is a coin flip on a gap, not a trade.

`fundamentals:` in `config/markets/{us,indian}_equities.yaml` configures the gates.
Three design decisions matter:

**Nothing here is weighted into a score.** `quality.py` ranks on cost-adjusted
expectancy because that is arithmetic — a breakeven win rate follows from the fee
schedule and the stop distance and needs no forecast. A weighted blend of P/E, ROE
and growth has no such derivation, so any weights would be invented, and fitting
them on the ~31 trades this repo can measure is noise mining. Fundamentals **veto**;
the ranking of whatever survives stays with the part of the system that can justify
itself. Approved names are listed alphabetically, because every one of them passed
every configured gate and there is nothing honest to order them by.

**Missing data is a third outcome.** A symbol whose P/E did not arrive is neither
approved nor rejected — it is `unrated`, with the missing field named. Passing it
would let an unfiltered name into the shortlist; rejecting it would turn an endpoint
gap into a market verdict.

**The earnings blackout is the gate that earns its place.** A swing position held
through an earnings release is an unhedgeable gap: the stop does not protect against
it, because the position opens at the new price. It needs nothing but a date and
makes no statistical claim. When the date is unknown the symbol is unrated, not waved
through — "we do not know when they report" is not a pass.

What is switched on by default is risk control, not alpha: the earnings blackout, a
market-cap floor, positive free cash flow, a debt/equity ceiling and a current-ratio
floor. The valuation thresholds (`max_pe_ttm`, `min_roe`, `min_analyst_rating`, …)
ship as `null` **on purpose** — the honest position is that this repo has no evidence
for a number there, and a threshold would look rigorous while being arbitrary. Set
one with `--set` if you have a view you can defend.

> **These thresholds have never been backtested.** No claim is made that any of them
> improves expectancy. They are what a conservative mandate implies, nothing more.

### The journal is the point

A daily recommendation you cannot look back at is unfalsifiable: in a month nobody
remembers what was suggested, so nothing is ever checked. Every run appends one JSON
line to `output/daily/journal.jsonl` and writes the day's full briefing to
`output/daily/<date>.json`. Re-running the same day appends to the journal and
overwrites the dated file, so a second attempt is not mistaken for a second morning.

Each line records the study verdict that was in force, so a past shortlist can be
judged by what was known when it was made. `--no-journal` turns it off; the dashboard
only writes with `?journal=true`, because a GET that writes is a surprise and a
polling browser would fill the file with duplicates.

### What it refuses to do

The advisory line is printed **above** the shortlist, in the terminal, in the JSON
payload and in the dashboard panel:

```
Daily briefing - 2026-11-24  us_equities / swing / smc_trend / 4h
  The strategy behind this shortlist is NOT VALIDATED (3/8 checks) - it was not
  distinguishable from random entry in the study that produced this verdict.
  Treat the list as a screen, not as advice.
```

`tvlab daily` is a screen with arithmetic attached. It is not a stock tip, and it
will not name "the best stock to buy today" — the strategy behind the ranking has
not been shown to beat random entry on the evidence available here. If that changes,
the verdict in the study report changes and this framing changes with it: the labels
are read from `report.py`, not hardcoded.

Exit codes follow the contract the rest of the CLI uses: `0` when a shortlist was
produced or the market was observed and nothing qualified, `2` when **every** symbol
failed to load. Those have opposite remedies — "nothing qualifies today" means wait,
"we could not look" means fix the data route — so an outage is never allowed to
render as a quiet market.

Fundamentals need outbound access to `scanner.tradingview.com`. Offline, the run
still completes, every surviving name is flagged `NOT fundamentally screened`, and
the reason is recorded in the briefing's `notes`. The dashboard shows the same
warning in red.

---

## Running it on live data

Everything above runs on synthetic bars. This is the part that needs a machine
with network access — **and none of it was verified from the environment this was
built in**, which has no route to NSE, Yahoo, `scanner.tradingview.com` or any
exchange. The adapters are written against documented endpoint behaviour and
tested against reconstructed payloads. Treat the first live run as the real test.

```bash
pip install -r requirements-optional.txt     # ccxt, yfinance, pynesys-pynecore
```

| Market | Source | Command |
|:-------|:-------|:--------|
| crypto | `ccxt` | `tvlab study --market crypto --style swing --data-source ccxt` |
| us_equities | `yfinance` | `tvlab study --market us_equities --style swing --data-source yfinance` |
| indian_equities | `nse` | `tvlab study --market indian_equities --style swing --data-source nse` |
| forex | `yfinance` | `tvlab study --market forex --style swing --data-source yfinance` |
| any | your own files | `tvlab study --data-source csv` after `tvlab data --export data/live` |

### Fetching once, then studying offline

`--data-source csv` reads a directory of per-symbol files, which is how you fetch
live bars once and then iterate without hammering an exchange:

```bash
tvlab data  --market crypto --style swing --data-source ccxt --export data/live
tvlab study --market crypto --style swing --data-source csv
```

`data/live` is the default for `data.csv_root`, so those two commands line up with
no extra flag. Point it elsewhere with `--set data.csv_root=/path/to/bars`; relative
paths resolve against the repo root, not the process working directory, so the same
config behaves identically from a shell, a cron job and the dashboard.

Accepted layouts:

- `<root>/BTC_USDT_4h.parquet` — what `--export` writes. Symbol separators are
  flattened and the `_<timeframe>` suffix is optional; where several timeframes are
  present the requested one wins.
- `<root>/BTC_USDT.csv` (also `.parquet`, `.pq`, `.json`, `.jsonl`)
- one long file with a `symbol` column

Exporting only the execution timeframe is fine: when the higher timeframe the trend
filter wants is not on disk, `bias_frame()` resamples it up from the execution bars
and says so in the log. That is a fallback, not an error — but note it means the
bias is derived from the same bars, so it is a weaker filter than native
higher-timeframe data. Fetch the bias timeframe too if you can.

### Indian equities specifically

`--data-source nse` needs **no API key**. The NSE endpoints only require a
`User-Agent` header, which `src/tvlab/data/nse.py` sets. But NSE publishes no
history endpoint, so bars come from Yahoo's chart API — Indian cash symbols need
the `.NS` suffix, which `yahoo_symbol()` adds for you.

Before the open, these answer the questions that decide whether there is anything
to trade at all:

```bash
tvlab nse session      # is the cash market open, and for how much longer?
tvlab nse breadth      # advances vs declines across every NSE index
tvlab nse liquidity    # the most tradeable names right now, by traded VALUE
tvlab nse universe     # live Nifty 500 + F&O membership
tvlab nse holidays     # upcoming trading holidays
```

`breadth` matters more than it looks: `allow_short` is `false` in
`config/markets/indian_equities.yaml`, so on a day when declines outnumber
advances **there is no trade for this configuration to take**. That is a fact
about the mandate, not a failure of the scan.

### What a network failure looks like

`tvlab scan` and `tvlab suggest` distinguish three outcomes, because collapsing
them is how a data outage gets mistaken for a quiet market:

| Situation | Output | Exit |
|:----------|:-------|:-----|
| every symbol failed to load | `this is a data failure, not a market verdict`, first five errors, a retry command | **2** |
| some loaded, none fired | the market verdict, qualified with `(based on 4 of 5 symbols — the rest failed to load)` | 0 |
| all loaded, none fired | `Nothing qualifies. … This is the normal state.` | 0 |

Exit code 2 is there so a script or a cron job can tell the difference without
parsing prose.

### The order to do things in

```bash
tvlab data    --market indian_equities --data-source nse --limit 50   # 1. do bars load at all?
tvlab study   --market indian_equities --style swing --data-source nse # 2. is the edge validated?
tvlab suggest --market indian_equities --style swing --data-source nse # 3. only then, what to trade
```

Skipping step 2 is the mistake this tool exists to prevent. `suggest` will happily
rank ideas on data that has never been validated, which is why it prints the
measured win rate from your last `study` next to every breakeven threshold — and
says `no previous study` when there isn't one.

One hard limit worth knowing before you try the intraday style: Yahoo serves 1m
bars for roughly 8 days and 5m–90m bars for the last **60 days**. Sixty days is
about 40 sessions, and the intraday style's walk-forward wants 6 splits. An
intraday edge on Indian large caps is therefore not validatable on free data — you
would need a paid history source. Daily bars are unaffected.

---

## What it does

```
universe scanner  ->  signal engine  ->  backtester  ->  validation stack  ->  report + dashboard
     screener/         strategy/        backtest/         validation/           report.py, dashboard/
```

| Stage | Module | Notes |
|:------|:-------|:------|
| Data | `data/` | synthetic, CSV/Parquet, ccxt, yfinance, NSE, TradingView scanner (universe **and** fundamentals). Cached, validated, resampled. |
| Indicators | `indicators/` | technical + **lag-safe** smart-money concepts, cross-checked against the vendored reference |
| Risk | `risk/sizing.py` | risk-per-trade sizing, position caps, drawdown throttle and halt |
| Strategies | `strategy/` | registry + 4 setups: `smc_trend`, `ema_pullback`, `donchian_breakout`, `buy_and_hold` |
| Backtest | `backtest/engine.py` | next-open fills, pessimistic intrabar resolution, fees + slippage both legs, mark-to-market |
| Statistics | `backtest/{walkforward,montecarlo,metrics}` | walk-forward, block bootstrap, deflated Sharpe |
| Validation | `validation/` | lookahead audit, permutation nulls, pattern recovery |
| Portfolio | `portfolio/` | correlation-aware allocation under the risk mandate |
| Screener | `screener/` | one implementation, shared by CLI and dashboard |
| Fundamentals | `data/fundamentals.py`, `screener/fundamental.py` | valuation / quality / balance-sheet / dividend / analyst gates + the earnings blackout. **Gates, never weights** — see [The daily run](#the-daily-run) |
| Daily | `screener/daily.py` | composes scan + rank + gates, carries the study verdict, appends a falsifiable journal |
| Report | `report.py`, `study.py` | verdict, charts, markdown + JSON |
| Dashboard | `dashboard/` | FastAPI + lightweight-charts v5.3.0 (vendored, no CDN) |

### Markets and styles

| Market | Symbols | Exchange | Fee / slip (bps) | Shorts | Corr. threshold |
|:-------|--------:|:---------|:-----------------|:-------|----------------:|
| `crypto` | 10 | binance | 10 / 5 | yes | 0.60 |
| `us_equities` | 12 | NASDAQ | 0.5 / 1 | yes | 0.65 |
| `indian_equities` | 12 | NSE | 3 / 5 | **no** | 0.65 |
| `forex` | 7 | FX_IDC | 1 / 1.5 | yes | 0.40 |

| Style | Execution | Bias | Default strategy | Max hold |
|:------|:----------|:-----|:-----------------|:---------|
| `hft` | 1m | 15m | `ema_pullback` | 12 bars |
| `intraday` | 5m | 1h | `smc_trend` | 36 bars |
| `swing` | 4h | 1d | `smc_trend` | 90 bars |
| `position` | 1d | 1w | `donchian_breakout` | 120 bars |

`hft` is an OHLCV-only stub and disables walk-forward: 1-minute bars carry far
less independent information, so fitting parameters there is close to pure noise
mining.

### Risk mandate (conservative, per your specification)

```yaml
risk_per_trade:      0.005   # 0.5% of equity at risk per trade
max_drawdown_pct:    0.10    # HARD STOP: halt all trading at 10% equity DD
daily_loss_limit_pct: 0.02   # no new entries after -2% on the day
max_position_pct:    0.25
max_open_positions:  5
max_correlated_positions: 2
max_leverage:        1.0
```

These are asserted in `tests/test_config.py` across **all 16 market × style
combinations**. A style may tighten them; none may loosen them. The `position`
style originally set a 3% daily loss limit and was corrected.

---

## The validation stack

A backtest that only reports P&L is a marketing document. Every check below
exists because the opposite failure is common and silent.

**Lookahead audit** (`validation/lookahead.py`) — recompute every indicator and
strategy on truncated data and assert the overlapping values are bit-identical.
The test suite includes *positive controls*: deliberately leaked functions
(centred rolling means, full-sample normalisation, un-lagged pivots) that the
detector **must** catch. A detector that cannot find a known leak is theatre.

Result: all strategies and indicators pass, 0 leaks across 4 cut points.

**Why this matters here specifically:** the vendored `smartmoneyconcepts`
reference computes `swing_highs_lows` and `fvg` with *negative* shifts — it
publishes a pivot on the pivot bar, before the bars that confirm it exist.
`indicators/smc.py` is a from-scratch lag-safe implementation;
`indicators/reference.py` compares the two and confirms the reference's own
convention is non-causal.

An earlier revision of this paragraph claimed that feeding the non-causal
convention to a backtester "inflates profit factor to 3–10". That figure was
unsourced folklore and it does not reproduce here: measured over 5 seeds, the
non-causal convention *lowered* profit factor in 4 of them and produced fewer
trades in all 5 (`tests/test_reference_quarantine.py`). The reason is structural —
this pipeline derives sweep ages, structure state and zone ages *from* the swing
series, so publishing pivots early re-times every feature the strategy was designed
around rather than handing it future prices.

That makes the quarantine more important, not less. A look-ahead bug does not
reliably make results look better, so a disappointing backtest is not evidence of
honesty; the number is meaningless in either direction. Cross-validation on
1,500 bars: swing points match 94.4% at the documented 12-bar lag, fair value
gaps match 100% and the gap *levels* are identical after the 1-bar lag.

**Permutation nulls** (`validation/null_test.py`) — shuffle bar returns, rebuild
a plausible OHLC path (with wicks, or stops would never trigger) and re-run the
*same* strategy including its higher-timeframe bias. Also a random-entry null.

**Walk-forward** (`backtest/walkforward.py`) — fit on a training window, score on
the next contiguous out-of-sample segment, slide forward. Windows tile the last
40% of the series with no gaps and no overlap. The trial count feeds the
deflated Sharpe, so searching 64 parameter sets is honestly penalised.

**Deflated Sharpe** — adjusts for the number of attempts. A Sharpe of 1.0 found
once is not a Sharpe of 1.0 found after 64 tries.

**Monte Carlo** — block bootstrap over the trade sequence (blocks, not iid draws,
or drawdown is understated), reporting P(loss), the drawdown distribution and
P(breaching the 10% mandate).

**Pattern recovery** (`validation/recovery.py`) — inject known sweep-and-reverse
events into synthetic data and score the detector's precision/recall against
ground truth, with the trend filter separated out via an eligibility mask.
Without the mask you mostly measure the filter, not the detector.

---

## Bugs this project found in itself

These are recorded because each one produced *plausible* output while being
wrong, and each is now a regression test.

| # | Bug | Symptom | Fix |
|--:|:----|:--------|:----|
| 1 | `SizingResult.accepted` tested `units > 0` | **every short silently rejected**, reason `"ok"` | `abs(units) > 0` |
| 2 | "no stop" expressed as `1e-9` | rejected as `stop_too_tight` | `±inf` sentinel throughout |
| 3 | `-inf` read as "missing" | buy-and-hold became a 90-bar swing trade; benchmark reported **−0.53% while BTC rose +141%** | `np.isinf` passthrough |
| 4 | Style params leaked across strategies | switching `strategy.name` inherited the old setup's params | `StrategyConfig.params_for` |
| 5 | A param named `max_bars_in_trade` shadowed the method of the same name | `TypeError: 'int' object is not callable`, surfaced as "no symbol produced a result" | renamed to `resolve_max_bars`; `__init__` refuses to shadow callables |
| 6 | **HTF prices forward-filled, then indicators computed** | a "50-day EMA" was really a 50-*bar* (~8 day) EMA — a different, much faster filter | `align_indicator`: compute on native HTF bars, align the resulting *state*, shifted one bar |
| 7 | Synthetic daily and 4h series were **independent draws** | bias frame correlated **−0.64** with the execution frame; every entry gated on an unrelated market | resample for synthetic; `_series_drift` consistency guard rejects any native HTF fetch that disagrees |
| 8 | `bias.reindex(exec_idx)` in walk-forward | daily stamps vs 4h stamps → all-NaN bias → windows ran **with no trend filter at all** | slice by time range, preserving warmup |
| 9 | Correlation measured on **equity curves** | a selective strategy is flat ~97% of bars, so correlation read 0.02 while assets correlated 0.65; `max_correlated_positions` never fired | measure on underlying asset returns |
| 10 | Cap ordering: position count before correlation | on an equal-weight tie the diversifiers were dropped by dict order, leaving **50% cash in two 90%-correlated positions** | correlation cap first; ties break by liquidity then A–Z |
| 11 | `bars_per_year` used a 6.5h US session for **every** non-crypto market | 4h forex annualised at 409 bars/yr instead of 1560 → vol and Sharpe understated **~1.95×** | per-market session hours |
| 12 | Supertrend seeded with a NaN ATR | `upper[i] = upper[i-1]` propagated NaN forever; direction never flipped once in 900 bars | start the recursion at the first finite ATR |
| 13 | Cache key omitted the generator version | after fix #9, a 1500-bar run read 0.00 correlation and an 1800-bar run read 0.67 — same code, no warning | `CACHE_SCHEMA_VERSION` in the key |
| 14 | Unallocated capital treated as nothing, not cash | report showed **"Net P&L −49,916"** next to **"Total return +0.17%"** for the same run | idle cash added back; curve starts at the account balance |
| 15 | `fan_chart` quantiles along `axis=1` | bands had one value per simulation, not per bar; plot raised a length error | `axis=0` |
| 16 | Walk-forward used a fixed 8-trade floor | a selective setup trading ~12 times in 2,500 bars had **every** window discarded — "no usable fit" ×3 | adaptive floor + graceful `expectancy` fallback with a `degraded` flag |
| 17 | `_regime`/`_realised_vol` assigned **after** `_apply_session` filtered the index | `ValueError: Length of values (4000) does not match length of index (667)` — **every non-crypto market, every intraday timeframe, every bar count**. Indian and US equities were unusable on synthetic data: scan, ohlcv, indicators and backtest all failed | attach ground truth *before* the filter, so it is dropped with the bars it describes |
| 18 | `_column_lookup` cleaned keys with a `[^A-Z0-9]` class | lowercased labels collapsed to `""`, one key matched every lookup, and **every fundamental silently read the same wrong column** | case-insensitive class; an empty key is never indexed |
| 19 | `build_briefing` treated "never fetched" as "fetched nothing" | with the gates on, an empty `snaps` silently moved every candidate to `no_data` and the shortlist vanished — reading as a market with no qualifying names | `snaps is None` is handled as an outage: names survive, flagged unscreened |
| 20 | `_set_outcome` keyed the no-signal branch off the shortlist length | two signals fired and were screened out on fundamentals, and the output said **"no strategy fired on the last bar of any symbol"** — sending the user looking for setups that were there | branch on `signals_found`; add `screened_out` and `unscoreable` outcomes |

Bug 17 was found by reading the running dashboard's log, not by review: it showed
every `*.NS` symbol failing with the same length error across four different
endpoints while the crypto study beside it worked. The fix is bit-identical for
crypto and daily bars — `tests/test_synthetic_session_filter.py` pins close-sum
fingerprints so a future reordering that shifts the RNG order fails loudly rather
than quietly invalidating every number quoted above.

Two were metric traps rather than code traps: `trade_stats()["exposure_pct"]` was
a bar *count* labelled as a percentage, and scoring a trend-following setup's
recall against direction-agnostic injected sweeps mostly measures the trend
filter. Both are documented at the call site.

---

## Layout

```
config/            default.yaml + markets/{crypto,indian_equities,us_equities,forex}.yaml
                            + styles/{hft,intraday,swing,position}.yaml
src/tvlab/         config, data/, indicators/, risk/, strategy/, backtest/,
                   validation/, portfolio/, screener/, pipeline, study, report, cli,
                   dashboard/
dashboard/static/  index.html + vendored lightweight-charts 5.3.0 (Apache-2.0, no CDN)
scripts/           setup.sh (venv + deps + vendor), fetch-vendor.sh (rebuild vendor/)
tests/             625 tests, 26 files - including lookahead positive controls
vendor/            18 reference repos, gitignored (632 MB); repos.txt + LOCK.json tracked
output/            generated reports + output/daily/ journal, gitignored
```

13,966 lines of source, 7,661 lines of tests.

---

## `vendor/` — the 18 cloned repositories

`vendor/` is **gitignored** (632 MB, 26,647 files). Two files are tracked so the
tree is reproducible:

- `vendor/repos.txt` — the 18 repos
- `vendor/LOCK.json` — pinned SHA, branch, date, stars and **license** for each

Rebuild with `./scripts/fetch-vendor.sh` (`--depth N` for a shallow fetch). It
strips nested `.git` directories. `koala73/worldmonitor` fails a plain clone
(`RPC failed; curl 56`) and is fetched as a codeload tarball instead.

**Licensing constrains how these may be used**, and the code respects it:

| License | Repos | Constraint honoured here |
|:--------|:------|:-------------------------|
| GPL-3.0 | `tradingview-pinescript-indicators`, `freqtrade` | copyleft — reference only, no code copied into `src/tvlab/` |
| AGPL-3.0 | `worldmonitor` | must not be embedded in the served dashboard |
| MIT | `market-lab`, `pynecore-strategies`, `TradingView-Screener`, `tradingview-mcp` (both), `pytrendline`, `smart-money-concepts`, `congress-trading-monitor` | attribution |
| none | `algorithmic-trading-python` | no reuse |
| NOASSERTION | `tradingview-mcp` (tradesdontlie), `journalit` | reference only |
| Apache-2.0 | `lightweight-charts`, `tvscreener`, `pynecore`, `KLineChart`, `VisualHFT` | attribution; the chart bundle is vendored with its LICENSE file |

Nothing from a GPL/AGPL/unlicensed repo is imported at runtime. The one runtime
PyPI dependency drawn from this family is `smartmoneyconcepts`, used **only** by
`indicators/reference.py` for cross-validation — never in a trading path, because
it looks ahead.

**[`VENDOR.md`](VENDOR.md) records what was taken from each of the 18 repos, and
why the other twelve were not.** Short version: 5 used at runtime, 1 evaluated
and rejected then reimplemented natively (`pytrendline` is incompatible with
pandas 2 and roughly cubic), 12 reference-only — four of those because copying
them would violate their licence.

---

## Limitations

Stated plainly, because each one changes what the output means.

- **No live data in this environment.** No route to exchanges, Yahoo or
  `scanner.tradingview.com`. `ccxt` and `yfinance` are not installed. The
  adapters fail loudly with the exact `pip install` line — they never return an
  empty frame, because an empty frame looks like "no trades" and produces a flat,
  plausible equity curve.
- **All published numbers are synthetic.** They characterise the *engine*, not a
  strategy's edge.
- **`hft` is a stub.** 1-minute OHLCV without order-book or latency modelling
  cannot support HFT conclusions. Costs alone would dominate.
- **Single-position-per-symbol.** No pyramiding, no partial exits, no hedging.
- **Correlation is estimated from realised returns**, not modelled. In a genuine
  risk-off move correlations go to 1 and the groups here would understate it.
- **The TradingView paths are untested live** for the reason above — both the
  universe scanner and the fundamentals adapter. `tvscreener`'s API is introspected
  and guarded with `hasattr`, every field name is resolved through a candidate list
  rather than hardcoded, and the parsing is tested against fixtures rather than
  captures. `tests/test_fundamentals.py` asserts that all 29 canonical fields still
  resolve against the installed enum, so an upstream rename fails loudly in CI
  instead of quietly producing a table of Nones.
- **Fundamental thresholds are not backtested.** The gates in
  `config/markets/*_equities.yaml` are what a conservative mandate implies, not a
  measured edge. No claim is made that any of them improves expectancy, and they
  veto rather than score precisely so that no invented weight can enter the ranking.
- **`tvscreener` also ships capability this repo does not use yet**: financial
  statement history (~20 fiscal years of income statement, balance sheet, cash flow
  and statistics), corporate events, monthly seasonality, earnings/dividend/IPO/
  economic calendars, peers, news, options surfaces and yield curves. See
  `VENDOR.md` §4 for the inventory and why each was left out.
- **The `smc_trend` setup is deliberately rare.** On the default 4,000 bars of
  synthetic 4h data it takes 31 trades across a 10-symbol universe, of which the
  risk caps fund 2. Loosening it to generate more trades was tried and **measured
  worse**: `entry_mode: sweep_or_zone` also accepts a retracement into an
  unmitigated FVG or order block, and across 10 samples it multiplied the
  opportunity set 10-23x while making total return, Sharpe and max drawdown worse
  in **10 of 10**. `tests/test_entry_mode.py` pins that.
- **Profit factor is the wrong instrument for that comparison**, and this README
  used to make the argument with it. The same bullet previously claimed zone entries
  degraded PF "from ~0.9 to ~0.58 over 93 trades"; no seed at any bar count tested
  reproduces those numbers. PF moved in *both* directions - worse in 5 samples of
  10, better in 5 - because it is gross profit over gross loss and is blind to how
  often capital is deployed and to fee drag. Taking 17x more trades at a similar PF
  is a much worse outcome that PF itself does not report. Judge that switch on
  return, Sharpe and drawdown.

---

## Notes for extending it

- **Strategies must not pre-shift signals.** A signal on bar `i` is filled at the
  open of bar `i+1` by the engine. Shifting in the strategy double-delays it.
- **Never name a method the same as a parameter.** `Strategy.__init__` sets
  params as attributes and now refuses to shadow a callable, but the collision is
  still confusing to read (bug #5).
- **Express "no stop" as `-inf`, never as a tiny number.** A tiny number is a
  valid stop and will be rejected by the stop-distance bounds.
- **Compute higher-timeframe indicators on native HTF bars**, then align the
  resulting state with `data.base.align_indicator`. Forward-filling HTF *prices*
  onto the execution index first silently shortens every lookback (bug #6).
- **Bump `CACHE_SCHEMA_VERSION`** whenever the meaning of a cached bar changes.
- **Generate config YAML with Python, not shell `sed`.** Quoting will produce
  malformed files that fail at load time in a different directory.
