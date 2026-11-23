# tvlab — a trading research workbench that tries to prove you wrong

Multi-market, config-driven backtesting, validation, screening and a live
lightweight-charts dashboard. One engine, 4 markets × 4 trading styles = 16
selectable combinations, conservative risk mandate enforced in code.

---

## Read this first: the honest result

**The default strategy is NOT validated, and on the bundled data it should not
be.**

Running the full study today:

```
VERDICT: NOT VALIDATED                          4 of 8 checks passed

 1  Enough trades to measure            FAIL    13 trades - below 30 every ratio is noise
 2  Deflated Sharpe (multiple testing)  FAIL    DSR=0.009 over 64 trials
 3  Drawdown within mandate             PASS    max DD 0.37% vs limit 10.0%
 4  Profit factor > 1 after costs       PASS    PF=1.09
 5  Walk-forward windows profitable     FAIL    1/4 OOS windows profitable
 6  OOS retains IS performance          FAIL    mean OOS/IS ratio = -0.27
 7  Beats shuffled-return null          FAIL    p=0.200
 8  Monte Carlo DD-breach risk          PASS    P(max DD >= 10%) = 0.0% of 500 sims

 Portfolio: +0.08% return, 0.37% max DD, 13 trades, PF 1.09, Sharpe 0.16
```

Why this is the correct answer rather than a failure of the project:

- **The default data is synthetic.** No exchange or broker is reachable from this
  environment, so `tvlab` ships a generator (Markov regime chain + GARCH(1,1)
  volatility + Student-t tails). Synthetic prices contain regime drift and
  volatility clustering, but **no exploitable microstructure**. A cost-bearing
  strategy on such data should converge to roughly zero minus costs. It does:
  +0.08% over 2,500 bars with 541 in fees.
- **13 trades cannot support any conclusion.** Check 1 exists to stop you reading
  a 38.5% win rate as information. It isn't.
- **Check 6 is the important one.** Parameters chosen in-sample lost their edge
  out of sample (ratio −0.27). That is what overfitting looks like when you
  measure it instead of hoping.

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
tvlab suggest  --market crypto --style swing          # the best idea, after costs
tvlab scan     --market crypto --style swing          # who is in a setup right now
tvlab backtest --market crypto --style swing          # single pass, portfolio roll-up
tvlab study    --market crypto --style swing          # the full validation run
tvlab serve    --market crypto --style swing          # dashboard on :8000
pytest                                                # 385 tests
```

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
your most recent `tvlab study`, and states the sample size:

```
$ tvlab suggest --market crypto --style swing
  round-trip cost 30.0 bp  ·  scanned 10 symbols  ·  max breakeven 60%
  measured on the last study: win rate 38.5% over 13 trades
  (far too few trades to trust), profit factor 1.09
```

Anything whose breakeven exceeds `--max-breakeven` (default 60%) is **rejected**,
not ranked lower: a lower rank implies a worse version of a good idea, and an idea
that cannot be taken at any win rate is not that. A stop below the risk mandate's
`min_stop_distance_pct` is rejected by that floor too, and the two gates are
independent - loosening one does not resurrect the other.

The score combines five weighted components (reward:risk 25%, cost drag 25%, stop
quality 20%, trend alignment 15%, location in a zone 15%). The weights are **not
fitted** - fitting them on 13 trades is noise mining, so they are fixed and stated
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
| Data | `data/` | synthetic, CSV/Parquet, ccxt, yfinance, TradingView scanner. Cached, validated, resampled. |
| Indicators | `indicators/` | technical + **lag-safe** smart-money concepts, cross-checked against the vendored reference |
| Risk | `risk/sizing.py` | risk-per-trade sizing, position caps, drawdown throttle and halt |
| Strategies | `strategy/` | registry + 4 setups: `smc_trend`, `ema_pullback`, `donchian_breakout`, `buy_and_hold` |
| Backtest | `backtest/engine.py` | next-open fills, pessimistic intrabar resolution, fees + slippage both legs, mark-to-market |
| Statistics | `backtest/{walkforward,montecarlo,metrics}` | walk-forward, block bootstrap, deflated Sharpe |
| Validation | `validation/` | lookahead audit, permutation nulls, pattern recovery |
| Portfolio | `portfolio/` | correlation-aware allocation under the risk mandate |
| Screener | `screener/` | one implementation, shared by CLI and dashboard |
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
publishes a pivot on the pivot bar, before the bars that confirm it exist. Fed to
a backtester that inflates profit factor to 3–10. `indicators/smc.py` is a
from-scratch lag-safe implementation; `indicators/reference.py` compares the two
and confirms the reference's own convention is non-causal. Cross-validation on
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
tests/             209 tests, 8 files - including lookahead positive controls
vendor/            18 reference repos, gitignored (632 MB); repos.txt + LOCK.json tracked
output/            generated reports, gitignored
```

9,877 lines of source, 2,043 lines of tests.

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
- **The scanner's TradingView path is untested live** for the reason above;
  `tvscreener`'s API is introspected and guarded with `hasattr`.
- **The `smc_trend` setup is deliberately rare.** On 2,500 bars of synthetic 4h
  data it takes 13 trades across a 10-symbol universe. Loosening it to generate
  more trades was tried and **measured worse** (entry on FVG/order-block
  retracement as well as sweeps: 93 trades, profit factor 0.58 vs 0.9). The
  default is the doctrinal sweep entry, and the alternative is left in the code
  behind `entry_mode` with that result documented beside it.

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
