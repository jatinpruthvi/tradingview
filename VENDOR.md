# `vendor/` — what was actually taken from each of the 18 repositories

This file exists because "we vendored 18 repos" is not the same as "we used 18
repos". Each entry below states what was taken, where it lives in `src/tvlab/`,
and — where nothing was taken — the specific reason.

`vendor/` itself is gitignored (632 MB, 26,647 files). `vendor/repos.txt` and
`vendor/LOCK.json` are tracked and pin SHA, branch, date, stars and license.
Rebuild with `./scripts/fetch-vendor.sh`.

**Score: 5 used at runtime, 1 evaluated and rejected then reimplemented natively,
12 reference-only.** The reference-only ones are not oversights; each has a
reason, and for four of them the reason is a licence that would be violated by
copying.

A second audit pass went looking for capability that was vendored but unused, on
the reasonable suspicion that "used" had been judged too narrowly. It found one:
`tvscreener` had been characterised as a universe screener when it also carries a
278-line fundamentals module, a websocket quote/study client, **3,666 `StockField`
members** (mostly fundamental) and calendar/news/peers/options/rates APIs. The
fundamentals are now used — see §4, which also records what is still unused and why
each item was left out rather than adopted. The other 17 entries were re-checked and
their verdicts stand.

| # | Repository | Licence | Verdict |
|--:|:-----------|:--------|:--------|
| 1 | heysanjeevv/market-lab | MIT | **Used** — NSE data layer |
| 2 | tradingview/lightweight-charts | Apache-2.0 | **Used** — dashboard charting |
| 3 | joshyattridge/smart-money-concepts | MIT | **Used** — cross-validation only |
| 4 | deepentropy/tvscreener | Apache-2.0 | **Used** — universe screener + fundamentals |
| 5 | ednunezg/pytrendline | MIT | **Rejected, reimplemented** |
| 6 | shner-elmo/TradingView-Screener | MIT | Reference — superseded for India by #1 |
| 7 | atilaahmettaner/tradingview-mcp | MIT | Reference — separate MCP server |
| 8 | tradesdontlie/tradingview-mcp | MIT | Reference — needs TradingView Desktop |
| 9 | PyneSys/pynecore | Apache-2.0 | **Used in tests** — Pine-semantics reference |
| 10 | theonetrade/pynecore-strategies | MIT | **Used in tests** — via #9's `ta` built-ins |
| 11 | klinecharts/KLineChart | Apache-2.0 | Reference — second charting library |
| 12 | kadoa-org/congress-trading-monitor | MIT | Reference — US-only alt data |
| 13 | visualHFT/VisualHFT | Apache-2.0 | Reference — needs order-book data |
| 14 | everget/tradingview-pinescript-indicators | GPL-3.0 | Reference — copyleft |
| 15 | freqtrade/freqtrade | GPL-3.0 | Reference — copyleft |
| 16 | koala73/worldmonitor | AGPL-3.0 | Reference — copyleft, network clause |
| 17 | Cursivez/journalit | NOASSERTION | Reference — licence unclear |
| 18 | nickmccullum/algorithmic-trading-python | none | Reference — all rights reserved |

---

## Used at runtime

### 1. `market-lab` (MIT) — the NSE data layer

The single most valuable repo here, and the one that was closest to being missed.
It is a self-contained NSE dashboard that fetches public data with nothing but
`urllib`, `numpy`, `pandas` and `tzdata`.

**Taken:** the endpoint knowledge, the IST regular-session window, and the
index-to-Yahoo-symbol map.

- `https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv` — Nifty
  500 constituents as CSV
- `https://www.nseindia.com/api/NextApi/apiClient/marketWatchApi?functionName=getIndicesData&symbol=…`
  — **every member of an index in one call**, with last price, change, day range,
  volume, traded value and 52-week range. Also accepts `SECURITIES IN F&O`.
- `https://www.nseindia.com/api/allIndices` — index levels plus **breadth**
  (advances/declines) and PE
- `https://www.nseindia.com/api/holiday-master?type=trading` — trading calendar
- `https://query1.finance.yahoo.com/v8/finance/chart/{sym}` — historical bars,
  **with no `yfinance` dependency**
- NSE needs only a `User-Agent` header; there is no cookie handshake
- Session is 09:15–15:30 IST (minutes 555–930); indices map to caret symbols
  (`NIFTY` → `^NSEI`, `BANKNIFTY` → `^NSEBANK`, `SENSEX` → `^BSESN`)

**Lives in:** `src/tvlab/data/nse.py` (`NseSource`, `YahooChartSource`,
`filter_ist_session`, `top_by_liquidity`, `yahoo_symbol`), wired into
`DataStore.source()` as `nse` / `yahoo_chart`, exposed as the `tvlab nse` command
group (`session`, `breadth`, `liquidity`, `universe`, `holidays`).

**Why it matters:** it removes the two blockers on live Indian data. `yfinance`
was an uninstalled optional dependency, and the Indian universe was 12 hand-typed
tickers that go stale. Both are now solved with the standard library.

**Not copied:** no code. `market-lab`'s modules are application-specific (its own
cache layout, its own SMC implementation, a threaded live poller). Only the
endpoint contracts and constants were taken, which are facts about NSE's API
rather than expression. Attribution is in the module docstring.

**48 tests** in `tests/test_nse_source.py` and `tests/test_cli_nse.py` cover the
parsers against recorded-shape payloads. **Those shapes were reconstructed from
market-lab's field usage, not captured live** — this sandbox reaches neither
host. Run `tvlab nse breadth` on a machine with network access before trusting
them.

### 2. `lightweight-charts` (Apache-2.0) — dashboard charting

**Taken:** the v5.3.0 standalone production bundle, pulled from npm because the
repo has no committed `dist/`.

**Lives in:** `dashboard/static/vendor/lightweight-charts.standalone.production.js`
(198 KB, tracked in git), served by the FastAPI app. The dashboard is fully
offline — no CDN, which is why the bundle is committed rather than linked.

### 3. `smart-money-concepts` (MIT) — cross-validation only

**Taken:** the PyPI package `smartmoneyconcepts`, used to check this repo's
from-scratch SMC implementation against a widely-used reference.

**Lives in:** `src/tvlab/indicators/reference.py`, and nowhere else.

**Why only there:** its `swing_highs_lows` and `fvg` use *negative* shifts, so
they publish a pivot on the pivot bar, before the bars that confirm it exist.
`indicators/smc.py` is a lag-safe reimplementation; `reference.py` compares the two
and demonstrates the reference's own non-causality. Cross-validation on 1,500 bars:
swing points match 94.4% at the documented 12-bar lag, fair value gaps match 100%
with identical levels after a 1-bar lag.

**It is never imported by a strategy, the scanner or the sizer, and no decision
depends on it.** `tests/test_reference_quarantine.py` enforces that three ways: an
AST scan pins the importer set to exactly `indicators/reference.py`, `cli.py` (the
`validate` command) and `pipeline.py`; and a full study is run with
`smartmoneyconcepts` blocked at `sys.meta_path`, recording every import attempt,
which must come back empty.

The `pipeline.py` import deserves precision, because an earlier revision of this
file claimed the engine never imports it, which was literally false. It does —
deferred, inside `if cross_validate:`, with its result attached to
`Study.cross_validation` and wrapped so a missing library only drops that section of
the report. It cannot reach a signal, an entry, an exit or a position size.

**Correction to a figure this file used to carry.** It claimed that feeding the
reference's non-causal convention to a backtester "inflates profit factor to 3–10".
That was unsourced and it does not reproduce: measured over 5 seeds the convention
*lowered* profit factor in 4 of them and produced fewer trades in all 5. This
pipeline derives sweep ages, structure state and zone ages from the swing series, so
early pivots re-time the features rather than leak future prices into them. The
quarantine stands on non-causality, not on the direction of the distortion — a
look-ahead bug that makes results look *worse* is still meaningless, and is arguably
more dangerous, because a bad backtest does not invite suspicion.

### 4. `tvscreener` (Apache-2.0) — universe screener **and** fundamentals

**Taken:** the PyPI package `tvscreener`, for pulling a ranked universe from
TradingView's scanner — and, since the audit below, for fundamentals.

**Lives in:** `src/tvlab/data/tv_screener_source.py` (universe), reachable via
`--data-source tvscreener` and `tvlab scan --screener`; and
`src/tvlab/data/fundamentals.py` (fundamentals), reachable via `tvlab daily`,
`GET /api/daily` and the `fundamentals:` config section.

**Caveats:** the documented import paths in its README (`tvscreener.core.stock`,
`tvscreener.screener`) do not exist; the working classes are `StockScreener` and
`CryptoScreener` from the package root, so every access is `hasattr`-guarded. It
has never been executed against the live endpoint from this repo — no network
route — so it is best treated as unverified.

#### What the audit found

This section used to describe `tvscreener` as a "universe screener", which
understated it badly enough to be worth recording. Introspecting the installed
0.8.2 package locally — no network needed for any of this — shows:

| Surface | Size | Used here? |
|:--------|:-----|:-----------|
| `StockField` enum members | **3,666**, mostly fundamental | 29 canonical fields via `FIELD_CANDIDATES` |
| `fundamentals.py` | 278 lines | no — see below |
| `_websocket.py` quote/study client | live streaming + `run_studies` | no |
| calendars | earnings, dividends, IPO, economic | no |
| screeners | `Index`, `ETF`, `Forex`, `Futures`, `Coin`, `Bond`, `GovernmentBond`, `Economics`, `Option` | only `Stock` and `Crypto` |
| other | `get_peers`, `get_news`, `get_earnings_calls`, `get_documents`, presets, options surfaces, yield curves, a `ta` module, an MCP server | no |

Each member's `.value` is a 5-tuple `(display_name, scanner_column, format, bool,
bool)`, e.g. `MARKET_CAPITALIZATION → ('Market Capitalization', 'market_cap_basic',
'currency', False, False)`. The **second** element is the column the scanner
actually returns, and `tvscreener` builds its request as `{field_name: label}` —
so a frame can be keyed by either, depending on version. `_column_lookup` indexes
both. Matching only one convention produces an all-`None` table that looks exactly
like a market with no fundamentals, which is why this is worth stating.

`tests/test_fundamentals.py` asserts that all 29 canonical fields still resolve
against the installed enum, so an upstream rename fails in CI rather than as a
silently empty column.

Field names verified present (probe with `hasattr` before coding against any of
them): `MARKET_CAPITALIZATION`, `PRICE_TO_EARNINGS_RATIO_TTM`, `PRICE_TO_BOOK_MRQ`
/ `_FY`, `PRICE_TO_SALES_FY`, `ENTERPRISE_VALUEEBITDA_TTM`, `GROSS_MARGIN_TTM`,
`OPERATING_MARGIN_TTM`, `NET_MARGIN_TTM`, `RETURN_ON_EQUITY_TTM` / `_FY`,
`RETURN_ON_ASSETS_TTM`, `RETURN_ON_INVESTED_CAPITAL_TTM`, `DIVIDENDS_YIELD_CURRENT`,
`DIVIDEND_PAYOUT_RATIO_TTM`, `BASIC_EPS_TTM`, `EPS_DILUTED_TTM`,
`NET_INCOME_FY`, `FREE_CASH_FLOW_TTM`, `DEBT_TO_EQUITY_FQ`, `CURRENT_RATIO_FQ`,
`CASH_DIVIDEND_COVERAGE_RATIO_TTM`, `SECTOR`, `INDUSTRY`, `RECENT_EARNINGS_DATE`,
`UPCOMING_EARNINGS_DATE`, `BETA_3_YEAR` / `_5_YEAR`, `RECOMMENDATION_MARK`.

Growth naming is irregular, so the canonical names were chosen from what actually
exists rather than from what reads naturally: revenue growth is
`REVENUE_TTM_YOY_GROWTH` (also `REVENUE_ANNUAL_YOY_GROWTH`), and EPS growth is
`EPS_DILUTED_TTM_YOY_GROWTH`.

**Names that do NOT exist** — verified by `hasattr` against the installed enum, and
recorded because guessing them is the fastest way to a silent `None`: there is no
`BETA_1_YEAR`, no `CLOSE` (prices are `PRICE`), no `MARKET_CAP` (it is
`MARKET_CAPITALIZATION`), no `REVENUE_GROWTH_FY`, no `EPS_GROWTH_FY`, no
`DEBT_TO_EQUITY_MRQ`, and no `ANALYST*` or `RATING*` stock field. Analyst consensus
is `RECOMMENDATION_MARK` plus `RECOMMENDATION_{BUY,HOLD,OVER,UNDER,SELL,TOTAL}` and
`RECOMMEND_ALL_1`.

An earlier draft of `FIELD_CANDIDATES` listed eight of those non-existent names as
"alternates", on the theory that a future version might use them. They were removed:
a fallback for a spelling that has never existed cannot be tested, so it is not a
safeguard, it is an unverified claim sitting in a table that looks authoritative.
`tests/test_fundamentals.py::test_no_candidate_name_is_speculative` now fails if any
name in the table is absent from the enum, and
`test_every_canonical_field_resolves_against_the_installed_enum` fails if a
canonical field stops resolving at all.

#### Why the rest is not used

Not neglect — each has a reason, and the reasons are the useful part:

- **`fundamentals.get_financials`** (statement history, ~20 fiscal years and 32
  quarters) is the right tool for computing growth and quality *trends* rather than
  reading TradingView's pre-computed ratios. It reads TradingView's data websocket.
  Left out because the gates here are deliberately simple vetoes, and a 20-year
  statement history invites exactly the fitted composite score that
  `screener/fundamental.py` refuses to build.
- **`get_corporate_events`** returns dividends, splits *and* earnings releases. The
  earnings date is already taken from `UPCOMING_EARNINGS_DATE` in the same scanner
  call, so this would be a second request for one field — but it is the correct
  source if the blackout ever needs dividends or splits too.
- **`_websocket.run_studies` / `stream_quotes`** would give live prices instead of a
  point-in-time snapshot. Not used because every consumer here is a daily batch;
  a streaming client that nothing polls is a connection that only fails.
- **The other screeners** (`Bond`, `Option`, `Economics`, …) have no counterpart in
  the four configured markets. `IndexScreener` is the nearest thing to a useful
  addition — regime context for the bias filter — and is recorded as the first
  candidate if this grows.
- **Calendars** would let the daily run warn about macro events the way it warns
  about earnings. Held back because an economic calendar has no obvious gate: unlike
  an earnings date, there is no defensible rule for how many days before a CPI print
  a swing trade becomes untradeable.
- **`tvscreener.mcp`** is a server exposing this same data to an MCP client. Not
  used; this repo is the client.

---

## 5. `pytrendline` (MIT) — evaluated, rejected, reimplemented

The clearest case of "taking advantage of a repo" meaning something other than
importing it. It does automatic trendline fitting, which `indicators/smc.py`
cannot: that module finds *horizontal* structure (swings, BOS, CHoCH, FVGs, equal
highs/lows, liquidity pools) and says nothing about *diagonal* structure.

Four independent blockers, each verified against the installed 1.0.1:

1. **Incompatible with pandas 2.** `detect.py` calls `DataFrame.append()`,
   removed in pandas 2.0. This repo pins `pandas>=2.1,<3`, so it raises
   `AttributeError` the first time a trendline is actually found. Fixing it means
   patching a vendored library or downgrading pandas and breaking everything else.
2. Its documented `datetime_col=None` path calls `self.df.set_index(None)` →
   `KeyError: 'None of [None] are in the columns'`.
3. Its constructor discards the results of its own `rename(...)` and
   `reset_index()`, so `detect()`'s `df.loc[0].name` requires a caller-supplied
   0-based integer index.
4. **Roughly cubic.** Measured: 120 daily bars = 3.9 s, 400 bars = 141 s.

**Taken:** the approach — fit lines through pivot pairs, score by how many pivots
respect them, group near-duplicates.

**Lives in:** `src/tvlab/indicators/trendlines.py`, implemented natively on top of
this repo's already-validated causal pivot detector (`smc.swing_points(lag=True)`).
**12 ms on 600 daily bars** versus 141 s on 400. Surfaced via
`tvlab scan --trendlines` (support %, resistance %, channel %) and
`GET /api/scan?trendlines=1`. 25 tests in `tests/test_trendlines.py`.

The rejection rationale is also recorded in `requirements-optional.txt`, so the
next person does not re-add it as a dependency and rediscover all four bugs.

**One real bug found while porting the idea:** `swing_points(lag=True)` publishes
a pivot at `i + L` carrying the price from bar `i`. Anchoring a line at the
publication index draws it through `(i + L, low[i])` — a point that never existed
— so every slope and every extension is wrong by an error that grows with
`swing_length`. `trendlines._pivots` subtracts `L` to recover the true anchor
while keeping the causal filtering. A test asserts each fitted line passes through
its own anchor pivot to within 1e-6.

---

## Reference only

Each of these was read and deliberately not integrated. The reason is recorded so
the decision is not silently reversed later.

### 6. `shner-elmo/TradingView-Screener` (MIT)

A cleaner TradingView scanner client than #4: a typed query builder
(`stocks('india')`, `crypto()`, `forex()`, `futures()`, `bond()`, `options()`),
`col(...)` expressions, and `get_scanner_data() -> (total, DataFrame)`.

**Not integrated** because for India — the market that motivated this audit — #1
is strictly better: NSE's own API is a documented public endpoint returning the
full index membership with traded value in a single call, whereas a scraper client
depends on an undocumented private endpoint that TradingView changes without
notice. Adding a second untestable screener backend would also repeat the
duplication that already caused drift between the CLI and the dashboard scan.

**Use it if** you want a non-Indian universe ranked by TradingView's own
technical rating. The API to start from is `tradingview_screener.stocks(market)`.

### 7. `atilaahmettaner/tradingview-mcp` (MIT)

An MCP server: real-time market data, 39 technical-analysis tools, screeners,
Yahoo Finance and backtesting for any MCP client.

**Not integrated** because it is a *server*, not a library, and tvlab already has
a data layer with its own validation, caching and lookahead discipline. Importing
its indicators would mean trusting a second implementation of ATR/RSI/ADX whose
causality has not been audited — the exact mistake #3 is quarantined for.

**Use it if** you want live TradingView data inside an MCP-capable client. Run it
as a separate process; do not vendor its indicator code into a backtest path.

### 8. `tradesdontlie/tradingview-mcp` (MIT)

Connects an AI client to a locally running **TradingView Desktop** app over the
Chrome DevTools Protocol.

**Not integrated.** It requires a paid TradingView subscription, a running desktop
app, and reads undocumented Electron internals that its own README warns "can
change or break without notice in any TradingView update". That is too fragile a
foundation for a data layer, and it cannot run headless on a server.

**Licence note:** `vendor/LOCK.json` recorded this as `NOASSERTION` because that
is what the GitHub API returned. The repo's `LICENSE` file reads *"MIT License /
Copyright (c) 2026 tradesdontlie"*. Corrected in the lock file with a note. A
lock file that disagrees with the licence it is supposed to record is worse than
no lock file.

### 9. `PyneSys/pynecore` (Apache-2.0)

Runs Pine Script-style code in Python via AST transformation, with genuinely
impressive verification: 1,000 published TradingView scripts, 99.771% of 132
million plotted values bit-identical to TradingView, 353,577 strategy trades
matching its timing.

**Not integrated as an engine.** It is a complete backtesting engine and tvlab is
one too; adopting it would mean two engines with two sets of fill, cost and sizing
semantics — the kind of duplication that already caused a real bug here — or
rebuilding tvlab's validation stack on top of it.

**Used as a reference oracle in the test suite.** Its `ta` module implements
TradingView's Pine built-ins, which makes it the only available authority on what
`ta.atr` or `ta.rsi` actually mean. `scripts/vendor_xval_indicator.py` plots those
built-ins over bars this repo generates, and `tests/test_pynecore_reference.py`
compares them against `indicators/technical.py`. Install with
`pip install -r requirements-optional.txt`; the tests skip without it.

What that comparison found is recorded in section 10.

**Genuinely worth reading** for two things tvlab does not have: its `na`
propagation semantics, and how it keeps variables persistent across bars without
introducing lookahead. If you already have Pine Script strategies, PyneCore is
the right way to run them; convert with PyneComp and compare its trades against
`tvlab backtest` as a cross-check.

### 10. `theonetrade/pynecore-strategies` (MIT)

449 strategies, 471 indicators and 85 libraries compiled from Pine Script to
PyneCore, runnable with no TradingView account, browser or chart.

**Not integrated as a corpus** — it requires #9 as its runtime, and the 471
compiled indicators are third-party scripts with their own input defaults,
smoothing and plotting choices. A disagreement with one of them could come from
the author's parameters rather than from a real difference in the underlying
formula, which makes them a poor oracle.

**Used instead through #9's `ta` built-ins**, which is the same Pine semantics
without the third-party parameters in the way. That closes what was previously
listed here as the best remaining use. 32 tests in
`tests/test_pynecore_reference.py`, run over 600 bars.

Results, all asserted rather than merely noted:

| Indicator | Against | Result |
|:----------|:--------|:-------|
| `sma`, `bollinger`, `stdev` | `ta.sma`, `ta.bb`, `ta.stdev` | **exact from bar 0** (~1e-14) |
| `atr(method="wilder")` | `ta.atr(14)` | converges to ~1e-15 |
| `rsi` | `ta.rsi(close,14)` | converges to ~1e-15 |
| `ema`, `rma` | `ta.ema`, `ta.rma` | converge to ~1e-15 |
| `macd` (all 3 outputs) | `ta.macd` | converge to ~1e-13 |
| `supertrend` band | `ta.supertrend(3,10)[0]` | converges to ~1e-16 |
| `adx` (all 3 outputs) | `ta.dmi(14,14)` | converge to ~1e-14 |
| `donchian` | `ta.highest`/`ta.lowest` | differs by exactly one bar, deliberately |

Four findings worth keeping:

1. **`bollinger`'s `ddof=0` is correct and now proven.** Pine's `ta.stdev`
   defaults to `biased=True`, i.e. population std. pandas' `.rolling().std()`
   defaults to `ddof=1`. Had that been left at the pandas default the bands would
   have been quietly too wide by `sqrt(n/(n-1))` — 1.3% at n=20, enough to move a
   band-cross signal without looking like a bug.

2. **Recursive indicators disagree during warmup, and by how much is modelled.**
   Pine seeds from bar zero; pandas recurses from bar zero but publishes NaN until
   `min_periods`. RSI differs by up to 27% relative in the first hundred bars and
   is indistinguishable after ~260. `bars_to_forget_seed()` in the test derives
   that number from `(1 - 1/period)**k` instead of hardcoding it. Consequence for
   anyone comparing indicators: a whole-series comparison of a recursive indicator
   fails for a reason that has nothing to do with the formula.

3. **`st_direction` is the exact negation of Pine's `direction`.** tvlab uses
   `+1 = uptrend`; TradingView's own built-in plots `direction < 0` as "Up
   Direction", so Pine uses `+1 = downtrend`. The band matches to machine
   precision — only the flag is flipped. Nothing in `src/` reads `st_direction`
   today, so this is a porting trap rather than a live bug, but 449 Pine
   strategies sit in this directory as reference material and copying one across
   would silently invert every entry. Documented in the `supertrend()` docstring
   where a porter would see it.

4. **`adx` was the one indicator in a live decision path with no external oracle,
   and it is now validated.** Pine has no `ta.adx`, so there is no function of that
   name to compare against — but `ta.dmi(diLength, adxSmoothing)` returns
   `(+DI, -DI, ADX)` as a triple, which is the function `adx()`'s docstring claim
   ("matches TradingView's `ta.dmi`") was actually about. tvlab uses one period for
   both the DI length and the ADX smoothing, so the mapping is `dmi(p, p)` — also
   TradingView's own `ta.dmi()` default of `(14, 14)`. All three columns converge to
   ~1e-14.

   Its warmup is the **largest** of anything cross-validated here: +DI disagrees by
   up to 28% and -DI by up to 46% relative over the first hundred bars, because
   tvlab's `true_range` fills bar zero with `high - low` where `ta.dmi`'s internal
   `tr()` is na there, and two chained RMAs then smooth that seed. This matters more
   than for the other indicators because `bias_adx_min: 22.0` is a *threshold* —
   inside the warmup a materially different number is compared against the same
   cutoff, so the higher-timeframe bias gate in `SmcTrendStrategy` can disagree with
   TradingView's on a short sample.

### 11. `klinecharts/KLineChart` (Apache-2.0)

Zero-dependency charting, ~50 kB gzipped, with built-in indicator panes and
drawing tools (trendlines, Fibonacci, annotations) that lightweight-charts does
not have.

**Not integrated** because #2 is already bundled, working and offline-served. A
second charting library doubles the frontend surface for one dashboard.

**Use it if** you want indicator sub-panes or interactive drawing overlays; that
is a real capability gap in the current dashboard, and the trendline output from
`indicators/trendlines.py` would render naturally on it.

### 12. `kadoa-org/congress-trading-monitor` (MIT)

Aggregates US congressional and executive-branch trade disclosures from three
official sources (House Clerk, Senate eFD, OGE) into one dataset.

**Not integrated.** It is US-only, and the disclosure lag is 30–45 days, so it
cannot inform the intraday or swing horizons this repo is configured for. There
is no Indian equivalent in the set — SEBI does not publish an analogous
insider-disclosure feed in this form.

**Use it if** you build a `us_equities` position-style alternative-data signal,
where a multi-week lag is acceptable.

### 13. `visualHFT/VisualHFT` (Apache-2.0)

Live Level-2 order book, liquidity and market-microstructure visualisation.

**Not integrated**, and it points at a real gap rather than a missed opportunity:
tvlab's `hft` style is documented as a stub precisely because OHLCV bars cannot
support HFT conclusions. There is no order-book data source anywhere in
`data/`, so there is nothing for this to visualise. Costs alone would dominate
any 1-minute edge.

**Use it if** you add a Level-2 feed. Until then the honest position is the one
already in the README: `hft` is a configuration placeholder, not a capability.

### 14. `everget/tradingview-pinescript-indicators` (GPL-3.0)

Pine Script source for a large set of TradingView indicators.

**Not used, and must not be copied.** tvlab is MIT; incorporating GPL-3.0 code
would relicense the whole project. Algorithms and formulae are not copyrightable,
so it remains useful as a *specification* of what TradingView users expect an
indicator to mean — which is how `indicators/technical.py` was written.

### 15. `freqtrade/freqtrade` (GPL-3.0)

The most mature open-source crypto backtester: hyperopt, protections, dry-run and
live execution.

**Not used, and must not be copied** — GPL-3.0, same reasoning as #14. Read as an
architecture reference, particularly for how it separates parameter optimisation
from evaluation, which informed the walk-forward/deflated-Sharpe split here.

### 16. `koala73/worldmonitor` (AGPL-3.0)

**Not used, and must not be embedded.** AGPL-3.0's network clause would extend to
anything served from the dashboard. Also the one repo that failed a plain
`git clone` (`RPC failed; curl 56`) and had to be fetched as a codeload tarball —
`scripts/fetch-vendor.sh` has that special case.

### 17. `Cursivez/journalit` (NOASSERTION)

A trading journal. **Not used** — the licence is unasserted, so the default is all
rights reserved and nothing may be copied. The *concept* is already covered:
`report.py` writes a markdown report, machine-readable JSON and a trade log per
run.

### 18. `nickmccullum/algorithmic-trading-python` (no licence)

Course material with no licence file. **Nothing may be reused** — absence of a
licence means all rights reserved, not public domain. Educational reference only.

---

## What this audit changed in the code

| Change | Files |
|:-------|:------|
| NSE + Yahoo-chart data source, no new dependencies | `src/tvlab/data/nse.py` (new) |
| `tvlab nse {session,breadth,liquidity,universe,holidays}` | `src/tvlab/cli.py` |
| `nse` / `yahoo_chart` as selectable sources | `src/tvlab/data/store.py`, `src/tvlab/cli.py` |
| Configurable `universe.nse_index` | `src/tvlab/config.py`, `config/markets/indian_equities.yaml` |
| Native trendline fitting | `src/tvlab/indicators/trendlines.py` (new) |
| `tvlab scan --trendlines`, `GET /api/scan?trendlines=1` | `src/tvlab/screener/scanner.py`, `cli.py`, `dashboard/app.py` |
| pytrendline rejection rationale | `requirements-optional.txt` |
| `tradesdontlie/tradingview-mcp` licence corrected | `vendor/LOCK.json` |
| Pine-semantics cross-validation of every TA indicator | `scripts/vendor_xval_indicator.py` (new), `tests/test_pynecore_reference.py` (new) |
| `supertrend` direction sign trap documented at the source | `src/tvlab/indicators/technical.py` |
| `pynesys-pynecore[cli]` as an optional test dependency | `requirements-optional.txt` |
| 73 new tests | `tests/test_nse_source.py`, `tests/test_cli_nse.py`, `tests/test_trendlines.py` |
| 32 more tests | `tests/test_pynecore_reference.py` |
| `adx` cross-validated against `ta.dmi` — the last unvalidated indicator in a live decision path | `scripts/vendor_xval_indicator.py`, `tests/test_pynecore_reference.py`, `src/tvlab/indicators/technical.py` |
| The #3 quarantine made machine-enforced instead of prose | `tests/test_reference_quarantine.py` (new) |
| The unsourced "inflates PF to 3–10" claim replaced with a measurement | `README.md`, `VENDOR.md`, `tests/test_reference_quarantine.py` |
| **#4's fundamentals surface adopted** — 29 canonical fields, resolved by name *and* label | `src/tvlab/data/fundamentals.py` (new) |
| Fundamental gates: veto, never weight; missing data is a third outcome | `src/tvlab/screener/fundamental.py` (new), `src/tvlab/config.py` |
| `fundamentals:` section, risk gates on and valuation gates deliberately `null` | `config/markets/us_equities.yaml`, `config/markets/indian_equities.yaml` |
| The daily workflow: scan + rank + gates, verdict attached, journal written | `src/tvlab/screener/daily.py` (new) |
| `tvlab daily`, `GET /api/daily`, dashboard Daily panel | `src/tvlab/cli.py`, `src/tvlab/dashboard/app.py`, `dashboard/static/index.html` |
| **Session-filter crash fixed** — every non-crypto market was unusable on synthetic data | `src/tvlab/data/synthetic.py` |
| 126 more tests | `tests/test_fundamentals.py`, `tests/test_daily_workflow.py`, `tests/test_cli_daily.py`, `tests/test_dashboard_daily.py`, `tests/test_synthetic_session_filter.py` |

**Still open, in rough order of value:**

1. ~~Cross-validate `indicators/technical.py` against the 471 Pine-compiled
   indicators in #10.~~ **Closed** — done through #9's `ta` built-ins rather than
   the compiled scripts, for the reason given in section 10. Results table and the
   four findings are in that section; the tests are
   `tests/test_pynecore_reference.py` (32) driven by
   `scripts/vendor_xval_indicator.py`. Coverage now includes `adx`, which had been
   the one indicator used in a live decision path (the `bias_adx_min` gate) with no
   external oracle; `ta.dmi` turned out to be the function its docstring claim was
   about all along.
2. Verify the NSE and Yahoo payload shapes against the live APIs. They are
   reconstructed, not captured. This one cannot be closed from a sandbox with no
   route to NSE or Yahoo — it needs a networked machine. **This now includes
   `tvscreener`'s fundamentals response**, which is the largest unverified surface
   in the repo: `data/fundamentals.py` has never reached
   `scanner.tradingview.com`, so its parsing is tested against fixtures built to
   match both column conventions rather than against a capture. What *is* verified
   without a network is the field table — all 29 canonical names resolve against
   the installed enum, and two tests fail if that stops being true.
4. `IndexScreener` for regime context. The bias filter currently reads the trend of
   the symbol's own higher timeframe; an index-level read would separate "this stock
   is trending" from "the market is trending", which is the distinction that decides
   whether a long setup is a trade or a counter-trend catch. Not built because it
   cannot be validated here, and an unvalidated extra filter is exactly what the
   `entry_mode` measurement showed makes results worse while looking more
   sophisticated.
3. ~~`screener/scanner.py` still carries its own `DEFAULT_CORR_THRESHOLD = 0.70`.~~
   **Closed, and this note was wrong when written.** `grep -n "corr" scanner.py`
   returns nothing: the scanner never grouped positions, so there was no second
   threshold to disagree with. Correlation lives only in
   `portfolio/allocate.py`, which reads `risk.correlation_threshold` from config
   (`DEFAULT_CORR_THRESHOLD = 0.60` is the fallback, not the value in force), and
   `pipeline.py` passes `corr_threshold=None` to mean "use the config". Covered by
   `test_correlation_threshold_is_per_market` and
   `test_correlation_threshold_controls_grouping`. Worth recording because a
   plausible-sounding note in a hand-written audit is not evidence.
