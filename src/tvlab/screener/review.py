"""Score past daily suggestions against what actually happened.

The daily journal exists so that recommendations are falsifiable. Until this module
existed, that was a claim rather than a capability: six places in the docs and the
CLI said "in a month you can score the shortlists against what actually happened",
and nothing in the repo did the scoring. A recommendation that is written down but
never checked is a horoscope with better arithmetic.

What this does
--------------
For each journaled suggestion it replays the quoted levels forward through the bars
that followed: entry at the quoted price, exit at whichever of stop or target the
market touched first, or mark-to-market when the horizon runs out. Then it compares
the **realised** win rate against the **breakeven** win rate that was quoted at the
time. That comparison is the whole point - it is the only place in this repo where
the cost arithmetic in ``quality.py`` meets an outcome.

What this is NOT
----------------
Five limitations, each of which changes what the numbers mean:

1. **It is a level replay, not a fill model.** The journal records the price at the
   time of the suggestion, not an executed entry. Nothing here knows whether the
   entry was ever hit, whether you were in the market at all, or what you actually
   paid. "Entry at the quoted price" is an assumption, and it is stated in the
   output.
2. **Intrabar ambiguity is resolved pessimistically**, matching
   ``backtest/engine.py``: when one bar touches both the stop and the target, OHLC
   cannot say which came first, so the stop is assumed. Using a different convention
   here than in the backtester would make the two disagree about the same trades.
3. **Open suggestions are neither wins nor losses.** An idea whose horizon has not
   elapsed is reported as ``open`` and excluded from the win rate. Counting it as a
   loss would punish a swing trade for still running; counting it as a win would be
   worse.
4. **Unscorable is not a failure.** No price data after the suggestion date means
   ``unscorable``, reported separately with its coverage count - the same three-way
   distinction the fundamental gates use, because an endpoint gap must not become a
   verdict on the strategy.
5. **The horizon is part of the result.** Scoring a 90-bar swing idea after 5 bars
   measures nothing. Every aggregate reports the horizon it used, and a win rate
   over a handful of scored suggestions is labelled with its sample size rather than
   presented as a rate.

On synthetic data this measures the generator, not a market: the bars after a
suggestion are deterministic given the seed, so a "realised" win rate there is a
property of the data builder. It is still useful - it verifies the scorer - but it
is not evidence about the strategy, and the output says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: The five states a journaled suggestion can be scored into, pinned as a vocabulary.
#:
#: Three of them are results and two are refusals to guess. Collapsing either refusal
#: into a result is the whole failure mode this module exists to avoid: ``open`` folded
#: into the loss column punishes a swing trade for still running, and ``unscorable``
#: folded in turns a data outage into a verdict on the strategy.
#:
#: Pinned rather than inferred so a new state cannot be introduced silently - the same
#: reason ``daily.py`` keeps ``KNOWN_VERDICTS``, after a guessed verdict string made
#: every real study read as unvalidated.
OUTCOMES = frozenset({"take_profit", "stop_loss", "time_exit", "open", "unscorable"})

#: The subset that represents a resolved result, and so belongs in a win rate.
SCORED_OUTCOMES = frozenset({"take_profit", "stop_loss", "time_exit"})

#: Fetcher signature: ``(symbol, start, timeframe) -> DataFrame`` of bars beginning at
#: the suggestion bar. Injected rather than importing the store, so the scoring can be
#: tested offline and the caller decides where prices come from.
#:
#: The timeframe is part of the contract because a journal accumulates across config
#: changes: if the style's execution timeframe moved from 1d to 4h last week, scoring
#: the older runs against 4h bars would silently change what "90 bars" means - 90
#: trading days versus 15 calendar days - and the horizon would no longer be the one
#: the suggestion was made under.
BarsFetcher = Callable[[str, Optional[str], Optional[str]], Optional[pd.DataFrame]]


@dataclass
class ScoredSuggestion:
    """One journaled suggestion, replayed against the bars that followed."""

    as_of: str
    symbol: str
    side: str
    entry: float
    stop: Optional[float]
    target: Optional[float]

    #: one of :data:`OUTCOMES`
    outcome: str = "unscorable"
    exit_price: Optional[float] = None
    exit_at: Optional[str] = None
    #: gross multiple of the initial risk; -1.0 at the stop, +rr at the target
    r_multiple: Optional[float] = None
    #: round-trip cost in R, as quoted when the suggestion was made
    cost_in_r: Optional[float] = None
    #: what the suggestion needed in order to be worth taking
    breakeven_win_rate: Optional[float] = None
    #: bars actually replayed (may be fewer than the horizon)
    bars_observed: int = 0
    #: bars the replay was allowed to run
    horizon_bars: int = 0
    #: relative drift between the journaled entry price and the close of the first
    #: replayed bar. Zero means these are the bars the idea was based on.
    entry_drift: Optional[float] = None
    #: worst and best excursion in R while the idea was alive
    max_adverse_r: Optional[float] = None
    max_favourable_r: Optional[float] = None
    note: str = ""

    @property
    def net_r(self) -> Optional[float]:
        """Gross R less the round-trip cost, in R.

        Costs are charged on notional while risk is measured in R, so the conversion
        is the same ``cost_in_r`` quoted at suggestion time. Comparing a *gross* win
        rate against a breakeven rate that was computed *net* of costs would flatter
        every suggestion by exactly the fee drag.
        """
        if self.r_multiple is None:
            return None
        if self.cost_in_r is None:
            return self.r_multiple
        return self.r_multiple - self.cost_in_r

    @property
    def is_win(self) -> Optional[bool]:
        """True/False only for a *closed* idea. ``None`` means it does not count.

        Defined through :data:`SCORED_OUTCOMES` rather than by naming the two refusals,
        so adding a sixth outcome cannot silently start counting as a win.
        """
        if self.outcome not in SCORED_OUTCOMES or self.r_multiple is None:
            return None
        return self.r_multiple > 0

    @property
    def scored(self) -> bool:
        return self.outcome in SCORED_OUTCOMES

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        # properties are not fields, so __dict__ omits them - the same trap that
        # dropped `verdict` from Candidate.to_dict and blanked a dashboard panel.
        d["net_r"] = self.net_r
        d["is_win"] = self.is_win
        d["scored"] = self.scored
        return d


@dataclass
class ReviewSummary:
    """The aggregate, with its coverage stated rather than implied."""

    #: when this review was run. Passed in rather than read from the clock here, so
    #: the module stays deterministic and testable.
    as_of: Optional[str] = None
    horizon_bars: int = 0
    #: the entry-drift tolerance the scoring used. Recorded for the same reason the
    #: horizon is: both change what the numbers mean, so neither may be implied.
    entry_tolerance: Optional[float] = None
    n_suggestions: int = 0
    n_scored: int = 0
    n_open: int = 0
    n_unscorable: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: Optional[float] = None
    avg_gross_r: Optional[float] = None
    avg_net_r: Optional[float] = None
    #: the mean breakeven rate quoted across the scored suggestions
    avg_breakeven: Optional[float] = None
    #: True when the realised win rate clears what the ideas needed
    clears_breakeven: Optional[bool] = None

    #: realised payoff: mean winning R divided by mean losing R. The quoted
    #: ``breakeven_win_rate`` assumes wins pay the promised R:R and losses pay -1R;
    #: a ``time_exit`` at +0.5R is not a +3R win, so the realised payoff is what
    #: decides whether that comparison is even meaningful.
    realised_payoff_r: Optional[float] = None
    #: mean R:R the suggestions quoted
    quoted_rr: Optional[float] = None
    #: breakeven recomputed from the realised payoff and the realised cost
    realised_breakeven: Optional[float] = None
    #: set when the realised payoff diverged from the quoted one, which makes
    #: ``clears_breakeven`` a comparison between unlike things
    payoff_diverged: bool = False
    #: expectancy per suggestion in R, net of costs. This is the bottom line, and
    #: unlike the win-rate comparison it assumes nothing about the payoff shape.
    expectancy_r: Optional[float] = None
    by_outcome: Dict[str, int] = field(default_factory=dict)
    #: symbols with no price data after their suggestion date
    unscorable_symbols: List[str] = field(default_factory=list)
    provenance: Optional[str] = None
    rows: List[ScoredSuggestion] = field(default_factory=list)

    @property
    def coverage(self) -> str:
        if not self.n_suggestions:
            return "no suggestions in the journal"
        return (f"{self.n_scored} of {self.n_suggestions} scored "
                f"({self.n_open} still open, {self.n_unscorable} unscorable)")

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "rows"}
        d["coverage"] = self.coverage
        d["rows"] = [r.to_dict() for r in self.rows]
        return d


# ----------------------------------------------------------------------- scoring


def _risk_unit(side: str, entry: float, stop: Optional[float]) -> Optional[float]:
    """Signed distance from entry to stop: positive for a long, negative for a short.

    Returns None when there is no usable stop, because without one there is no R to
    measure in - and inventing a default stop would make the replay incomparable
    with the suggestion, which quoted its own.
    """
    if stop is None or not np.isfinite(stop):
        return None
    unit = entry - stop if side == "long" else stop - entry
    return float(unit) if unit > 0 else None


def _to_r(side: str, entry: float, risk_unit: float, price: float) -> float:
    """Express a price as a multiple of the initial risk."""
    if side == "long":
        return (price - entry) / risk_unit
    return (entry - price) / risk_unit


def score_one(
    entry_row: Dict[str, Any],
    bars: Optional[pd.DataFrame],
    horizon_bars: int,
    entry_tolerance: float = 0.005,
) -> ScoredSuggestion:
    """Replay one journal entry against the bars that followed it.

    ``bars`` must start at the suggestion bar and run forward. Passing the full
    history would let the replay exit on a bar that happened *before* the suggestion
    was made, which is a look-ahead bug in the one place this repo is trying to
    measure honesty. The first bar is used to check that these really are the bars
    the idea was based on, and is then excluded from the replay, because it closed
    before the suggestion existed.

    ``entry_tolerance`` is the relative drift allowed between the journaled entry
    price and the first replayed bar's close before the row is reported as
    ``unscorable`` rather than scored. It defaults to 0.5%: wide enough to absorb an
    adjusted-close revision or a bar-boundary difference, narrow enough to catch a
    fetch that returned a different series entirely.
    """
    s = ScoredSuggestion(
        as_of=str(entry_row.get("as_of") or ""),
        symbol=str(entry_row.get("symbol") or ""),
        side=str(entry_row.get("side") or "long"),
        entry=float(entry_row.get("last") or float("nan")),
        stop=entry_row.get("stop"),
        target=entry_row.get("target"),
        cost_in_r=entry_row.get("cost_in_r"),
        breakeven_win_rate=entry_row.get("breakeven_win_rate"),
        horizon_bars=int(horizon_bars),
    )
    s.cost_in_r = float(s.cost_in_r) if s.cost_in_r is not None else None
    s.breakeven_win_rate = (float(s.breakeven_win_rate)
                            if s.breakeven_win_rate is not None else None)

    if not np.isfinite(s.entry) or s.entry <= 0:
        s.note = "journal entry has no usable entry price"
        return s

    risk_unit = _risk_unit(s.side, s.entry, s.stop)
    if risk_unit is None:
        s.note = "no usable stop, so there is no R to measure in"
        return s

    if bars is None or len(bars) == 0:
        s.note = "no price data at or after the suggestion date"
        return s

    # The window keeps bar 0 *plus* `horizon_bars` bars after it. Slicing at
    # `horizon_bars` would spend the first slot on the suggestion bar and leave the
    # replay one bar short of the horizon it was asked for - every idea would then be
    # reported as still open, which reads as "too early to judge" rather than as the
    # off-by-one it is.
    n_keep = horizon_bars + 1 if horizon_bars else None
    window = bars.iloc[:n_keep] if n_keep else bars
    s.bars_observed = int(len(window))

    # Consistency guard. The window starts at the suggestion date, so its first bar is
    # the bar the idea was made on and its close should equal the journaled entry
    # price. When it does not, these are not the bars this suggestion was based on, and
    # replaying them would produce a confident R multiple measuring nothing.
    #
    # This is not a hypothetical. On the synthetic source, re-requesting from a later
    # `start` regenerates an unrelated series - the seed fixes the stream from the start
    # date, so shifting the start shifts every subsequent bar. Measured on BTC/USDT 4h:
    # bar 400 of a 1200-bar fetch closes at 30703.32, while a fresh 800-bar fetch
    # beginning at bar 400's date closes at 41475.43 - 35% apart. A reviewer that
    # fetched naively would score every synthetic suggestion against prices that never
    # existed, and report a win rate that measured the fetch, not the advice.
    first_close = float(window["close"].iloc[0])
    if np.isfinite(first_close) and first_close > 0:
        s.entry_drift = abs(first_close - s.entry) / s.entry
        if s.entry_drift > entry_tolerance:
            s.note = (f"replayed bars do not match the suggestion (entry "
                      f"{s.entry:,.6g} vs first bar close {first_close:,.6g}, "
                      f"{s.entry_drift * 100:.2f}% apart) - these are not the prices "
                      f"this idea was based on")
            return s

    # The suggestion is made on the CLOSE of bar 0, so bar 0's high and low already
    # happened before the decision existed. Evaluating an exit on that bar would let an
    # idea be stopped out by a price that preceded its own entry. backtest/engine.py
    # does not do this either: entry is assigned in the step that runs *after* the exit
    # check, so the earliest bar its exits are evaluated on is the one following entry.
    post = window.iloc[1:]
    if len(post) == 0:
        s.outcome = "open"
        s.note = ("the suggestion bar is the last bar available - nothing has happened "
                  "since, not counted either way")
        return s

    hi = post["high"].to_numpy(dtype="float64")
    lo = post["low"].to_numpy(dtype="float64")
    cl = post["close"].to_numpy(dtype="float64")
    idx = post.index

    long_ = s.side == "long"
    has_target = s.target is not None and np.isfinite(s.target)

    worst = best = 0.0
    for i in range(len(post)):
        # Adverse excursion uses the unfavourable extreme (low for a long, high for a
        # short); favourable excursion uses the other one. Getting these the wrong way
        # round would report a long's best move as its worst.
        adverse_px = float(lo[i]) if long_ else float(hi[i])
        favour_px = float(hi[i]) if long_ else float(lo[i])
        worst = min(worst, _to_r(s.side, s.entry, risk_unit, adverse_px))
        best = max(best, _to_r(s.side, s.entry, risk_unit, favour_px))

        hit_stop = (lo[i] <= s.stop) if long_ else (hi[i] >= s.stop)
        hit_target = bool(has_target) and ((hi[i] >= s.target) if long_ else (lo[i] <= s.target))
        if hit_stop and hit_target:
            # Same convention as backtest/engine.py: OHLC cannot say which came
            # first inside one bar, so take the adverse branch.
            s.outcome, s.exit_price = "stop_loss", float(s.stop)
        elif hit_stop:
            s.outcome, s.exit_price = "stop_loss", float(s.stop)
        elif hit_target:
            s.outcome, s.exit_price = "take_profit", float(s.target)
        else:
            continue
        s.exit_at = str(idx[i])
        s.r_multiple = _to_r(s.side, s.entry, risk_unit, s.exit_price)
        s.max_adverse_r, s.max_favourable_r = worst, best
        return s

    # Horizon elapsed with neither level touched. The horizon counts bars *after* the
    # suggestion bar, matching the engine's `i - entry_i`.
    s.max_adverse_r, s.max_favourable_r = worst, best
    if len(post) < horizon_bars:
        s.outcome = "open"
        s.note = (f"only {len(post)} bar(s) since the suggestion, horizon is "
                  f"{horizon_bars} - still running, not counted either way")
        return s
    s.outcome = "time_exit"
    s.exit_price = float(cl[-1])
    s.exit_at = str(idx[-1])
    s.r_multiple = _to_r(s.side, s.entry, risk_unit, s.exit_price)
    s.note = f"neither level touched within {horizon_bars} bars; marked to market"
    return s


def review(
    entries: Sequence[Dict[str, Any]],
    fetch_bars: BarsFetcher,
    horizon_bars: int = 90,
    provenance: Optional[str] = None,
    entry_tolerance: float = 0.005,
    as_of: Optional[str] = None,
) -> ReviewSummary:
    """Score every shortlisted suggestion in the journal.

    ``entries`` are journal lines, i.e. one dict per *run*, each carrying a
    ``shortlist`` of suggestions. Runs with an empty shortlist contribute nothing but
    are still counted, because "the tool ran 20 days and found nothing" is a fact
    worth having next to the days it did find something.
    """
    rows: List[ScoredSuggestion] = []
    n_suggestions = 0
    cache: Dict[Tuple[str, str, str], Optional[pd.DataFrame]] = {}

    for run in entries or []:
        # Deliberately not called `as_of`: that name is the parameter carrying the
        # date this review was run, and a loop variable of the same name would still be
        # bound to the last run's date when the loop ended - silently replacing the
        # review date with a journal date.
        run_date = run.get("as_of")
        tf = run.get("timeframe")
        for item in run.get("shortlist") or []:
            n_suggestions += 1
            row = dict(item)
            row.setdefault("as_of", run_date)
            key = (str(row.get("symbol")), str(run_date), str(tf))
            if key not in cache:
                try:
                    cache[key] = fetch_bars(str(row.get("symbol")), str(run_date), tf)
                except Exception as exc:  # noqa: BLE001 - one symbol must not kill the review
                    log.warning("could not fetch bars for %s from %s: %s", key[0], key[1], exc)
                    cache[key] = None
            rows.append(score_one(row, cache[key], horizon_bars,
                                  entry_tolerance=entry_tolerance))

    return summarise(rows, n_suggestions, horizon_bars, provenance, as_of=as_of,
                     entry_tolerance=entry_tolerance)


def _rr_of(r: ScoredSuggestion) -> Optional[float]:
    """The R:R a suggestion promised, from its own entry/stop/target.

    Kept separate from the realised payoff because the gap between the two is the
    interesting number: a shortlist quoting 3:1 that realises 0.8:1 is either
    exiting early or picking levels the market does not respect.
    """
    if r.stop is None or r.target is None or not np.isfinite(r.entry):
        return None
    if not np.isfinite(r.stop) or not np.isfinite(r.target):
        return None
    risk = abs(r.entry - r.stop)
    if risk <= 0:
        return None
    return abs(r.target - r.entry) / risk


def summarise(rows: Sequence[ScoredSuggestion], n_suggestions: int,
              horizon_bars: int, provenance: Optional[str] = None,
              as_of: Optional[str] = None,
              entry_tolerance: Optional[float] = None) -> ReviewSummary:
    """Aggregate scored rows. Split out from :func:`review` so it can be tested
    without a price source at all."""
    scored = [r for r in rows if r.scored]
    open_ = [r for r in rows if r.outcome == "open"]
    unscorable = [r for r in rows if r.outcome == "unscorable"]
    wins = [r for r in scored if r.is_win]

    def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
        vals = [float(x) for x in xs if x is not None and np.isfinite(x)]
        return float(np.mean(vals)) if vals else None

    by_outcome: Dict[str, int] = {}
    for r in rows:
        by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1

    losses = [r for r in scored if r.is_win is False]
    win_rate = (len(wins) / len(scored)) if scored else None
    avg_be = _mean([r.breakeven_win_rate for r in scored])

    # Edge cases are decided explicitly rather than left to fall out of the means,
    # because "no wins" and "no losses" have opposite answers and both would
    # otherwise produce None -> clears_breakeven None, which reads as "cannot tell".
    # A book with zero winners plainly does not clear its breakeven; a book with zero
    # losers plainly does.
    avg_cost = _mean([r.cost_in_r for r in scored])
    payoff: Optional[float] = None
    if scored and not wins:
        payoff = 0.0
    elif scored and not losses:
        payoff = None                      # infinite payoff; no rate could fail it
    else:
        avg_win = _mean([r.r_multiple for r in wins])
        avg_loss = _mean([abs(r.r_multiple) for r in losses])
        if avg_win is not None and avg_loss is not None and avg_loss > 0:
            payoff = float(avg_win / avg_loss)

    # Expectancy in R, net of costs, is the number that needs no assumption about
    # payoff shape: it is simply what one journaled suggestion was worth on average.
    expectancy = _mean([r.net_r for r in scored])

    # Divergence is measured against the R:R the suggestions *promised* (from their
    # own entry/stop/target), not against the realised R of the take-profit exits.
    # Using the latter would report no divergence in exactly the case that matters:
    # a book where nothing ever reached its target, so there are no take-profit rows
    # to compare against at all.
    quoted = _mean([_rr_of(r) for r in scored])
    diverged = bool(payoff is not None and quoted is not None
                    and abs(payoff - quoted) > 0.25 * max(quoted, 1e-9))

    # Compare against the breakeven implied by what actually happened, not only the
    # one that was promised. If wins came in well short of the quoted R:R, the
    # quoted breakeven understates what was required.
    realised_be: Optional[float] = None
    if scored and not losses:
        realised_be = 0.0                  # nothing lost, so any rate clears
    elif payoff is not None:
        realised_be = float((1.0 + (avg_cost or 0.0)) / (1.0 + payoff))
    clears = None
    if win_rate is not None and realised_be is not None:
        clears = bool(win_rate >= realised_be)

    return ReviewSummary(
        as_of=as_of,
        horizon_bars=int(horizon_bars),
        entry_tolerance=entry_tolerance,
        n_suggestions=int(n_suggestions),
        n_scored=len(scored),
        n_open=len(open_),
        n_unscorable=len(unscorable),
        n_wins=len(wins),
        n_losses=len(scored) - len(wins),
        win_rate=win_rate,
        avg_gross_r=_mean([r.r_multiple for r in scored]),
        avg_net_r=_mean([r.net_r for r in scored]),
        avg_breakeven=avg_be,
        clears_breakeven=clears,
        realised_payoff_r=payoff,
        quoted_rr=quoted,
        realised_breakeven=realised_be,
        payoff_diverged=diverged,
        expectancy_r=expectancy,
        by_outcome=dict(sorted(by_outcome.items())),
        unscorable_symbols=sorted({r.symbol for r in unscorable}),
        provenance=provenance,
        rows=list(rows),
    )


# --------------------------------------------------------------------- fetching
def store_fetcher(
    store: Any,
    scan_bars: int = 1200,
    errors: Optional[List[str]] = None,
    no_future: Optional[Dict[str, str]] = None,
) -> BarsFetcher:
    """Build a :data:`BarsFetcher` over anything with ``.ohlcv(symbol, timeframe, …)``.

    Duck-typed rather than importing ``DataStore``, so this module keeps no dependency
    on the data layer and the anchoring below stays unit-testable against a stub. Both
    ``tvlab review`` and ``GET /api/review`` build their fetcher here, because two
    copies of price-anchoring logic is how a terminal and a browser end up reporting
    different outcomes for the same journal.

    ``errors`` and ``no_future``, when passed, are filled in as a side effect so the
    caller can explain *why* something was unscorable instead of only that it was.

    The anchoring is the subtle part
    --------------------------------
    The window is fetched with the same parameters the scan used and then sliced,
    rather than re-requested from the suggestion date. Re-requesting is not a harmless
    optimisation: on the synthetic source the seed fixes the generated stream from the
    start date, so asking for a later start returns an unrelated series. Measured on
    BTC/USDT 4h, bar 400 of a 1200-bar fetch closes at 30703.32 while a fresh 800-bar
    fetch beginning at bar 400's date closes at 41475.43 - 35% apart at the same index.

    The slice is anchored on the last bar at or before the end of the suggestion date,
    because that is the bar whose close the journal recorded as the entry price. A
    daily timeframe makes the two coincide; on an intraday one, anchoring on the first
    bar *of* that date would compare the entry against a close from earlier in the day
    and trip the drift guard on data that is perfectly good.
    """

    def fetch(symbol: str, start: Optional[str], timeframe: Optional[str]):
        try:
            df = store.ohlcv(symbol, timeframe, limit=scan_bars)
        except Exception as exc:  # noqa: BLE001 - one symbol must not kill the review
            if errors is not None:
                errors.append(f"{symbol}: {type(exc).__name__}: "
                              f"{str(exc).splitlines()[0]}")
            return None
        if df is None or len(df) == 0:
            if errors is not None:
                errors.append(f"{symbol}: the source returned no bars at all")
            return None

        idx = df.index
        ts = pd.Timestamp(start)
        if getattr(idx, "tz", None) is not None and ts.tzinfo is None:
            ts = ts.tz_localize(idx.tz)
        day_end = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        pos = int(idx.searchsorted(day_end, side="right")) - 1
        if pos < 0:
            if no_future is not None:
                no_future[symbol] = f"the series starts at {idx[0]}, after {start}"
            return None
        if pos >= len(idx) - 1 and no_future is not None:
            no_future[symbol] = f"the series ends at {idx[-1]}"
        return df.iloc[pos:]

    return fetch
