"""Scoring past suggestions: the arithmetic, and the ways it can lie.

Built from explicit bar frames rather than generated data, so each outcome can be
checked exactly. Bar 0 of every fixture is the suggestion bar itself - the bar whose
close the journal recorded as the entry - and the replay starts at bar 1.

The cases that matter are not the happy path. They are the ones where a
plausible-looking number would be wrong:

* an idea still running is ``open``, not a loss;
* a symbol with no later bars is ``unscorable``, not a failure;
* a bar touching both levels resolves to the **stop**, matching
  ``backtest/engine.py`` - a different convention here would make the reviewer and
  the backtester disagree about the same trades;
* the suggestion bar itself cannot exit the position, because it closed before the
  suggestion existed;
* bars that do not reproduce the journaled entry price are not the bars the idea was
  based on, and must not be scored;
* realised R is compared to a breakeven recomputed from the **realised** payoff, since
  a win rate alone says nothing when wins stop paying what was promised.

Fixtures use entry 100, stop 98, target 106, so the risk unit is 2.0 and the quoted
R:R is 3.0. Cost is 0.10R.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tvlab.screener.review import (
    OUTCOMES,
    SCORED_OUTCOMES,
    ScoredSuggestion,
    review,
    score_one,
    summarise,
)


def _bars(rows, start="2026-01-05"):
    """rows: (high, low, close) or (high, low, close, open). Row 0 is the suggestion
    bar. The three-field form means a bar that opens at its close - the quiet case
    most outcomes need; the four-field form exists so a gap can be described."""
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(
        {"high": [r[0] for r in rows], "low": [r[1] for r in rows],
         "close": [r[2] for r in rows],
         "open": [r[3] if len(r) > 3 else r[2] for r in rows],
         "volume": [1.0] * len(rows)},
        index=idx)


#: the suggestion bar: a quiet bar closing at the entry price, inside both levels
_BAR0 = (101.0, 99.5, 100.0)
#: the same bar for a short, whose stop sits above the entry
_BAR0_SHORT = (100.5, 99.5, 100.0)


def _entry(**kw):
    base = dict(as_of="2026-01-04", symbol="AAPL", side="long", last=100.0,
                stop=98.0, target=106.0, cost_in_r=0.10, breakeven_win_rate=0.30)
    base.update(kw)
    return base


# ------------------------------------------------------------------- outcomes


def test_target_hit_first_scores_a_win_at_the_quoted_rr():
    s = score_one(_entry(), _bars([_BAR0, (106.5, 100.0, 106.0)]), horizon_bars=90)
    assert s.outcome == "take_profit"
    assert s.exit_price == pytest.approx(106.0)
    assert s.r_multiple == pytest.approx(3.0)     # (106-100)/2
    assert s.is_win is True
    assert s.net_r == pytest.approx(2.9)          # 3R less 0.10R of cost
    assert s.bars_observed == 2
    assert s.entry_drift == pytest.approx(0.0)


def test_stop_hit_first_scores_a_loss_at_minus_one_r():
    s = score_one(_entry(), _bars([_BAR0, (100.0, 97.5, 98.0)]), horizon_bars=90)
    assert s.outcome == "stop_loss"
    assert s.exit_price == pytest.approx(98.0)
    assert s.r_multiple == pytest.approx(-1.0)
    assert s.is_win is False
    assert s.net_r == pytest.approx(-1.1)


def test_a_bar_touching_both_levels_resolves_to_the_stop():
    """Same convention as backtest/engine.py. OHLC cannot say which came first."""
    s = score_one(_entry(), _bars([_BAR0, (107.0, 97.0, 101.0)]), horizon_bars=90)
    assert s.outcome == "stop_loss"
    assert s.exit_price == pytest.approx(98.0)
    assert s.r_multiple == pytest.approx(-1.0)


def test_a_gap_through_the_stop_scores_at_the_open_not_the_stop():
    """Same rule as backtest/engine.py. A journal that scored gaps at the stop would
    report a flattering win rate that the backtester does not endorse."""
    # opens at 94, well below the 98 stop, on the first bar after the suggestion
    s = score_one(_entry(), _bars([_BAR0, (99.0, 92.0, 93.0, 94.0)]), horizon_bars=90)
    assert s.outcome == "stop_loss"
    assert s.exit_price == pytest.approx(94.0)
    assert s.r_multiple == pytest.approx(-3.0)     # (94-100)/2
    assert s.is_win is False


def test_a_bar_that_only_dips_intrabar_still_scores_at_the_stop():
    s = score_one(_entry(), _bars([_BAR0, (100.0, 97.5, 98.0, 99.5)]), horizon_bars=90)
    assert s.outcome == "stop_loss"
    assert s.exit_price == pytest.approx(98.0)
    assert s.r_multiple == pytest.approx(-1.0)


def test_neither_level_within_the_horizon_marks_to_market():
    s = score_one(_entry(), _bars([_BAR0] + [(102.0, 99.0, 101.0)] * 4), horizon_bars=4)
    assert s.outcome == "time_exit"
    assert s.exit_price == pytest.approx(101.0)
    assert s.r_multiple == pytest.approx(0.5)     # (101-100)/2
    assert "marked to market" in s.note


def test_an_idea_that_has_not_run_long_enough_is_open_not_a_loss():
    """Counting a still-running swing trade as a loss punishes it for existing."""
    s = score_one(_entry(), _bars([_BAR0, (102.0, 100.0, 101.5)]), horizon_bars=90)
    assert s.outcome == "open"
    assert s.is_win is None
    assert s.scored is False
    assert "still running, not counted either way" in s.note


def test_a_suggestion_with_only_its_own_bar_available_is_open():
    """Nothing has happened since the idea was written. That is not a failure and not
    a loss - it is the state every suggestion is in the morning it is made."""
    s = score_one(_entry(), _bars([_BAR0]), horizon_bars=90)
    assert s.outcome == "open"
    assert s.scored is False
    assert s.bars_observed == 1
    assert "nothing has happened since" in s.note


def test_the_horizon_counts_bars_after_the_suggestion():
    """`horizon_bars=1` means one bar of opportunity, so the suggestion bar plus one
    more is enough to time-exit. Counting the suggestion bar toward the horizon would
    mark every idea to market before it had a bar to move."""
    s = score_one(_entry(), _bars([_BAR0, (102.0, 99.0, 101.0)]), horizon_bars=1)
    assert s.outcome == "time_exit"
    assert s.r_multiple == pytest.approx(0.5)
    assert s.exit_at.startswith("2026-01-06")


def test_no_price_data_is_unscorable_not_a_failure():
    s = score_one(_entry(), None, horizon_bars=90)
    assert s.outcome == "unscorable"
    assert s.scored is False
    assert "no price data" in s.note
    assert score_one(_entry(), _bars([]), horizon_bars=90).outcome == "unscorable"


def test_a_missing_stop_cannot_be_replayed():
    """Without a stop there is no R to measure in, and inventing one would make the
    replay incomparable with the suggestion that quoted its own."""
    s = score_one(_entry(stop=None), _bars([_BAR0, (106.5, 99.0, 106.0)]), horizon_bars=90)
    assert s.outcome == "unscorable"
    assert "no usable stop" in s.note


def test_a_missing_entry_price_is_unscorable():
    s = score_one(_entry(last=float("nan")), _bars([_BAR0]), horizon_bars=90)
    assert s.outcome == "unscorable"
    assert "no usable entry price" in s.note


def test_a_suggestion_with_no_target_still_scores_against_its_stop():
    s = score_one(_entry(target=None), _bars([_BAR0, (101.0, 97.5, 98.0)]), horizon_bars=90)
    assert s.outcome == "stop_loss"
    s2 = score_one(_entry(target=None),
                   _bars([_BAR0] + [(101.0, 99.0, 100.0)] * 3), horizon_bars=3)
    assert s2.outcome == "time_exit"
    assert s2.r_multiple == pytest.approx(0.0)


# --------------------------------------------- the suggestion bar is not tradeable


def test_the_suggestion_bar_itself_cannot_stop_the_position_out():
    """Bar 0's low of 97.0 is below the 98 stop - but that low happened *before* the
    close that generated the suggestion. Scoring it would report a loss on a bar the
    idea did not exist for, and would make the reviewer harsher than the backtester,
    which assigns entry after its exit check and so never evaluates the entry bar."""
    bars = _bars([(101.0, 97.0, 100.0), (102.0, 99.0, 101.0)])
    s = score_one(_entry(), bars, horizon_bars=1)
    assert s.outcome == "time_exit"
    # only bar 1's low of 99 is counted (-0.5R); bar 0's 97.0 would have read -1.5R
    assert s.max_adverse_r == pytest.approx(-0.5)


def test_the_suggestion_bar_itself_cannot_take_profit():
    bars = _bars([(112.0, 99.5, 100.0), (102.0, 99.0, 101.0)])
    s = score_one(_entry(), bars, horizon_bars=1)
    assert s.outcome == "time_exit"
    # only bar 1's high of 102 is counted (+1.0R); bar 0's 112 would have read +6.0R
    # and would also have looked like the target had been hit before entry
    assert s.max_favourable_r == pytest.approx(1.0)


# ------------------------------------------------------- entry-price consistency


def test_bars_that_do_not_reproduce_the_entry_price_are_unscorable():
    """A fetch that returned a different series must not be scored. On the synthetic
    source this is the default behaviour, not an edge case: shifting `start` shifts the
    whole RNG stream, so a re-fetch from a later date shares no bars with the original.
    Scoring it anyway would print a win rate measuring the fetch."""
    bars = _bars([_BAR0, (106.5, 100.0, 106.0)])
    bars.iloc[0, bars.columns.get_loc("close")] = 135.0     # 35% away from the entry
    s = score_one(_entry(), bars, horizon_bars=90)
    assert s.outcome == "unscorable"
    assert s.entry_drift == pytest.approx(0.35)
    assert "not the prices this idea was based on" in s.note


def test_a_small_entry_drift_is_tolerated():
    """Adjusted-close revisions and bar-boundary differences are normal on live data;
    refusing to score them would make the reviewer useless for exactly the journals
    that matter."""
    bars = _bars([_BAR0, (106.5, 100.0, 106.0)])
    bars.iloc[0, bars.columns.get_loc("close")] = 100.2     # 0.2% away
    s = score_one(_entry(), bars, horizon_bars=90)
    assert s.entry_drift == pytest.approx(0.002)
    assert s.outcome == "take_profit"


def test_the_drift_tolerance_is_configurable():
    bars = _bars([_BAR0, (106.5, 100.0, 106.0)])
    bars.iloc[0, bars.columns.get_loc("close")] = 100.2
    strict = score_one(_entry(), bars, horizon_bars=90, entry_tolerance=0.001)
    assert strict.outcome == "unscorable"


# --------------------------------------------------------------------- shorts


def test_a_short_is_scored_in_its_own_direction():
    e = dict(side="short", last=100.0, stop=102.0, target=94.0)
    win = score_one(_entry(**e), _bars([_BAR0_SHORT, (100.5, 93.5, 94.0)]), horizon_bars=90)
    assert win.outcome == "take_profit"
    assert win.r_multiple == pytest.approx(3.0)       # (100-94)/2
    loss = score_one(_entry(**e), _bars([_BAR0_SHORT, (102.5, 99.5, 102.0)]), horizon_bars=90)
    assert loss.outcome == "stop_loss"
    assert loss.r_multiple == pytest.approx(-1.0)


def test_excursions_use_the_favourable_extreme_for_the_side():
    """A long's best move is its high; its worst is its low. Swapped, a long that ran
    up 4R before stopping out would report a best excursion of 0. The target is 112 so
    the 108 high does not take profit and the idea survives to stop out."""
    bars = _bars([_BAR0, (108.0, 99.0, 107.0), (107.0, 97.5, 98.0)])
    s = score_one(_entry(target=112.0), bars, horizon_bars=90)
    assert s.outcome == "stop_loss"
    assert s.max_favourable_r == pytest.approx(4.0)   # (108-100)/2
    assert s.max_adverse_r == pytest.approx(-1.25)    # (97.5-100)/2


def test_a_short_excursion_is_mirrored():
    bars = _bars([_BAR0_SHORT, (101.0, 92.0, 93.0), (102.5, 93.0, 102.0)])
    s = score_one(_entry(side="short", last=100.0, stop=102.0, target=88.0),
                  bars, horizon_bars=90)
    assert s.outcome == "stop_loss"
    assert s.max_favourable_r == pytest.approx(4.0)   # (100-92)/2
    assert s.max_adverse_r == pytest.approx(-1.25)    # (100-102.5)/2


# ------------------------------------------------------------------ fetch contract


def test_the_fetcher_is_called_with_the_suggestion_date_and_timeframe():
    """The timeframe is part of the contract because a journal accumulates across
    config changes; scoring a 1d suggestion against 4h bars would silently change what
    the horizon means."""
    seen = []

    def fetch(symbol, start, timeframe):
        seen.append((symbol, start, timeframe))
        return _bars([_BAR0, (106.5, 100.0, 106.0)])

    out = review([{"as_of": "2026-01-04", "timeframe": "1d",
                   "shortlist": [_entry()]}], fetch, horizon_bars=90)
    assert seen == [("AAPL", "2026-01-04", "1d")]
    assert out.rows[0].outcome == "take_profit"


def test_a_run_with_no_recorded_timeframe_still_passes_none_through():
    seen = []

    def fetch(symbol, start, timeframe):
        seen.append(timeframe)
        return None

    review([{"as_of": "2026-01-04", "shortlist": [_entry()]}], fetch, horizon_bars=90)
    assert seen == [None]


def test_the_review_date_is_not_overwritten_by_a_journal_date():
    """The summary's `as_of` is when the review was run. A loop variable of the same
    name inside `review()` would still be bound to the last run's date once the loop
    ended, and the header would then claim a 2026 review happened in 2022 - a small
    wrongness in the one line the reader trusts most."""
    entries = _journal(("2022-03-22", [_entry(symbol="A")]),
                       ("2022-04-11", [_entry(symbol="B")]))
    s = review(entries, lambda sym, start, tf: _win_bars(), horizon_bars=90,
               as_of="2026-11-24")
    assert s.as_of == "2026-11-24"
    assert s.to_dict()["as_of"] == "2026-11-24"


def test_three_dates_are_kept_apart():
    """The review date, the run date and the suggestion's own date are three different
    things and none of them may silently become another.

    Precedence is deliberate: a suggestion carrying its own ``as_of`` keeps it, because
    the journal is written per run but a row may have been dated when it was built;
    a row with none inherits its run's date, which is how the CLI's journals are
    actually shaped.
    """
    entries = [{"as_of": "2022-03-22", "timeframe": "4h",
                "shortlist": [_entry(symbol="A", as_of="2022-03-20"),   # keeps its own
                              {"symbol": "B", "side": "long", "last": 100.0,
                               "stop": 98.0, "target": 106.0}]}]         # inherits
    s = review(entries, lambda sym, start, tf: _win_bars(), horizon_bars=90,
               as_of="2026-11-24")
    assert s.as_of == "2026-11-24"                    # when the review ran
    assert s.rows[0].as_of == "2022-03-20"            # the row's own date
    assert s.rows[1].as_of == "2022-03-22"            # inherited from the run
    # the inherited row is also fetched at the run's date, not the review date
    assert s.rows[1].outcome == "take_profit"


def test_one_bars_fetch_per_symbol_per_date_per_timeframe():
    """A 30-day journal with 5 names is 150 suggestions; re-fetching per row would be
    150 network calls where 2 distinct (symbol, date) pairs exist."""
    calls = []

    def fetch(symbol, start, timeframe):
        calls.append((symbol, start, timeframe))
        return _bars([_BAR0, (106.5, 100.0, 106.0)])

    entries = [{"as_of": "2026-01-04", "timeframe": "4h",
                "shortlist": [_entry(symbol="A"), _entry(symbol="A")]},
               {"as_of": "2026-01-05", "timeframe": "4h", "shortlist": [_entry(symbol="A")]}]
    review(entries, fetch, horizon_bars=90)
    assert len(calls) == 2, calls


def test_the_same_symbol_and_date_at_two_timeframes_is_two_fetches():
    """The timeframe must be part of the cache key, not just an argument. A journal
    spans config changes, so one symbol on one morning can legitimately appear under
    both 4h and 1d - and a shared cache entry would score the 1d suggestion against
    4h bars, where 'horizon' silently means 15 days instead of 90."""
    calls = []

    def fetch(symbol, start, timeframe):
        calls.append((symbol, start, timeframe))
        # 4h bars are tight and never reach the target; 1d bars run to it
        if timeframe == "4h":
            return _bars([_BAR0, (102.0, 99.0, 101.0)])
        return _win_bars()

    entries = [{"as_of": "2026-01-04", "timeframe": "4h",
                "shortlist": [_entry(symbol="A")]},
               {"as_of": "2026-01-04", "timeframe": "1d",
                "shortlist": [_entry(symbol="A")]}]
    s = review(entries, fetch, horizon_bars=90)

    assert calls == [("A", "2026-01-04", "4h"), ("A", "2026-01-04", "1d")]
    assert len(s.rows) == 2
    # the two rows were scored against genuinely different price series
    assert s.rows[0].outcome == "open"
    assert s.rows[1].outcome == "take_profit"


def test_a_fetcher_that_raises_does_not_kill_the_review():
    def fetch(symbol, start, timeframe):
        if symbol == "BOOM":
            raise RuntimeError("network down")
        return _bars([_BAR0, (106.5, 100.0, 106.0)])

    entries = [{"as_of": "2026-01-04",
                "shortlist": [_entry(symbol="BOOM"), _entry(symbol="OK")]}]
    s = review(entries, fetch, horizon_bars=90)
    assert s.n_scored == 1 and s.n_unscorable == 1
    assert s.unscorable_symbols == ["BOOM"]


# ------------------------------------------------------------------ aggregate


def _journal(*runs):
    return [{"as_of": r[0], "timeframe": "4h", "shortlist": r[1]} for r in runs]


def _win_bars():
    return _bars([_BAR0, (106.5, 100.0, 106.0)])          # take_profit, +3R


def _lose_bars():
    return _bars([_BAR0, (100.0, 97.5, 98.0)])            # stop_loss, -1R


def test_the_summary_states_its_coverage():
    entries = _journal(
        ("2026-01-04", [_entry(symbol="WIN"), _entry(symbol="LOSE"),
                        _entry(symbol="NODATA")]),
        ("2026-01-05", [_entry(symbol="OPEN")]),
    )

    def fetch(symbol, start, timeframe):
        if symbol == "NODATA":
            return None
        if symbol == "WIN":
            return _win_bars()
        if symbol == "LOSE":
            return _lose_bars()
        return _bars([_BAR0])                              # OPEN: its own bar only

    s = review(entries, fetch, horizon_bars=90)
    assert s.n_suggestions == 4
    assert s.n_scored == 2 and s.n_open == 1 and s.n_unscorable == 1
    assert s.win_rate == pytest.approx(0.5)
    assert s.n_wins == 1 and s.n_losses == 1
    assert s.unscorable_symbols == ["NODATA"]
    assert "2 of 4 scored" in s.coverage
    assert s.by_outcome == {"stop_loss": 1, "take_profit": 1, "open": 1, "unscorable": 1}


def test_an_empty_journal_reports_no_suggestions_rather_than_a_zero_rate():
    s = review([], lambda sym, start, tf: None, horizon_bars=90)
    assert s.n_suggestions == 0
    assert s.win_rate is None
    assert s.clears_breakeven is None
    assert s.coverage == "no suggestions in the journal"


def test_runs_with_empty_shortlists_are_counted_but_contribute_nothing():
    """'The tool ran 20 days and found nothing' belongs next to the days it did."""
    s = review(_journal(("2026-01-04", []), ("2026-01-05", [])),
               lambda sym, start, tf: None, horizon_bars=90)
    assert s.n_suggestions == 0
    assert s.win_rate is None


def test_a_win_rate_without_scored_rows_has_no_payoff_and_no_comparison():
    s = review(_journal(("2026-01-04", [_entry(symbol="NODATA")])),
               lambda sym, start, tf: None, horizon_bars=90)
    assert s.n_unscorable == 1
    assert s.realised_payoff_r is None
    assert s.realised_breakeven is None
    assert s.expectancy_r is None
    assert s.clears_breakeven is None
    assert s.payoff_diverged is False


# --------------------------------------- payoff divergence and expectancy


def test_expectancy_is_reported_and_needs_no_payoff_assumption():
    """One 3R win and two 1R losses is a 33% win rate that still makes money. A
    summary reporting only the win rate would call this a failure."""
    entries = _journal(("2026-01-04", [_entry(symbol=sym) for sym in ("A", "B", "C")]))

    def fetch(symbol, start, timeframe):
        return _win_bars() if symbol == "A" else _lose_bars()

    s = review(entries, fetch, horizon_bars=90)
    assert s.win_rate == pytest.approx(1 / 3)
    assert s.expectancy_r == pytest.approx((2.9 - 1.1 - 1.1) / 3)
    assert s.expectancy_r > 0
    assert s.realised_payoff_r == pytest.approx(3.0)
    assert s.quoted_rr == pytest.approx(3.0)
    assert s.payoff_diverged is False
    assert s.clears_breakeven is True


def test_three_losses_clear_nothing():
    entries = _journal(("2026-01-04", [_entry(symbol=sym) for sym in ("A", "B", "C")]))
    s = review(entries, lambda sym, start, tf: _lose_bars(), horizon_bars=90)
    assert s.win_rate == pytest.approx(0.0)
    assert s.n_losses == 3 and s.n_wins == 0
    assert s.clears_breakeven is False
    assert s.avg_net_r == pytest.approx(-1.1)
    assert s.expectancy_r == pytest.approx(-1.1)


def test_a_book_with_no_losses_clears_its_breakeven():
    """The mirror edge case. Leaving `clears_breakeven` as None here would read as
    "cannot tell" for a book that lost nothing."""
    entries = _journal(("2026-01-04", [_entry(symbol=sym) for sym in ("A", "B")]))
    s = review(entries, lambda sym, start, tf: _win_bars(), horizon_bars=90)
    assert s.win_rate == pytest.approx(1.0)
    assert s.clears_breakeven is True
    assert s.realised_breakeven == pytest.approx(0.0)


def test_time_exits_make_the_realised_payoff_diverge_from_the_quoted_one():
    """The case that breaks a naive win-rate comparison: nobody hits the target, so
    wins are small marks-to-market rather than the promised 3R."""
    entries = _journal(("2026-01-04",
                        [_entry(symbol=sym) for sym in ("A", "B", "C", "D")]))

    def fetch(symbol, start, timeframe):
        # low 98.5 is above the 98 stop and high 104 below the 106 target, so neither
        # level is touched and all four run to the horizon. Winners close at 101.5
        # (+0.75R), losers at 99.0 (-0.5R) -> realised payoff 1.5 against a promised 3.
        close = 101.5 if symbol in ("A", "B") else 99.0
        return _bars([_BAR0] + [(104.0, 98.5, close)] * 4)

    s = review(entries, fetch, horizon_bars=4)
    assert s.n_scored == 4
    assert all(r.outcome == "time_exit" for r in s.rows)
    assert s.win_rate == pytest.approx(0.5)
    assert s.realised_payoff_r == pytest.approx(1.5)
    assert s.quoted_rr == pytest.approx(3.0)
    assert s.payoff_diverged is True
    assert s.realised_breakeven == pytest.approx(0.44)    # (1+0.1)/(1+1.5)
    assert s.clears_breakeven is True
    assert s.expectancy_r == pytest.approx(0.025)


def test_the_quoted_breakeven_is_not_used_when_wins_came_in_short():
    """A shortlist that promised 3:1 and delivered 1:1 needs a much higher win rate
    than it quoted. Comparing against the promise would pass a losing book - which is
    the whole reason the realised payoff is computed."""
    entries = _journal(("2026-01-04", [
        _entry(symbol="A", breakeven_win_rate=0.28),      # as quoted at 3:1
        _entry(symbol="B", breakeven_win_rate=0.28),
        _entry(symbol="C", breakeven_win_rate=0.28),
    ]))

    def fetch(symbol, start, timeframe):
        # One small winner (+0.5R) and two small losers (-0.5R). The range stays inside
        # the levels - a low of 97.5 would breach the 98 stop and turn every row into a
        # stop_loss - so all three run to the horizon and time-exit.
        close = 101.0 if symbol == "A" else 99.0
        return _bars([_BAR0] + [(102.0, 98.5, close)] * 4)

    s = review(entries, fetch, horizon_bars=4)
    assert s.win_rate == pytest.approx(1 / 3)
    assert s.avg_breakeven == pytest.approx(0.28)
    assert s.win_rate > s.avg_breakeven, "the naive comparison would pass this book"
    assert s.realised_payoff_r == pytest.approx(1.0)
    assert s.quoted_rr == pytest.approx(3.0)
    assert s.payoff_diverged is True
    assert s.realised_breakeven == pytest.approx(0.55)    # (1+0.1)/(1+1.0)
    assert s.clears_breakeven is False
    assert s.expectancy_r == pytest.approx(-0.2666666666, rel=1e-6)


def test_divergence_is_measured_against_the_promise_not_the_take_profit_exits():
    """If no suggestion ever reached its target there are no take-profit rows, so a
    divergence test built on them would report no divergence in exactly the case where
    the promise was missed most completely."""
    entries = _journal(("2026-01-04", [_entry(symbol=sym) for sym in ("A", "B")]))

    def fetch(symbol, start, timeframe):
        close = 101.0 if symbol == "A" else 99.0
        return _bars([_BAR0] + [(102.0, 98.5, close)] * 4)

    s = review(entries, fetch, horizon_bars=4)
    assert s.by_outcome == {"time_exit": 2}
    assert "take_profit" not in s.by_outcome
    assert s.quoted_rr == pytest.approx(3.0)
    assert s.realised_payoff_r == pytest.approx(1.0)
    assert s.payoff_diverged is True


# ------------------------------------------------------- the pinned vocabulary


def test_an_outcome_outside_the_vocabulary_counts_as_neither_scored_nor_a_win():
    """The reason ``OUTCOMES`` and ``SCORED_OUTCOMES`` are pinned rather than inferred.

    A sixth state added later must land on the *refusal* side until someone decides
    otherwise. Defining ``scored`` by exclusion - ``outcome not in ("open",
    "unscorable")`` - would let any new state count as a result and walk straight into
    the win rate, which is how a partially-filled or cancelled idea would end up being
    reported as a loss.
    """
    row = ScoredSuggestion(as_of="2026-01-04", symbol="X", side="long", entry=100.0,
                           stop=98.0, target=106.0, outcome="partially_filled",
                           r_multiple=1.5, cost_in_r=0.10)
    assert row.outcome not in OUTCOMES
    assert row.scored is False
    assert row.is_win is None
    # the arithmetic is still available; it is only the *counting* that is refused
    assert row.net_r == pytest.approx(1.4)


def test_an_unknown_outcome_never_enters_the_win_rate():
    rows = [score_one(_entry(symbol="A"), _win_bars(), 90),
            ScoredSuggestion(as_of="2026-01-04", symbol="X", side="long", entry=100.0,
                             stop=98.0, target=106.0, outcome="cancelled",
                             r_multiple=-1.0, cost_in_r=0.10)]
    s = summarise(rows, n_suggestions=2, horizon_bars=90)
    assert s.n_scored == 1, "an unpinned outcome was counted as a result"
    assert s.n_wins == 1 and s.n_losses == 0
    assert s.win_rate == pytest.approx(1.0)
    assert s.expectancy_r == pytest.approx(2.9)


def test_the_vocabulary_partitions_into_results_and_refusals():
    """Every state is either a result that belongs in a rate or a refusal that does
    not. A state in neither set would be silently dropped from both the win rate and
    the coverage line, so the two sets must be disjoint and cover everything."""
    refusals = OUTCOMES - SCORED_OUTCOMES
    assert refusals == {"open", "unscorable"}
    assert not (SCORED_OUTCOMES & refusals)
    assert SCORED_OUTCOMES == {"take_profit", "stop_loss", "time_exit"}


def test_every_outcome_the_scorer_can_produce_is_in_the_vocabulary():
    """Guards the other direction: a string assigned in ``score_one`` that is not in
    ``OUTCOMES`` would be invisible to both the docs test and the partition above.

    ``unscorable`` is never assigned - it is the dataclass default, and every
    early-return path leaves it in place. That default is itself the property worth
    pinning: a newly constructed row must start as a *refusal*, so that any guard that
    returns early reports "could not score" rather than a result.
    """
    import re
    src = (Path(__file__).resolve().parents[1] / "src" / "tvlab" / "screener"
           / "review.py").read_text(encoding="utf-8")
    assigned = set(re.findall(r's\.outcome\s*(?:,\s*s\.exit_price)?\s*=\s*"(\w+)"', src))
    assert assigned, "the regex found no outcome assignments; it needs updating"
    assert assigned <= OUTCOMES, f"score_one assigns states outside OUTCOMES: {assigned - OUTCOMES}"

    default = re.search(r'outcome:\s*str\s*=\s*"(\w+)"', src)
    assert default, "the outcome default was not found; the regex needs updating"
    assert default.group(1) in OUTCOMES
    assert default.group(1) not in SCORED_OUTCOMES, \
        "a fresh row defaults to a result, so an early return would report a score"

    # together the assignments and the default must cover the vocabulary exactly
    assert assigned | {default.group(1)} == OUTCOMES, \
        f"unreachable states in OUTCOMES: {OUTCOMES - assigned - {default.group(1)}}"


def test_a_fresh_row_is_a_refusal_not_a_result():
    """The runtime form of the default check above: every guard in ``score_one``
    returns early, so the state a row is born in is the state an unscoreable
    suggestion is reported in."""
    bare = ScoredSuggestion(as_of="2026-01-04", symbol="X", side="long", entry=100.0,
                            stop=98.0, target=106.0)
    assert bare.outcome == "unscorable"
    assert bare.scored is False
    assert bare.is_win is None


# ------------------------------------------------------------- serialisation


def test_summary_to_dict_keeps_the_properties():
    """`coverage` is a property; a to_dict built from __dict__ would drop it, which is
    the bug that blanked a dashboard panel once already."""
    s = review(_journal(("2026-01-04", [_entry()])),
               lambda sym, start, tf: _win_bars(), horizon_bars=90)
    d = s.to_dict()
    assert d["coverage"] == s.coverage
    assert d["expectancy_r"] == pytest.approx(s.expectancy_r)
    assert d["realised_breakeven"] == pytest.approx(s.realised_breakeven)
    row = d["rows"][0]
    assert row["net_r"] == pytest.approx(s.rows[0].net_r)
    assert row["is_win"] is True
    assert row["scored"] is True
    assert row["entry_drift"] == pytest.approx(0.0)
    json.dumps(d, default=str)


def test_summarise_can_be_driven_without_a_price_source():
    rows = [score_one(_entry(symbol="A"), _win_bars(), 90),
            score_one(_entry(symbol="B"), None, 90)]
    s = summarise(rows, n_suggestions=2, horizon_bars=90, provenance="SYNTHETIC")
    assert s.n_scored == 1 and s.n_unscorable == 1
    assert s.provenance == "SYNTHETIC"


def test_a_win_rate_on_two_suggestions_still_travels_with_its_sample():
    """Not a property of the code so much as a guard on how it gets read: the count is
    emitted everywhere the rate is."""
    s = review(_journal(("2026-01-04", [_entry(symbol="A"), _entry(symbol="B")])),
               lambda sym, start, tf: _win_bars(), horizon_bars=90)
    assert s.win_rate == pytest.approx(1.0)
    assert s.n_scored == 2
    assert "2 of 2 scored" in s.coverage
