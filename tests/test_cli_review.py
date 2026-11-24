"""The ``tvlab review`` command, and the prose that promises it.

Driven by a journal written to a patched ``REPO_ROOT`` and a stubbed ``ohlcv``, so the
anchoring, the framing and the exit codes can be checked exactly rather than hoping
the synthetic generator happens to have bars after today (it does not - which is
itself asserted below).

Two things are protected here that a unit test of ``screener/review.py`` cannot see:

* **Framing.** Coverage leads, the synthetic caveat precedes any rate, and a journal
  that could not be scored says so instead of printing 0%.
* **The prose.** Six places in this repo used to say "in a month you can score the
  shortlists against what actually happened" while nothing did the scoring. Those
  sites must now name the command that exists, and the unbacked wording must be gone -
  asserted as *absent*, because a test that only checks for new wording passes while
  the old claim is still sitting next to it.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from tvlab.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _series(rows, start="2026-01-01", tz="UTC"):
    """rows: list of (high, low, close). Daily bars, so one bar per date and the
    anchoring is easy to reason about."""
    idx = pd.date_range(start, periods=len(rows), freq="D", tz=tz)
    return pd.DataFrame({"open": [r[2] for r in rows], "high": [r[0] for r in rows],
                         "low": [r[1] for r in rows], "close": [r[2] for r in rows],
                         "volume": [1.0] * len(rows)}, index=idx)


def _sugg(symbol="AAA", last=100.0, stop=98.0, target=106.0, side="long", **kw):
    d = dict(symbol=symbol, side=side, last=last, stop=stop, target=target, rr=3.0,
             risk_pct=0.02, cost_in_r=0.10, breakeven_win_rate=0.275,
             quality_score=0.5, reasons=[], warnings=[], fundamental_status=None)
    d.update(kw)
    return d


def _run(as_of, shortlist, timeframe="1d", **kw):
    d = dict(as_of=as_of, market="crypto", style="swing", strategy="smc_trend",
             timeframe=timeframe, shortlist=shortlist, rejected_technical=[],
             rejected_fundamental=[], data_errors={}, symbols_scanned=10,
             signals_found=len(shortlist), outcome="shortlist" if shortlist
             else "no_signals", outcome_detail="", study_verdict="NOT VALIDATED",
             notes=[])
    d.update(kw)
    return d


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich truncates to the terminal width; at CliRunner's 80 columns a symbol
    renders as "AA..." and substring assertions silently pass or fail."""
    monkeypatch.setenv("COLUMNS", "250")


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    """Keep the journal out of the real repo, and out of whatever a live run left in
    ``output/daily/`` - a test that reads the artifact on disk is a time bomb."""
    monkeypatch.setattr("tvlab.cli.REPO_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def runner():
    return CliRunner()


def _write_journal(root, runs):
    d = root / "output" / "daily"
    d.mkdir(parents=True, exist_ok=True)
    (d / "journal.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in runs), encoding="utf-8")
    return d / "journal.jsonl"


@pytest.fixture
def patch_ohlcv(monkeypatch):
    """Stub DataStore.ohlcv. `store_fetcher` is duck-typed on `.ohlcv`, so this also
    exercises the real anchoring rather than a copy of it."""
    calls = []

    def _apply(frames, exc=None):
        import tvlab.data.store as store_mod

        def fake(self, symbol, timeframe=None, start=None, end=None, limit=None,
                 **kwargs):
            calls.append({"symbol": symbol, "timeframe": timeframe, "start": start,
                          "limit": limit})
            if exc is not None:
                raise exc
            df = frames.get(symbol)
            if df is None:
                raise RuntimeError(f"no frame stubbed for {symbol}")
            return df.iloc[:limit] if limit else df

        monkeypatch.setattr(store_mod.DataStore, "ohlcv", fake)
        return calls
    return _apply


# ------------------------------------------------------------------ no journal


def test_no_journal_explains_how_to_make_one(runner, isolated_output):
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "Nothing to review yet" in out
    assert "tvlab daily" in out, "the remedy must name the command that writes it"
    # It must not look like a scored review.
    assert "win rate" not in out
    assert "expectancy" not in out


# --------------------------------------------------------------------- framing


#: the suggestion bar: closes at the entry price, inside both levels
_BAR0 = (101.0, 99.5, 100.0)


def _scoreable_journal(root):
    """One suggestion per outcome class, each with its own frame so the paths differ.

    At ``--horizon 1`` the single bar after the suggestion decides every one of them:

    ======  ==================================================  ============
    WIN     high 106.5 clears the 106 target                    take_profit
    LOSE    low 97.5 breaches the 98 stop                         stop_loss
    TIME    stays inside both levels                              time_exit
    OPEN    the frame holds only the suggestion bar               open
    NODATA  no frame stubbed, so the fetch raises                 unscorable
    ======  ==================================================  ============
    """
    runs = [_run("2026-01-01", [_sugg(sym)])
            for sym in ("WIN", "LOSE", "TIME", "OPEN", "NODATA")]
    frames = {
        "WIN": _series([_BAR0, (106.5, 100.0, 106.0)]),
        "LOSE": _series([_BAR0, (100.0, 97.5, 98.0)]),
        "TIME": _series([_BAR0, (102.0, 99.0, 101.0)]),
        "OPEN": _series([_BAR0]),
        # NODATA deliberately absent: patch_ohlcv raises for an unstubbed symbol
    }
    return runs, frames


def _rate_line(text):
    """The rendered win-rate line, as distinct from the phrase "win rate".

    The synthetic caveat reads "any win rate computed here measures the data
    generator", so searching for the phrase finds the caveat itself - which is printed
    first by design. A rendered rate always has a digit after it.
    """
    return re.search(r"win rate\s+[0-9]", text)


def test_a_journal_with_no_winners_renders_instead_of_crashing(runner, isolated_output,
                                                               patch_ohlcv):
    """Zero winners is a real state, not an edge case - and it used to crash.

    The `n_wins == 0` branch emitted `[red]0.0%[/red] [dim](n=1)[/red][/dim]`: a third
    `[/red]` with no tag open to close. rich raises `MarkupError` at render time, so
    `tvlab review` exited 1 on a journal whose suggestions had all lost, and the user
    saw a traceback about markup instead of the result. Every other branch of this
    block closes with the generic `[/]`, which is why only this one broke.

    It survived this long because the only way to reach it is a journal that scored
    at least one suggestion and won none of them. A journal where everything failed
    to fetch exits 2 before the rate block; a journal with a mix takes the `clears_
    breakeven` branch. This fixture is the all-losers case: two symbols on one date,
    each anchored on bar 0 of its own frame, both stopped out.

    (Two runs on two dates with one shared frame does not work: the second anchors on
    bar 1, whose close is the stop price, so the drift guard correctly refuses it.)
    """
    runs = [_run("2026-01-01", [_sugg("LOSEA"), _sugg("LOSEB")])]
    patch_ohlcv({"LOSEA": _series([_BAR0, (100.0, 97.5, 98.0)]),
                 "LOSEB": _series([_BAR0, (100.0, 97.0, 97.5)])})
    _write_journal(isolated_output, runs)

    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing",
                              "--horizon", "1"])
    assert res.exit_code == 0, f"crashed with {res.exception!r}\n{res.output}"
    out = _flat(res.output)
    assert _rate_line(out), "the win-rate line did not render"
    assert "0.0%" in out
    assert "not one suggestion reached its target" in out
    # and it must still say what was scored, rather than implying nothing was measured
    assert "2 of 2 scored" in out
    # A zero-win book has no winners to pay the promised R, so the divergence note is
    # about there being nothing to apply the breakeven to - not about underpaying.
    assert "no winners at all" in out
    # Both rows are losses and both were scored, so the rates below describe them.
    assert out.count("stop_loss") >= 2


def test_the_no_winners_branch_has_no_stray_closing_tag():
    """The crash was one unbalanced tag; pin the exact sequence that caused it.

    A test that re-renders every literal in the block would be testing the test's own
    string surgery - concatenating the three branches' literals is not valid markup
    even when each branch is correct. Asserting the broken sequence is absent is the
    honest guard, and it is specific enough that only reintroducing this bug trips it.
    """
    import tvlab.cli as climod

    src = Path(climod.__file__).read_text(encoding="utf-8")
    start = src.index("if summary.n_wins == 0:")
    block = src[start:src.index("exp_c = ", start)]
    assert "[/red][/dim]" not in block, (
        "a [/red] with no opener to close: rich raises MarkupError at render time")
    # Counting `[red]` against `[/red]` per literal would be wrong here: the branches
    # split their markup across f-string lines and close with the generic `[/]`, so a
    # correct block does not balance by that measure. Only the defect is pinned.
    assert "[red]" in block and "[/]" in block, "expected the styled branch to be here"


def test_coverage_is_printed_before_any_rate(runner, isolated_output, patch_ohlcv):
    runs, frames = _scoreable_journal(isolated_output)
    _write_journal(isolated_output, runs)
    patch_ohlcv(frames)
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing",
                              "--horizon", "1"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "scored" in out
    m = _rate_line(out)
    assert m, "no win rate was rendered at all"
    assert out.index("scored") < m.start()


def test_the_synthetic_caveat_precedes_the_win_rate(runner, isolated_output, patch_ohlcv):
    runs, frames = _scoreable_journal(isolated_output)
    _write_journal(isolated_output, runs)
    patch_ohlcv(frames)
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing",
                              "--horizon", "1"])
    out = _flat(res.output)
    assert "measures the data generator" in out
    m = _rate_line(out)
    assert m, "no win rate was rendered at all"
    # The caveat sentence itself contains the phrase "win rate", which is why this
    # compares against a rendered rate rather than the phrase.
    assert out.index("measures the data generator") < m.start()


def test_an_unscorable_row_is_listed_with_its_reason(runner, isolated_output, patch_ohlcv):
    """NODATA has no stubbed frame, so its fetch raises. It must be named, not folded
    into a count - an outage and a young idea have opposite remedies."""
    runs, frames = _scoreable_journal(isolated_output)
    _write_journal(isolated_output, runs)
    patch_ohlcv(frames)
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing",
                              "--horizon", "1"])
    out = _flat(res.output)
    assert "Unscorable" in out
    assert "NODATA" in out


def test_a_journal_that_cannot_be_scored_does_not_print_a_rate(runner, isolated_output,
                                                               patch_ohlcv):
    """Every symbol fails to fetch. That is a data outage, and rendering it as a 0%
    win rate would turn an infrastructure problem into a verdict on the strategy."""
    runs = [_run("2026-01-01", [_sugg("AAA"), _sugg("BBB")])]
    _write_journal(isolated_output, runs)
    patch_ohlcv({}, exc=RuntimeError("network down"))
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing"])
    out = _flat(res.output)
    assert "Nothing could be scored" in out
    assert "not a verdict on the strategy" in out
    # The message says "No win rate, no expectancy", which contains the phrase; what
    # must not appear is a rendered rate.
    assert _rate_line(out) is None, "a rate was rendered for a book that has none"
    assert res.exit_code == 2, "a total fetch failure is a data failure, exit 2"


def test_an_empty_shortlist_run_is_counted_but_scores_nothing(runner, isolated_output,
                                                              patch_ohlcv):
    runs = [_run("2026-01-01", [], outcome="no_signals")]
    _write_journal(isolated_output, runs)
    patch_ohlcv({})
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    assert "no suggestions" in _flat(res.output).lower() or "Nothing could be scored" \
        in res.output


# -------------------------------------------------------------------- options


def test_json_output_is_parseable_and_carries_the_new_fields(runner, isolated_output,
                                                             patch_ohlcv):
    runs, frames = _scoreable_journal(isolated_output)
    _write_journal(isolated_output, runs)
    patch_ohlcv(frames)
    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing",
                              "--horizon", "3", "--json"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.output[res.output.index("{"):])
    for k in ("as_of", "coverage", "n_scored", "win_rate", "expectancy_r",
              "quoted_rr", "realised_payoff_r", "realised_breakeven",
              "clears_breakeven", "payoff_diverged", "by_outcome", "rows"):
        assert k in d, f"--json dropped {k}"
    assert d["rows"][0]["net_r"] is not None


def test_the_horizon_option_reaches_the_scorer(runner, isolated_output, patch_ohlcv):
    """With a horizon of 1 the time-exit row resolves; with 500 it is still open."""
    runs, frames = _scoreable_journal(isolated_output)
    _write_journal(isolated_output, runs)
    patch_ohlcv(frames)
    short = runner.invoke(cli, ["review", "--horizon", "1", "--json"])
    long_ = runner.invoke(cli, ["review", "--horizon", "500", "--json"])
    d_short = json.loads(short.output[short.output.index("{"):])
    d_long = json.loads(long_.output[long_.output.index("{"):])
    assert d_short["horizon_bars"] == 1
    assert d_long["horizon_bars"] == 500
    assert d_long["n_open"] > d_short["n_open"]


def test_days_limits_how_many_runs_are_reviewed(runner, isolated_output, patch_ohlcv):
    runs, frames = _scoreable_journal(isolated_output)
    _write_journal(isolated_output, runs)
    patch_ohlcv(frames)
    res = runner.invoke(cli, ["review", "--days", "2", "--json"])
    d = json.loads(res.output[res.output.index("{"):])
    assert d["n_suggestions"] == 2


def test_the_fetcher_is_given_the_timeframe_the_run_recorded(runner, isolated_output,
                                                             patch_ohlcv):
    """A journal accumulates across config changes; replaying a 1d suggestion against
    4h bars would silently change what the horizon means."""
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA")], timeframe="1d")])
    calls = patch_ohlcv({"AAA": bars})
    runner.invoke(cli, ["review", "--horizon", "1"])
    assert calls and calls[0]["timeframe"] == "1d", calls


def test_a_timeframe_override_replaces_the_recorded_one(runner, isolated_output,
                                                        patch_ohlcv):
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA")], timeframe="1d")])
    calls = patch_ohlcv({"AAA": bars})
    runner.invoke(cli, ["review", "--timeframe", "4h", "--horizon", "1"])
    assert calls and calls[0]["timeframe"] == "4h", calls


# ------------------------------------------------------------------ anchoring


def test_the_window_is_sliced_not_refetched_from_the_suggestion_date(runner,
                                                                     isolated_output,
                                                                     patch_ohlcv):
    """`ohlcv` must be called without a `start`, then sliced. Passing a later start is
    not a harmless optimisation: on the synthetic source it returns an unrelated
    series, because the seed fixes the stream from the start date."""
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA")])])
    calls = patch_ohlcv({"AAA": bars})
    runner.invoke(cli, ["review", "--horizon", "1"])
    assert calls[0]["start"] is None, f"the fetcher re-requested from a start: {calls}"
    assert calls[0]["limit"] == 1200, "the scan's bar count must be reused"


def test_a_tz_naive_index_is_handled(runner, isolated_output, patch_ohlcv):
    """The journal stores a bare date; a naive index must not raise when compared."""
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)], tz=None)
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA")])])
    patch_ohlcv({"AAA": bars})
    res = runner.invoke(cli, ["review", "--horizon", "1", "--json"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.output[res.output.index("{"):])
    assert d["rows"][0]["outcome"] == "take_profit"


def test_a_drifting_entry_price_is_refused(runner, isolated_output, patch_ohlcv):
    """The guard that stops a fabricated series from being scored as though it were
    the one the idea was based on."""
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output,
                   [_run("2026-01-01", [_sugg("AAA", last=140.0)])])
    patch_ohlcv({"AAA": bars})
    res = runner.invoke(cli, ["review", "--horizon", "1", "--json"])
    d = json.loads(res.output[res.output.index("{"):])
    assert d["rows"][0]["outcome"] == "unscorable"
    assert d["n_scored"] == 0


def test_the_entry_tolerance_flag_actually_reaches_the_scorer(runner, isolated_output,
                                                              patch_ohlcv):
    """The README documents this flag, so it has to change behaviour - a documented
    option that is parsed and dropped is the same class of unbacked claim as the
    scoring promise this whole feature exists to retire.

    A 2% drift is refused at the 0.5% default and accepted at 5%.
    """
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA", last=102.0)])])
    patch_ohlcv({"AAA": bars})

    strict = runner.invoke(cli, ["review", "--horizon", "1", "--json"])
    d_strict = json.loads(strict.output[strict.output.index("{"):])
    assert d_strict["rows"][0]["outcome"] == "unscorable"
    # drift is relative to the journaled entry, so |100 - 102| / 102, not / 100
    assert d_strict["rows"][0]["entry_drift"] == pytest.approx(2 / 102)
    assert d_strict["entry_tolerance"] == pytest.approx(0.005)

    loose = runner.invoke(cli, ["review", "--horizon", "1", "--json",
                                "--entry-tolerance", "0.05"])
    d_loose = json.loads(loose.output[loose.output.index("{"):])
    assert d_loose["rows"][0]["outcome"] == "take_profit", \
        "widening the tolerance did not change the outcome; the flag is not wired"
    assert d_loose["n_scored"] == 1
    assert d_loose["entry_tolerance"] == pytest.approx(0.05)


def test_the_tolerance_used_is_printed_in_the_header(runner, isolated_output, patch_ohlcv):
    """Two reviews of one journal can disagree, and the tolerance is one reason why.
    It has to be on screen next to the horizon."""
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA")])])
    patch_ohlcv({"AAA": bars})
    res = runner.invoke(cli, ["review", "--horizon", "1", "--entry-tolerance", "0.02"])
    out = _flat(res.output)
    assert "entry tolerance 2.00%" in out
    assert "horizon 1 bars" in out


def test_set_cannot_reach_the_tolerance_and_the_docs_say_so(runner, isolated_output,
                                                            patch_ohlcv):
    """`--set` only reaches fields in the config schema, and an unknown path is a
    silent no-op rather than an error. That is why the tolerance is a flag: an earlier
    draft of the README claimed it was `--set`-able, which would have been a plausible
    instruction that did nothing."""
    bars = _series([(101.0, 99.5, 100.0), (106.5, 100.0, 106.0)])
    _write_journal(isolated_output, [_run("2026-01-01", [_sugg("AAA", last=102.0)])])
    patch_ohlcv({"AAA": bars})

    res = runner.invoke(cli, ["review", "--horizon", "1", "--json",
                              "--set", "review.entry_tolerance=0.5"])
    assert res.exit_code == 0, res.output          # no error - it is silently ignored
    d = json.loads(res.output[res.output.index("{"):])
    assert d["entry_tolerance"] == pytest.approx(0.005), "--set changed the tolerance"
    assert d["rows"][0]["outcome"] == "unscorable", "--set appeared to take effect"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "--entry-tolerance" in readme, "the README must name the flag that exists"
    assert "silent no-op" in readme, \
        "the README should warn that --set cannot reach this"


# ------------------------------------------------- the real-data round trip


def test_the_csv_round_trip_scores_real_bars_with_zero_drift(runner, isolated_output,
                                                             tmp_path, monkeypatch):
    """The only path where this feature is useful, and the one every other test here
    stubs out.

    On synthetic data the generated window ends years in the past, so suggestions have
    no bars after them and come back ``open`` or ``unscorable`` - correct, but it never
    exercises the scoring. A CSV source that has actually accumulated history does.

    This writes bars to disk under the filename convention ``tvlab data --export``
    uses, journals entries whose prices are read back out of those bars, and runs the
    real command with ``--data-source csv``. Nothing is monkeypatched. What it proves:

    * ``CsvSource`` finds the exported file, including for a symbol whose name
      contains a separator (``BTC/USDT`` -> ``BTC_USDT_1d.parquet``);
    * the anchoring lands on exactly the bar the journal recorded, so ``entry_drift``
      is 0.0 and the guard passes for the right reason rather than by tolerance;
    * outcomes resolve against real prices, and the arithmetic hand-checks.
    """
    root = tmp_path / "bars"
    root.mkdir()
    n = 120
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(11)
    close = 50000.0 + np.cumsum(rng.normal(0, 400.0, n))
    frame = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close, "volume": 1.0}, index=idx)
    frame.index.name = "timestamp"
    frame.to_parquet(root / "BTC_USDT_1d.parquet")

    # Journal entries priced from the bars themselves, at three points with enough
    # history after them to resolve, plus one AT the final bar so there is no future
    # to score against at all. Note that "few bars left" is not the same thing as
    # "still open": if the stop is breached within two bars the outcome is known and
    # is reported, correctly, as stop_loss. Only the absence of any bar after the
    # suggestion forces `open`.
    runs = []
    for i in (20, 50, 80, n - 1):
        e = float(close[i])
        runs.append(_run(str(idx[i].date()), [
            _sugg("BTC/USDT", last=e, stop=round(e * 0.98, 2),
                  target=round(e * 1.06, 2))], timeframe="1d"))
    (isolated_output / "output" / "daily").mkdir(parents=True, exist_ok=True)
    (isolated_output / "output" / "daily" / "journal.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in runs), encoding="utf-8")

    res = runner.invoke(cli, ["review", "--market", "crypto", "--style", "swing",
                              "--data-source", "csv", "--horizon", "10", "--json",
                              "--set", f"data.csv_root={root}"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.output[res.output.index("{"):])

    assert d["provenance"] == "csv"
    assert d["n_suggestions"] == 4
    # every row anchored on exactly the bar the journal recorded
    assert [r["entry_drift"] for r in d["rows"]] == [0.0, 0.0, 0.0, 0.0]
    # the last one has only 2 bars after it, so it must be open rather than scored
    assert d["rows"][-1]["outcome"] == "open"
    assert d["n_scored"] == 3 and d["n_open"] == 1 and d["n_unscorable"] == 0
    # and the three that resolved did so against real prices
    assert {r["outcome"] for r in d["rows"][:3]} <= {"take_profit", "stop_loss",
                                                     "time_exit"}
    for r in d["rows"][:3]:
        assert r["bars_observed"] == 11           # the suggestion bar + 10 of horizon
        assert np.isfinite(r["r_multiple"])


def test_no_synthetic_warning_appears_for_a_csv_source(runner, isolated_output,
                                                       tmp_path):
    """The generator caveat is true of synthetic data and false of everything else.
    Printing it for a real source would train the reader to ignore it."""
    root = tmp_path / "bars"
    root.mkdir()
    # validate_ohlcv refuses fewer than 60 bars, so a 30-bar fixture fails before the
    # code under test is ever reached - and fails as a "fetch failure", which would
    # have looked like a real regression in the warning logic.
    idx = pd.date_range("2022-01-01", periods=80, freq="D", tz="UTC")
    close = np.full(80, 100.0)
    frame = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close, "volume": 1.0}, index=idx)
    frame.index.name = "timestamp"
    frame.to_parquet(root / "BTC_USDT_1d.parquet")
    _write_journal(isolated_output,
                   [_run(str(idx[5].date()), [_sugg("BTC/USDT")], timeframe="1d")])

    res = runner.invoke(cli, ["review", "--data-source", "csv", "--horizon", "5",
                              "--set", f"data.csv_root={root}"])
    assert res.exit_code == 0, res.output
    assert "measures the data generator" not in res.output
    assert "source csv" in _flat(res.output)


# ---------------------------------------------------------------------- prose


def test_the_unbacked_scoring_claim_is_gone_from_every_site():
    """The wording that promised a capability the repo did not have. Asserted absent,
    not merely superseded: a test that only looks for the new sentence passes while
    the old claim is still sitting beside it."""
    banned = "in a month you can score the shortlists"
    sites = {
        "README.md": REPO_ROOT / "README.md",
        "cli.py": REPO_ROOT / "src" / "tvlab" / "cli.py",
        "screener/daily.py": REPO_ROOT / "src" / "tvlab" / "screener" / "daily.py",
        "dashboard/static/index.html": REPO_ROOT / "dashboard" / "static" / "index.html",
    }
    for name, path in sites.items():
        text = _flat(path.read_text(encoding="utf-8"))
        assert banned not in text.lower(), f"{name} still carries the unbacked claim"


def test_the_sites_that_promise_scoring_now_name_the_command():
    sites = {
        "README.md": REPO_ROOT / "README.md",
        "cli.py": REPO_ROOT / "src" / "tvlab" / "cli.py",
        "dashboard/static/index.html": REPO_ROOT / "dashboard" / "static" / "index.html",
    }
    for name, path in sites.items():
        text = path.read_text(encoding="utf-8")
        assert "review" in text.lower(), f"{name} never mentions the review capability"
    # The command itself must be spelled out where a reader would go looking for it.
    assert "tvlab review" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "tvlab review" in (REPO_ROOT / "src" / "tvlab" / "cli.py").read_text(encoding="utf-8")


def test_the_daily_command_points_at_the_review_command(runner, isolated_output,
                                                        monkeypatch):
    """The line printed after journaling is where a user learns the record can be
    scored. It used to say only that a recommendation should be falsifiable."""
    import tvlab.screener.daily as daily_mod
    from tvlab.screener.scanner import ScanRow

    def fake_scan(store, cfg, symbols, **kwargs):
        row = ScanRow(symbol="AAA", last=100.0, signal=1, stop=98.0, target=106.0,
                      reward_risk=3.0, trend=1, adx=25.0, atr_pct=2.0, bars=1200)
        return [row], {}

    monkeypatch.setattr("tvlab.screener.scan_universe", fake_scan)
    monkeypatch.setattr("tvlab.screener.scanner.scan_universe", fake_scan)
    monkeypatch.setattr("tvlab.cli._measured_win_rate", lambda *a, **k: None)
    monkeypatch.setattr(daily_mod, "append_journal",
                        lambda b, out_dir: out_dir / "daily" / "journal.jsonl")

    res = runner.invoke(cli, ["daily", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    assert "tvlab review" in _flat(res.output)


def test_the_daily_docstring_does_not_reference_a_module_that_does_not_exist():
    """`journal.py` was cited for years and never existed; the journal functions live
    in `daily.py`. Every module a docstring names must be importable."""
    text = (REPO_ROOT / "src" / "tvlab" / "screener" / "daily.py").read_text(encoding="utf-8")
    assert "``journal.py``" not in text
    for mod in re.findall(r"``screener/(\w+)\.py``", text):
        assert (REPO_ROOT / "src" / "tvlab" / "screener" / f"{mod}.py").exists(), \
            f"daily.py cites screener/{mod}.py, which does not exist"


def test_the_review_command_is_registered_and_documented():
    from tvlab.cli import cli as group
    from tvlab.screener.review import OUTCOMES, SCORED_OUTCOMES

    assert "review" in group.commands
    help_text = group.commands["review"].help or ""
    assert "journal" in help_text.lower()
    # Every outcome the scorer can produce must be documented, so a sixth state cannot
    # appear in the payload while the help text still describes five.
    for oc in sorted(OUTCOMES):
        assert oc in help_text, f"{oc} is a real outcome but is not documented"
    # And the two refusals must be described as refusals, not as results.
    assert "neither win nor loss" in help_text
    assert SCORED_OUTCOMES < OUTCOMES
