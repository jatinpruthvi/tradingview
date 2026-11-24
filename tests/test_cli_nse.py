"""CLI surface for the NSE commands, driven by an injected fetcher.

These run the real click commands end to end against recorded payloads, so the
rendering path is tested and not just the error path. No network is touched.
"""
import re
from datetime import datetime

import pandas as pd
import pytest
from click.testing import CliRunner
from zoneinfo import ZoneInfo

from _nse_payloads import _all_indices_payload

from tvlab.cli import cli
from tvlab.data import nse as nse_mod
from tvlab.data.nse import NseSource

IST = ZoneInfo("Asia/Kolkata")


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive rich's line wrapping."""
    return re.sub(r"\s+", " ", text)


class _FrozenDateTime:
    """Stand-in for ``datetime`` that returns a fixed 'now'.

    ``tvlab.cli`` imports datetime at module level precisely so this works - a
    function-local import cannot be patched, which is how the first version of
    these tests came to assert nothing at all.
    """

    def __init__(self, fixed: datetime):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed.astimezone(tz) if tz else self._fixed.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """TVLAB_HTTP_DELAY exists so politeness is tunable without editing code."""
    monkeypatch.setenv("TVLAB_HTTP_DELAY", "0")


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_nse(monkeypatch):
    """Route every NSE and Yahoo URL to a fixture payload."""
    import json

    from _nse_payloads import (
        NIFTY500_CSV_BODY,
        _holiday_payload,
        _market_watch_payload,
        _yahoo_payload,
    )

    def fetcher(url, timeout):
        if "holiday-master" in url:
            return json.dumps(_holiday_payload()).encode()
        if "allIndices" in url:
            return json.dumps(_all_indices_payload()).encode()
        if "marketWatchApi" in url:
            return json.dumps(_market_watch_payload()).encode()
        if "nsearchives" in url:
            return NIFTY500_CSV_BODY.encode()
        if "finance/chart" in url:
            return json.dumps(_yahoo_payload()).encode()
        raise AssertionError(f"unexpected URL requested: {url}")

    monkeypatch.setattr(nse_mod, "_default_fetcher", fetcher)
    return fetcher


# ------------------------------------------------------------------- group


def test_nse_group_is_registered(runner):
    res = runner.invoke(cli, ["nse", "--help"])
    assert res.exit_code == 0, res.output
    for sub in ("session", "breadth", "liquidity", "universe", "holidays"):
        assert sub in res.output, sub


def test_nse_group_help_disclaims_being_a_tip(runner):
    """This group describes the market. Its help text must not read as a
    recommendation engine, because the project's own verdict is NOT VALIDATED."""
    out = _flat(runner.invoke(cli, ["nse", "--help"]).output)
    assert "NOT VALIDATED" in out
    assert "None of them is a signal, a tip or a prediction" in out


# ----------------------------------------------------------------- session


def test_session_reports_an_open_market_and_minutes_left(runner, fake_nse, monkeypatch):
    monkeypatch.setattr("tvlab.cli.datetime",
                        _FrozenDateTime(datetime(2026, 11, 20, 14, 3, tzinfo=IST)))
    res = runner.invoke(cli, ["nse", "session"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "market open YES" in out, "a Friday 14:03 IST is inside the session"
    assert "minutes to close 87" in out


def test_session_reports_a_weekend_as_closed(runner, fake_nse, monkeypatch):
    monkeypatch.setattr("tvlab.cli.datetime",
                        _FrozenDateTime(datetime(2026, 11, 21, 12, 0, tzinfo=IST)))
    out = _flat(runner.invoke(cli, ["nse", "session"]).output)
    assert "market open NO" in out
    assert "weekend" in out.lower()


def test_session_reports_a_market_holiday_as_closed(runner, fake_nse, monkeypatch):
    """26-Nov-2026 is in the holiday fixture. A weekday must not be reported as
    open just because it is not a weekend."""
    monkeypatch.setattr("tvlab.cli.datetime",
                        _FrozenDateTime(datetime(2026, 11, 26, 12, 0, tzinfo=IST)))
    out = _flat(runner.invoke(cli, ["nse", "session"]).output)
    assert "market open NO" in out
    assert "holiday" in out.lower()


def test_session_warns_in_the_closing_minutes(runner, fake_nse, monkeypatch):
    """A same-day trade opened with minutes left is mostly a bet on the closing
    auction; the command says so instead of just printing a number."""
    monkeypatch.setattr("tvlab.cli.datetime",
                        _FrozenDateTime(datetime(2026, 11, 20, 15, 25, tzinfo=IST)))
    out = _flat(runner.invoke(cli, ["nse", "session"]).output)
    assert "minutes to close 5" in out
    assert "closing auction" in out.lower()


# ----------------------------------------------------------------- breadth


def test_breadth_renders_indices_and_a_regime_read(runner, fake_nse):
    res = runner.invoke(cli, ["nse", "breadth"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "NIFTY 50" in out and "NIFTY BANK" in out
    # fixture: 34 advances vs 15 declines -> positive breadth
    assert "advances dominate" in out.lower()


def test_breadth_flags_a_long_only_dead_day(runner, fake_nse, monkeypatch):
    """With allow_short=false, negative breadth means there is no trade to take.
    That has to be stated, not left for the user to infer from a table."""
    monkeypatch.setattr(NseSource, "indices", lambda self: pd.DataFrame([{
        "index": "NIFTY 50", "last": 24500.0, "pct_change": -1.1, "variation": -270.0,
        "previousClose": 24770.0, "advances": 4, "declines": 46, "unchanged": 0,
        "pe": 22.0, "high_52w": 26300.0, "low_52w": 21000.0, "open": 24600.0,
        "breadth": (4 - 46) / 50,
    }]))
    out = _flat(runner.invoke(cli, ["nse", "breadth"]).output).lower()
    assert "declines dominate" in out
    assert "allow_short=false" in out, "must name the config reason there is no trade"


def test_breadth_reports_mixed_breadth_honestly(runner, fake_nse, monkeypatch):
    """Near-zero breadth is not neutral-good: it means a few weights are moving
    the index while the market does nothing."""
    monkeypatch.setattr(NseSource, "indices", lambda self: pd.DataFrame([{
        "index": "NIFTY 50", "last": 24500.0, "pct_change": 0.4, "variation": 98.0,
        "previousClose": 24402.0, "advances": 25, "declines": 25, "unchanged": 0,
        "pe": 22.0, "high_52w": 26300.0, "low_52w": 21000.0, "open": 24450.0,
        "breadth": 0.0,
    }]))
    assert "mixed breadth" in _flat(runner.invoke(cli, ["nse", "breadth"]).output).lower()


# --------------------------------------------------------------- liquidity


def test_liquidity_ranks_by_traded_value_not_volume(runner, fake_nse):
    """YESBANK trades 90M shares against RELIANCE's 8M, but a third of the
    rupees. Ranking on volume would put you in the name you cannot exit."""
    res = runner.invoke(cli, ["nse", "liquidity", "--top", "10"])
    assert res.exit_code == 0, res.output
    out = res.output
    assert out.index("RELIANCE") < out.index("TCS") < out.index("YESBANK"), \
        "must be ordered by traded value"


def test_liquidity_explains_the_range_column(runner, fake_nse):
    out = _flat(runner.invoke(cli, ["nse", "liquidity", "--top", "3"]).output).lower()
    assert "range_pct" in out
    assert "room a same-day trade actually has" in out


def test_liquidity_handles_an_empty_index(runner, fake_nse, monkeypatch):
    monkeypatch.setattr(NseSource, "snapshot",
                        lambda self, index=None: pd.DataFrame(
                            columns=["symbol", "ltp", "pct_change", "open", "high", "low",
                                     "prev_close", "volume", "traded_value", "high_52w", "low_52w"]))
    out = runner.invoke(cli, ["nse", "liquidity"]).output
    assert "no EQ-series rows" in out


# ---------------------------------------------------------------- universe


def test_universe_unions_the_csv_and_the_fo_list(runner, fake_nse):
    """Both legs must contribute. In the shared fixtures HDFCBANK appears only in
    the Nifty 500 CSV and YESBANK only in the market-watch response, so seeing
    both proves a union rather than one leg winning."""
    out = runner.invoke(cli, ["nse", "universe"]).output
    assert "RELIANCE" in out and "TCS" in out          # in both legs
    assert "HDFCBANK" in out, "the CSV-only name must appear"
    assert "YESBANK" in out, "the market-watch-only name must appear"
    assert out.startswith("4 symbols"), out.splitlines()[0]


def test_universe_top_n_narrows_by_liquidity(runner, fake_nse):
    out = _flat(runner.invoke(cli, ["nse", "universe", "--top", "2"]).output).lower()
    assert "most liquid" in out


def test_universe_suggests_the_next_command(runner, fake_nse):
    """A list of symbols is only useful if the user knows what to do with it."""
    out = _flat(runner.invoke(cli, ["nse", "universe", "--top", "3"]).output)
    assert "tvlab scan" in out and "--data-source nse" in out


# ---------------------------------------------------------------- holidays


def test_holidays_lists_upcoming_days(runner, fake_nse):
    out = runner.invoke(cli, ["nse", "holidays", "--days", "400"]).output
    assert "2026-11-26" in out and "Thursday" in out
    assert "2026-12-25" in out


def test_holidays_handles_a_clear_calendar(runner, fake_nse, monkeypatch):
    monkeypatch.setattr(NseSource, "trading_holidays", lambda self: set())
    assert "no nse trading holidays" in _flat(
        runner.invoke(cli, ["nse", "holidays"]).output).lower()


def test_holidays_window_excludes_the_far_future(runner, fake_nse):
    out = runner.invoke(cli, ["nse", "holidays", "--days", "1"]).output
    assert "2026-12-25" not in out


# ------------------------------------------------------------- failure mode


def test_an_unreachable_host_is_a_message_not_a_traceback(runner, monkeypatch):
    """NSE 403s and Yahoo rate-limits are normal operating conditions for this
    command group, so they must read as a hint rather than a stack trace."""
    from tvlab.data.base import DataError

    def boom(url, timeout):
        raise DataError("could not reach nseindia.com: no route to host")

    monkeypatch.setattr(nse_mod, "_default_fetcher", boom)
    res = runner.invoke(cli, ["nse", "breadth"])
    assert res.exit_code == 1, res.output
    out = _flat(res.output)
    assert "nseindia.com" in out
    assert "undocumented public endpoints" in out, "should explain why and what to do"
    assert res.exception is None or isinstance(res.exception, SystemExit), \
        "the DataError must be caught, not propagated"
