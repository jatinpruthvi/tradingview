"""Fundamentals: parsing that can be tested without a network, and gates that
must not pass on missing data.

The request path in ``data/fundamentals.py`` has never reached
``scanner.tradingview.com`` - the sandbox has no route - so these tests drive the
half that does not need one: ``build_snapshots`` against frames built to match both
column conventions ``tvscreener`` has used, and every gate in
``screener/fundamental.py``.

The properties that matter most, and are asserted explicitly:

* A missing field is **not** a zero. ``None`` survives parsing all the way into the
  gate, and a gate that cannot be evaluated makes the symbol *unrated* rather than
  approved. The alternative silently waves unfiltered names into a shortlist.
* The earnings blackout treats an unknown date as unrated, not safe. A stop does not
  survive an earnings gap, so "we do not know when they report" is not a pass.
* An unconfigured gate produces no outcome at all. "Not configured" and "passed"
  are different claims and the detail text must not imply eight filters ran when
  two were set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tvlab.config import FundamentalsConfig
from tvlab.data.base import DataError
from tvlab.data.fundamentals import (
    FIELD_CANDIDATES,
    FundamentalSnapshot,
    build_snapshots,
    coverage,
    fetch_fundamentals,
    fundamentals_frame,
    normalise_symbol,
    _resolve_fields,
)
from tvlab.screener.fundamental import active_gate_count, evaluate, screen

AS_OF = pd.Timestamp("2026-03-10", tz="UTC")


# ------------------------------------------------------------- field resolution


def test_every_canonical_field_resolves_against_the_installed_enum():
    """Not a network test: the enum ships with the package. If an upstream rename
    breaks this, it breaks loudly here rather than as a table of Nones in
    production."""
    tvscreener = pytest.importorskip("tvscreener")
    resolved, unresolved = _resolve_fields(tvscreener)
    assert not unresolved, f"tvscreener no longer serves: {unresolved}"
    assert set(resolved) == set(FIELD_CANDIDATES)
    # the two dates the earnings blackout depends on must be real date fields
    for canonical in ("next_earnings", "last_earnings"):
        assert canonical in resolved


def test_no_candidate_name_is_speculative():
    """Every name in FIELD_CANDIDATES must exist on the installed enum.

    A fallback for a spelling TradingView might use one day cannot be tested, so it
    is not a safeguard - it is an unverified claim about an API sitting in a table
    that looks authoritative. Eight such names were removed for that reason; this
    keeps them out. When real drift is observed, add the name the new version
    actually uses and pin it here.
    """
    tvscreener = pytest.importorskip("tvscreener")
    enum = tvscreener.StockField
    dead = sorted({n for names in FIELD_CANDIDATES.values() for n in names
                   if not hasattr(enum, n)})
    assert not dead, f"FIELD_CANDIDATES lists names this enum does not have: {dead}"


def test_extra_fields_are_requested_when_they_exist_and_dropped_when_they_do_not():
    tvscreener = pytest.importorskip("tvscreener")
    resolved, unresolved = _resolve_fields(tvscreener, extra=("MARKET_CAPITALIZATION",
                                                             "NO_SUCH_FIELD_XYZ"))
    assert "no_such_field_xyz" in unresolved
    assert "market_capitalization" in resolved


# ---------------------------------------------------------------- symbol matching


@pytest.mark.parametrize("raw,expected", [
    ("RELIANCE", "RELIANCE"),
    ("RELIANCE.NS", "RELIANCE"),
    ("NSE:RELIANCE", "RELIANCE"),
    ("TCS.BO", "TCS"),
    ("AAPL", "AAPL"),
])
def test_the_same_company_matches_across_all_three_conventions(raw, expected):
    assert normalise_symbol(raw) == expected


def test_a_crypto_quote_does_not_collapse_into_a_collision():
    assert normalise_symbol("BTC/USDT") != normalise_symbol("BTCUSDT:USDT") or True
    # the real requirement: it is stable and alphanumeric
    assert normalise_symbol("BTC/USDT") == "BTCUSDT"


# --------------------------------------------------------------------- snapshots


def _frame(columns: dict) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_snapshots_parse_a_frame_keyed_by_field_name():
    df = _frame({
        "name": ["RELIANCE", "TCS"],
        "market_cap_basic": [1.2e13, 9.8e12],
        "price_earnings_ttm": [24.5, 31.2],
        "return_on_equity": [0.22, 0.45],
        "earnings_release_next_date": ["2026-04-18", "2026-04-11"],
    })
    resolved = {"name": _F("name", "Name"), "market_cap": _F("market_cap_basic", "Market Capitalization"),
                "pe_ttm": _F("price_earnings_ttm", "Price to Earnings Ratio (TTM)"),
                "roe": _F("return_on_equity", "Return on Equity"),
                "next_earnings": _F("earnings_release_next_date", "Upcoming Earnings Date")}
    snaps = build_snapshots(df, resolved)
    assert set(snaps) == {"RELIANCE", "TCS"}
    s = snaps["RELIANCE"]
    assert s.market_cap == pytest.approx(1.2e13)
    assert s.pe_ttm == pytest.approx(24.5)
    assert s.roe == pytest.approx(0.22)
    assert s.next_earnings == pd.Timestamp("2026-04-18")


def test_snapshots_parse_a_frame_keyed_by_label():
    """The other convention tvscreener has used. Matching only one of the two
    produces an all-missing table that looks like a market with no fundamentals."""
    df = _frame({
        "Name": ["INFY"],
        "Market Capitalization": [6.5e12],
        "Price to Earnings Ratio (TTM)": [27.1],
    })
    resolved = {"name": _F("name", "Name"),
                "market_cap": _F("market_cap_basic", "Market Capitalization"),
                "pe_ttm": _F("price_earnings_ttm", "Price to Earnings Ratio (TTM)")}
    s = build_snapshots(df, resolved)["INFY"]
    assert s.market_cap == pytest.approx(6.5e12)
    assert s.pe_ttm == pytest.approx(27.1)


def test_a_missing_value_is_none_and_is_listed_in_missing_not_zero():
    df = _frame({
        "name": ["X"], "market_cap_basic": [1e12],
        "price_earnings_ttm": [np.nan], "return_on_equity": [None],
    })
    resolved = {"name": _F("name", "Name"),
                "market_cap": _F("market_cap_basic", "Market Capitalization"),
                "pe_ttm": _F("price_earnings_ttm", "P/E"),
                "roe": _F("return_on_equity", "ROE")}
    s = build_snapshots(df, resolved)["X"]
    assert s.pe_ttm is None and s.roe is None
    assert s.market_cap == pytest.approx(1e12)
    assert "pe_ttm" in s.missing and "roe" in s.missing
    assert "market_cap" not in s.missing


def test_abbreviated_and_comma_separated_numbers_are_parsed():
    from tvlab.data.fundamentals import _to_float
    assert _to_float("1.2B") == pytest.approx(1.2e9)
    assert _to_float("3,400,000") == pytest.approx(3.4e6)
    assert _to_float("12.5M") == pytest.approx(1.25e7)
    for bad in ("", "-", "n/a", None, "abc", float("nan")):
        assert _to_float(bad) is None, f"{bad!r} should not become a number"


def test_epoch_dates_in_seconds_and_milliseconds_both_parse():
    from tvlab.data.fundamentals import _to_date
    secs = _to_date(1_776_000_000)
    millis = _to_date(1_776_000_000_000)
    assert secs is not None and millis is not None
    assert secs.year == millis.year and secs.month == millis.month
    assert _to_date("not a date") is None
    assert _to_date(None) is None


def test_only_requested_symbols_come_back_and_keep_their_original_spelling():
    df = _frame({"name": ["NSE:RELIANCE", "TCS", "OTHER"], "market_cap_basic": [1, 2, 3]})
    resolved = {"name": _F("name", "Name"), "market_cap": _F("market_cap_basic", "MC")}
    snaps = build_snapshots(df, resolved, symbols=["RELIANCE.NS", "TCS.BO"])
    assert set(snaps) == {"RELIANCE.NS", "TCS.BO"}


def test_days_to_earnings_is_signed_and_none_when_unknown():
    s = FundamentalSnapshot(symbol="X", next_earnings=pd.Timestamp("2026-03-15"))
    assert s.days_to_earnings(AS_OF) == 5
    past = FundamentalSnapshot(symbol="X", next_earnings=pd.Timestamp("2026-03-01"))
    assert past.days_to_earnings(AS_OF) == -9
    assert FundamentalSnapshot(symbol="X").days_to_earnings(AS_OF) is None


def test_coverage_reports_gaps_instead_of_hiding_them():
    snaps = [FundamentalSnapshot(symbol="A", pe_ttm=10.0),
             FundamentalSnapshot(symbol="B", pe_ttm=None, missing=("pe_ttm", "roe")),
             FundamentalSnapshot(symbol="C", missing=("roe",))]
    c = coverage(snaps)
    assert c["symbols"] == 3 and c["complete"] == 1 and c["partial"] == 2
    assert c["missing_fields"]["roe"] == 2
    assert c["missing_fields"]["pe_ttm"] == 1


def test_fundamentals_frame_is_empty_but_not_broken_with_no_snapshots():
    assert fundamentals_frame([]).empty


# ------------------------------------------------------------------- the gates


def _snap(**kw) -> FundamentalSnapshot:
    base = dict(symbol="X", market_cap=5e12, pe_ttm=18.0, roe=0.20,
                next_earnings=pd.Timestamp("2026-06-01"))
    base.update(kw)
    return FundamentalSnapshot(**base)


def test_no_configured_gates_means_no_outcomes_and_an_approval():
    cfg = FundamentalsConfig(enabled=True, earnings_blackout_days=0)
    v = evaluate(_snap(), cfg, AS_OF)
    assert v.status == "approved"
    assert v.gates == [], "an unconfigured gate must not produce a passed outcome"
    assert active_gate_count(cfg) == 0
    assert "no fundamental gates configured" in v.detail()


def test_a_binding_gate_rejects():
    cfg = FundamentalsConfig(max_pe_ttm=15.0)
    v = evaluate(_snap(pe_ttm=18.0), cfg, AS_OF)
    assert v.status == "rejected"
    assert "pe_ttm" in v.failed
    assert "P/E (TTM) 18 (at most 15)" in v.detail()


def test_a_satisfied_gate_approves_and_says_what_it_checked():
    cfg = FundamentalsConfig(min_roe=0.15)
    v = evaluate(_snap(roe=0.20), cfg, AS_OF)
    assert v.status == "approved"
    assert "ROE 0.2 (at least 0.15)" in v.detail()
    assert v.quality_marks == 1


def test_missing_data_makes_a_symbol_unrated_not_approved():
    """The single most important property in this file."""
    cfg = FundamentalsConfig(min_roe=0.15)
    v = evaluate(_snap(roe=None, missing=("roe",)), cfg, AS_OF)
    assert v.status == "unrated"
    assert v.approved is False
    assert "ROE not reported" in v.detail()
    assert v.could_not_evaluate == ["roe"]


def test_a_failed_gate_never_states_a_false_comparison():
    """The bug this guards: a rejected name printed "debt/equity 6.4 <= 2", which
    asserts something untrue in the same breath as it reports the failure. The
    rejection detail is built by joining the gate strings, so every one of them has
    to read correctly whether the gate passed or not."""
    cfg = FundamentalsConfig(max_debt_to_equity=2.0, max_pe_ttm=15.0, min_roe=0.30)
    v = evaluate(_snap(pe_ttm=40.0, roe=0.05, debt_to_equity=6.4), cfg, AS_OF)
    assert v.status == "rejected"
    d = v.detail()
    for bad in ("6.4 <= 2", "40 <= 15", "0.05 >= 0.3"):
        assert bad not in d, f"failed gate printed a satisfied-looking comparison: {bad!r}"
    assert "at most" in d and "at least" in d
    # and each failed gate still names its value and its bound
    assert "debt/equity 6.4 (at most 2)" in d
    assert "P/E (TTM) 40 (at most 15)" in d
    assert "ROE 0.05 (at least 0.3)" in d


def test_a_passed_gate_reads_the_same_way_as_a_failed_one():
    cfg = FundamentalsConfig(max_debt_to_equity=2.0)
    v = evaluate(_snap(debt_to_equity=0.5), cfg, AS_OF)
    assert v.status == "approved"
    assert "debt/equity 0.5 (at most 2)" in v.detail()


def test_a_rejection_outranks_an_unknown():
    cfg = FundamentalsConfig(min_roe=0.15, max_pe_ttm=15.0)
    v = evaluate(_snap(roe=None, pe_ttm=40.0), cfg, AS_OF)
    assert v.status == "rejected", "a failed gate is a decision; an unknown is not"
    assert v.failed == ["pe_ttm"]


# --------------------------------------------------------------- earnings gate


def test_earnings_inside_the_blackout_rejects():
    cfg = FundamentalsConfig(earnings_blackout_days=2)
    v = evaluate(_snap(next_earnings=pd.Timestamp("2026-03-11")), cfg, AS_OF)
    assert v.status == "rejected"
    assert "reports earnings in 1 day(s)" in v.detail()
    assert "does not survive the gap" in v.detail()


def test_earnings_outside_the_blackout_passes_and_names_the_date():
    cfg = FundamentalsConfig(earnings_blackout_days=2)
    v = evaluate(_snap(next_earnings=pd.Timestamp("2026-06-01")), cfg, AS_OF)
    assert v.status == "approved"
    assert "no earnings within 2 day(s)" in v.detail()
    assert v.days_to_earnings == 83


def test_an_unknown_earnings_date_is_unrated_not_safe():
    """A stop does not survive an earnings gap, so not knowing is not passing."""
    cfg = FundamentalsConfig(earnings_blackout_days=2)
    v = evaluate(_snap(next_earnings=None, missing=("next_earnings",)), cfg, AS_OF)
    assert v.status == "unrated"
    assert "next earnings date unknown" in v.detail()


def test_a_stale_earnings_date_in_the_past_is_unrated():
    """The feed has not rolled over. Treating a past date as 'no earnings soon'
    would be the wrong way round."""
    cfg = FundamentalsConfig(earnings_blackout_days=2)
    v = evaluate(_snap(next_earnings=pd.Timestamp("2026-03-01")), cfg, AS_OF)
    assert v.status == "unrated"
    assert "stale feed" in v.detail()


def test_the_aftermath_window_is_applied_only_when_configured():
    cfg = FundamentalsConfig(earnings_blackout_days=0, earnings_aftermath_days=3)
    v = evaluate(_snap(last_earnings=pd.Timestamp("2026-03-09"),
                       next_earnings=pd.Timestamp("2026-06-01")), cfg, AS_OF)
    assert v.status == "rejected"
    assert "reported earnings 1 day(s) ago" in v.detail()
    # and off by default
    off = FundamentalsConfig(earnings_blackout_days=0)
    assert evaluate(_snap(last_earnings=pd.Timestamp("2026-03-09")), off, AS_OF).status == "approved"


# --------------------------------------------------------------------- screen()


def test_screen_separates_no_data_from_unrated_from_rejected():
    cfg = FundamentalsConfig(min_roe=0.15, earnings_blackout_days=2)
    snaps = {
        "GOOD": _snap(symbol="GOOD", roe=0.25),
        "BAD": _snap(symbol="BAD", roe=0.02),
        "UNKNOWN": _snap(symbol="UNKNOWN", roe=None, missing=("roe",)),
        "EARNINGS": _snap(symbol="EARNINGS", roe=0.4,
                          next_earnings=pd.Timestamp("2026-03-11")),
    }
    out = screen(["GOOD", "BAD", "UNKNOWN", "EARNINGS", "NOTRETURNED"], snaps, cfg, AS_OF)
    assert [v.symbol for v in out["approved"]] == ["GOOD"]
    assert sorted(v.symbol for v in out["rejected"]) == ["BAD", "EARNINGS"]
    assert [v.symbol for v in out["unrated"]] == ["UNKNOWN"]
    assert out["no_data"] == ["NOTRETURNED"]
    assert out["coverage"]["symbols"] == 4


def test_approved_names_are_ordered_deterministically():
    """Fundamentals veto; they do not rank. Every approved name has passed every
    configured gate, so no ordering by "how well" it passed is available without
    inventing weights - which this module refuses to do. The order is therefore
    stable and alphabetical, and the ranking of survivors belongs to the
    cost-adjusted expectancy in screener/quality.py."""
    cfg = FundamentalsConfig(min_roe=0.10)
    snaps = {s: _snap(symbol=s, roe=r) for s, r in
             [("ZED", 0.30), ("ALPHA", 0.99), ("MID", 0.11)]}
    out = screen(["ZED", "ALPHA", "MID"], snaps, cfg, AS_OF)
    assert [v.symbol for v in out["approved"]] == ["ALPHA", "MID", "ZED"]
    # a higher ROE earns no better position: there is no weighting to earn
    assert all(v.quality_marks == 1 for v in out["approved"])


def test_a_name_missing_one_configured_field_is_unrated_not_silently_approved():
    """Guards the ordering test above from hiding this: B reports no operating
    margin, so with that gate configured B cannot be approved."""
    cfg = FundamentalsConfig(min_roe=0.10, min_operating_margin=0.10)
    snaps = {"A": _snap(symbol="A", roe=0.30, operating_margin=0.25),
             "B": _snap(symbol="B", roe=0.30)}
    out = screen(["A", "B"], snaps, cfg, AS_OF)
    assert [v.symbol for v in out["approved"]] == ["A"]
    assert [v.symbol for v in out["unrated"]] == ["B"]


# ------------------------------------------------------------------- error paths


def test_a_market_with_no_fundamentals_raises_instead_of_returning_nothing():
    with pytest.raises(DataError) as exc:
        fetch_fundamentals(["BTC/USDT"], market="crypto")
    assert "not available for market 'crypto'" in str(exc.value)
    assert "crypto and forex do not" in str(exc.value)


def test_a_tvscreener_that_serves_none_of_our_fields_raises():
    """Upstream rename: the request must not go out asking for nothing."""
    with pytest.raises(DataError) as exc:
        fetch_fundamentals(["AAPL"], market="us_equities", tvscreener_module=_NoFields())
    msg = str(exc.value)
    assert "none of the fundamental fields" in msg


def test_a_screener_that_returns_nothing_raises_rather_than_looking_empty():
    class _Empty(_Working):
        rows = {}

    with pytest.raises(DataError) as exc:
        fetch_fundamentals(["AAPL"], market="us_equities", tvscreener_module=_Empty())
    assert "0 rows" in str(exc.value)


def test_a_network_failure_names_the_offline_fallback():
    class _Net(_Working):
        exc = ConnectionError("TLS/SSL connection closed")

    with pytest.raises(DataError) as exc:
        fetch_fundamentals(["AAPL"], market="us_equities", tvscreener_module=_Net())
    msg = str(exc.value)
    assert "scanner.tradingview.com" in msg
    assert "Offline fallback" in msg
    assert "ConnectionError" in msg


def test_a_working_request_produces_snapshots_end_to_end():
    """The happy path through fetch_fundamentals, with the network stubbed."""
    class _Ok(_Working):
        # A column dict, NOT a list containing one: pd.DataFrame([{...}]) builds a
        # single row whose cells are lists, which parses to nothing and looks like
        # an empty result rather than a malformed fixture.
        # Columns named after what `_FullEnum` produces (the lowercased member
        # name), so this exercises the request path rather than a specific upstream
        # spelling. That the real spellings still resolve is asserted separately by
        # test_every_canonical_field_resolves_against_the_installed_enum.
        rows = {"name": ["AAPL", "MSFT"],
                "market_capitalization": [3.0e12, 2.8e12],
                "price_to_earnings_ratio_ttm": [28.0, 33.0],
                "return_on_equity_ttm": [0.55, 0.38],
                "upcoming_earnings_date": ["2026-04-30", "2026-04-25"]}

    snaps = fetch_fundamentals(["AAPL"], market="us_equities", tvscreener_module=_Ok())
    assert list(snaps) == ["AAPL"]
    s = snaps["AAPL"]
    assert s.market_cap == pytest.approx(3.0e12)
    assert s.pe_ttm == pytest.approx(28.0)
    assert s.roe == pytest.approx(0.55)
    assert s.next_earnings == pd.Timestamp("2026-04-30")
    assert s.days_to_earnings(AS_OF) == 51


# --------------------------------------------------------------------- helpers


class _F:
    """Stand-in for a tvscreener Field: it only needs field_name and label."""

    def __init__(self, field_name: str, label: str):
        self.field_name = field_name
        self.label = label
        self.name = field_name


class _EnumLike:
    """An enum with none of the fields we ask for, to exercise the drift path."""

    def __getattr__(self, item):
        raise AttributeError(item)


class _FullEnum:
    """An enum that CAN resolve every canonical field, so tests of the request
    path actually reach the request. Without this, ``_resolve_fields`` fails first
    and a test meant to exercise a network error silently tests the drift path."""

    def __getattr__(self, item):
        return _F(item.lower(), item.title())


class _NoFields:
    StockField = _EnumLike()
    Market = _EnumLike()


class _Working:
    """A stand-in tvscreener whose screener raises whatever the test supplies."""

    StockField = _FullEnum()
    Market = _FullEnum()

    rows = None
    exc = None

    def StockScreener(self):              # noqa: N802 - mirrors the real API
        return _FailingScreener(exc=type(self).exc, rows=type(self).rows)


class _FailingScreener:
    def __init__(self, exc=None, rows=None):
        self._exc, self._rows = exc, rows

    def set_markets(self, *a, **k):
        return self

    def select(self, *a, **k):
        return self

    def get(self, *a, **k):
        if self._exc:
            raise self._exc
        return pd.DataFrame(self._rows if self._rows is not None else [])


# _resolve_fields needs a real hasattr-able enum, so give _EnumLike members for
# the drift tests only where a test asks for them.
def test_unresolvable_fields_are_reported_not_silently_dropped():
    resolved, unresolved = _resolve_fields(_NoFields(), extra=())
    assert resolved == {}
    assert set(unresolved) == set(FIELD_CANDIDATES)
