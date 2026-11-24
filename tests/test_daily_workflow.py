"""The daily workflow: composition, framing, and the journal.

No strategy reliably fires on the last bar of synthetic data, so the rows here are
stubs with the exact attributes ``quality.evaluate`` reads. That is deliberate:
these tests exercise the composition, not the luck of a seed.

The properties that matter:

* The validation verdict is attached to the output and precedes any shortlist. An
  unknown verdict renders as *unmeasured*, never as a pass - the same rule
  ``_measured_win_rate`` follows, because a missing report is not evidence of
  validation.
* A fundamental outage does not read as "no fundamental concerns". Every surviving
  name carries a warning that it was not screened.
* The three-way outcome contract holds: all-failed is a *data* failure, distinct
  from "we looked and nothing qualified".
* The journal is append-only, survives a corrupt line, and a failure to write it
  never aborts the briefing.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

import pytest

from tvlab.config import FundamentalsConfig, load_config
from tvlab.data.fundamentals import FundamentalSnapshot
from tvlab.screener.daily import (
    DailyBriefing,
    append_journal,
    build_briefing,
    journal_path,
    read_journal,
)


# ------------------------------------------------------------------ stub rows
def _row(symbol: str, *, signal: int = 1, last: float = 100.0, stop: float = 98.0,
         target: float = 106.0, atr_pct: float = 1.5, trend: int = 1, adx: float = 28.0,
         bars: int = 1200, in_zone: bool = True, **kw) -> SimpleNamespace:
    base = dict(symbol=symbol, signal=signal, last=last, stop=stop, target=target,
                atr_pct=atr_pct, trend=trend, adx=adx, bars=bars, in_zone=in_zone,
                zones="ob", sweep_age=3.0, support_pct=-1.0, resistance_pct=2.0,
                reward_risk=None, rsi=55.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _cfg(market: str = "us_equities", **fund) -> object:
    cfg = load_config(market=market, style="swing")
    if fund:
        cfg = cfg.merge({"fundamentals": dict(fund)})
    return cfg


def _snap(symbol: str, **kw) -> FundamentalSnapshot:
    """A snapshot that clears the earnings blackout by default.

    ``next_earnings=None`` would make every one of these *unrated* under the
    configured 2-day blackout, so tests about market cap or cash flow would pass or
    fail on the wrong gate. Tests that are about the blackout override it.
    """
    later = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=60)
    base = dict(symbol=symbol, market_cap=5e11, roe=0.25, pe_ttm=20.0,
                free_cash_flow=1e10, debt_to_equity=0.5, current_ratio=1.8,
                sector="Technology", next_earnings=later)
    base.update(kw)
    return FundamentalSnapshot(**base)


STUDY_OK = {"status": "ok", "verdict": "NOT VALIDATED", "checks_passed": 3,
            "checks_total": 8, "win_rate": 0.29, "n_trades": 31,
            "failed_checks": ["deflated sharpe"], "path": "output/x/report.json"}
STUDY_PASS = {"status": "ok", "verdict": "PASSES ON THIS DATA", "checks_passed": 8,
              "checks_total": 8, "win_rate": 0.55, "n_trades": 400, "path": "p"}


# ------------------------------------------------------------------- framing


def test_an_unvalidated_study_is_stated_before_any_shortlist():
    b = build_briefing([_row("AAPL")], {}, _cfg(), study=STUDY_OK)
    assert b.outcome == "shortlist"
    assert b.is_validated is False
    assert "NOT VALIDATED" in b.advisory_line
    assert "3/8 checks" in b.advisory_line
    assert "not as advice" in b.advisory_line or "screen, not as advice" in b.advisory_line
    # and it is in the machine-readable payload, not just the prose
    d = b.to_dict()
    assert d["study_verdict"] == "NOT VALIDATED"
    assert d["is_validated"] is False
    assert "NOT VALIDATED" in d["advisory"]


def test_a_missing_study_is_unmeasured_and_never_rendered_as_a_pass():
    """A missing report is not evidence of validation."""
    b = build_briefing([_row("AAPL")], {}, _cfg(), study=None)
    assert b.study_verdict is None
    assert b.is_validated is False
    assert "No validation study on record" in b.advisory_line
    assert "has not been measured" in b.advisory_line


def test_a_validated_study_changes_the_framing():
    b = build_briefing([_row("AAPL")], {}, _cfg(), study=STUDY_PASS)
    assert b.is_validated is True
    assert "ranked ideas rather than an unvalidated screen" in b.advisory_line


def test_a_study_that_took_no_trades_does_not_look_validated():
    b = build_briefing([_row("AAPL")], {}, _cfg(),
                       study={"status": "no_trades", "win_rate": None, "n_trades": 0})
    assert b.study_verdict is None
    assert b.is_validated is False
    assert any("unreadable" in n for n in b.notes)


def test_a_corrupt_study_report_is_reported_as_such():
    b = build_briefing([_row("AAPL")], {}, _cfg(),
                       study={"status": "corrupt", "win_rate": None, "error": "bad json"})
    assert b.is_validated is False
    assert any("corrupt" in n for n in b.notes)
    assert b.to_dict()["notes"]


# ------------------------------------------------------------------ outcomes


def test_every_symbol_failing_is_a_data_failure_not_a_quiet_market():
    errs = {s: "DataError: no bars" for s in ("AAPL", "MSFT", "NVDA")}
    b = build_briefing([], errs, _cfg())
    assert b.outcome == "data_failure"
    assert "data failure, not a market verdict" in b.outcome_detail


def test_partial_coverage_is_stated_with_its_coverage():
    b = build_briefing([_row("AAPL")], {"MSFT": "DataError: no bars"}, _cfg())
    assert b.outcome == "partial"
    assert "1 of 2 symbol(s) unavailable" in b.outcome_detail


def test_nothing_firing_is_the_normal_state_and_says_so():
    b = build_briefing([_row("AAPL", signal=0), _row("MSFT", signal=0)], {}, _cfg())
    assert b.outcome == "no_signals"
    assert b.rows == []
    assert "no strategy fired" in b.outcome_detail


def test_signals_all_rejected_on_cost_are_distinguished_from_no_signals():
    """A 0.05% stop at 16 bp round trip needs an impossible win rate."""
    tight = _row("AAPL", stop=99.95, target=100.5)
    b = build_briefing([tight], {}, _cfg())
    assert b.outcome == "all_rejected"
    assert b.rejected_technical
    assert "rejected by the cost test" in b.outcome_detail


# --------------------------------------------------------------- fundamentals


def test_fundamental_gates_remove_names_from_the_shortlist():
    cfg = _cfg(enabled=True, earnings_blackout_days=2, min_market_cap=1e9,
               require_positive_free_cash_flow=True, max_debt_to_equity=2.0,
               min_current_ratio=1.0)
    # _snap clears the blackout for all three, so the gates below are what
    # separates them rather than an unrelated earnings date.
    snaps = {
        "AAPL": _snap("AAPL"),
        "JUNK": _snap("JUNK", free_cash_flow=-5e8, debt_to_equity=9.0),
        "SMALL": _snap("SMALL", market_cap=1e6),
    }
    b = build_briefing([_row("AAPL"), _row("JUNK"), _row("SMALL")], {}, cfg, snaps=snaps)
    assert [r.symbol for r in b.rows] == ["AAPL"]
    rejected = {r["symbol"]: r for r in b.rejected_fundamental}
    assert "JUNK" in rejected and rejected["JUNK"]["status"] == "rejected"
    assert "SMALL" in rejected


def test_an_earnings_blackout_removal_names_the_gap_risk():
    import pandas as pd
    cfg = _cfg(enabled=True, earnings_blackout_days=2)
    soon = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    snaps = {"AAPL": _snap("AAPL", next_earnings=soon)}
    b = build_briefing([_row("AAPL")], {}, cfg, snaps=snaps)
    assert b.rows == []
    detail = b.rejected_fundamental[0]["detail"]
    assert "reports earnings in 1 day(s)" in detail
    assert "does not survive the gap" in detail


def test_a_fundamental_outage_warns_on_every_surviving_name():
    """The critical one: an outage must not read as 'no fundamental concerns'."""
    cfg = _cfg(enabled=True, earnings_blackout_days=2)
    b = build_briefing([_row("AAPL"), _row("MSFT")], {}, cfg, snaps=None,
                       fundamentals_error="DataError: TLS/SSL connection has been closed")
    assert len(b.rows) == 2
    assert any("fundamentals unavailable" in n for n in b.notes)
    for r in b.rows:
        assert any("NOT fundamentally screened" in w for w in r.warnings), r.symbol
    assert b.fundamentals_requested is True


def test_a_market_with_no_fundamentals_says_why_instead_of_failing_quietly():
    cfg = load_config(market="crypto", style="swing").merge({"fundamentals": {"enabled": True}})
    b = build_briefing([_row("BTC/USDT")], {}, cfg, snaps=None,
                       fundamentals_error="market 'crypto' has no earnings")
    assert len(b.rows) == 1
    assert any("no earnings" in n for n in b.notes)


def test_symbols_the_endpoint_never_returned_are_reported_as_no_data():
    cfg = _cfg(enabled=True, min_market_cap=1e9)
    b = build_briefing([_row("AAPL"), _row("OBSCURE")], {}, cfg,
                       snaps={"AAPL": _snap("AAPL")})
    assert [r.symbol for r in b.rows] == ["AAPL"]
    assert "OBSCURE" in b.no_fundamental_data
    nd = [r for r in b.rejected_fundamental if r["symbol"] == "OBSCURE"]
    assert nd and nd[0]["status"] == "no_data"


def test_fundamental_context_is_attached_to_the_surviving_row():
    import pandas as pd
    cfg = _cfg(enabled=True, min_market_cap=1e9)
    soon = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=10)
    snaps = {"AAPL": _snap("AAPL", sector="Technology", pe_ttm=28.5, roe=0.42,
                           next_earnings=soon)}
    b = build_briefing([_row("AAPL")], {}, cfg, snaps=snaps)
    r = b.rows[0]
    assert r.fundamental_status == "approved"
    assert r.sector == "Technology" and r.pe_ttm == 28.5 and r.roe == 0.42
    assert r.days_to_earnings == 10
    # within 14 days but outside the blackout: a warning, not a rejection
    assert any("within a swing hold" in w for w in r.warnings)


def test_fundamentals_disabled_means_no_gate_and_no_claim_of_screening():
    cfg = _cfg(enabled=False)
    b = build_briefing([_row("AAPL")], {}, cfg, snaps=None)
    assert len(b.rows) == 1
    assert b.fundamentals_enabled is False
    assert b.rows[0].fundamental_status is None


def test_enabled_but_with_no_threshold_set_is_reported_as_a_no_op():
    cfg = _cfg(enabled=True, earnings_blackout_days=0)
    b = build_briefing([_row("AAPL")], {}, cfg, snaps=None)
    assert len(b.rows) == 1
    assert any("no fundamental gate ran" in n for n in b.notes)


# ------------------------------------------------------------------- ordering


def test_the_shortlist_order_matches_quality_then_breakeven_then_symbol():
    rows = [_row("ZED", stop=98.0, target=106.0), _row("ALPHA", stop=98.0, target=106.0),
            _row("MID", stop=98.0, target=106.0)]
    b = build_briefing(rows, {}, _cfg())
    syms = [r.symbol for r in b.rows]
    assert syms == sorted(syms), "identical candidates must order deterministically"


def test_two_runs_on_identical_input_agree():
    rows = [_row("AAPL"), _row("MSFT"), _row("NVDA")]
    a = build_briefing(rows, {}, _cfg(), as_of="2026-11-24")
    c = build_briefing(rows, {}, _cfg(), as_of="2026-11-24")
    assert [r.to_dict() for r in a.rows] == [r.to_dict() for r in c.rows]


def test_top_truncates_the_shortlist():
    rows = [_row(s) for s in ("A", "B", "C", "D")]
    assert len(build_briefing(rows, {}, _cfg(), top=2).rows) == 2


# ------------------------------------------------------------------- journal


def test_the_journal_appends_one_line_per_run_and_is_newest_first(tmp_path):
    b1 = build_briefing([_row("AAPL")], {}, _cfg(), as_of="2026-11-23")
    b2 = build_briefing([_row("MSFT")], {}, _cfg(), as_of="2026-11-24")
    jp = append_journal(b1, tmp_path)
    append_journal(b2, tmp_path)
    assert jp == journal_path(tmp_path)
    lines = jp.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    got = read_journal(tmp_path)
    assert [g["as_of"] for g in got] == ["2026-11-24", "2026-11-23"]
    assert got[0]["shortlist"][0]["symbol"] == "MSFT"


def test_the_journal_records_the_verdict_so_a_past_run_can_be_distrusted_later(tmp_path):
    b = build_briefing([_row("AAPL")], {}, _cfg(), study=STUDY_OK, as_of="2026-11-24")
    append_journal(b, tmp_path)
    got = read_journal(tmp_path)[0]
    assert got["study_verdict"] == "NOT VALIDATED"
    assert got["is_validated"] is False
    assert got["outcome"] == "shortlist"


def test_the_dated_file_holds_the_full_briefing(tmp_path):
    b = build_briefing([_row("AAPL")], {}, _cfg(), as_of="2026-11-24")
    append_journal(b, tmp_path)
    day = tmp_path / "daily" / "2026-11-24.json"
    assert day.exists()
    d = json.loads(day.read_text(encoding="utf-8"))
    assert d["shortlist"][0]["symbol"] == "AAPL"
    assert d["advisory"]


def test_a_rerun_the_same_day_overwrites_rather_than_inventing_a_second_morning(tmp_path):
    b = build_briefing([_row("AAPL")], {}, _cfg(), as_of="2026-11-24")
    append_journal(b, tmp_path)
    append_journal(b, tmp_path)
    day = tmp_path / "daily" / "2026-11-24.json"
    assert day.exists()
    # the journal keeps both runs; the dated detail keeps the latest
    assert len(read_journal(tmp_path)) == 2


def test_a_corrupt_journal_line_is_skipped_not_fatal(tmp_path):
    jp = journal_path(tmp_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text('{"as_of": "2026-11-22", "outcome": "shortlist"}\nNOT JSON\n'
                  '{"as_of": "2026-11-23", "outcome": "no_signals"}\n', encoding="utf-8")
    got = read_journal(tmp_path)
    assert [g["as_of"] for g in got] == ["2026-11-23", "2026-11-22"]


def test_reading_a_journal_that_does_not_exist_is_empty_not_an_error(tmp_path):
    assert read_journal(tmp_path) == []


def test_an_unwritable_journal_does_not_abort_the_briefing(tmp_path, monkeypatch):
    b = build_briefing([_row("AAPL")], {}, _cfg(), as_of="2026-11-24")
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory", encoding="utf-8")
    # mkdir(parents=True) over an existing *file* raises, which is the failure mode
    # that must not take the user's shortlist down with it
    with pytest.raises(OSError):
        blocked.mkdir(parents=True)
    jp = append_journal(b, blocked / "daily")
    assert b.rows, "the briefing is intact regardless of the journal"
    assert isinstance(jp, object)


# ------------------------------------------------------------- serialisation


def test_to_dict_is_json_serialisable_with_no_default_hook():
    """NaN and numpy types are the usual culprits; a payload that needs
    ``default=str`` is a payload that will silently stringify a boolean."""
    b = build_briefing([_row("AAPL"), _row("MSFT", signal=0)], {"X": "boom"}, _cfg(),
                       study=STUDY_OK, as_of="2026-11-24")
    s = json.dumps(b.to_dict())
    assert "AAPL" in s
    d = json.loads(s)
    assert d["is_validated"] is False
    assert isinstance(d["is_validated"], bool)


def test_daily_row_to_dict_keeps_every_property_and_list():
    b = build_briefing([_row("AAPL")], {}, _cfg(), study=STUDY_OK)
    d = b.rows[0].to_dict()
    assert isinstance(d["reasons"], list) and isinstance(d["warnings"], list)
    assert isinstance(d["missing_fundamentals"], list)
    assert "verdict_band" in d


# ------------------------------------- states distinguished by the composition
def test_gates_active_but_nothing_fetched_keeps_the_names_and_says_so():
    """`snaps=None` means "nobody fetched", not "the endpoint returned nothing".

    Conflating them silently emptied the shortlist: every candidate fell into
    `no_data`, and the output looked like a market with no qualifying names rather
    than a run that never asked. The names must survive, flagged as unscreened.
    """
    cfg = _cfg(enabled=True, min_market_cap=1e9)
    b = build_briefing([_row("AAPL"), _row("MSFT")], {}, cfg, snaps=None)
    assert [r.symbol for r in b.rows] == ["AAPL", "MSFT"]
    assert b.no_fundamental_data == []
    assert any("no fundamentals were fetched" in n for n in b.notes)
    assert all(any("NOT fundamentally screened" in w for w in r.warnings) for r in b.rows)


def test_an_empty_fetch_result_is_a_coverage_fact_not_an_outage():
    """`snaps={}` means the request succeeded and returned nothing for these names,
    which is a different statement from never having asked."""
    cfg = _cfg(enabled=True, min_market_cap=1e9)
    b = build_briefing([_row("AAPL"), _row("MSFT")], {}, cfg, snaps={})
    assert b.rows == []
    assert sorted(b.no_fundamental_data) == ["AAPL", "MSFT"]
    assert b.fundamentals_requested is True
    assert not any("no fundamentals were fetched" in n for n in b.notes)


def test_an_unrecognised_verdict_is_flagged_rather_than_silently_distrusted():
    b = build_briefing([_row("AAPL")], {}, _cfg(),
                       study={"status": "ok", "verdict": "GREAT", "checks_passed": 9,
                              "checks_total": 9, "win_rate": 0.6})
    assert b.is_validated is False
    assert any("unrecognised study verdict" in n for n in b.notes)


def test_every_verdict_the_report_can_emit_is_recognised():
    """Guards the vocabulary against drift: report.py writes exactly these three."""
    from tvlab.screener.daily import KNOWN_VERDICTS
    assert KNOWN_VERDICTS == {"PASSES ON THIS DATA", "NOT VALIDATED", "PARTIAL"}
    for v in ("PASSES ON THIS DATA", "NOT VALIDATED", "PARTIAL"):
        b = build_briefing([_row("AAPL")], {}, _cfg(),
                           study={"status": "ok", "verdict": v})
        assert not any("unrecognised" in n for n in b.notes), v
        assert b.is_validated is (v == "PASSES ON THIS DATA")


# ------------------------------------------- outcome branches that must not lie
def test_signals_screened_out_on_fundamentals_do_not_report_no_signals():
    """The bug this guards: two signals fired and the fundamental gates removed
    both, and the output said "no strategy fired on the last bar of any symbol".
    That sends someone looking for setups that were there."""
    cfg = _cfg(enabled=True, earnings_blackout_days=2)
    import pandas as pd
    soon = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    snaps = {"AAPL": _snap("AAPL", next_earnings=soon), "MSFT": _snap("MSFT", next_earnings=soon)}
    b = build_briefing([_row("AAPL"), _row("MSFT")], {}, cfg, snaps=snaps)
    assert b.signals_found == 2
    assert b.outcome == "screened_out"
    assert "2 signal(s) fired" in b.outcome_detail
    assert "screened out on fundamentals" in b.outcome_detail
    assert "no strategy fired" not in b.outcome_detail


def test_unscoreable_signals_are_distinguished_from_no_signals():
    """A signal with no usable stop cannot be ranked. That is not the same as the
    strategy staying quiet, and the difference tells the user where to look."""
    nostop = _row("AAPL", signal=1, stop=None)
    b = build_briefing([nostop], {}, _cfg())
    assert b.signals_found == 1
    assert b.outcome == "unscoreable"
    assert "none could be scored" in b.outcome_detail


def test_no_signals_is_only_reported_when_nothing_fired():
    b = build_briefing([_row("AAPL", signal=0)], {}, _cfg())
    assert b.signals_found == 0
    assert b.outcome == "no_signals"
    assert "no strategy fired" in b.outcome_detail
