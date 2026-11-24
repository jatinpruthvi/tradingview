"""The Daily panel's JS and ``/api/daily`` have no shared types either.

Same failure mode as ``test_dashboard_suggest.py``: a renamed or dropped key surfaces
as ``undefined`` in a browser, silently, and only on the branch that has a shortlist.
The bug that motivated that file - a ``@property`` missing from a ``to_dict`` built
off ``self.__dict__`` - is exactly the shape of bug this panel can have, because
``DailyBriefing.advisory_line`` and ``is_validated`` are properties too.

Two properties get their own tests because they are the reason the panel exists:

* ``advisory`` and ``is_validated`` must survive serialisation. If they did not, the
  panel would render a ranked shortlist with no statement about whether the strategy
  behind it had ever been validated - the precise framing error this dashboard is
  built to avoid.
* The JS must render the advisory *before* the table. Order is the substance here,
  not cosmetics: a footnote about validity under a ranked list arrives after the
  reader has already formed an impression.
"""
import json
import re

import pandas as pd
import pytest

from tvlab.config import load_config
from tvlab.data.fundamentals import FundamentalSnapshot
from tvlab.screener.daily import build_briefing
from tvlab.screener.scanner import ScanRow

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "dashboard" / "static" / "index.html"


def _row(symbol, signal=1, last=100.0, stop=98.0, target=104.0, **kw):
    d = dict(last=last, signal=signal, stop=stop, target=target,
             reward_risk=abs(target - last) / abs(last - stop),
             trend=1, adx=25.0, atr_pct=2.0, bars=1200)
    d.update(kw)
    return ScanRow(symbol=symbol, **d)


def _snap(symbol, **kw):
    later = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=60)
    base = dict(symbol=symbol, market_cap=3.2e12, roe=0.42, pe_ttm=28.5,
                free_cash_flow=1e11, debt_to_equity=0.4, current_ratio=1.9,
                sector="Technology", next_earnings=later)
    base.update(kw)
    return FundamentalSnapshot(**base)


def _cfg():
    return load_config(market="us_equities", style="swing")


STUDY_FAIL = {"status": "ok", "verdict": "NOT VALIDATED", "checks_passed": 3,
              "checks_total": 8, "win_rate": 0.29, "n_trades": 31,
              "failed_checks": ["Deflated Sharpe"], "path": "output/x/report.json"}


@pytest.fixture(scope="module")
def daily_payload():
    """A payload from the real code, with a shortlist, a rejection and an outage note."""
    cfg = _cfg()
    snaps = {"WIDE": _snap("WIDE"), "SOON": _snap("SOON")}
    soon = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    snaps["SOON"] = _snap("SOON", next_earnings=soon)
    b = build_briefing([_row("WIDE"), _row("SOON"), _row("HAIR", stop=99.95, target=100.20)],
                       {"DEAD": "DataError: no bars"}, cfg, snaps=snaps,
                       study=STUDY_FAIL, as_of="2026-11-24")
    d = b.to_dict()
    d["fundamentals_error"] = None
    d["journal_path"] = "output/daily/journal.jsonl"
    d["recent_runs"] = []
    return d


@pytest.fixture(scope="module")
def outage_payload():
    """The state the sandbox actually produces: gates configured, endpoint down."""
    cfg = _cfg()
    b = build_briefing([_row("WIDE")], {}, cfg, snaps=None,
                       fundamentals_error="DataError: TLS/SSL connection has been closed",
                       study=None, as_of="2026-11-24")
    d = b.to_dict()
    d["fundamentals_error"] = "DataError: TLS/SSL connection has been closed"
    d["journal_path"] = "output/daily/journal.jsonl"
    d["recent_runs"] = []
    return d


@pytest.fixture(scope="module")
def script_body():
    html = INDEX.read_text()
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "no inline script in index.html"
    return m.group(1)


def _daily_fn(script_body):
    start = script_body.index("async function runDaily")
    end = script_body.index("async function runSuggest")
    assert start < end
    return script_body[start:end]


# --------------------------------------------------------------- serialisation


def test_the_advisory_and_verdict_survive_serialisation(daily_payload):
    """Named directly: these are properties, and a to_dict built from __dict__
    would drop both, leaving the panel with a shortlist and no caveat."""
    d = json.loads(json.dumps(daily_payload, default=str))
    assert d["advisory"]
    assert "NOT VALIDATED" in d["advisory"]
    assert d["is_validated"] is False
    assert isinstance(d["is_validated"], bool), "a stringified numpy bool is truthy"


def test_an_unvalidated_payload_never_serialises_as_validated(daily_payload, outage_payload):
    for p in (daily_payload, outage_payload):
        d = json.loads(json.dumps(p, default=str))
        assert d["is_validated"] is not True
        assert d["study_verdict"] != "PASSES ON THIS DATA"


def test_a_missing_study_serialises_with_a_null_verdict_not_a_pass():
    cfg = _cfg()
    b = build_briefing([_row("WIDE")], {}, cfg, study=None, as_of="2026-11-24")
    d = json.loads(json.dumps(b.to_dict(), default=str))
    assert d["study_verdict"] is None
    assert d["is_validated"] is False
    assert "has not been measured" in d["advisory"]


# ------------------------------------------------------------- field contracts


def test_every_shortlist_field_the_js_reads_is_in_the_payload(daily_payload, script_body):
    body = _daily_fn(script_body)
    table = body[body.index("r.shortlist.forEach"):]
    read = set(re.findall(r"\b(?:x|top)\.([a-z_]+)", table))
    provided = set(daily_payload["shortlist"][0].keys())
    missing = sorted(read - provided)
    assert not missing, f"runDaily reads shortlist fields the API never sends: {missing}"


def test_top_level_keys_the_js_reads_are_all_present(daily_payload, script_body):
    body = _daily_fn(script_body)
    read = set(re.findall(r"\br\.([a-z_]+)", body))
    missing = sorted(read - set(daily_payload.keys()))
    assert not missing, f"runDaily reads response keys the endpoint never sends: {missing}"


def test_rejected_entries_carry_what_the_js_reads(daily_payload, script_body):
    body = _daily_fn(script_body)
    read = set()
    # `.slice(0,N)` is optional here: pinning the exact call shape would make this
    # test break on a legitimate truncation change while nothing became unsafe.
    for name in ("rejected_fundamental", "rejected_technical"):
        pat = rf"for\(const x of \(r\.{name}\|\|\[\]\)(?:\.slice\([^)]*\))?\)(.*?)(?=\n\s*(?:\$\(|for\(|let |const |return|\}}))"
        found = re.findall(pat, body, re.S)
        assert found, f"the {name} loop was not found; the regex needs updating"
        for loop in found:
            read |= set(re.findall(r"\bx\.([a-z_]+)", loop))
    fund = set(daily_payload["rejected_fundamental"][0].keys())
    tech = set(daily_payload["rejected_technical"][0].keys())
    assert not (read - (fund | tech)), \
        f"rejection rows read fields not sent: {sorted(read - (fund | tech))}"


# ------------------------------------------------------------------- rendering


def test_the_advisory_is_rendered_before_the_shortlist_table(script_body):
    """Order is the substance. A validity caveat printed under a ranked table
    arrives after the reader has already decided."""
    body = _daily_fn(script_body)
    assert body.index("r.advisory") < body.index("r.shortlist.forEach"), \
        "the advisory must be emitted before the shortlist table"
    assert body.index("r.advisory") < body.index("<table>")


def test_an_unvalidated_verdict_is_coloured_as_a_warning_not_a_success(script_body):
    body = _daily_fn(script_body)
    assert "r.is_validated ? '#30a46c'" in body, \
        "only a validated study may render green"
    assert "#e5484d" in body, "a known failure verdict must render red"


def test_a_fundamental_outage_is_rendered_loudly(script_body):
    """Silence would read as 'screened and clean'."""
    body = _daily_fn(script_body)
    assert "r.fundamentals_error" in body
    assert "NOT" in body and "applied" in body
    # and the outage branch states what was not screened
    assert "not screened for valuation" in body


def test_data_failure_is_distinguished_from_a_quiet_market(script_body):
    body = _daily_fn(script_body)
    assert "r.outcome === 'data_failure'" in body
    assert "not a" in body and "market verdict" in body
    # the data_failure branch must return before any table is drawn
    df_branch = body[body.index("r.outcome === 'data_failure'"):]
    assert df_branch.index("return;") < df_branch.index("r.shortlist.forEach")


def test_the_panel_and_button_exist_and_are_wired():
    html = INDEX.read_text()
    assert 'id="btnDaily"' in html
    assert 'id="dailyTable"' in html
    assert "$('btnDaily').addEventListener('click', runDaily)" in html


def test_days_to_earnings_is_highlighted_inside_a_fortnight(script_body):
    """Outside the blackout but inside a swing hold is still a gap risk, and the row
    should say so rather than show a bare number."""
    body = _daily_fn(script_body)
    assert "x.days_to_earnings" in body
    assert "days_to_earnings<=14" in body.replace(" ", "")


def test_fmtCap_exists_and_is_used_for_market_cap(script_body):
    assert "function fmtCap(" in script_body
    assert "fmtCap(x.market_cap)" in _daily_fn(script_body)
