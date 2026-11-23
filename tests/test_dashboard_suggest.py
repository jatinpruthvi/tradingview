"""The dashboard's JS and the Python API are two languages with no shared types.

Nothing at import time links ``Candidate.to_dict()`` to the fields
``runSuggest`` reads out of the response, so a renamed or dropped key surfaces as
``undefined`` in a browser - silently, and only on the branch that has candidates.
These tests parse both sides and compare them.

They exist because of a real bug: ``verdict`` is a ``@property``, and ``to_dict``
built itself from ``self.__dict__``, which holds only dataclass *fields*. So every
serialised candidate lacked ``verdict`` while the JS called
``x.verdict.replace(...)``, which throws on ``undefined`` and blanks the panel.
"""
import json
import re
import subprocess
import shutil
from pathlib import Path

import pytest

from tvlab.screener.quality import suggest
from tvlab.screener.scanner import ScanRow

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "dashboard" / "static" / "index.html"

CRYPTO_FEE, CRYPTO_SLIP = 10.0, 5.0


def _row(symbol, last=100.0, stop=98.0, target=104.0, **kw):
    d = dict(signal=1, last=last, stop=stop, target=target,
             reward_risk=abs(target - last) / abs(last - stop),
             trend=1, adx=25.0, atr_pct=2.0, zones="-", bars=1200)
    d.update(kw)
    return ScanRow(symbol=symbol, **d)


@pytest.fixture(scope="module")
def suggest_payload():
    """A payload built by the real code, with candidates AND a rejection."""
    res = suggest([_row("WIDE"), _row("HAIR", stop=99.95, target=100.20)],
                  CRYPTO_FEE, CRYPTO_SLIP)
    return {
        "candidates": [c.to_dict() for c in res["candidates"]],
        "rejected": res["rejected"],
        "unscoreable": res["unscoreable"],
        "n_candidates": res["n_candidates"],
        "n_rejected": res["n_rejected"],
        "round_trip_cost_bps": res["round_trip_cost_bps"],
        "max_breakeven": res["max_breakeven"],
        "weights": res["weights"],
        "measured": {"win_rate": 0.55, "n_trades": 400, "profit_factor": 1.4,
                     "path": "output/x/report.json"},
        "strategy": "smc_trend", "timeframe": "4h", "market": "crypto",
        "scanned": 2, "errors": [],
        "note": "A ranked shortlist is not a recommendation.",
    }


@pytest.fixture(scope="module")
def script_body():
    html = INDEX.read_text()
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "no inline script in index.html"
    return m.group(1)


def _suggest_fn(script_body):
    start = script_body.index("async function runSuggest")
    end = script_body.index("async function runScan")
    assert start < end
    return script_body[start:end]


# --------------------------------------------------------------- contract tests


def test_every_candidate_field_the_js_reads_is_in_the_payload(suggest_payload, script_body):
    """The regression test for the missing `verdict`. Parse what the renderer reads
    off a candidate and require the API to actually send it."""
    body = _suggest_fn(script_body)
    # fields read as `x.<name>` or `top.<name>` inside the candidate table + detail
    read = set(re.findall(r"\b(?:x|top)\.([a-z_]+)", body))
    # the rejected loop reuses `x`, so restrict to the candidate branch by removing
    # the rejected-loop statements before collecting
    for loop in re.findall(r"for\(const x of \(r\.rejected\|\|\[\]\)\)[^\n]*(?:\n[^\n]*)?", body):
        read -= set(re.findall(r"\bx\.([a-z_]+)", loop))
    read |= {"verdict", "symbol"}          # read via template literals / detail block

    provided = set(suggest_payload["candidates"][0].keys())
    missing = sorted(read - provided)
    assert not missing, f"runSuggest reads candidate fields the API never sends: {missing}"


def test_rejected_entries_carry_what_the_js_reads(suggest_payload, script_body):
    body = _suggest_fn(script_body)
    loops = re.findall(r"for\(const x of \(r\.rejected\|\|\[\]\)\)(.*?)(?=\n\s*(?:\$\(|for\(|let |const |return|\}))",
                       body, re.S)
    assert loops, "the rejected loop was not found; the regex needs updating"
    read = set()
    for loop in loops:
        read |= set(re.findall(r"\bx\.([a-z_]+)", loop))
    provided = set(suggest_payload["rejected"][0].keys())
    assert not (read - provided), f"rejected rows read fields not sent: {sorted(read - provided)}"


def test_top_level_keys_the_js_reads_are_all_present(suggest_payload, script_body):
    body = _suggest_fn(script_body)
    read = set(re.findall(r"\br\.([a-z_]+)", body))
    assert not (read - set(suggest_payload.keys())), \
        f"runSuggest reads response keys the endpoint never sends: {sorted(read - set(suggest_payload.keys()))}"


def test_verdict_survives_serialisation():
    """Named directly because it is the bug that motivated this file."""
    c = suggest([_row("WIDE")], CRYPTO_FEE, CRYPTO_SLIP)["candidates"][0]
    d = json.loads(json.dumps(c.to_dict(), default=str))
    assert "verdict" in d and d["verdict"] == c.verdict


def test_measured_none_is_a_shape_the_js_handles(suggest_payload, script_body):
    """The 'no previous study' branch reads no fields off `measured`, so a null
    must not reach an `m.win_rate` dereference."""
    body = _suggest_fn(script_body)
    assert "if(m && m.win_rate!=null)" in body or "m && m.win_rate" in body, \
        "measured is dereferenced without a null guard"


# ---------------------------------------------------------------- render harness

node = shutil.which("node")
requires_node = pytest.mark.skipif(node is None, reason="node is not installed")


@requires_node
@pytest.mark.parametrize("case", ["populated", "empty", "no_study",
                                  "data_failure", "partial_failure"])
def test_runSuggest_renders_without_throwing(case, suggest_payload, script_body, tmp_path):
    """Actually execute the renderer. The populated branch cannot be reached with
    synthetic data - no strategy reliably fires on the final bar - so it would
    otherwise ship unexercised."""
    payload = dict(suggest_payload)
    if case == "empty":
        payload["candidates"] = []
        payload["n_candidates"] = 0
    elif case == "no_study":
        payload["measured"] = None
    elif case == "data_failure":
        # every symbol failed to load: nothing was observed, so no market verdict
        payload.update(candidates=[], n_candidates=0, rejected=[], scanned=0,
                       errors=[{"symbol": f"S{i}.NS", "error": "DataError: unreachable"}
                               for i in range(9)])
    elif case == "partial_failure":
        payload.update(candidates=[], n_candidates=0, scanned=4,
                       errors=[{"symbol": "DEAD.NS", "error": "DataError: unreachable"}])

    js = script_body
    assert js.count("\nboot();") == 1, "boot() invocation moved; update the strip"
    js = js.replace("\nboot();", "\n")   # boot() needs a real /api/config

    head = '''
const els = {};
function mk(id){ const e = { _id:id, _html:'', style:{}, value:'4h', textContent:'',
  set innerHTML(v){ e._html = String(v) }, get innerHTML(){ return e._html },
  insertAdjacentHTML(pos,h){ e._html += String(h) }, addEventListener(){} }; return e; }
global.document = { getElementById:(id)=> (els[id] ||= mk(id)), body:mk('body') };
global.window = {};
global.fetch = async ()=>({ ok:true, json:async()=> (__PAYLOAD__) });
'''
    tail = '''
state.cfg = {risk:{risk_per_trade:0.005, max_position_pct:0.25}};
(async()=>{ await runSuggest(); console.log(els['suggestTable']._html); })();
'''
    # Build the full string, THEN substitute: `A + js + B.replace(...)` binds the
    # call to B only, which silently leaves the token in A unreplaced.
    harness = (head + js + tail).replace("__PAYLOAD__", json.dumps(payload))
    assert "__PAYLOAD__" not in harness

    script = tmp_path / "harness.js"
    script.write_text(harness)
    r = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"node crashed: {r.stderr[:800]}"

    out = r.stdout
    flat = re.sub(r"\s+", " ", out)     # what a browser would render as text
    assert not out.strip().startswith('<div class="err">'), f"hit the error path: {out[:200]}"
    for junk in ("undefined", "NaN", "__PAYLOAD__"):
        assert junk not in out, f"{junk!r} leaked into the rendered HTML"

    if case == "populated":
        for probe in ("WIDE", "38.3%", "33.3%", "PASS", "Top candidate", "25.0%",
                      "HAIR", "not a recommendation", "plausible only with a real edge"):
            assert probe in out, f"{probe!r} missing from the populated render"
    elif case == "empty":
        assert "Nothing qualifies" in out
    elif case == "no_study":
        assert "no previous study" in out
    elif case == "data_failure":
        # The point of the branch: a total data failure must not present itself as
        # a quiet market. Assert the reassuring wording is ABSENT, not just that the
        # new wording is present - the old text could otherwise be printed too.
        assert "data failure, not a market verdict" in out
        assert "Nothing qualifies" not in out
        assert "normal state" not in out
        assert "All 9 symbol(s) failed to load" in flat
        assert "and 4 more" in flat, "should truncate rather than dump every error"
        assert "tvlab data --market" in out, "must give a concrete next step"
        assert 'class="err"' in out, "should use the error styling, not a muted note"
    elif case == "partial_failure":
        # A partial view must not present itself as a complete one, but it is still
        # a legitimate observation, so the verdict stays.
        assert "Nothing qualifies" in out
        assert "data failure" not in out
        assert "based on 4 of 5 symbols" in flat, "must state the coverage it had"


@requires_node
def test_index_html_is_syntactically_valid_javascript(script_body, tmp_path):
    script = tmp_path / "page.js"
    script.write_text(script_body)
    r = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:800]
