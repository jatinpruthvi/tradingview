"""The dashboard's JS and the Python API are two languages with no shared types.

Nothing at import time links ``ReviewSummary.to_dict()`` to the fields ``runReview``
reads out of the response, so a renamed or dropped key surfaces as ``undefined`` in a
browser - silently, and only on the branch that has scored rows. These tests parse
both sides and compare them.

They follow ``test_dashboard_suggest.py``, which exists because of a real bug: a
``@property`` was missing from a ``to_dict`` built off ``self.__dict__``, so the JS
called ``x.verdict.replace(...)`` on ``undefined`` and blanked the panel.
``ReviewSummary.coverage`` and ``ScoredSuggestion.{net_r,is_win,scored}`` are
properties of exactly that kind, so they are named directly below.

One improvement over the earlier pattern: the endpoint's payload keys are read from
``api_review``'s own source rather than from a hand-copied list in this file. A list
copied by hand drifts with the endpoint and then tests the copy, which is how a
contract test comes to assert nothing.
"""
import json
import re
from pathlib import Path

import pytest

from tvlab.screener.review import ReviewSummary, ScoredSuggestion, review

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "dashboard" / "static" / "index.html"
APP = REPO_ROOT / "src" / "tvlab" / "dashboard" / "app.py"


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def script_body():
    html = INDEX.read_text()
    m = re.search(r"<script>(.*)</script>", html, re.S)
    assert m, "no inline script in index.html"
    return m.group(1)


@pytest.fixture(scope="module")
def review_fn(script_body):
    """Sliced at *whatever function comes next*, not at a hardcoded neighbour.

    ``test_dashboard_daily.py`` pinned ``runSuggest`` as the end of ``runDaily`` and
    broke the first time a function was inserted between them: the slice absorbed the
    new function and its contract test then demanded fields the endpoint never sent.
    Function ordering in ``index.html`` is not part of any contract, so neither test
    may depend on it.
    """
    start = script_body.index("async function runReview")
    rest = script_body[start + len("async function runReview"):]
    m = re.search(r"\n(?:async\s+)?function\s+\w+\s*\(", rest)
    assert m, "runReview is the last function in the script; the boundary moved"
    return script_body[start:start + len("async function runReview") + m.start()]


@pytest.fixture(scope="module")
def endpoint_source():
    src = APP.read_text()
    start = src.index("def api_review(")
    end = src.index("# ------------------------------------------------------------------ report",
                    start)
    return src[start:end]


def _bars(rows, start="2026-01-05"):
    import pandas as pd
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame({"high": [r[0] for r in rows], "low": [r[1] for r in rows],
                         "close": [r[2] for r in rows], "open": [r[2] for r in rows],
                         "volume": [1.0] * len(rows)}, index=idx)


_BAR0 = (101.0, 99.5, 100.0)


@pytest.fixture(scope="module")
def summary_payload():
    """A payload built by the real scorer, covering every outcome class at once.

    Built from ``review()`` rather than a literal so the keys under test are the keys
    the code actually produces.
    """
    entries = [{"as_of": "2026-01-04", "timeframe": "4h", "shortlist": [
        {"symbol": "WIN", "side": "long", "last": 100.0, "stop": 98.0,
         "target": 106.0, "cost_in_r": 0.10, "breakeven_win_rate": 0.275},
        {"symbol": "LOSE", "side": "long", "last": 100.0, "stop": 98.0,
         "target": 106.0, "cost_in_r": 0.10, "breakeven_win_rate": 0.275},
        {"symbol": "TIME", "side": "short", "last": 100.0, "stop": 102.0,
         "target": 94.0, "cost_in_r": 0.10, "breakeven_win_rate": 0.275},
        {"symbol": "OPEN", "side": "long", "last": 100.0, "stop": 98.0,
         "target": 106.0, "cost_in_r": 0.10, "breakeven_win_rate": 0.275},
        {"symbol": "NODATA", "side": "long", "last": 100.0, "stop": 98.0,
         "target": 106.0, "cost_in_r": 0.10, "breakeven_win_rate": 0.275},
    ]}]

    def fetch(symbol, start, timeframe):
        if symbol == "NODATA":
            return None
        if symbol == "OPEN":
            return _bars([_BAR0])
        if symbol == "WIN":
            return _bars([_BAR0, (106.5, 100.0, 106.0)])
        if symbol == "LOSE":
            return _bars([_BAR0, (100.0, 97.5, 98.0)])
        # TIME: a short at 100 with a 102 stop and a 94 target. High 101 never reaches
        # the stop and low 99 never reaches the target, so with a full 90 bars of
        # horizon neither level is touched and it marks to market at 100.5 (-0.25R).
        # A shorter frame would come back `open` instead, which is a different class.
        return _bars([_BAR0] + [(101.0, 99.0, 100.5)] * 90)

    s = review(entries, fetch, horizon_bars=90, provenance="synthetic",
               as_of="2026-11-24")
    assert {r.outcome for r in s.rows} >= {"take_profit", "stop_loss", "time_exit",
                                          "open", "unscorable"}, \
        "the fixture no longer covers every outcome class"
    return s.to_dict()


@pytest.fixture(scope="module")
def endpoint_keys(endpoint_source, summary_payload):
    """Every key ``GET /api/review`` can send, derived from the endpoint's source."""
    keys = set(summary_payload)                       # from summary.to_dict()
    keys |= set(re.findall(r'payload\["([a-z_]+)"\]\s*=', endpoint_source))
    # the no-journal branch returns its own literal
    m = re.search(r"return _json_safe\(\{(.*?)\n            \}\)", endpoint_source, re.S)
    assert m, "the unavailable-branch dict literal was not found; regex needs updating"
    keys |= set(re.findall(r'"([a-z_]+)":', m.group(1)))
    return keys


# --------------------------------------------------------------- contract tests


def test_every_top_level_key_the_js_reads_is_sent(endpoint_keys, review_fn):
    """The regression test for a dropped or renamed summary field."""
    read = set(re.findall(r"\br\.([a-z_]+)", review_fn))
    missing = sorted(read - endpoint_keys)
    assert not missing, f"runReview reads response keys /api/review never sends: {missing}"


def test_every_row_field_the_js_reads_is_in_a_scored_row(summary_payload, review_fn):
    """Row fields are read as `x.<name>` off both the scored table and the
    unscorable/open listings."""
    read = set(re.findall(r"\bx\.([a-z_]+)", review_fn))
    provided = set(summary_payload["rows"][0].keys())
    missing = sorted(read - provided)
    assert not missing, f"runReview reads row fields to_dict() never sends: {missing}"


def test_the_properties_survive_serialisation():
    """Named directly: these are ``@property``, and a ``to_dict`` built from
    ``self.__dict__`` would drop every one of them while the JS reads them."""
    s_keys = set(ReviewSummary(horizon_bars=90).to_dict())
    for prop in ("coverage",):
        assert prop in s_keys, f"ReviewSummary.{prop} is missing from to_dict()"

    row = ScoredSuggestion(as_of="2026-01-04", symbol="X", side="long", entry=100.0,
                           stop=98.0, target=106.0, outcome="take_profit",
                           r_multiple=3.0, cost_in_r=0.10, exit_price=106.0)
    r_keys = set(row.to_dict())
    for prop in ("net_r", "is_win", "scored"):
        assert prop in r_keys, f"ScoredSuggestion.{prop} is missing from to_dict()"
    assert row.to_dict()["net_r"] == pytest.approx(2.9)
    assert row.to_dict()["is_win"] is True
    assert row.to_dict()["scored"] is True


def test_the_payload_is_json_serialisable(summary_payload):
    """JSON has no NaN literal and numpy bools are not `bool`; both have broken a
    dashboard payload before."""
    raw = json.dumps(summary_payload, default=str)
    assert "NaN" not in raw
    back = json.loads(raw)
    assert back["coverage"] == summary_payload["coverage"]
    for r in back["rows"]:
        assert isinstance(r["scored"], bool), f"{r['symbol']} scored is not a JSON bool"
        assert r["is_win"] is None or isinstance(r["is_win"], bool)


# ------------------------------------------------------- what the panel must say


def test_coverage_is_rendered_before_any_rate(review_fn):
    """The first honest fact about a review is how much of the journal could be
    scored. A win rate above its denominator reads as a verdict."""
    body = review_fn
    i_cov = body.index("r.coverage")
    i_win = body.index("r.win_rate")
    assert i_cov < i_win, "the win rate is rendered before the coverage line"


def test_the_synthetic_warning_is_rendered_before_any_number(review_fn):
    """On synthetic data every rate measures the generator. That has to be on screen
    before the reader meets a win rate, not in a footnote after it."""
    i_gen = review_fn.index("measures_generator_not_market")
    i_win = review_fn.index("r.win_rate")
    assert i_gen < i_win


def test_the_unavailable_branch_is_handled_without_rendering_a_rate(review_fn):
    """No journal is not a 0% win rate. The panel must stop at the explanation."""
    i_avail = review_fn.index("if(!r.available)")
    i_ret = review_fn.index("return;", i_avail)
    i_win = review_fn.index("r.win_rate")
    assert i_avail < i_ret < i_win, \
        "the no-journal branch does not return before the win rate is rendered"


def test_a_zero_scored_book_does_not_render_a_win_rate(review_fn):
    """`n_scored === 0` must short-circuit to an explanation. Rendering 0.0% for a
    journal that could not be priced turns an outage into a verdict."""
    m = re.search(r"if\(r\.n_scored === 0\)\{(.*?)\}else\{", review_fn, re.S)
    assert m, "the n_scored === 0 branch was not found"
    assert "win_rate" not in m.group(1), \
        "the unscoreable branch renders a win rate"
    assert "not a verdict" in m.group(1).lower() or "no outcome to measure" in m.group(1)


def test_unscorable_and_open_rows_are_listed_with_their_reason(review_fn):
    """Hiding them behind a count leaves the reader unable to tell a data outage from
    a young idea, which have opposite remedies."""
    assert "filter(x=>!x.scored && x.outcome !== 'open')" in review_fn
    assert "filter(x=>x.outcome === 'open')" in review_fn
    assert review_fn.count("x.note") >= 2


def test_the_divergence_flag_is_surfaced(review_fn):
    """The payoff divergence is the reason the realised breakeven exists at all; if
    the panel never shows it, the reader is left comparing unlike rates."""
    assert "payoff_diverged" in review_fn
    i_flag = review_fn.index("payoff_diverged")
    i_rate = review_fn.index("r.win_rate")
    assert i_flag < i_rate or "does not apply" in review_fn


def test_sample_size_travels_with_the_rate(review_fn):
    """A win rate without n is an invitation to over-read it."""
    i_win = review_fn.index("r.win_rate")
    nearby = review_fn[i_win:i_win + 400]
    assert "n=${r.n_scored}" in nearby or "r.n_scored" in nearby


def test_row_fields_are_escaped(review_fn):
    """Symbol and note come from a journal file on disk and from fetcher error text;
    both are attacker-influenceable in principle and must not be injected raw."""
    assert review_fn.count("escapeHtml(x.as_of") >= 1
    assert review_fn.count("escapeHtml(x.symbol") >= 2
    assert review_fn.count("escapeHtml(x.note") >= 2
    # no raw interpolation of a string row field into the table
    assert "${x.symbol}" not in review_fn
    assert "${x.note}" not in review_fn
    assert "${x.as_of}" not in review_fn


def test_the_endpoint_never_writes_the_journal(endpoint_source):
    """A GET that writes to the record it is measuring is a surprise, and a polling
    dashboard would corrupt the very journal it reports on."""
    assert "append_journal" not in endpoint_source
    assert ".write_text" not in endpoint_source
    assert "open(" not in endpoint_source


def test_the_endpoint_delegates_to_the_shared_scorer_and_fetcher(endpoint_source):
    """Two implementations of price anchoring is how a terminal and a browser end up
    reporting different outcomes for one journal line."""
    assert "from ..screener.review import review as score_journal, store_fetcher" \
        in endpoint_source
    assert "store_fetcher(" in endpoint_source
    assert "score_journal(" in endpoint_source


def test_cli_and_endpoint_use_the_same_horizon_default(endpoint_source):
    src = (REPO_ROOT / "src" / "tvlab" / "cli.py").read_text()
    cli_default = re.search(r'@click\.option\("--horizon", type=int, default=(\d+)', src)
    api_default = re.search(r'horizon: int = Query\((\d+)', endpoint_source)
    assert cli_default and api_default, "a horizon default was not found"
    assert cli_default.group(1) == api_default.group(1), \
        "the CLI and the API disagree about the default horizon"
