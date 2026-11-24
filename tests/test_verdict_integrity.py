"""The verdict must classify correctly and survive serialisation.

Three defects, all found by asking check 6 to explain its own ``n/a``:

1. ``passed`` could be ``np.False_`` rather than ``False``. Several checks are
   written ``np.isfinite(x) and x >= t``, and when ``x`` is NaN that expression
   evaluates to a numpy bool, which is *not* a subclass of ``bool``.

2. ``hard_fail`` tested ``c["passed"] is False``. ``np.False_ is False`` is False,
   so a failing walk-forward or shuffled-null check - both in the hard-fail set -
   was skipped, and the verdict came out **PARTIAL** where it should have said
   **NOT VALIDATED**. The bug was masked in every shipped report because the
   deflated-Sharpe check happened to fail too, and that one produces a real bool.

3. ``json.dumps(..., default=str)`` cannot serialise ``np.False_`` and falls back to
   ``str()``, producing the *string* ``"False"`` - which is truthy in Python, in
   JavaScript and in every template. A failed check would have rendered as passed.

A fourth gap is pinned here too: the verdict existed only in the markdown. The most
decision-relevant output in the project was unreadable by the dashboard, the suggest
endpoint or any script, so nothing could warn a user that the win rate being quoted
back to them came from a study that failed its own validity checks.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tvlab.config import load_config
from tvlab.report import _verdict

HARD_FAIL_CHECKS = {
    "Beats shuffled-return null",
    "Deflated Sharpe (multiple testing)",
    "Walk-forward windows profitable",
    "Enough trades to measure",
}


def _study(n_trades=40, pf=1.5, dsr=0.99):
    """A study that passes every check it is given the chance to pass."""
    return SimpleNamespace(
        portfolio_metrics={
            "n_trades": n_trades, "profit_factor": pf, "max_drawdown_pct": 0.5,
            "total_return_pct": 2.0, "sharpe": 1.2, "win_rate": 0.5,
            "deflated_sharpe": {"dsr": dsr, "verdict": "pass", "n_trials": 320},
        },
        config=load_config(market="crypto", style="swing"),
    )


def _check(v, name):
    for c in v["checks"]:
        if c["check"] == name:
            return c
    raise AssertionError(f"{name!r} not in the verdict")


def test_a_nan_walk_forward_hit_rate_still_hard_fails():
    """No windows at all -> hit rate NaN -> check 5 fails. It is in the hard-fail
    set, so the verdict must be NOT VALIDATED, not PARTIAL."""
    wf = {"n_windows": 0, "windows_profitable": 0,
          "window_hit_rate": float("nan"), "mean_oos_efficiency": float("nan")}
    v = _verdict(_study(), wf, None, None)
    c = _check(v, "Walk-forward windows profitable")
    assert c["passed"] is False, f"expected a real False, got {c['passed']!r}"
    assert v["verdict"] == "NOT VALIDATED", (
        f"a failed hard-fail check produced {v['verdict']!r}")


def test_a_nan_p_value_still_hard_fails():
    """Same shape of bug on the shuffled-return null."""
    nulls = {"shuffled_returns": {"p_value": float("nan"),
                                  "better_than_null_pct": float("nan"),
                                  "n_sims": 60}}
    v = _verdict(_study(), None, nulls, None)
    c = _check(v, "Beats shuffled-return null")
    assert c["passed"] is False
    assert v["verdict"] == "NOT VALIDATED"


def test_every_hard_fail_check_downgrades_the_verdict_on_its_own():
    """Each one separately, with everything else passing. This is the case that was
    broken: a single hard-fail failure used to be enough to be missed."""
    cases = {
        "Enough trades to measure": lambda: _verdict(_study(n_trades=5), None, None, None),
        "Deflated Sharpe (multiple testing)": lambda: _verdict(_study(dsr=0.01), None, None, None),
        "Walk-forward windows profitable": lambda: _verdict(_study(), {
            "n_windows": 5, "windows_profitable": 1, "window_hit_rate": 0.2,
            "mean_oos_efficiency": 0.5}, None, None),
        "Beats shuffled-return null": lambda: _verdict(_study(), None, {
            "shuffled_returns": {"p_value": 0.53, "better_than_null_pct": 47.0,
                                 "n_sims": 60}}, None),
    }
    assert set(cases) == HARD_FAIL_CHECKS, "the hard-fail set changed; update this test"
    for name, build in cases.items():
        v = build()
        assert _check(v, name)["passed"] is False, f"{name} should have failed"
        assert v["verdict"] == "NOT VALIDATED", (
            f"{name} failed but the verdict was {v['verdict']!r}")


def test_passed_is_always_a_plain_bool_or_none():
    wf = {"n_windows": 0, "windows_profitable": 0,
          "window_hit_rate": float("nan"), "mean_oos_efficiency": float("nan")}
    nulls = {"shuffled_returns": {"p_value": float("nan"),
                                  "better_than_null_pct": float("nan"), "n_sims": 60}}
    mc = {"ruin_probabilities": {"dd_exceeds_10pct": 0.0}, "n_sims": 2000}
    v = _verdict(_study(), wf, nulls, mc)
    for c in v["checks"]:
        assert c["passed"] is None or type(c["passed"]) is bool, (
            f"{c['check']}: passed is {type(c['passed']).__name__}, which is not "
            f"comparable with `is False` and not JSON-serialisable as a boolean")


def test_a_failed_check_is_falsy_after_a_json_round_trip():
    """The dashboard reads this back. `default=str` would have made it the truthy
    string "False"."""
    wf = {"n_windows": 0, "windows_profitable": 0,
          "window_hit_rate": float("nan"), "mean_oos_efficiency": float("nan")}
    v = _verdict(_study(), wf, None, None)
    text = json.dumps(v, default=str)
    back = json.loads(text)
    by_name = {c["check"]: c for c in back["checks"]}
    # this scenario fails the walk-forward check via a NaN hit rate, which is the
    # exact path that used to produce np.False_
    c = by_name["Walk-forward windows profitable"]
    assert c["passed"] is False, (
        f"round-tripped to {c['passed']!r} - a truthy value for a failed check "
        f"would render green in the dashboard")
    assert c["check"] in HARD_FAIL_CHECKS
    assert '"False"' not in text, "a numpy bool leaked into the JSON as a string"
    assert '"True"' not in text, "a numpy bool leaked into the JSON as a string"


def test_the_written_report_json_carries_the_verdict(tmp_path):
    """End to end: the artifact on disk must be machine-readable."""
    from tvlab.pipeline import run_study
    from tvlab.report import write_report

    cfg = load_config(market="crypto", style="swing",
                      overrides={"data.synthetic_bars": 900})
    study = run_study(cfg, with_benchmark=False, cross_validate=False)
    paths = write_report(study, tmp_path, plots=False)

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "verdict" in payload, "the verdict is still markdown-only"
    v = payload["verdict"]
    assert v["verdict"] in {"NOT VALIDATED", "PARTIAL", "PASSES ON THIS DATA"}
    assert v["n_checks"] == len(v["checks"]) >= 4
    assert all(isinstance(c["passed"], bool) or c["passed"] is None
               for c in v["checks"])
    # and it agrees with the markdown that sits next to it
    md = paths["markdown"].read_text(encoding="utf-8")
    assert f"## Verdict: **{v['verdict']}**" in md
