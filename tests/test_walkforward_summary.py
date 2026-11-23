"""An unmeasurable validity check has to say why.

Check 6 of the report ("OOS retains IS performance") compares each walk-forward
window's out-of-sample objective against its in-sample one. When no window can
produce an in-sample objective the mean efficiency is NaN, and the check used to
print a bare ``n/a``.

That is the worst possible output for a validation report. It reads like an
unimplemented feature, when the actual situation is a finding: the sample is too
small for the ratio to have a denominator. A user who sees ``n/a`` learns nothing;
a user who sees "5/5 windows had too few in-sample trades to score (best 7 vs 8
required)" knows exactly what to go and get.

These tests pin both halves - that ``WalkForwardResult.summary()`` carries the
diagnostic, and that the report renders it - including the two distinct reasons a
window can fail to score, which need different wording.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tvlab.backtest.walkforward import WalkForwardResult, WindowResult
from tvlab.config import load_config
from tvlab.report import _verdict

TS = pd.Timestamp("2022-01-01")


def _window(index: int, train_trades: int, required: int,
            objective_train: float, objective_test: float,
            test_trades: int | None = None) -> WindowResult:
    """test_trades defaults to something above the floor; pass it explicitly when
    the scenario is a too-short out-of-sample segment, or the rendered detail will
    quote a trade count that contradicts its own explanation."""
    if test_trades is None:
        test_trades = max(required, train_trades // 2)
    return WindowResult(
        index=index,
        best_params={"atr_stop_mult": 1.8},
        trials=64,
        train_metrics={"n_trades": train_trades, "net_pnl": 10.0 * train_trades},
        test_metrics={"n_trades": test_trades, "net_pnl": 5.0},
        train_slice=(TS, TS + pd.Timedelta(days=100)),
        test_slice=(TS + pd.Timedelta(days=100), TS + pd.Timedelta(days=150)),
        objective_train=objective_train,
        objective_test=objective_test,
        min_trades_required=required,
    )


def _result(windows) -> WalkForwardResult:
    return WalkForwardResult(config=load_config(market="crypto", style="swing"),
                             symbol="BTC/USDT", strategy="smc_trend", windows=windows)


def _stub_study(n_trades: int = 13, profit_factor: float = 1.09):
    """``_verdict`` only reads portfolio_metrics and the drawdown mandate."""
    cfg = load_config(market="crypto", style="swing")
    return SimpleNamespace(
        portfolio_metrics={
            "n_trades": n_trades, "profit_factor": profit_factor,
            "max_drawdown_pct": 0.37, "total_return_pct": 0.08, "sharpe": 0.16,
            "win_rate": 0.38, "deflated_sharpe": {"dsr": 0.002, "verdict": "fail",
                                                  "n_trials": 320},
        },
        config=cfg,
    )


def _check6(verdict) -> dict:
    for c in verdict["checks"]:
        if c["check"] == "OOS retains IS performance":
            return c
    raise AssertionError("check 6 missing from the verdict")


# ------------------------------------------------- the summary carries the reason


def test_summary_reports_how_many_windows_could_not_be_scored():
    windows = [_window(i, train_trades=7, required=8,
                       objective_train=float("-inf"), objective_test=0.4)
               for i in range(5)]
    s = _result(windows).summary()
    assert s["n_windows"] == 5
    assert s["windows_scorable"] == 0
    assert s["windows_unscoreable"] == 5
    assert s["unscoreable_train_trades"] == [7] * 5
    assert s["unscoreable_required"] == [8] * 5
    assert s["mean_oos_efficiency"] != s["mean_oos_efficiency"]   # NaN


def test_summary_names_which_side_could_not_be_scored():
    """The case the shipped config actually produces: healthy in-sample fits, and an
    out-of-sample segment too short to score. Naming the wrong side would send
    someone to add bars per window when the fix is fewer splits or more history."""
    windows = [_window(i, train_trades=10, required=3, test_trades=1,
                       objective_train=1.7, objective_test=float("-inf"))
               for i in range(5)]
    s = _result(windows).summary()
    assert s["windows_unscoreable"] == 5
    assert s["windows_scorable"] == 0
    assert s["unscoreable_side"] == "test"
    assert s["unscoreable_train_trades"] == [10] * 5


def test_summary_distinguishes_a_train_side_failure():
    windows = [_window(i, train_trades=2, required=8,
                       objective_train=float("-inf"), objective_test=0.4)
               for i in range(3)]
    assert _result(windows).summary()["unscoreable_side"] == "train"


def test_summary_reports_mixed_when_both_sides_fail():
    windows = [
        _window(0, 2, 8, float("-inf"), 0.4),      # train side
        _window(1, 10, 3, 1.7, float("-inf"), test_trades=1),   # test side
    ]
    assert _result(windows).summary()["unscoreable_side"] == "mixed"


def test_check6_blames_the_out_of_sample_segment_when_that_is_the_cause():
    """Regression test for a wrong explanation this check briefly shipped with.

    The first version of the diagnostic reported the *in-sample* trade counts for
    every unmeasurable window. On the real default config those counts are 8-11
    against a requirement of 8 - i.e. the in-sample fits were fine - so the report
    claimed a cause that the numbers beside it contradicted.
    """
    windows = [_window(i, train_trades=10, required=3, test_trades=1,
                       objective_train=1.7, objective_test=float("-inf"))
               for i in range(5)]
    c = _check6(_verdict(_stub_study(), _result(windows).summary(), None, None))
    assert c["passed"] is False
    assert "not measurable" in c["detail"]
    assert "out-of-sample" in c["detail"]
    assert "in-sample fits scored fine" in c["detail"]
    assert "5/5 windows" in c["detail"]
    assert "undefined, not zero" in c["detail"]
    # the remedy has to be the right one for this side
    assert "Fewer walk-forward splits" in c["detail"]
    # and it must NOT blame the in-sample side, which was healthy
    assert "too few in-sample trades" not in c["detail"]


def test_summary_counts_scorable_windows_separately():
    windows = [
        _window(0, 7, 8, float("-inf"), 0.4),          # unscoreable
        _window(1, 20, 8, 1.0, 0.5),                    # efficiency 0.5
        _window(2, 22, 8, 2.0, 0.4),                    # efficiency 0.2
    ]
    s = _result(windows).summary()
    assert s["windows_scorable"] == 2
    assert s["windows_unscoreable"] == 1
    assert s["unscoreable_train_trades"] == [7]
    assert s["mean_oos_efficiency"] == pytest.approx(0.35)


# ------------------------------------------------------- and the report renders it


def test_check6_says_too_few_trades_instead_of_a_bare_na():
    windows = [_window(i, train_trades=7, required=8,
                       objective_train=float("-inf"), objective_test=0.4)
               for i in range(5)]
    v = _verdict(_stub_study(), _result(windows).summary(), None, None)
    c = _check6(v)
    assert c["passed"] is False
    assert c["detail"] != "n/a", "the bare n/a is exactly what this replaces"
    assert "not measurable" in c["detail"]
    assert "5/5 windows" in c["detail"]
    assert "too few in-sample trades" in c["detail"]
    assert "best 7 vs 8 required" in c["detail"]
    # the distinction that matters most: undefined is not the same as zero
    assert "undefined, not zero" in c["detail"]


def test_check6_uses_the_other_wording_when_trades_were_not_the_problem():
    """A window can fail to score with plenty of trades - e.g. a non-finite Sharpe
    makes ``objective_value`` return -inf regardless. Claiming "too few trades"
    there would send the user after the wrong fix."""
    windows = [_window(i, train_trades=40, required=8,
                       objective_train=float("-inf"), objective_test=0.4)
               for i in range(3)]
    c = _check6(_verdict(_stub_study(), _result(windows).summary(), None, None))
    assert "not measurable" in c["detail"]
    assert "too few in-sample trades" not in c["detail"]
    assert "no usable in-sample objective" in c["detail"]
    assert "40" in c["detail"] and "8" in c["detail"]


def test_check6_reports_the_ratio_when_it_is_measurable():
    windows = [_window(i, 20, 8, 1.0, 0.5) for i in range(3)]
    c = _check6(_verdict(_stub_study(), _result(windows).summary(), None, None))
    assert c["passed"] is True
    assert "mean OOS/IS objective ratio = 0.50" in c["detail"]
    assert "not measurable" not in c["detail"]


def test_check6_fails_on_an_inverted_fit():
    """A negative ratio means the in-sample fit stopped working out of sample."""
    windows = [_window(i, 20, 8, 1.0, -0.27) for i in range(4)]
    c = _check6(_verdict(_stub_study(), _result(windows).summary(), None, None))
    assert c["passed"] is False
    assert "-0.27" in c["detail"]


def test_check6_survives_a_summary_with_no_diagnostic_fields():
    """Older serialised reports will not carry the new keys. The check must still
    render something honest rather than KeyError on a round-tripped JSON."""
    c = _check6(_verdict(_stub_study(), {
        "n_windows": 5, "window_hit_rate": 0.2, "windows_profitable": 1,
        "mean_oos_efficiency": float("nan"),
    }, None, None))
    assert c["passed"] is False
    assert "not measurable" in c["detail"]
    assert c["detail"] != "n/a"
