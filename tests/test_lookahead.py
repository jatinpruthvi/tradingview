"""The single most important test file in the project.

A strategy with lookahead bias produces beautiful backtests and losing accounts.
These tests fail loudly if any calculation in the trading path uses future bars.
"""
import numpy as np
import pandas as pd
import pytest

from tvlab.data.base import align_to_execution, resample
from tvlab.data.synthetic import basket
from tvlab.indicators.smc import smc_features
from tvlab.strategy import available, get
from tvlab.validation import lookahead as LA


@pytest.fixture(scope="module")
def df():
    return basket(["BTC/USDT"], n_bars=900, timeframe="4h")["BTC/USDT"]


@pytest.fixture(scope="module")
def bias(df):
    return align_to_execution(resample(df, "1d"), df)


CUTS = (400, 550, 700)


def test_smc_has_no_lookahead(df):
    r = LA.check_smc(df, cuts=CUTS, swing_length=12)
    assert r.ok, f"SMC leak: {r.leaks}"
    assert r.columns_checked >= 20


def test_technicals_have_no_lookahead(df):
    r = LA.check_technicals(df, cuts=CUTS)
    assert r.ok, f"technical indicator leak: {r.leaks}"


@pytest.mark.parametrize("name", available())
def test_every_registered_strategy_is_causal(df, bias, name):
    strat = get(name)({})
    r = LA.check_strategy(strat, df, bias, cuts=CUTS)
    assert r.ok, f"{name} leaks future information: {r.leaks}"


def test_naive_swing_detection_would_leak(df):
    """Proves the test can actually detect a leak.

    Publishing a fractal pivot at the pivot bar (the reference convention) is
    non-causal, and this test must catch it - otherwise the suite is theatre.
    """
    def leaky(d):
        s = smc_features(d, swing_length=12)
        # shift confirmed pivots BACK to the pivot bar, i.e. undo the lag
        out = s.copy()
        out["swing_high"] = np.roll(s["swing_high"].to_numpy(), -12)
        out["swing_low"] = np.roll(s["swing_low"].to_numpy(), -12)
        return out

    r = LA.check_function(leaky, df, target="deliberately_leaky", cuts=CUTS,
                          columns=["swing_high", "swing_low"])
    assert not r.ok, "the lookahead detector failed to catch a known leak"
    assert r.n_leaks > 0


def test_full_sample_normalisation_would_leak(df):
    """Second positive control: normalising by the full-sample max is a classic
    silent leak (it uses the future to scale the past).

    Note ``bfill`` is NOT a valid control here - the synthetic close series has
    no NaN, so backward-filling changes nothing and the test would pass for the
    wrong reason.
    """
    r = LA.check_function(lambda d: (d[["close"]] / d["close"].max()).rename(columns={"close": "norm"}),
                          df, target="fullsample_norm_control", cuts=CUTS, columns=["norm"])
    assert not r.ok, "detector missed full-sample normalisation"


def test_centred_rolling_window_would_leak(df):
    """Third positive control: a centred rolling mean peeks forward."""
    r = LA.check_function(
        lambda d: d["close"].rolling(20, center=True, min_periods=1).mean().to_frame("cma"),
        df, target="centred_ma_control", cuts=CUTS, columns=["cma"])
    assert not r.ok, "detector missed a centred rolling window"


def test_bias_frame_alignment_is_causal(df, bias):
    a = LA.check_alignment(df, bias)
    assert a["ok"]
    assert a["exec_rows"] == len(df)
