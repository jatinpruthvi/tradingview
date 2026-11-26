"""Tests for the Black-Scholes option-buying simulator."""
import numpy as np
import pandas as pd
import pytest

from tvlab.backtest.options_sim import (
    OptionSimConfig,
    bs_delta,
    bs_price,
    causal_session_rv,
    run_option_study,
    simulate_options,
    strike_for_delta,
)
from tvlab.config import load_config
from tvlab.strategy import get
from tvlab.strategy.base import blank_signals


def _sessions(n_sessions=8, bars=75, step_per_bar=0.0, start="2026-01-05"):
    idx = []
    rows = []
    for d in pd.bdate_range(start, periods=n_sessions):
        base = d + pd.Timedelta(hours=9, minutes=15)
        for b in range(bars):
            idx.append(base + pd.Timedelta(minutes=5 * b))
            if b < 3:
                o = c = 1000.0
                hi, lo = 1005.0, 995.0
            else:
                c = 1000.0 + (b - 2) * step_per_bar
                o = c - step_per_bar
                hi = max(o, c) + (6.0 if b == 3 and step_per_bar > 0 else 0.5)
                lo = min(o, c) - 0.5
            rows.append({"open": o, "high": hi, "low": lo, "close": c, "volume": 10_000.0})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_black_scholes_put_call_parity_and_delta_strike_solver():
    s, k, t, vol, r = 25_000.0, 25_200.0, 7.0 / 365.0, 0.16, 0.065
    call = bs_price(s, k, t, vol, side=1, rate=r)
    put = bs_price(s, k, t, vol, side=-1, rate=r)
    assert call - put == pytest.approx(s - k * np.exp(-r * t), rel=1e-6)

    for tgt in (0.30, 0.50, 0.65):
        kc = strike_for_delta(s, t, vol, tgt, side=1, rate=r)
        assert bs_delta(s, kc, t, vol, side=1, rate=r) == pytest.approx(tgt, abs=1e-6)
        kp = strike_for_delta(s, t, vol, tgt, side=-1, rate=r)
        assert abs(bs_delta(s, kp, t, vol, side=-1, rate=r)) == pytest.approx(tgt, abs=1e-6)


def test_causal_rv_only_uses_prior_completed_sessions():
    df = _sessions(n_sessions=6, bars=10, step_per_bar=0.0)
    rv_before = causal_session_rv(df)
    # Shock the final session massively; all prior sessions must remain bit-identical
    df_shock = df.copy()
    df_shock.iloc[-1, df_shock.columns.get_loc("close")] = 1500.0
    rv_after = causal_session_rv(df_shock)
    assert np.array_equal(rv_before[:-10], rv_after[:-10])


def test_flat_underlying_loses_theta_and_spread():
    df = _sessions(n_sessions=3, bars=20, step_per_bar=0.0)
    sig = blank_signals(df)
    sig.iloc[5, sig.columns.get_loc("signal")] = 1.0
    sig.iloc[5, sig.columns.get_loc("stop")] = 900.0
    cfg = load_config("indian_equities", "intraday")
    opt = OptionSimConfig(target_delta=0.50, dte_days=3.0, max_hold_bars=10)
    res = simulate_options(df, sig, cfg, opt, symbol="TEST")
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "time_exit"
    assert res.trades.iloc[0]["pnl"] < 0


def test_fast_upward_orb_profits_for_call_buyer():
    df = _sessions(n_sessions=3, bars=20, step_per_bar=3.0)
    strat = get("orb")({"allow_short": True})
    sig = strat.signals(df)
    cfg = load_config("indian_equities", "intraday")
    opt = OptionSimConfig(target_delta=0.65, dte_days=7.0, max_hold_bars=12,
                          premium_target_mult=2.0)
    res = simulate_options(df, sig, cfg, opt, symbol="TEST")
    assert len(res.trades) == 3
    assert (res.trades["option_type"] == "CE").all()
    assert res.trades["pnl"].sum() > 0


def test_run_option_study_on_synthetic_intraday():
    cfg = load_config("indian_equities", "intraday").merge({
        "strategy.name": "orb",
        "data.synthetic_bars": 800,
    })
    df = run_option_study(cfg, symbols=["RELIANCE.NS", "HDFCBANK.NS"])
    assert not df.empty
    assert set(["atm_7dte_60m", "itm65_7dte_60m", "otm30_7dte_60m"]).issubset(set(df["config"]))
