"""Pins the opening-range breakout and the session-close exit it depends on.

The strategy is only meaningful if it (a) takes exactly one break per session,
(b) stops at the other side of the range, and (c) is flat by the close. Point
(c) needs the engine's ``exit`` column, so it is tested through the engine too -
a strategy that silently holds overnight is a different strategy.
"""
import numpy as np
import pandas as pd
import pytest

from tvlab.backtest.engine import Backtester
from tvlab.config import load_config
from tvlab.strategy import get

BARS = 75          # 09:15 -> 15:25 on 5m bars
RANGE = 3          # 09:15-09:30


def _index(sessions=3, bars=BARS, start="2026-01-05"):
    out = []
    for d in pd.bdate_range(start, periods=sessions):
        base = d + pd.Timedelta(hours=9, minutes=15)
        out.extend(base + pd.Timedelta(minutes=5 * b) for b in range(bars))
    return pd.DatetimeIndex(out)


def _frame(direction="up", sessions=3, ambiguous_bar=None):
    """Flat opening range at 100 (99.5-100.5), then a steady trend."""
    rows = []
    for s in range(sessions):
        for b in range(BARS):
            if b < RANGE:
                o = c = 100.0
                hi, lo = 100.5, 99.5
            else:
                step = (b - RANGE + 1) * (0.5 if direction == "up" else -0.5)
                c = 100.0 + step
                o = c - (0.5 if direction == "up" else -0.5)
                hi = max(o, c) + 0.05
                lo = min(o, c) - 0.05
            if ambiguous_bar is not None and b == ambiguous_bar:
                hi, lo = 101.0, 99.0      # breaks both sides at once
            rows.append({"open": o, "high": hi, "low": lo, "close": c, "volume": 1000.0})
    return pd.DataFrame(rows, index=_index(sessions))


def test_takes_one_long_break_and_stops_at_range_low():
    df = _frame("up")
    out = get("orb")({}).signals(df)
    sig = out["signal"].to_numpy()
    assert (sig == 1).sum() == 3, "one entry per session"
    first = int(np.flatnonzero(sig == 1)[0])
    assert df["high"].iloc[first] > 100.5
    assert out["stop"].iloc[first] == pytest.approx(99.5)
    assert out["tag"].iloc[first] == "orb_long"
    # nothing is signalled while price is still inside the opening range
    assert (sig[:RANGE] == 0).all()


def test_short_break_respects_allow_short():
    df = _frame("down")
    allowed = get("orb")({}).signals(df)["signal"].to_numpy()
    assert (allowed == -1).sum() == 3
    blocked = get("orb")({"allow_short": False}).signals(df)["signal"].to_numpy()
    assert (blocked == 0).all()


def test_ambiguous_bar_skips_the_session():
    df = _frame("up", ambiguous_bar=RANGE)
    out = get("orb")({}).signals(df)
    assert (out["signal"] == 0).all(), "a bar breaking both sides cannot be traded"


def test_session_exit_is_flagged_on_the_clock_not_the_next_row():
    df = _frame("up")
    out = get("orb")({}).signals(df)
    exits = out["exit"].to_numpy()
    # the 15:25 bar of every session is flagged, and only it
    assert exits.sum() == df.index.normalize().nunique()
    last_of_session = df.index.hour * 60 + df.index.minute == 15 * 60 + 25
    assert np.array_equal(exits.astype(bool), last_of_session)


def test_a_break_on_the_final_bar_is_not_traded():
    """The engine enters on the next bar's open, so a last-bar break would be
    filled on the next session's open and carried overnight - not the rule."""
    rows = []
    for _ in range(3):
        for b in range(BARS):
            if b < RANGE:
                o = c = 100.0
                hi, lo = 100.5, 99.5
            elif b == BARS - 1:
                o, c = 100.0, 101.0      # breaks the range only at 15:25
                hi, lo = 101.0, 100.0
            else:
                o = c = 100.0
                hi, lo = 100.4, 99.6
            rows.append({"open": o, "high": hi, "low": lo, "close": c, "volume": 1000.0})
    df = pd.DataFrame(rows, index=_index(3))
    out = get("orb")({}).signals(df)
    assert (out["signal"] == 0).all()


def test_engine_refuses_to_open_on_the_exit_bar():
    """A signal on the bar before the close would fill on the closing bar, which
    cannot be held - the engine must not open there."""
    df = _frame("up", sessions=2)
    strat = get("orb")({})
    sig = strat.signals(df)
    sig["signal"] = 0.0
    exit_bar = int(np.flatnonzero(sig["exit"].to_numpy() == 1)[0])
    sig.iloc[exit_bar - 1, sig.columns.get_loc("signal")] = 1.0
    sig.iloc[exit_bar - 1, sig.columns.get_loc("stop")] = 99.0
    cfg = load_config(market="indian_equities", style="swing")
    res = Backtester(cfg).run(df, sig, symbol="TEST", timeframe="5m",
                             max_bars_in_trade=strat.resolve_max_bars(None))
    assert res.trades.empty, res.trades[['entry_time', 'exit_time', 'exit_reason']]


def test_daily_frames_produce_no_signals():
    idx = pd.bdate_range("2026-01-05", periods=120)
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0, "volume": 1.0}, index=idx)
    out = get("orb")({}).signals(df)
    assert (out["signal"] == 0).all()
    assert (out["exit"] == 0).all(), "no session clock on daily bars"


def test_engine_closes_the_position_before_the_session_ends():
    df = _frame("up", sessions=4)
    strat = get("orb")({})
    cfg = load_config(market="indian_equities", style="swing")
    res = Backtester(cfg).run(df, strat.signals(df), symbol="TEST",
                             timeframe="5m", max_bars_in_trade=strat.resolve_max_bars(None))
    trades = res.trades
    assert not trades.empty
    assert (trades["exit_reason"] == "session_exit").any()
    # no position is carried across the overnight gap
    assert (trades["entry_time"].dt.normalize()
            == trades["exit_time"].dt.normalize()).all()
