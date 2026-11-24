"""Risk sizing and the governor. Conservative mandate: 0.5% per trade, 10% max DD."""
import numpy as np
import pandas as pd
import pytest

from tvlab.config import load_config
from tvlab.risk.sizing import RiskGovernor, Sizer


@pytest.fixture(scope="module")
def cfg():
    return load_config(market="crypto", style="swing")


def test_mandate_values_are_conservative(cfg):
    assert cfg.risk.risk_per_trade == 0.005
    assert cfg.risk.max_drawdown_pct == 0.10
    assert cfg.risk.daily_loss_limit_pct == 0.02
    assert cfg.risk.max_leverage == 1.0


def test_position_size_matches_risk_budget(cfg):
    """units * (entry - stop) must equal risk_per_trade * equity."""
    s = Sizer(cfg.risk)
    r = s.size(side=1, equity=100_000.0, price=50_000.0, stop=49_000.0, atr=800.0)
    assert r.accepted
    assert abs(r.units) * abs(50_000.0 - 49_000.0) == pytest.approx(500.0, rel=1e-6)


def test_wider_stop_means_smaller_position(cfg):
    s = Sizer(cfg.risk)
    near = s.size(side=1, equity=100_000.0, price=100.0, stop=98.0, atr=1.0)
    far = s.size(side=1, equity=100_000.0, price=100.0, stop=90.0, atr=5.0)
    assert near.accepted and far.accepted
    assert abs(far.units) < abs(near.units)
    # both risk the same dollars - that is the whole point of risk-based sizing
    assert abs(near.units) * 2.0 == pytest.approx(abs(far.units) * 10.0, rel=1e-6)


def test_position_cap_binds_before_risk_budget(cfg):
    """max_position_pct must truncate a tight-stop trade that would over-leverage.

    A 0.5% stop on a 0.5% risk budget implies 100% of equity in notional, well
    over the 25% cap. The cap must win, which means risk_amount falls BELOW the
    budget - the account is never levered up to hit the risk number.
    """
    c = cfg.risk
    s = Sizer(c)
    r = s.size(equity=100_000.0, price=100.0, stop=99.5, side=1, atr=1.0)
    assert r.accepted, r.reason
    max_notional = c.max_position_pct * 100_000.0 * c.max_leverage
    assert abs(r.units) * 100.0 == pytest.approx(max_notional, rel=1e-6)
    assert r.risk_amount < 100_000.0 * c.risk_per_trade


def test_stop_distance_bounds_are_enforced(cfg):
    s = Sizer(cfg.risk)
    too_tight = s.size(equity=100_000.0, price=100.0, stop=99.9999, side=1, atr=1.0)
    assert not too_tight.accepted and too_tight.reason == "stop_too_tight"
    too_wide = s.size(equity=100_000.0, price=100.0, stop=50.0, side=1, atr=10.0)
    assert not too_wide.accepted and too_wide.reason == "stop_too_wide"


def test_stop_inside_noise_is_rejected(cfg):
    """A stop tighter than 5% of ATR is just noise; sizing must refuse it."""
    s = Sizer(cfg.risk)
    r = s.size(equity=100_000.0, price=100.0, stop=99.5, side=1, atr=20.0)
    assert not r.accepted and r.reason == "stop_inside_noise"


def test_stop_on_the_wrong_side_is_rejected(cfg):
    s = Sizer(cfg.risk)
    assert s.size(equity=100_000.0, price=100.0, stop=101.0, side=1, atr=1.0).reason == "stop_on_wrong_side"
    assert s.size(equity=100_000.0, price=100.0, stop=99.0, side=-1, atr=1.0).reason == "stop_on_wrong_side"


def test_infinity_is_the_no_stop_sentinel(cfg):
    """Regression: 'no stop' used to be expressed as 1e-9, which the
    max-stop-distance check rejected as stop_too_tight."""
    s = Sizer(cfg.risk)
    r = s.size(equity=100_000.0, price=100.0, stop=-np.inf, side=1, atr=1.0)
    assert r.accepted and r.reason == "ok", r.reason
    # with no stop there is no risk distance, so it sizes on the notional cap
    assert abs(r.units) * 100.0 == pytest.approx(
        100_000.0 * cfg.risk.max_position_pct * cfg.risk.max_leverage, rel=1e-6)
    assert np.isnan(r.risk_amount)


def test_below_min_lot_is_skipped_not_rounded(cfg):
    s = Sizer(cfg.risk)
    r = s.size(equity=100_000.0, price=100.0, stop=98.0, side=1, atr=1.0, min_units=1000.0)
    assert not r.accepted and r.reason == "below_min_lot"


def test_short_is_rejected_when_shorts_disabled():
    """Indian equities are long-only in config; sizing must enforce it."""
    cfg2 = load_config(market="indian_equities", style="swing")
    assert cfg2.risk.allow_short is False
    s = Sizer(cfg2.risk)
    r = s.size(equity=100_000.0, price=100.0, stop=102.0, side=-1, atr=1.0)
    assert not r.accepted and r.reason == "shorts_disabled"
    # and the same trade is accepted on crypto, where shorts are allowed
    s3 = Sizer(load_config(market="crypto", style="swing").risk)
    assert s3.size(equity=100_000.0, price=100.0, stop=102.0, side=-1, atr=1.0).accepted


def test_throttle_ramps_linearly_to_zero_at_the_limit(cfg):
    s = Sizer(cfg.risk)
    hard = cfg.risk.max_drawdown_pct
    soft = hard * cfg.risk.dd_throttle_start
    assert s.throttle_at(0.0) == 1.0
    assert s.throttle_at(soft) == 1.0
    assert s.throttle_at(hard) == 0.0
    assert s.throttle_at(hard * 2) == 0.0
    mid = s.throttle_at((soft + hard) / 2)
    assert 0.0 < mid < 1.0
    assert mid == pytest.approx(0.5, abs=1e-9)


def test_throttle_reduces_size_as_drawdown_grows(cfg):
    s = Sizer(cfg.risk)
    kw = dict(price=100.0, stop=98.0, side=1, atr=1.0)
    full = s.size(equity=100_000.0, drawdown=0.0, **kw)
    throttled = s.size(equity=97_000.0, drawdown=0.075, **kw)
    deep = s.size(equity=90_000.0, drawdown=0.10, **kw)
    assert full.accepted and throttled.accepted
    assert abs(throttled.units) < abs(full.units)
    assert not deep.accepted and deep.reason == "dd_limit_reached"
    assert abs(deep.units) == 0, "sizing must stop at the max drawdown"


def test_peak_equity_implies_the_same_throttle(cfg):
    s = Sizer(cfg.risk)
    kw = dict(price=100.0, stop=98.0, side=1, atr=1.0)
    a = s.size(equity=95_000.0, peak_equity=100_000.0, **kw)
    b = s.size(equity=95_000.0, drawdown=0.05, **kw)
    assert a.units == pytest.approx(b.units)


TS = pd.date_range("2022-01-01", periods=400, freq="4h", tz="UTC")
DAILY = pd.date_range("2022-01-01", periods=400, freq="1D", tz="UTC")


def test_governor_halts_at_the_drawdown_limit(cfg):
    g = RiskGovernor(cfg.risk, initial_capital=100_000.0)
    assert g.can_open()[0]
    for ts, eq in zip(TS, [101_000, 102_000, 99_000, 96_000, 93_000, 91_000, 90_000, 88_000]):
        g.on_bar(ts, eq)
    assert g.drawdown > 0.10, g.drawdown
    assert g.halted
    ok, why = g.can_open()
    assert not ok, "the governor kept trading through a >10% drawdown"
    assert why, "a halt must carry a human-readable reason"


def test_governor_throttles_before_it_halts(cfg):
    """A slow bleed across SEPARATE DAYS must throttle exposure, not halt.

    Daily timestamps are deliberate: on a single day this same path would trip
    the 2% daily-loss lock instead, which is a different (and stricter) breaker.
    """
    g = RiskGovernor(cfg.risk, initial_capital=100_000.0)
    for ts, eq in zip(DAILY[:6], [100_000, 99_000, 97_500, 96_000, 94_500, 93_000]):
        g.on_bar(ts, eq)
    ok, why = g.can_open()
    assert ok, f"must still be trading at a 7% drawdown, got {why}"
    assert not g.day_loss_locked, "no single day lost 2%"
    assert 0.0 < g.throttle() < 1.0, "exposure should be throttled, not full"


def test_daily_loss_lock_fires_before_the_drawdown_halt(cfg):
    """A 7% loss inside ONE day must lock the day even though total DD is only 7%."""
    g = RiskGovernor(cfg.risk, initial_capital=100_000.0)
    for ts, eq in zip(TS[:6], [100_000, 99_000, 97_000, 95_000, 94_000, 93_000]):
        g.on_bar(ts, eq)
    ok, why = g.can_open()
    assert not ok and why == "daily_loss_limit"
    assert g.day_loss_locked
    assert not g.halted                        # DD limit is 10%, not yet breached


def test_governor_snapshot_is_json_safe(cfg):
    import json
    g = RiskGovernor(cfg.risk, initial_capital=100_000.0)
    for ts, eq in zip(TS[:3], [100_000, 98_000, 95_000]):
        g.on_bar(ts, eq)
    snap = g.snapshot()
    assert isinstance(snap, dict)
    json.dumps(snap, default=str)             # must survive serialisation


def test_sizer_is_deterministic(cfg):
    s = Sizer(cfg.risk)
    kw = dict(equity=100_000.0, price=100.0, stop=98.0, side=1, atr=1.0)
    a, b = s.size(**kw), s.size(**kw)
    assert a.units == b.units and a.reason == b.reason and a.notional == b.notional


def test_invalid_inputs_do_not_crash(cfg):
    s = Sizer(cfg.risk)
    assert s.size(equity=0.0, price=100.0, stop=98.0, side=1).reason == "invalid_inputs"
    assert s.size(equity=-1.0, price=100.0, stop=98.0, side=1).reason == "invalid_inputs"
    assert s.size(equity=100_000.0, price=0.0, stop=98.0, side=1).reason == "invalid_inputs"
    assert s.size(equity=100_000.0, price=np.nan, stop=98.0, side=1).reason == "invalid_inputs"
    assert s.size(equity=100_000.0, price=100.0, stop=98.0, side=0).reason == "no_signal"


def test_governor_respects_max_open_positions(cfg):
    g = RiskGovernor(cfg.risk, initial_capital=100_000.0)
    g.on_bar(TS[0], 100_000.0, open_positions=cfg.risk.max_open_positions)
    ok, why = g.can_open()
    assert not ok and why == "max_open_positions"
    g.on_bar(TS[1], 100_000.0, open_positions=cfg.risk.max_open_positions - 1)
    assert g.can_open()[0]
