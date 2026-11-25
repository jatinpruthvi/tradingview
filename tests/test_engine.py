"""Backtester correctness. Each regression test names the bug it prevents."""
import numpy as np
import pandas as pd
import pytest

from tvlab.backtest.engine import UNSET, Backtester
from tvlab.config import load_config
from tvlab.data.synthetic import basket
from tvlab.strategy import get
from tvlab.strategy.base import blank_signals


@pytest.fixture(scope="module")
def cfg():
    return load_config(market="crypto", style="swing")


def flat_prices(n=200, price=100.0, start="2022-01-01", band=0.01):
    """A flat series whose bars span price*(1 +- band).

    ``band`` must stay inside the config's stop-distance bounds
    (min 0.15%, max 25% of price) or the sizer rejects the trade for a reason
    unrelated to whatever the test is checking.
    """
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": price, "high": price * (1 + band), "low": price * (1 - band),
                         "close": price, "volume": 1000.0}, index=idx)


def set_wide_bar(df, i, high, low):
    """Widen ONE bar so it can touch both a stop and a target."""
    df.loc[df.index[i], "high"] = high
    df.loc[df.index[i], "low"] = low
    return df


def set_gap_bar(df, i, open_, high, low):
    """Move a bar's open through a level, the way an overnight gap does."""
    df.loc[df.index[i], "open"] = open_
    df.loc[df.index[i], "high"] = high
    df.loc[df.index[i], "low"] = low
    return df


# stops/targets must respect risk.min_stop_distance_pct (0.15%) and
# risk.max_stop_distance_pct (25%): 2% and 4% are comfortably inside both.
STOP = 98.0
TARGET = 104.0


def test_entry_happens_on_the_next_bars_open(cfg):
    """The core no-lookahead guarantee of the execution model."""
    df = flat_prices(100, 100.0)
    sig = blank_signals(df)
    sig.loc[df.index[40], "signal"] = 1.0
    sig.loc[df.index[40], "stop"] = STOP
    sig.loc[df.index[40], "target"] = TARGET
    res = Backtester(cfg).run(df, sig, symbol="T")
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["entry_time"] == df.index[41]


def test_costs_are_charged_both_sides(cfg):
    """A round trip on a flat market must lose exactly the round-trip cost."""
    df = flat_prices(100, 100.0, band=0.001)      # tight bars: stop/target never hit
    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = 1.0
    sig.loc[df.index[10], "stop"] = STOP
    sig.loc[df.index[10], "target"] = TARGET
    res = Backtester(cfg).run(df, sig, symbol="T")
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "end_of_data"
    assert t["fees"] > 0
    rt = (cfg.costs.fee_bps + cfg.costs.slippage_bps) * 2 / 1e4
    assert t["pnl"] < 0
    assert abs(t["pnl"]) / t["notional"] == pytest.approx(rt, rel=0.15)


def test_pessimistic_intrabar_fill_takes_the_stop(cfg):
    """When one bar touches both stop and target, the stop wins by default."""
    df = flat_prices(60, 100.0, band=0.001)
    sig = blank_signals(df)
    sig.loc[df.index[20], "signal"] = 1.0         # -> entered at bar 21's open
    sig.loc[df.index[20], "stop"] = STOP
    sig.loc[df.index[20], "target"] = TARGET
    # bar 22 (the first bar the open position is managed on) spans 90..110, so it
    # touches the stop at 98 AND the target at 104 in the same bar
    set_wide_bar(df, 22, 110.0, 90.0)

    pess = Backtester(cfg).run(df, sig, symbol="T")
    assert pess.trades.iloc[0]["exit_reason"] == "stop_loss"

    opt = cfg.merge({"backtest.intrabar_fill": "optimistic"})
    assert Backtester(opt).run(df, sig, symbol="T").trades.iloc[0]["exit_reason"] == "take_profit"


def test_a_gap_through_the_stop_fills_at_the_open(cfg):
    """A stop is a market order once triggered: if the bar OPENS below it, the
    market never traded at the stop. Filling there invents a price nobody could
    have got - and on a limit-down open it is pure fiction."""
    df = flat_prices(60, 100.0, band=0.001)
    sig = blank_signals(df)
    sig.loc[df.index[20], "signal"] = 1.0        # entered at bar 21's open
    sig.loc[df.index[20], "stop"] = STOP
    sig.loc[df.index[20], "target"] = TARGET
    set_gap_bar(df, 22, 94.0, 95.0, 92.0)        # opens 4 below a 98 stop

    t = Backtester(cfg).run(df, sig, symbol="T").trades.iloc[0]
    slip = cfg.costs.slippage_bps / 1e4
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(94.0 * (1 - slip)), \
        "filled at the open, with adverse slippage - not at the 98 stop"
    assert t["r_multiple"] < -2.0, "a 4-point gap on a 2-point risk is worse than -1R"


def test_a_gap_through_the_stop_beats_an_optimistic_target(cfg):
    """The stop triggers at the open, before anything intrabar. A bar that gaps
    through the stop and later trades up through the target was already flat."""
    df = flat_prices(60, 100.0, band=0.001)
    sig = blank_signals(df)
    sig.loc[df.index[20], "signal"] = 1.0
    sig.loc[df.index[20], "stop"] = STOP
    sig.loc[df.index[20], "target"] = TARGET
    set_gap_bar(df, 22, 94.0, 106.0, 93.0)       # opens through the stop, then rallies

    opt = cfg.merge({"backtest.intrabar_fill": "optimistic"})
    t = Backtester(opt).run(df, sig, symbol="T").trades.iloc[0]
    slip = cfg.costs.slippage_bps / 1e4
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(94.0 * (1 - slip))


def test_a_bar_that_only_dips_intrabar_still_fills_at_the_stop(cfg):
    """The gap rule must not become 'fill at the open whenever the stop is hit'."""
    df = flat_prices(60, 100.0, band=0.001)
    sig = blank_signals(df)
    sig.loc[df.index[20], "signal"] = 1.0
    sig.loc[df.index[20], "stop"] = STOP
    sig.loc[df.index[20], "target"] = TARGET
    set_gap_bar(df, 22, 100.0, 101.0, 97.0)      # opens above, dips through the stop

    t = Backtester(cfg).run(df, sig, symbol="T").trades.iloc[0]
    slip = cfg.costs.slippage_bps / 1e4
    assert t["exit_price"] == pytest.approx(STOP * (1 - slip))


def test_a_short_gapped_through_its_stop_fills_at_the_open(cfg):
    """Symmetric: a gap UP through a short's stop fills above it, not at it."""
    df = flat_prices(60, 100.0, band=0.001)
    sig = blank_signals(df)
    sig.loc[df.index[20], "signal"] = -1.0
    sig.loc[df.index[20], "stop"] = 102.0
    sig.loc[df.index[20], "target"] = 96.0
    set_gap_bar(df, 22, 105.0, 106.0, 104.0)

    t = Backtester(cfg).run(df, sig, symbol="T").trades.iloc[0]
    slip = cfg.costs.slippage_bps / 1e4
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(105.0 * (1 + slip))
    assert t["r_multiple"] < -1.0


def test_shorts_are_not_silently_rejected(cfg):
    """Regression: SizingResult.accepted tested units > 0, and shorts have
    negative units, so every short was skipped with reason 'ok'."""
    df = flat_prices(100, 100.0)
    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = -1.0
    sig.loc[df.index[10], "stop"] = 102.0
    sig.loc[df.index[10], "target"] = 96.0
    res = Backtester(cfg).run(df, sig, symbol="T")
    assert res.skipped.get("ok", 0) == 0, f"short was rejected: {res.skipped}"
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["side"] == -1
    assert res.trades.iloc[0]["units"] < 0


def test_shorts_profit_when_price_falls(cfg):
    df = flat_prices(100, 100.0)
    for i in range(30, 100):
        df.iloc[i, df.columns.get_loc("close")] = 80.0
        df.iloc[i, df.columns.get_loc("open")] = 80.0
        df.iloc[i, df.columns.get_loc("high")] = 81.0
        df.iloc[i, df.columns.get_loc("low")] = 79.0
    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = -1.0
    sig.loc[df.index[10], "stop"] = 102.0
    sig.loc[df.index[10], "target"] = 79.0
    res = Backtester(cfg).run(df, sig, symbol="T")
    assert res.trades.iloc[0]["pnl"] > 0


def test_shorts_can_be_disabled(cfg):
    c = cfg.merge({"risk.allow_short": False})
    df = flat_prices(60, 100.0)
    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = -1.0
    sig.loc[df.index[10], "stop"] = 102.0
    res = Backtester(c).run(df, sig, symbol="T")
    assert len(res.trades) == 0
    assert res.skipped.get("shorts_disabled", 0) == 1


def test_infinite_stop_means_no_time_or_stop_exit(cfg):
    """Regression: -inf was treated as 'missing' and replaced by an ATR stop."""
    c = cfg.merge({"backtest.max_bars_in_trade": 10})
    df = flat_prices(200, 100.0)
    sig = blank_signals(df)
    sig.loc[df.index[5], "signal"] = 1.0
    sig.loc[df.index[5], "stop"] = -np.inf
    sig.loc[df.index[5], "target"] = np.inf
    res = Backtester(c).run(df, sig, symbol="T", max_bars_in_trade=None)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "end_of_data"
    assert t["bars_held"] > 100


def test_unset_falls_back_to_config_but_none_means_no_limit(cfg):
    c = cfg.merge({"backtest.max_bars_in_trade": 5})
    df = flat_prices(60, 100.0)
    sig = blank_signals(df)
    sig.loc[df.index[2], "signal"] = 1.0
    sig.loc[df.index[2], "stop"] = STOP
    sig.loc[df.index[2], "target"] = TARGET
    bt = Backtester(c)
    assert bt.run(df, sig, symbol="T").trades.iloc[0]["exit_reason"] == "time_exit"
    assert bt.run(df, sig, symbol="T", max_bars_in_trade=None
                  ).trades.iloc[0]["exit_reason"] == "end_of_data"
    assert bt.run(df, sig, symbol="T", max_bars_in_trade=UNSET
                  ).trades.iloc[0]["exit_reason"] == "time_exit"


def test_max_drawdown_halt_stops_trading(cfg):
    """The governor must stop opening positions once the DD limit is breached."""
    c = cfg.merge({"risk.max_drawdown_pct": 0.01, "risk.risk_per_trade": 0.05})
    df = flat_prices(200, 100.0)
    for i in range(5, 200):                      # steady bleed
        px = 100.0 * (1 - 0.004 * (i - 4))
        df.iloc[i, df.columns.get_loc("close")] = px
        df.iloc[i, df.columns.get_loc("open")] = px
        df.iloc[i, df.columns.get_loc("high")] = px * 1.001
        df.iloc[i, df.columns.get_loc("low")] = px * 0.999
    sig = blank_signals(df)
    sig["signal"] = 1.0
    sig["stop"] = df["close"] * 0.98      # inside the max-stop-distance cap
    sig["target"] = df["close"] * 1.04
    res = Backtester(c).run(df, sig, symbol="T")
    assert res.equity.iloc[-1] < c.backtest.initial_capital
    assert len(res.trades) < 100, "trading continued after the drawdown halt"


def test_no_trades_is_reported_not_silently_zero(cfg):
    df = flat_prices(100, 100.0)
    res = Backtester(cfg).run(df, blank_signals(df), symbol="T")
    assert len(res.trades) == 0
    assert res.warnings, "a zero-trade run must warn, not look like a flat success"
    assert np.allclose(res.equity.to_numpy(), cfg.backtest.initial_capital)


def test_equity_is_marked_to_market(cfg):
    """Unrealised P&L must show up in equity, or drawdown is understated."""
    df = flat_prices(100, 100.0, band=0.001)
    for i in range(30, 100):
        df.iloc[i, df.columns.get_loc("close")] = 120.0
        df.iloc[i, df.columns.get_loc("open")] = 120.0
        df.iloc[i, df.columns.get_loc("high")] = 120.12
        df.iloc[i, df.columns.get_loc("low")] = 119.88
    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = 1.0
    sig.loc[df.index[10], "stop"] = STOP
    # +inf means "no target", so the position rides the move to 120 instead of
    # being taken out at TARGET on the very first bar of the rally.
    sig.loc[df.index[10], "target"] = np.inf
    res = Backtester(cfg).run(df, sig, symbol="T")
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "end_of_data"
    # unrealised P&L must appear bar by bar, not only at the exit
    assert res.equity.iloc[20] == pytest.approx(cfg.backtest.initial_capital, rel=2e-3)
    assert res.equity.iloc[50] > res.equity.iloc[20]
    # price is constant from bar 30 on, so a correctly marked-to-market curve
    # must be flat there too - it should not drift or keep accruing
    assert res.equity.iloc[90] == pytest.approx(res.equity.iloc[50], rel=1e-9)
    # a ~20% price move on ~25% of equity is roughly +5% on the account
    assert res.equity.iloc[90] > cfg.backtest.initial_capital * 1.02


def test_real_strategies_produce_trades(cfg):
    df = basket(["BTC/USDT"], n_bars=1200, timeframe="4h")["BTC/USDT"]
    for name in ("smc_trend", "ema_pullback", "donchian_breakout"):
        strat = get(name)({})
        res = Backtester(cfg).run(df, strat.signals(df), symbol="BTC/USDT")
        assert len(res.trades) > 0, f"{name} produced no trades"
        longs = (res.trades["side"] > 0).sum()
        shorts = (res.trades["side"] < 0).sum()
        assert longs + shorts == len(res.trades)


def test_r_multiple_is_consistent_with_stop_distance(cfg):
    df = flat_prices(100, 100.0, band=0.001)
    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = 1.0
    sig.loc[df.index[10], "stop"] = 98.0          # risk = 2.0 from a 100 entry
    sig.loc[df.index[10], "target"] = 106.0       # reward = 6.0 => a clean +3R
    set_wide_bar(df, 12, 107.0, 99.5)             # first managed bar hits the target
    res = Backtester(cfg).run(df, sig, symbol="T")
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "take_profit"
    # entry slippage widens the realised risk (98 vs a 100.02 fill), so the
    # realised R lands just under the clean +3R. Fees do not enter r_multiple.
    assert 2.80 < t["r_multiple"] < 3.00, t["r_multiple"]


# --------------------------------------------------------------- trailing exits
# Both rules are opt-in. Their whole value is that they can only ever move the
# stop in the trade's favour, and that the move lands on the NEXT bar - a bar
# that raises the stop with its own high and then gets stopped by its own low
# would be reading an intrabar order nobody knows.

def _atr(df, value=1.0):
    return pd.Series(value, index=df.index, name="atr")


def test_trailing_stop_follows_the_best_price_and_cuts_the_trade_early(cfg):
    """A trail replaces the fixed stop once price has run, so the round trip
    ends at the trail instead of giving the whole move back to the original stop."""
    df = flat_prices(60, 100.0, band=0.002)
    # entry lands on bar 11's open; bar 12 is the first managed bar
    set_wide_bar(df, 12, 108.0, 100.0)            # best = 108, trail -> 106
    df.loc[df.index[12], "close"] = 107.0
    set_gap_bar(df, 13, 106.5, 107.0, 105.0)      # opens above the trail, dips through it

    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = 1.0
    sig.loc[df.index[10], "stop"] = STOP          # 98.0, risk 2.0
    sig.loc[df.index[10], "target"] = 120.0       # far enough not to fire

    trail = cfg.merge({"backtest.trail_atr_mult": 2.0})
    res = Backtester(trail).run(df, sig, symbol="T", atr=_atr(df, 1.0))
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_time"] == df.index[13]
    # trail after bar 12 = 108 - 2*1 = 106; exit at that level less exit slippage
    assert 105.9 < t["exit_price"] < 106.0, t["exit_price"]
    # R is measured against the ORIGINAL 2.0 of risk, so a trailed winner is > 0
    assert t["r_multiple"] > 2.5, t["r_multiple"]
    assert t["stop"] == STOP                     # the risk basis, not the trail

    # the same bars without the trail never reach 98, so nothing is stopped
    base = Backtester(cfg).run(df, sig, symbol="T")
    assert base.trades.iloc[0]["exit_reason"] != "stop_loss"


def test_breakeven_moves_the_stop_to_entry_and_R_uses_the_original_risk(cfg):
    """If R were measured against the moved stop, the denominator would collapse
    to zero and every breakeven exit would report an enormous R."""
    df = flat_prices(60, 100.0, band=0.002)
    set_wide_bar(df, 12, 102.5, 100.0)            # +1R reached
    df.loc[df.index[12], "close"] = 102.0
    set_gap_bar(df, 13, 101.5, 101.8, 99.5)

    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = 1.0
    sig.loc[df.index[10], "stop"] = STOP          # risk 2.0 => 1R at 102
    sig.loc[df.index[10], "target"] = 120.0

    be = cfg.merge({"backtest.breakeven_at_r": 1.0})
    res = Backtester(be).run(df, sig, symbol="T")
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_time"] == df.index[13]
    assert 99.9 < t["exit_price"] < 100.1, t["exit_price"]     # stopped at entry
    assert -0.1 < t["r_multiple"] < 0.1, t["r_multiple"]       # ~0R, not ~ +inf
    assert t["stop"] == STOP


def test_a_ratcheted_stop_is_not_applied_to_the_bar_that_raised_it(cfg):
    """One bar raises the trail with its own high and dips below it with its own
    low. Applying the trail inside that bar would exit at 106 on an intrabar path
    we cannot know; the stop in force at the open is 98, and 99 never touches it."""
    df = flat_prices(60, 100.0, band=0.002)
    # ONE bar: high 108 would trail the stop to 106, low 99 would then hit it -
    # but 99 is above the only stop in force when the bar opened, at 98
    set_wide_bar(df, 12, 108.0, 99.0)
    df.loc[df.index[12], "close"] = 107.0

    sig = blank_signals(df)
    sig.loc[df.index[10], "signal"] = 1.0
    sig.loc[df.index[10], "stop"] = STOP
    sig.loc[df.index[10], "target"] = 120.0

    trail = cfg.merge({"backtest.trail_atr_mult": 2.0})
    res = Backtester(trail).run(df, sig, symbol="T", atr=_atr(df, 1.0))
    t = res.trades.iloc[0]
    assert t["exit_time"] != df.index[12], "the bar raised and hit its own trail"
    assert t["exit_time"] > df.index[12]


def test_trailing_is_off_unless_configured(cfg):
    """Default config must keep the stop the signal chose, or every study
    already published would silently change."""
    assert cfg.backtest.trail_atr_mult is None
    assert cfg.backtest.breakeven_at_r is None
