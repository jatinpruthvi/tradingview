"""Tests for the 10 ORB profitability enhancements, execution, sizing, and variants."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tvlab.backtest.engine import Backtester
from tvlab.config import CONFIG_DIR, load_config
from tvlab.pipeline import run_study
from tvlab.portfolio import causal_rs_mask
from tvlab.risk.sizing import Sizer
from tvlab.strategy import get
from tvlab.validation import lookahead as LA
from tvlab.variants import VariantSpec, orb_improvement_variants, run_variants

BARS = 75
RANGE = 3


def _index(sessions=5, bars=BARS, start="2026-01-05"):
    out = []
    for d in pd.bdate_range(start, periods=sessions):
        base = d + pd.Timedelta(hours=9, minutes=15)
        out.extend(base + pd.Timedelta(minutes=5 * b) for b in range(bars))
    return pd.DatetimeIndex(out)


def _orb_frame(sessions=5, direction="up", wick_only_at=None, break_bar=RANGE,
               range_half=0.5, break_step=0.5, gap_offset=0.0):
    rows = []
    prev_close = 100.0
    for s in range(sessions):
        base = (prev_close + gap_offset) if s > 0 else 100.0
        for b in range(BARS):
            if b < RANGE:
                o = c = base
                hi, lo = base + range_half, base - range_half
            elif wick_only_at is not None and b == wick_only_at:
                o = base
                c = base + (0.1 if direction == "up" else -0.1)
                hi = base + range_half + 0.5 if direction == "up" else base + 0.2
                lo = base - range_half - 0.5 if direction == "down" else base - 0.2
            elif b >= break_bar:
                step = (b - break_bar + 1) * (break_step if direction == "up" else -break_step)
                c = base + range_half * (1 if direction == "up" else -1) + step
                o = c - (break_step if direction == "up" else -break_step)
                hi = max(o, c) + 0.05
                lo = min(o, c) - 0.05
            else:
                o = c = base
                hi, lo = base + range_half * 0.6, base - range_half * 0.6
            rows.append({"open": o, "high": hi, "low": lo, "close": c, "volume": 10_000.0})
        prev_close = rows[-1]["close"]
    return pd.DataFrame(rows, index=_index(sessions))


# ------------------------------------------------ confirm_close & chase & tod
def test_confirm_close_ignores_intrabar_wick_fakeout():
    df = _orb_frame(sessions=2, direction="up", wick_only_at=RANGE, break_bar=RANGE + 2)
    wick_sig = get("orb")({"confirm_close": False}).signals(df)
    close_sig = get("orb")({"confirm_close": True}).signals(df)

    # Without confirm_close it fires immediately on the wick bar (index RANGE)
    assert wick_sig["signal"].iloc[RANGE] == 1.0
    # With confirm_close it skips the wick bar and enters on bar RANGE + 2
    assert close_sig["signal"].iloc[RANGE] == 0.0
    assert close_sig["signal"].iloc[RANGE + 2] == 1.0


def test_max_entry_tod_skips_late_day_breakouts():
    # Break occurs at bar 40 (09:15 + 200m = 12:35), past 11:30 cutoff
    df = _orb_frame(sessions=2, direction="up", break_bar=40)
    unfiltered = get("orb")({}).signals(df)
    cutoff = get("orb")({"max_entry_tod": "11:30"}).signals(df)
    assert (unfiltered["signal"] == 1.0).sum() == 2
    assert (cutoff["signal"] == 0.0).all()


def test_max_chase_ratio_skips_overextended_breakout_bar():
    # Opening range width = 1.0 (99.5..100.5); break_step=2.0 -> close > rhi + 0.75 * 1.0
    df = _orb_frame(sessions=2, direction="up", range_half=0.5, break_step=2.0)
    sig = get("orb")({"confirm_close": True, "max_chase_ratio": 0.75}).signals(df)
    assert (sig["signal"] == 0.0).all()


# -------------------------------------------- range width & overnight gap gates
def test_min_range_pct_skips_tight_ranges():
    # range width = 0.2 / 100 = 0.2% < 0.5%
    df = _orb_frame(sessions=2, direction="up", range_half=0.1, break_step=0.5)
    sig = get("orb")({"min_range_pct": 0.005}).signals(df)
    assert (sig["signal"] == 0.0).all()


def test_max_range_atr_and_max_gap_atr_skip_exhausted_sessions():
    # Session 0 has range ~1.0; session 1 opens with a massive +25 gap and range 10
    df0 = _orb_frame(sessions=2, direction="up", range_half=0.5, break_step=0.05)
    # Build a 3-session frame where session 2 has a huge gap
    df_gap = _orb_frame(sessions=3, direction="up", range_half=0.5, break_step=0.02, gap_offset=15.0)
    sig_gap = get("orb")({"max_gap_atr": 1.5}).signals(df_gap)
    # Session 0 has no prior session -> trades; sessions 1 & 2 gap +15 >> 1.5x prior range -> skipped
    assert sig_gap["signal"].iloc[:BARS].sum() == 1.0
    assert sig_gap["signal"].iloc[BARS:].sum() == 0.0

    # Huge opening range on session 1 relative to session 0
    df_wide = df0.copy()
    s1 = slice(BARS, BARS + RANGE)
    df_wide.iloc[s1, df_wide.columns.get_loc("high")] = 120.0
    df_wide.iloc[s1, df_wide.columns.get_loc("low")] = 90.0
    sig_wide = get("orb")({"max_range_atr": 1.5}).signals(df_wide)
    assert sig_wide["signal"].iloc[:BARS].sum() == 1.0
    assert sig_wide["signal"].iloc[BARS:].sum() == 0.0


# --------------------------------------------------- stop_mode & r_multiple
def test_midpoint_stop_and_r_multiple_target():
    df = _orb_frame(sessions=2, direction="up", range_half=2.0, break_step=0.5)
    # range = [98, 102], width = 4.0 (~4% >= 1.5%), midpoint = 100.0
    sig = get("orb")({
        "stop_mode": "midpoint_on_wide",
        "wide_range_pct": 0.015,
        "r_multiple": 2.5,
    }).signals(df)
    first = int(np.flatnonzero(sig["signal"] == 1.0)[0])
    assert sig["stop"].iloc[first] == pytest.approx(100.0)
    ref = df["close"].iloc[first]
    expected_tgt = ref + 2.5 * (ref - 100.0)
    assert sig["target"].iloc[first] == pytest.approx(expected_tgt)


# ------------------------------------------------ bias & market regime filters
def test_use_bias_and_market_filter_gate_countertrend_breaks():
    df = _orb_frame(sessions=3, direction="up", range_half=0.5, break_step=0.5)
    # Bearish regime on all bars blocks long breakouts
    regime = df.copy()
    regime["close"] = 90.0
    regime["high"] = 100.0
    regime["low"] = 95.0
    # Make opening range [95, 100] and post-range close 90 (< 95 -> bearish regime -1)
    for s in range(3):
        regime.iloc[s * BARS : s * BARS + RANGE, regime.columns.get_loc("close")] = 98.0

    sig_blocked = get("orb")({"market_filter": True}).signals(df, regime=regime)
    assert (sig_blocked["signal"] == 0.0).all()

    # Bullish regime (+1) allows long breakouts
    regime_bull = regime.copy()
    for s in range(3):
        regime_bull.iloc[s * BARS + RANGE : (s + 1) * BARS, regime_bull.columns.get_loc("close")] = 105.0
    sig_ok = get("orb")({"market_filter": True}).signals(df, regime=regime_bull)
    assert (sig_ok["signal"] == 1.0).sum() == 3


# ------------------------------------------------ limit entry in Backtester
def test_limit_entry_fills_only_when_next_bar_touches_level_with_zero_slippage():
    df = _orb_frame(sessions=2, direction="up", range_half=0.5, break_step=0.5)
    # Break bar is RANGE (close 101.0, rhi=100.5); next bar RANGE+1 has low > 100.5 so limit misses
    df_miss = df.copy()
    df_miss.iloc[RANGE + 1, df_miss.columns.get_loc("open")] = 101.2
    df_miss.iloc[RANGE + 1, df_miss.columns.get_loc("low")] = 101.0

    strat = get("orb")({"limit_entry": True})
    sig = strat.signals(df_miss)
    assert "entry_price" in sig.columns

    cfg = load_config(market="indian_equities", style="intraday").merge({
        "backtest.use_limit_entries": True,
        "risk.allow_short": True,
    })
    res_miss = Backtester(cfg).run(df_miss.iloc[:BARS], sig.iloc[:BARS], symbol="T",
                                   max_bars_in_trade=None)
    assert res_miss.trades.empty
    assert res_miss.skipped.get("limit_not_filled", 0) == 1

    # Now let bar RANGE+1 dip to rhi=100.5 -> fills cleanly at 100.5 with 0 slippage
    df_hit = df_miss.copy()
    df_hit.iloc[RANGE + 1, df_hit.columns.get_loc("low")] = 100.4
    res_hit = Backtester(cfg).run(df_hit.iloc[:BARS], sig.iloc[:BARS], symbol="T",
                                  max_bars_in_trade=None)
    assert len(res_hit.trades) == 1
    assert res_hit.trades.iloc[0]["entry_price"] == pytest.approx(100.5)


# ----------------------------------------- sizing clipping diagnostics
def test_sizer_reports_clipped_flag_and_backtester_warns():
    cfg = load_config(market="indian_equities", style="intraday")
    sizer = Sizer(cfg.risk)
    # 0.4% stop with 0.5% risk wants 125% notional, clipped to max_position_pct=25%
    res = sizer.size(equity=100_000.0, price=100.0, stop=99.6, side=1)
    assert res.accepted
    assert res.clipped is True
    assert res.unclipped_notional == pytest.approx(125_000.0)
    assert res.notional == pytest.approx(25_000.0)

    df = _orb_frame(sessions=3, direction="up", range_half=0.2, break_step=0.2)
    sig = get("orb")({}).signals(df)
    bt_res = Backtester(cfg).run(df, sig, symbol="T", max_bars_in_trade=None)
    assert any("clipped by max_position_pct" in w for w in bt_res.warnings)


# -------------------------------------- causal RS mask & holdout slicing
def test_causal_rs_mask_uses_strictly_prior_sessions():
    idx = _index(sessions=6, bars=10)
    # Strong leader vs weak laggard over sessions 0..4, then reversed on session 5
    closes_a = []
    closes_b = []
    for s in range(6):
        val_a = 100.0 + s * 5.0 if s < 5 else 80.0
        val_b = 100.0 - s * 2.0 if s < 5 else 150.0
        closes_a.extend([val_a] * 10)
        closes_b.extend([val_b] * 10)
    df_a = pd.DataFrame({"open": closes_a, "high": closes_a, "low": closes_a,
                         "close": closes_a, "volume": 1000.0}, index=idx)
    df_b = pd.DataFrame({"open": closes_b, "high": closes_b, "low": closes_b,
                         "close": closes_b, "volume": 1000.0}, index=idx)

    masks = causal_rs_mask({"A": df_a, "B": df_b}, lookback_sessions=2, top_pct=0.5)
    # On session 5 (even though A crashes and B rallies during session 5),
    # day 5's eligibility was fixed at session 4's close -> A is leader, B is laggard.
    s5 = slice(50, 60)
    assert masks["A"]["allow_long"].iloc[s5].all()
    assert not masks["B"]["allow_long"].iloc[s5].any()
    assert masks["B"]["allow_short"].iloc[s5].all()


# ---------------------------------------------- lookahead audit on enhanced ORB
def test_enhanced_orb_with_all_filters_is_strictly_causal():
    idx = _index(sessions=25, bars=BARS)
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, 0.002, len(idx))
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close * 1.003,
        "low": close * 0.997,
        "close": close,
        "volume": 50_000.0,
    }, index=idx)
    df.iloc[0, df.columns.get_loc("open")] = 100.0

    strat = get("orb")({
        "confirm_close": True,
        "max_entry_tod": "11:30",
        "min_range_pct": 0.002,
        "max_range_atr": 2.0,
        "max_gap_atr": 2.0,
        "max_chase_ratio": 1.0,
        "stop_mode": "midpoint_on_wide",
        "r_multiple": 2.0,
        "limit_entry": True,
        "use_bias": True,
        "market_filter": True,
    })
    cuts = [BARS * 8 + 15, BARS * 15 + 30, BARS * 20 + 5]
    r = LA.check_function(lambda d: strat.signals(d, regime=d), df,
                          target="orb_enhanced_all", cuts=cuts)
    assert r.ok, f"Enhanced ORB leaked future bars: {r.leaks}"


# ----------------------------------------- preset config & variants runner
def test_indian_orb_mis_preset_loads_and_variants_run():
    preset_path = CONFIG_DIR / "presets" / "indian_orb_mis.yaml"
    assert preset_path.exists()
    cfg = load_config(market="indian_equities", style="intraday", config_path=preset_path)
    assert cfg.risk.allow_short is True
    assert cfg.risk.max_position_pct == 0.40
    assert cfg.risk.max_leverage == 2.0
    assert cfg.backtest.use_limit_entries is True
    assert cfg.strategy.name == "orb"
    assert cfg.strategy.params["confirm_close"] is True

    # Run variants runner on fast synthetic intraday data
    syn_cfg = load_config(market="indian_equities", style="intraday").merge({
        "data.synthetic_bars": 800,
    })
    specs = [
        VariantSpec("baseline", {"strategy.name": "orb"}),
        VariantSpec("confirm_close", {"strategy.name": "orb", "strategy.params.confirm_close": True}),
    ]
    out = run_variants(syn_cfg, specs, symbols=["RELIANCE.NS", "TCS.NS"])
    assert len(out) == 2
    assert list(out["variant"]) == ["baseline", "confirm_close"]
    assert (out["trials"] == 2).all()
    assert len(orb_improvement_variants()) >= 10
