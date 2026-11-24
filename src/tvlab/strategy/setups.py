"""The strategies. Three of them, on purpose.

``smc_trend`` is the intended edge: a liquidity sweep into a confirmed swing, in
the direction of the higher-timeframe trend, with the stop beyond the sweep and
the target at the next resting liquidity pool. ``ema_pullback`` and
``donchian_breakout`` are deliberately dull baselines - if the SMC setup cannot
beat a 20-line moving-average pullback out of sample, it is not an edge, it is
decoration. Always run the baselines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..indicators.smc import liquidity_pool, smc_features
from ..indicators.technical import atr, donchian, ema, rsi
from .base import Strategy, blank_signals, register, trend_filter


@register("smc_trend")
class SmcTrendStrategy(Strategy):
    """Liquidity-sweep reversal in the direction of the higher-timeframe trend.

    The setup, in words
    -------------------
    1. **Bias.** The higher timeframe must be trending (close vs EMA, ADX above a
       floor, +DI vs -DI). We only take sweeps that point the same way.
    2. **Trigger.** Price wicks through a *confirmed* swing extreme - taking the
       stop cluster resting there - and closes back inside. That is the sweep.
    3. **Confirmation.** Either a CHoCH/BOS in the bias direction, or price
       retracing into an unmitigated fair-value gap / order block left behind by
       the impulse. Confirmation avoids buying a knife mid-cascade.
    4. **Stop.** Beyond the sweep extreme plus an ATR buffer, so normal noise
       cannot tag it.
    5. **Target.** The next pool of resting liquidity on the far side, or a fixed
       R-multiple, whichever is nearer. Taking the nearer one is what keeps the
       realised win rate honest.
    """

    defaults: Dict[str, Any] = {
        "swing_length": 12,
        "atr_period": 14,
        "atr_stop_mult": 1.8,
        "atr_stop_buffer": 0.25,
        "r_multiple": 2.5,
        "bias_ema": 50,
        "bias_mode": "ema_slope",
        "bias_slope_lookback": 5,
        "bias_adx_period": 14,
        "bias_adx_min": 22.0,
        # "sweep" is the doctrinal setup and the default. "sweep_or_zone" also
        # accepts a retracement into an unmitigated FVG/order block.
        #
        # MEASURED, not a hunch: tests/test_entry_mode.py re-runs both modes over
        # the crypto/swing universe and asserts the direction. Across 10 samples
        # (5 seeds x 2500/4000 bars) zone entries multiplied the opportunity set
        # 10-23x and made total return, Sharpe and max drawdown worse in 10 of 10.
        #
        # Profit factor is NOT the way to judge this, and an earlier revision of
        # this comment did exactly that, quoting "~0.9 to ~0.58 over 93 trades" -
        # numbers that reproduce on no seed at any bar count tested. PF moved in
        # both directions (worse in 5 of 10, better in 5; e.g. seed 99 at 2500 bars
        # goes 0.18 -> 0.58 while the return goes -1.01% -> -4.33%). PF is gross
        # profit over gross loss: it is blind to how often capital is deployed and
        # to fee drag, so taking 17x more trades at a similar PF is a much worse
        # outcome that PF itself does not report.
        #
        # Re-measure on your own data before switching, and do not switch just to
        # get more trades. If you do, read the return and the drawdown, not the PF.
        "entry_mode": "sweep",
        "require_fvg": False,
        "require_structure": True,
        "liquidity_sweep_lookback": 20,
        "max_sweep_age": 4,
        "max_confirm_age": 8,
        "require_confirmation": True,
        "use_liquidity_target": True,
        "max_bars_in_trade": 90,
        "rsi_filter": True,
        "rsi_period": 14,
        "rsi_long_max": 70.0,
        "rsi_short_min": 30.0,
    }

    grid: Dict[str, tuple] = {
        "swing_length": (8, 12, 16),
        "atr_stop_mult": (1.2, 1.8, 2.5),
        "r_multiple": (1.8, 2.5, 3.5),
        "bias_adx_min": (18.0, 22.0, 26.0),
        "max_sweep_age": (2, 4, 6),
    }

    def signals(self, df: pd.DataFrame, bias: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        p = self.params
        out = blank_signals(df)

        feats = smc_features(
            df,
            swing_length=int(p["swing_length"]),
            ob_lookback=int(p["liquidity_sweep_lookback"]),
        )
        a = atr(df, int(p["atr_period"]))
        trend = trend_filter(df, bias, int(p["bias_ema"]), int(p["bias_adx_period"]),
                             float(p["bias_adx_min"]), mode=str(p["bias_mode"]),
                             slope_lookback=int(p["bias_slope_lookback"]))

        sweep_low = feats["sweep_low"].to_numpy()
        sweep_high = feats["sweep_high"].to_numpy()
        # per-side ages: a bearish sweep must never satisfy a long's freshness test
        bull_sweep_age = feats["bull_sweep_age"].to_numpy(dtype="float64")
        bear_sweep_age = feats["bear_sweep_age"].to_numpy(dtype="float64")
        choch = feats["choch"].to_numpy()
        bos = feats["bos"].to_numpy()
        choch_age = feats["choch_age"].to_numpy(dtype="float64")
        bos_age = feats["bos_age"].to_numpy(dtype="float64")
        structure = feats["structure"].to_numpy()
        fvg_zone = feats["fvg_in_zone"].to_numpy()
        ob_zone = feats["ob_in_zone"].to_numpy()
        bull_fvg = feats["bull_fvg"].to_numpy()
        bear_fvg = feats["bear_fvg"].to_numpy()

        close = df["close"].to_numpy(dtype="float64")
        low = df["low"].to_numpy(dtype="float64")
        high = df["high"].to_numpy(dtype="float64")
        a_arr = a.to_numpy(dtype="float64")
        # hoisted out of the loop: .to_numpy() per bar was O(n^2) work
        last_low_lvl = feats["last_low_level"].to_numpy(dtype="float64")
        last_high_lvl = feats["last_high_level"].to_numpy(dtype="float64")

        rsi_arr = (rsi(df, int(p["rsi_period"])).to_numpy(dtype="float64")
                   if p["rsi_filter"] else None)

        # liquidity targets: where the next resting stop cluster sits
        liq_long = (liquidity_pool(df, feats, "long")["liquidity_target"].to_numpy()
                    if p["use_liquidity_target"] else None)
        liq_short = (liquidity_pool(df, feats, "short")["liquidity_target"].to_numpy()
                     if p["use_liquidity_target"] else None)

        trend_arr = trend.to_numpy()
        sig = np.zeros(len(df))
        stop = np.full(len(df), np.nan)
        target = np.full(len(df), np.nan)
        tags = [""] * len(df)
        max_age = int(p["max_sweep_age"])
        conf_age = int(p["max_confirm_age"])
        entry_mode = str(p["entry_mode"])

        for i in range(len(df)):
            if not np.isfinite(a_arr[i]) or a_arr[i] <= 0:
                continue

            # ---------- LONG: sweep or zone retracement, uptrend bias ----------
            # entry_mode "sweep" needs a fresh liquidity sweep; "sweep_or_zone"
            # also accepts a retracement into an unmitigated FVG/order block,
            # which is the other half of the doctrine and roughly doubles the
            # opportunity set without loosening the trend requirement.
            fresh_bull_sweep = np.isfinite(bull_sweep_age[i]) and bull_sweep_age[i] <= max_age
            if entry_mode == "sweep_or_zone":
                bull_trigger = fresh_bull_sweep or bool(fvg_zone[i] or ob_zone[i])
            else:
                bull_trigger = fresh_bull_sweep
            if trend_arr[i] == 1 and bull_trigger:
                confirmed = True
                why = []
                if p["require_confirmation"]:
                    # a *fresh* structural event, or price sitting in an
                    # unmitigated FVG/order block. "structure >= 0" alone is
                    # almost always true inside an uptrend and filters nothing.
                    fresh_event = (
                        (np.isfinite(choch_age[i]) and choch_age[i] <= conf_age and choch_age[i] >= 0)
                        or (np.isfinite(bos_age[i]) and bos_age[i] <= conf_age)
                    )
                    confirmed = confirmed and bool(fresh_event or fvg_zone[i] or ob_zone[i])
                    why.append("confirm")
                if p["require_structure"]:
                    confirmed = confirmed and structure[i] >= 0
                    why.append("structure")
                if p["require_fvg"]:
                    confirmed = confirmed and bool(bull_fvg[i] or fvg_zone[i] or ob_zone[i])
                    why.append("fvg")
                if rsi_arr is not None and np.isfinite(rsi_arr[i]):
                    confirmed = confirmed and rsi_arr[i] <= p["rsi_long_max"]
                    why.append("rsi")
                if confirmed:
                    lvl = last_low_lvl[i]
                    if fresh_bull_sweep:
                        anchor = min(low[i], lvl) if np.isfinite(lvl) else low[i]
                    else:
                        # zone entry: anchor on the recent low, not a sweep extreme
                        anchor = float(np.min(low[max(0, i - max_age):i + 1]))
                    stop[i] = anchor - p["atr_stop_buffer"] * a_arr[i]
                    risk = close[i] - stop[i]
                    if risk > 0:
                        tgt_r = close[i] + p["r_multiple"] * risk
                        tgt_liq = liq_long[i] if liq_long is not None else np.nan
                        target[i] = (min(tgt_r, tgt_liq) if np.isfinite(tgt_liq) and tgt_liq > close[i]
                                     else tgt_r)
                        sig[i] = 1.0
                        trig = "sweep" if fresh_bull_sweep else "zone"
                        tags[i] = f"long_{trig}" + ("+" + ",".join(why) if why else "")

            # ---------- SHORT: bearish sweep, downtrend bias ----------
            fresh_bear_sweep = np.isfinite(bear_sweep_age[i]) and bear_sweep_age[i] <= max_age
            if entry_mode == "sweep_or_zone":
                bear_trigger = fresh_bear_sweep or bool(fvg_zone[i] or ob_zone[i])
            else:
                bear_trigger = fresh_bear_sweep
            if sig[i] == 0 and trend_arr[i] == -1 and bear_trigger:
                confirmed = True
                why = []
                if p["require_confirmation"]:
                    fresh_event = (
                        (np.isfinite(choch_age[i]) and choch_age[i] <= conf_age)
                        or (np.isfinite(bos_age[i]) and bos_age[i] <= conf_age)
                    )
                    confirmed = confirmed and bool(fresh_event or fvg_zone[i] or ob_zone[i])
                    why.append("confirm")
                if p["require_structure"]:
                    confirmed = confirmed and structure[i] <= 0
                    why.append("structure")
                if p["require_fvg"]:
                    confirmed = confirmed and bool(bear_fvg[i] or fvg_zone[i] or ob_zone[i])
                    why.append("fvg")
                if rsi_arr is not None and np.isfinite(rsi_arr[i]):
                    confirmed = confirmed and rsi_arr[i] >= p["rsi_short_min"]
                    why.append("rsi")
                if confirmed:
                    lvl = last_high_lvl[i]
                    if fresh_bear_sweep:
                        anchor = max(high[i], lvl) if np.isfinite(lvl) else high[i]
                    else:
                        anchor = float(np.max(high[max(0, i - max_age):i + 1]))
                    stop[i] = anchor + p["atr_stop_buffer"] * a_arr[i]
                    risk = stop[i] - close[i]
                    if risk > 0:
                        tgt_r = close[i] - p["r_multiple"] * risk
                        tgt_liq = liq_short[i] if liq_short is not None else np.nan
                        target[i] = (max(tgt_r, tgt_liq) if np.isfinite(tgt_liq) and tgt_liq < close[i]
                                     else tgt_r)
                        sig[i] = -1.0
                        trig = "sweep" if fresh_bear_sweep else "zone"
                        tags[i] = f"short_{trig}" + ("+" + ",".join(why) if why else "")

        out["signal"] = sig
        out["stop"] = stop
        out["target"] = target
        out["tag"] = tags
        return out


@register("ema_pullback")
class EmaPullbackStrategy(Strategy):
    """Baseline: buy the pullback to the fast EMA inside a slow-EMA uptrend.

    This is the strategy every retail system eventually reduces to. It is here so
    ``smc_trend`` has something honest to be compared against.
    """

    defaults: Dict[str, Any] = {
        "fast": 21, "slow": 50, "bias_ema": 200, "atr_period": 14,
        "atr_stop_mult": 2.0, "r_multiple": 2.0, "rsi_period": 14,
        "rsi_long_max": 60.0, "rsi_short_min": 40.0,
        "bias_adx_period": 14, "bias_adx_min": 0.0, "max_bars_in_trade": 60,
    }
    grid: Dict[str, tuple] = {
        "fast": (9, 21, 34), "slow": (50, 100),
        "atr_stop_mult": (1.5, 2.0, 3.0), "r_multiple": (1.5, 2.0, 3.0),
    }

    def signals(self, df: pd.DataFrame, bias: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        p = self.params
        out = blank_signals(df)
        close = df["close"]
        fast = ema(close, int(p["fast"]))
        slow = ema(close, int(p["slow"]))
        big = ema(close, int(p["bias_ema"]))
        a = atr(df, int(p["atr_period"]))
        r = rsi(df, int(p["rsi_period"]))

        touched_fast = (df["low"] <= fast) & (close > fast)
        touched_fast_s = (df["high"] >= fast) & (close < fast)

        long_ = (fast > slow) & (close > big) & touched_fast & (r <= p["rsi_long_max"])
        short_ = (fast < slow) & (close < big) & touched_fast_s & (r >= p["rsi_short_min"])

        sig = np.where(long_, 1.0, np.where(short_, -1.0, 0.0))
        stop = np.where(sig > 0, close - p["atr_stop_mult"] * a,
                        np.where(sig < 0, close + p["atr_stop_mult"] * a, np.nan))
        risk = np.abs(close - stop)
        target = np.where(sig > 0, close + p["r_multiple"] * risk,
                          np.where(sig < 0, close - p["r_multiple"] * risk, np.nan))

        out["signal"] = sig
        out["stop"] = stop
        out["target"] = target
        out["tag"] = np.where(sig > 0, "ema_pullback_long",
                              np.where(sig < 0, "ema_pullback_short", ""))
        return out


@register("donchian_breakout")
class DonchianBreakoutStrategy(Strategy):
    """Baseline: classic Turtle-style channel breakout with a trend filter.

    The oldest published trend-following system there is. If a new setup cannot
    beat this out of sample on the same universe, the new setup is noise.
    """

    defaults: Dict[str, Any] = {
        "donchian_period": 55, "exit_period": 20, "atr_period": 20,
        "atr_stop_mult": 3.0, "r_multiple": 4.0, "trend_ema": 200,
        "bias_ema": 200, "bias_adx_period": 14, "bias_adx_min": 18.0,
        "require_trend_filter": True, "max_bars_in_trade": 120,
    }
    grid: Dict[str, tuple] = {
        "donchian_period": (20, 55, 90), "exit_period": (10, 20),
        "atr_stop_mult": (2.0, 3.0, 4.0), "r_multiple": (3.0, 4.0, 6.0),
    }

    def signals(self, df: pd.DataFrame, bias: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        p = self.params
        out = blank_signals(df)
        ch = donchian(df, int(p["donchian_period"]))
        a = atr(df, int(p["atr_period"]))
        close = df["close"]
        trend = ema(close, int(p["trend_ema"]))

        long_ = close > ch["dc_upper"]
        short_ = close < ch["dc_lower"]
        if p["require_trend_filter"]:
            long_ &= close > trend
            short_ &= close < trend

        sig = np.where(long_, 1.0, np.where(short_, -1.0, 0.0))
        stop = np.where(sig > 0, close - p["atr_stop_mult"] * a,
                        np.where(sig < 0, close + p["atr_stop_mult"] * a, np.nan))
        risk = np.abs(close - stop)
        target = np.where(sig > 0, close + p["r_multiple"] * risk,
                          np.where(sig < 0, close - p["r_multiple"] * risk, np.nan))

        out["signal"] = sig
        out["stop"] = stop
        out["target"] = target
        out["tag"] = np.where(sig > 0, "breakout_long",
                              np.where(sig < 0, "breakout_short", ""))
        return out


@register("buy_and_hold")
class BuyAndHold(Strategy):
    """The benchmark every long-only claim must clear.

    Runs with ``max_position_pct = 1.0`` and no time exit, so it is a genuine
    buy-and-hold rather than a capital-constrained approximation of one.
    """

    defaults: Dict[str, Any] = {
        "atr_period": 14, "atr_stop_mult": 0.0, "r_multiple": 0.0,
        "max_bars_in_trade": None,
    }
    grid: Dict[str, tuple] = {}
    #: risk-config overrides applied only to this strategy
    risk_overrides: Dict[str, Any] = {"max_position_pct": 1.0}

    def signals(self, df: pd.DataFrame, bias: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        out = blank_signals(df)
        sig = np.zeros(len(df))
        sig[0] = 1.0
        out["signal"] = sig
        # -inf / +inf are the engine's explicit "no stop" / "no target" sentinels
        out["stop"] = np.full(len(df), -np.inf)
        out["target"] = np.full(len(df), np.inf)
        out["tag"] = "buy_and_hold"
        return out
