"""The strategies. Four of them, on purpose.

``smc_trend`` is the intended edge: a liquidity sweep into a confirmed swing, in
the direction of the higher-timeframe trend, with the stop beyond the sweep and
the target at the next resting liquidity pool. ``ema_pullback`` and
``donchian_breakout`` are deliberately dull baselines - if the SMC setup cannot
beat a 20-line moving-average pullback out of sample, it is not an edge, it is
decoration. Always run the baselines.

``orb`` is the opening-range breakout, added because it is the only rule measured
on live NSE intraday data with a gross edge that is not noise (t = 3.2). It is
registered so the walk-forward harness can judge it, not because it is known to
make money - at the config's 16 bp round trip it does not. See
``docs/PROFITABILITY_BACKLOG.md``.
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


@register("orb")
class OpeningRangeBreakout(Strategy):
    """Break of the first 15 minutes of the session, flat by the close.

    The rule
    --------
    1. Take the high and low of the first ``range_bars`` bars of the session
       (3 bars = 09:15-09:30 on 5m NSE data).
    2. Take the first break of that range, whichever side breaks first. If one
       bar breaks both sides the session is skipped - OHLC cannot tell which
       side came first, and guessing is how a backtest invents an entry.
    3. Stop at the opposite side of the range, optionally buffered by ATR.
    4. Exit at the session close (``exit_tod``), so nothing is held overnight.

    One entry per session: a range is a single decision, and re-entering it every
    time price pokes back through turns one idea into five correlated trades.

    What has been measured, so this is not sold as a winner
    -------------------------------------------------------
    On 60 live NSE names x 60 sessions of 5m bars (3,510 breakouts) the basket
    grossed **+13.4 bp per day, t = 3.23** - the only non-noise gross edge found
    in this project. At the Indian config's 16 bp round trip it nets **-2.6 bp**,
    and it is green on only 40% of days with an 8-day losing run. A relative-
    volume filter (the published "stocks in play" refinement) did not survive a
    causal test: it turns negative once the volume window stops peeking past the
    entry. Full numbers in ``docs/PROFITABILITY_BACKLOG.md``.

    So: run it through the walk-forward harness to see whether a longer, cleaner
    sample rescues it. Do not read this docstring as a recommendation.

    Frames that are not intraday (median bar spacing >= 20h) produce no signals,
    because a session is a single bar and there is no range to break.
    """

    defaults: Dict[str, Any] = {
        "range_bars": 3,
        "atr_period": 14,
        "stop_buffer_atr": 0.0,
        #: last bar of the session, as "HH:MM" in ``session_tz`` (or the frame's tz)
        "exit_tod": "15:25",
        #: optional timezone name (e.g. "Asia/Kolkata") to convert tz-aware indices
        #: before checking ``exit_tod`` / ``max_entry_tod`` and session boundaries
        "session_tz": None,
        "allow_short": True,
        "max_bars_in_trade": None,   # the session exit is the exit
        #: require bar close beyond the opening range instead of an intrabar wick
        "confirm_close": False,
        #: optional "HH:MM" cutoff after which no new breakouts are entered
        "max_entry_tod": None,
        #: skip sessions whose opening range is tighter than this price fraction
        "min_range_pct": None,
        #: skip sessions whose opening range exceeds this multiple of prior-session range ATR
        "max_range_atr": None,
        #: skip sessions whose overnight gap exceeds this multiple of prior-session range ATR
        "max_gap_atr": None,
        #: skip breakouts whose signal bar already extended beyond k * opening_range
        "max_chase_ratio": None,
        #: "opposite" | "midpoint" | "midpoint_on_wide"
        "stop_mode": "opposite",
        "wide_range_pct": 0.015,
        #: optional finite R-multiple profit target (None = hold until exit_tod)
        "r_multiple": None,
        #: emit a resting limit entry level at the opening range boundary
        "limit_entry": False,
        "limit_offset_atr": 0.0,
        #: optional higher-timeframe trend alignment via trend_filter(df, bias)
        "use_bias": False,
        "bias_ema": 50,
        "bias_adx_period": 14,
        "bias_adx_min": 18.0,
        "bias_mode": "ema_slope",
        "bias_slope_lookback": 5,
        #: optional index/market regime alignment (reads regime frame or _regime_dir)
        "market_filter": False,
    }

    grid: Dict[str, tuple] = {
        "range_bars": (2, 3, 6),
        "stop_buffer_atr": (0.0, 0.5),
    }

    @staticmethod
    def _parse_tod(hhmm: Optional[str]) -> Optional[int]:
        if not hhmm:
            return None
        hh, mm = str(hhmm).split(":")
        return int(hh) * 60 + int(mm)

    @staticmethod
    def _causal_session_stats(
        starts: np.ndarray, ends: np.ndarray,
        hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, lookback: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prior-session close and rolling session-range ATR using strictly completed sessions."""
        m = len(starts)
        prev_close = np.full(m, np.nan)
        daily_atr = np.full(m, np.nan)
        ranges = np.empty(m)
        for k in range(m):
            s, e = int(starts[k]), int(ends[k])
            ranges[k] = float(hi[s:e].max() - lo[s:e].min())
            if k >= 1:
                prev_close[k] = float(cl[int(ends[k - 1]) - 1])
                w_start = max(0, k - max(int(lookback), 1))
                daily_atr[k] = float(ranges[w_start:k].mean())
        return prev_close, daily_atr

    @staticmethod
    def _causal_regime_dir(
        df: pd.DataFrame, regime: Optional[pd.DataFrame],
        starts: np.ndarray, ends: np.ndarray, rb: int,
    ) -> Optional[np.ndarray]:
        """Causal index/market direction aligned to df (+1 bullish, -1 bearish, 0 neutral)."""
        if "_regime_dir" in df.columns:
            return df["_regime_dir"].fillna(0).to_numpy(dtype="int8")
        if regime is None or len(regime) == 0 or "close" not in regime.columns:
            return None
        aligned = regime.reindex(df.index).ffill()
        rc = aligned["close"].to_numpy(dtype="float64")
        rh = aligned["high"].to_numpy(dtype="float64") if "high" in aligned else rc
        rl = aligned["low"].to_numpy(dtype="float64") if "low" in aligned else rc
        out = np.zeros(len(df), dtype="int8")
        for s, e in zip(starts, ends):
            if e - s < rb + 1:
                continue
            hi_win = rh[s:s + rb]
            lo_win = rl[s:s + rb]
            if not (np.isfinite(hi_win).any() and np.isfinite(lo_win).any()):
                continue
            irhi = float(np.nanmax(hi_win))
            irlo = float(np.nanmin(lo_win))
            for i in range(s + rb, e):
                if np.isfinite(rc[i]):
                    if rc[i] > irhi:
                        out[i] = 1
                    elif rc[i] < irlo:
                        out[i] = -1
        return out

    def signals(self, df: pd.DataFrame, bias: Optional[pd.DataFrame] = None,
                regime: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        p = self.params
        out = blank_signals(df)
        n = len(df)
        idx = df.index
        if n == 0 or not isinstance(idx, pd.DatetimeIndex):
            return out

        sess_tz = p.get("session_tz")
        local_idx = idx.tz_convert(sess_tz) if (sess_tz and idx.tz is not None) else idx
        tod = (local_idx.hour * 60 + local_idx.minute).to_numpy()
        day = (local_idx.tz_localize(None) if local_idx.tz is not None else local_idx).normalize()
        spacing = pd.Series(idx).diff().median()
        intraday = bool(pd.notna(spacing) and spacing < pd.Timedelta(hours=20))

        # A session close is knowable from the clock alone. Deriving it from
        # "this is the last row of the date" would need the NEXT bar's date and
        # is therefore lookahead - the audit in validation/lookahead.py catches it.
        exit_flags = np.zeros(n)
        exit_cut = self._parse_tod(p["exit_tod"]) if intraday else None
        if exit_cut is not None:
            exit_flags[tod >= exit_cut] = 1.0
        out["exit"] = exit_flags
        max_entry_cut = self._parse_tod(p["max_entry_tod"]) if intraday else None

        op = df["open"].to_numpy(dtype="float64")
        hi = df["high"].to_numpy(dtype="float64")
        lo = df["low"].to_numpy(dtype="float64")
        cl = df["close"].to_numpy(dtype="float64")
        a = atr(df, int(p["atr_period"])).to_numpy(dtype="float64")
        rb = int(p["range_bars"])
        buf = float(p["stop_buffer_atr"])

        trend_arr = (
            trend_filter(
                df, bias, int(p["bias_ema"]), int(p["bias_adx_period"]),
                float(p["bias_adx_min"]), mode=str(p["bias_mode"]),
                slope_lookback=int(p["bias_slope_lookback"]),
            ).to_numpy()
            if p["use_bias"] else None
        )

        codes = pd.factorize(day)[0]
        cuts = np.flatnonzero(np.diff(codes)) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, n]

        prev_close, daily_atr = self._causal_session_stats(
            starts, ends, hi, lo, cl, int(p["atr_period"]),
        )
        reg_dir = (
            self._causal_regime_dir(df, regime, starts, ends, rb)
            if p["market_filter"] else None
        )

        sig = np.zeros(n)
        stop = np.full(n, np.nan)
        target = np.full(n, np.inf)
        entry_px_arr = np.full(n, np.nan)
        tags = [""] * n

        confirm_close = bool(p["confirm_close"])
        min_range_pct = p["min_range_pct"]
        max_range_atr = p["max_range_atr"]
        max_gap_atr = p["max_gap_atr"]
        max_chase = p["max_chase_ratio"]
        stop_mode = str(p["stop_mode"])
        wide_pct = float(p["wide_range_pct"])
        r_mult = p["r_multiple"]
        limit_entry = bool(p["limit_entry"])
        limit_off = float(p["limit_offset_atr"])

        for sess_idx, (s, e) in enumerate(zip(starts, ends)):
            if e - s < rb + 1:
                continue
            rhi = float(hi[s:s + rb].max())
            rlo = float(lo[s:s + rb].min())
            rng = rhi - rlo
            rmid = 0.5 * (rhi + rlo)

            if min_range_pct is not None and rmid > 0 and (rng / rmid) < float(min_range_pct):
                continue
            d_atr = daily_atr[sess_idx]
            if max_range_atr is not None and np.isfinite(d_atr) and d_atr > 0:
                if (rng / d_atr) > float(max_range_atr):
                    continue
            p_close = prev_close[sess_idx]
            if (max_gap_atr is not None and np.isfinite(p_close)
                    and np.isfinite(d_atr) and d_atr > 0):
                if abs(op[s] - p_close) / d_atr > float(max_gap_atr):
                    continue

            use_mid = (stop_mode == "midpoint") or (
                stop_mode == "midpoint_on_wide" and rmid > 0 and (rng / rmid) >= wide_pct
            )
            step = 0
            for i in range(s + rb, e):
                if i > s:
                    d = int(tod[i] - tod[i - 1])
                    step = d if 0 < d <= 60 else step
                # The engine fills on the NEXT bar's open, so a break needs a bar
                # left to hold it. Both tests are clock-only: reading the next
                # bar's flag would be lookahead, and the audit rejects it.
                if exit_cut is not None and (exit_flags[i]
                                             or (step and tod[i] + 2 * step > exit_cut)):
                    break
                if max_entry_cut is not None and tod[i] > max_entry_cut:
                    break

                # An outside bar that pierces both sides is always ambiguous
                if hi[i] > rhi and lo[i] < rlo:
                    break
                up = (cl[i] > rhi) if confirm_close else (hi[i] > rhi)
                dn = (cl[i] < rlo) if confirm_close else (lo[i] < rlo)
                if up and dn:
                    break

                a_val = a[i] if np.isfinite(a[i]) else 0.0
                if up:
                    if max_chase is not None and rng > 0 and (cl[i] - rhi) / rng > float(max_chase):
                        break
                    if trend_arr is not None and trend_arr[i] != 1:
                        break
                    if reg_dir is not None and reg_dir[i] != 1:
                        break
                    base_stop = rmid if use_mid else rlo
                    stop_val = base_stop - buf * a_val
                    lim_val = rhi + limit_off * a_val if limit_entry else np.nan
                    ref_px = lim_val if np.isfinite(lim_val) else cl[i]
                    risk_dist = ref_px - stop_val
                    if risk_dist <= 0:
                        break
                    sig[i] = 1.0
                    stop[i] = stop_val
                    if limit_entry:
                        entry_px_arr[i] = lim_val
                    if r_mult is not None:
                        target[i] = ref_px + float(r_mult) * risk_dist
                    tags[i] = "orb_long"
                    break
                if dn:
                    if not p["allow_short"]:
                        break
                    if max_chase is not None and rng > 0 and (rlo - cl[i]) / rng > float(max_chase):
                        break
                    if trend_arr is not None and trend_arr[i] != -1:
                        break
                    if reg_dir is not None and reg_dir[i] != -1:
                        break
                    base_stop = rmid if use_mid else rhi
                    stop_val = base_stop + buf * a_val
                    lim_val = rlo - limit_off * a_val if limit_entry else np.nan
                    ref_px = lim_val if np.isfinite(lim_val) else cl[i]
                    risk_dist = stop_val - ref_px
                    if risk_dist <= 0:
                        break
                    sig[i] = -1.0
                    stop[i] = stop_val
                    if limit_entry:
                        entry_px_arr[i] = lim_val
                    if r_mult is not None:
                        target[i] = ref_px - float(r_mult) * risk_dist
                    tags[i] = "orb_short"
                    break

        out["signal"] = sig
        out["stop"] = stop
        # +inf is the engine's explicit "no target" sentinel when r_multiple is None.
        out["target"] = target
        out["tag"] = tags
        if limit_entry:
            out["entry_price"] = entry_px_arr
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
