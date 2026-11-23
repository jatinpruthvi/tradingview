"""Smart Money Concepts, implemented WITHOUT lookahead.

The trap this module exists to avoid
------------------------------------
Fractal swing detection is *not* causal by nature: to know that bar ``i`` is a
swing high you must see the next ``L`` bars fail to exceed it. The widely-used
reference implementation (``joshyattridge/smart-money-concepts``, vendored at
``vendor/smart-money-concepts``) marks the swing at bar ``i`` using
``rolling(2L).max()`` applied to ``high.shift(-L)`` - a **negative shift**, i.e.
a forward look. Its output is correct as a *chart annotation* (drawn after the
fact) but if you feed it straight into a backtester you are trading with a
perfect preview of where the pivots were.

The consequences are not subtle: an SMC strategy with that leak typically shows
a profit factor of 3-10 in backtest and loses money live.

What this module does instead
-----------------------------
Every column is **published at confirmation time**. A swing high occurring at
bar ``i`` with swing length ``L`` appears in the output at bar ``i + L`` - the
first bar on which a real trader could actually know about it. Structure breaks,
sweeps, order blocks and FVGs are all derived from those *confirmed* levels, so
the whole feature set is causal and :mod:`tvlab.validation.lookahead` can prove
it by truncation.

``vendor/smart-money-concepts`` is still used, but only as a cross-check: see
:func:`tvlab.indicators.reference.compare_swings`, which asserts the two
implementations find the same pivots once the lag is accounted for.

Columns produced by :func:`smc_features`
----------------------------------------
=========================  =====================================================
``swing_high``             bool, True on the bar a swing high is *confirmed*
``swing_low``              bool, ditto for swing lows
``last_high_level``        price of the most recent confirmed swing high (ffill)
``last_low_level``         price of the most recent confirmed swing low
``bos``                    +1 bullish / -1 bearish break of structure (continuation)
``choch``                  +1 / -1 change of character (reversal)
``structure``              +1 bullish / -1 bearish current market structure
``bull_fvg`` / ``bear_fvg``bool, 3-candle imbalance present
``fvg_top``/``fvg_bottom`` active bullish FVG bounds (NaN when none)
``ob_bull`` / ``ob_bear``  bool, order block created this bar
``ob_top``/``ob_bottom``   active order-block bounds
``sweep_low``              bullish liquidity sweep: wicked below the confirmed
                           swing low and closed back above it (stop hunt)
``sweep_high``             bearish equivalent above the confirmed swing high
``eq_high`` / ``eq_low``   equal highs/lows - resting liquidity pools
``sweep_age``              bars since the most recent sweep (either side)
``bull_sweep_age``         bars since the most recent BULLISH sweep
``bear_sweep_age``         bars since the most recent BEARISH sweep
``choch_age``              bars since the most recent change of character
``bos_age``                bars since the most recent break of structure
=========================  =====================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def swing_points(
    df: pd.DataFrame,
    swing_length: int = 12,
    lag: bool = True,
) -> pd.DataFrame:
    """Detect fractal swing highs/lows.

    Parameters
    ----------
    swing_length:
        half-window ``L``. A pivot at ``i`` needs ``high[i]`` to be the max over
        ``[i-L, i+L]``.
    lag:
        **True (default) = causal.** The pivot is published at ``i + L``, the
        first bar where it is knowable. ``lag=False`` reproduces the reference
        implementation's chart-annotation convention and is only for plotting or
        cross-checking - never feed it to the backtester.
    """
    if swing_length < 1:
        raise ValueError("swing_length must be >= 1")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)
    L = int(swing_length)

    raw_high = np.zeros(n, dtype=bool)
    raw_low = np.zeros(n, dtype=bool)
    raw_high_level = np.full(n, np.nan)
    raw_low_level = np.full(n, np.nan)

    if n > 2 * L:
        # rolling max/min over the centred window, computed causally-on-the-window
        win = 2 * L + 1
        hi_max = pd.Series(high).rolling(win, center=True, min_periods=win).max().to_numpy()
        lo_min = pd.Series(low).rolling(win, center=True, min_periods=win).min().to_numpy()
        raw_high = (high == hi_max) & np.isfinite(hi_max)
        raw_low = (low == lo_min) & np.isfinite(lo_min)
        # a bar cannot be both; prefer the high (matches the reference behaviour)
        raw_low &= ~raw_high
        raw_high_level = np.where(raw_high, high, np.nan)
        raw_low_level = np.where(raw_low, low, np.nan)

    if not lag:
        return pd.DataFrame(
            {"swing_high": raw_high, "swing_low": raw_low,
             "swing_high_level": raw_high_level, "swing_low_level": raw_low_level},
            index=df.index,
        )

    # publish L bars later: shift the *labels and their levels* forward in time
    sh_h = np.zeros(n, dtype=bool)
    sh_l = np.zeros(n, dtype=bool)
    lv_h = np.full(n, np.nan)
    lv_l = np.full(n, np.nan)
    if n > L:
        sh_h[L:] = raw_high[:-L]
        sh_l[L:] = raw_low[:-L]
        lv_h[L:] = raw_high_level[:-L]
        lv_l[L:] = raw_low_level[:-L]

    return pd.DataFrame(
        {"swing_high": sh_h, "swing_low": sh_l,
         "swing_high_level": lv_h, "swing_low_level": lv_l},
        index=df.index,
    )


def fair_value_gaps(df: pd.DataFrame, join_consecutive: bool = False) -> pd.DataFrame:
    """3-candle imbalances. Causal: bar ``i``'s FVG uses bars ``i-2, i-1, i``.

    Bullish FVG at ``i``: ``low[i] > high[i-2]`` - the gap ``[high[i-2], low[i]]``.
    Bearish FVG at ``i``: ``high[i] < low[i-2]`` - the gap ``[high[i], low[i-2]]``.
    """
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)

    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull_top = np.full(n, np.nan)
    bull_bot = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bot = np.full(n, np.nan)

    if n >= 3:
        bull[2:] = low[2:] > high[:-2]
        bear[2:] = high[2:] < low[:-2]
        bull_top[2:] = np.where(bull[2:], low[2:], np.nan)
        bull_bot[2:] = np.where(bull[2:], high[:-2], np.nan)
        bear_top[2:] = np.where(bear[2:], low[:-2], np.nan)
        bear_bot[2:] = np.where(bear[2:], high[2:], np.nan)

    out = pd.DataFrame(
        {"bull_fvg": bull, "bear_fvg": bear,
         "bull_fvg_top": bull_top, "bull_fvg_bottom": bull_bot,
         "bear_fvg_top": bear_top, "bear_fvg_bottom": bear_bot},
        index=df.index,
    )
    if join_consecutive:
        for side in ("bull", "bear"):
            flag = out[f"{side}_fvg"].to_numpy()
            top, bot = out[f"{side}_fvg_top"].to_numpy(), out[f"{side}_fvg_bottom"].to_numpy()
            for i in range(1, n):
                if flag[i] and flag[i - 1]:
                    top[i] = np.nanmax([top[i], top[i - 1]])
                    bot[i] = np.nanmin([bot[i], bot[i - 1]])
            out[f"{side}_fvg_top"], out[f"{side}_fvg_bottom"] = top, bot
    return out


def equal_highs_lows(df: pd.DataFrame, swings: pd.DataFrame, tolerance: float = 0.002) -> pd.DataFrame:
    """Flag resting liquidity: two confirmed swings at ~the same price.

    Equal highs above price are buy-stops; equal lows below are sell-stops. Both
    are magnets, and both get swept. ``tolerance`` is a fraction of price.
    """
    n = len(df)
    eq_high = np.zeros(n, dtype=bool)
    eq_low = np.zeros(n, dtype=bool)
    eq_high_level = np.full(n, np.nan)
    eq_low_level = np.full(n, np.nan)

    hi_conf = swings["swing_high"].to_numpy()
    lo_conf = swings["swing_low"].to_numpy()
    hi_lvl = swings["swing_high_level"].to_numpy()
    lo_lvl = swings["swing_low_level"].to_numpy()

    last_hi, last_hi_i = np.nan, -1
    last_lo, last_lo_i = np.nan, -1
    for i in range(n):
        if hi_conf[i]:
            if np.isfinite(last_hi) and abs(hi_lvl[i] - last_hi) <= tolerance * max(last_hi, 1e-12):
                eq_high[i] = True
                eq_high_level[i] = max(hi_lvl[i], last_hi)
            last_hi, last_hi_i = hi_lvl[i], i
        if lo_conf[i]:
            if np.isfinite(last_lo) and abs(lo_lvl[i] - last_lo) <= tolerance * max(last_lo, 1e-12):
                eq_low[i] = True
                eq_low_level[i] = min(lo_lvl[i], last_lo)
            last_lo, last_lo_i = lo_lvl[i], i

    return pd.DataFrame(
        {"eq_high": eq_high, "eq_low": eq_low,
         "eq_high_level": eq_high_level, "eq_low_level": eq_low_level},
        index=df.index,
    )


def smc_features(
    df: pd.DataFrame,
    swing_length: int = 12,
    close_break: bool = True,
    ob_lookback: int = 12,
    ob_max_age: int = 40,
    eq_tolerance: float = 0.002,
    fvg_max_age: int = 30,
) -> pd.DataFrame:
    """Full causal SMC feature set. See the module docstring for column meanings.

    The structure loop is sequential by design: BOS/CHoCH, order blocks and the
    active FVG/OB zones are path-dependent state, not a rolling window.
    """
    swings = swing_points(df, swing_length, lag=True)
    fvg = fair_value_gaps(df)
    eqhl = equal_highs_lows(df, swings, eq_tolerance)

    open_ = df["open"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    n = len(df)

    sh_conf = swings["swing_high"].to_numpy()
    sl_conf = swings["swing_low"].to_numpy()
    sh_lvl_raw = swings["swing_high_level"].to_numpy()
    sl_lvl_raw = swings["swing_low_level"].to_numpy()
    ref = close if close_break else None

    bos = np.zeros(n, dtype=np.int8)
    choch = np.zeros(n, dtype=np.int8)
    structure = np.zeros(n, dtype=np.int8)
    sweep_low = np.zeros(n, dtype=bool)
    sweep_high = np.zeros(n, dtype=bool)
    last_high_level = np.full(n, np.nan)
    last_low_level = np.full(n, np.nan)

    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)
    ob_top = np.full(n, np.nan)
    ob_bottom = np.full(n, np.nan)
    ob_bull_top = ob_bull_bot = None
    ob_bear_top = ob_bear_bot = None

    active_fvg_top = np.full(n, np.nan)
    active_fvg_bot = np.full(n, np.nan)
    bull_fvg_top = fvg["bull_fvg_top"].to_numpy()
    bull_fvg_bot = fvg["bull_fvg_bottom"].to_numpy()
    bear_fvg_top = fvg["bear_fvg_top"].to_numpy()
    bear_fvg_bot = fvg["bear_fvg_bottom"].to_numpy()
    bull_fvg_flag = fvg["bull_fvg"].to_numpy()
    bear_fvg_flag = fvg["bear_fvg"].to_numpy()

    sweep_age = np.full(n, np.inf)
    bull_sweep_age = np.full(n, np.inf)
    bear_sweep_age = np.full(n, np.inf)
    structure_age = np.full(n, np.inf)
    choch_age = np.full(n, np.inf)
    bos_age = np.full(n, np.inf)

    trend = 0            # +1 bullish structure, -1 bearish, 0 undetermined
    cur_hi = np.nan      # most recent CONFIRMED swing high level
    cur_lo = np.nan      # most recent CONFIRMED swing low level
    fvg_side = 0         # which FVG zone is currently active
    fvg_t, fvg_b = np.nan, np.nan
    fvg_created = -10**9
    ob_created = -10**9
    last_sweep = -10**9
    last_bull_sweep = -10**9
    last_bear_sweep = -10**9
    last_struct = -10**9
    last_choch = -10**9
    last_bos = -10**9

    for i in range(n):
        # ---- 1. ingest swings confirmed as of this bar -----------------------
        if sh_conf[i] and np.isfinite(sh_lvl_raw[i]):
            cur_hi = sh_lvl_raw[i]
        if sl_conf[i] and np.isfinite(sl_lvl_raw[i]):
            cur_lo = sl_lvl_raw[i]

        # ---- 2. liquidity sweeps (wick through, close back inside) -----------
        # Bullish sweep: something ran the stops below the confirmed swing low,
        # then the bar closed back above it. Only testable once cur_lo exists.
        # NOTE: the two sides are aged separately. A shared age would let a
        # bearish sweep two bars ago satisfy a long entry's freshness test.
        if np.isfinite(cur_lo) and low[i] < cur_lo and close[i] > cur_lo:
            sweep_low[i] = True
            last_sweep = i
            last_bull_sweep = i
        if np.isfinite(cur_hi) and high[i] > cur_hi and close[i] < cur_hi:
            sweep_high[i] = True
            last_sweep = i
            last_bear_sweep = i

        # ---- 3. structure breaks --------------------------------------------
        px = ref[i] if ref is not None else close[i]
        if np.isfinite(cur_hi) and px > cur_hi:
            if trend == -1:
                choch[i] = 1        # character changed: down -> up
                last_choch = i
            else:
                bos[i] = 1          # continuation (or first break)
                last_bos = i
            last_struct = i
            trend = 1
            cur_hi = np.nan         # consumed; wait for the next confirmed swing
        elif np.isfinite(cur_lo) and px < cur_lo:
            if trend == 1:
                choch[i] = -1
                last_choch = i
            else:
                bos[i] = -1
                last_bos = i
            last_struct = i
            trend = -1
            cur_lo = np.nan

        structure[i] = trend

        # ---- 4. order block: last opposing candle before the break -----------
        if bos[i] == 1 or choch[i] == 1:
            j = _find_ob(open_, close, high, low, i, ob_lookback, bearish=False)
            if j >= 0:
                ob_bull[i] = True
                ob_bull_top, ob_bull_bot = high[j], low[j]
                ob_created = i
        elif bos[i] == -1 or choch[i] == -1:
            j = _find_ob(open_, close, high, low, i, ob_lookback, bearish=True)
            if j >= 0:
                ob_bear[i] = True
                ob_bear_top, ob_bear_bot = high[j], low[j]
                ob_created = i

        if ob_bull_top is not None and (i - ob_created) <= ob_max_age:
            # zone dies once price trades through it (mitigation)
            if close[i] < (ob_bull_bot if ob_bull_top is not None else np.nan):
                ob_bull_top = ob_bull_bot = None
        if ob_bear_top is not None and (i - ob_created) <= ob_max_age:
            if close[i] > ob_bear_top:
                ob_bear_top = ob_bear_bot = None

        if ob_bull_top is not None and (i - ob_created) <= ob_max_age:
            ob_top[i], ob_bottom[i] = ob_bull_top, ob_bull_bot
        elif ob_bear_top is not None and (i - ob_created) <= ob_max_age:
            ob_top[i], ob_bottom[i] = ob_bear_top, ob_bear_bot
        else:
            ob_bull_top = ob_bull_bot = ob_bear_top = ob_bear_bot = None

        # ---- 5. active FVG zone ---------------------------------------------
        if bull_fvg_flag[i]:
            fvg_side, fvg_t, fvg_b, fvg_created = 1, bull_fvg_top[i], bull_fvg_bot[i], i
        elif bear_fvg_flag[i]:
            fvg_side, fvg_t, fvg_b, fvg_created = -1, bear_fvg_top[i], bear_fvg_bot[i], i

        if fvg_side != 0 and (i - fvg_created) <= fvg_max_age:
            filled = (fvg_side == 1 and low[i] <= fvg_b) or (fvg_side == -1 and high[i] >= fvg_t)
            if filled:
                fvg_side, fvg_t, fvg_b = 0, np.nan, np.nan
            else:
                active_fvg_top[i], active_fvg_bot[i] = fvg_t, fvg_b
        else:
            fvg_side, fvg_t, fvg_b = 0, np.nan, np.nan

        last_high_level[i] = cur_hi if np.isfinite(cur_hi) else np.nan
        last_low_level[i] = cur_lo if np.isfinite(cur_lo) else np.nan
        sweep_age[i] = (i - last_sweep) if last_sweep > -10**8 else np.inf
        bull_sweep_age[i] = (i - last_bull_sweep) if last_bull_sweep > -10**8 else np.inf
        bear_sweep_age[i] = (i - last_bear_sweep) if last_bear_sweep > -10**8 else np.inf
        structure_age[i] = (i - last_struct) if last_struct > -10**8 else np.inf
        choch_age[i] = (i - last_choch) if last_choch > -10**8 else np.inf
        bos_age[i] = (i - last_bos) if last_bos > -10**8 else np.inf

    # keep the *level* of the last consumed swing available (structure memory)
    last_high_level = _carry_levels(last_high_level, sh_lvl_raw, sh_conf)
    last_low_level = _carry_levels(last_low_level, sl_lvl_raw, sl_conf)

    out = pd.DataFrame(index=df.index)
    out["swing_high"] = sh_conf
    out["swing_low"] = sl_conf
    out["swing_high_level"] = sh_lvl_raw
    out["swing_low_level"] = sl_lvl_raw
    out["last_high_level"] = last_high_level
    out["last_low_level"] = last_low_level
    out["bos"] = bos
    out["choch"] = choch
    out["structure"] = structure
    out["structure_age"] = structure_age
    out["choch_age"] = choch_age
    out["bos_age"] = bos_age
    out["sweep_low"] = sweep_low
    out["sweep_high"] = sweep_high
    out["sweep_age"] = sweep_age
    out["bull_sweep_age"] = bull_sweep_age
    out["bear_sweep_age"] = bear_sweep_age
    out["ob_bull"] = ob_bull
    out["ob_bear"] = ob_bear
    out["ob_top"] = ob_top
    out["ob_bottom"] = ob_bottom
    out = out.join(fvg[["bull_fvg", "bear_fvg"]])
    out["active_fvg_top"] = active_fvg_top
    out["active_fvg_bottom"] = active_fvg_bot
    out = out.join(eqhl[["eq_high", "eq_low", "eq_high_level", "eq_low_level"]])
    out["fvg_in_zone"] = _in_zone(low, high, active_fvg_top, active_fvg_bot)
    out["ob_in_zone"] = _in_zone(low, high, ob_top, ob_bottom)
    return out


def _find_ob(
    open_: np.ndarray, close: np.ndarray, high: np.ndarray, low: np.ndarray,
    i: int, lookback: int, bearish: bool
) -> int:
    """Index of the order-block candle: the last counter-colour bar before bar i."""
    start = max(0, i - lookback)
    for j in range(i - 1, start - 1, -1):
        is_bear = close[j] < open_[j]
        if (not bearish) and is_bear:      # bullish OB = last down candle
            return j
        if bearish and (not is_bear):      # bearish OB = last up candle
            return j
    return -1


def _carry_levels(levels: np.ndarray, raw: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Keep the last seen swing level alive after it is consumed by a break."""
    out = np.array(levels, dtype="float64")
    last = np.nan
    for i in range(len(out)):
        if conf[i] and np.isfinite(raw[i]):
            last = raw[i]
        if not np.isfinite(out[i]):
            out[i] = last
    return out


def _in_zone(low: np.ndarray, high: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (high >= bottom) & (low <= top) & np.isfinite(top) & np.isfinite(bottom)


def liquidity_pool(df: pd.DataFrame, features: pd.DataFrame, side: str) -> pd.DataFrame:
    """Nearest resting liquidity above/below price - the natural take-profit target.

    ``side='long'`` returns the next pool of buy-stops above (equal highs, then
    the last confirmed swing high). ``side='short'`` mirrors it below.
    """
    if side == "long":
        cands = np.vstack([
            features["eq_high_level"].to_numpy(dtype="float64"),
            features["last_high_level"].to_numpy(dtype="float64"),
        ])
    elif side == "short":
        cands = np.vstack([
            features["last_low_level"].to_numpy(dtype="float64"),
            features["eq_low_level"].to_numpy(dtype="float64"),
        ])
    else:
        raise ValueError("side must be 'long' or 'short'")

    close = df["close"].to_numpy(dtype="float64")
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(n):
        col = cands[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        if side == "long":
            above = col[col > close[i]]
            out[i] = above.min() if above.size else np.nan
        else:
            below = col[col < close[i]]
            out[i] = below.max() if below.size else np.nan
    return pd.DataFrame({"liquidity_target": out}, index=df.index)


def summarize(features: pd.DataFrame) -> Dict[str, float]:
    """Counts used by the report and the recovery test."""
    n = max(len(features), 1)
    return {
        "bars": float(n),
        "swing_highs": float(features["swing_high"].sum()),
        "swing_lows": float(features["swing_low"].sum()),
        "bos_up": float((features["bos"] == 1).sum()),
        "bos_down": float((features["bos"] == -1).sum()),
        "choch_up": float((features["choch"] == 1).sum()),
        "choch_down": float((features["choch"] == -1).sum()),
        "bull_sweeps": float(features["sweep_low"].sum()),
        "bear_sweeps": float(features["sweep_high"].sum()),
        "bull_fvgs": float(features["bull_fvg"].sum()),
        "bear_fvgs": float(features["bear_fvg"].sum()),
        "bull_obs": float(features["ob_bull"].sum()),
        "bear_obs": float(features["ob_bear"].sum()),
        "equal_highs": float(features["eq_high"].sum()),
        "equal_lows": float(features["eq_low"].sum()),
        "pct_bars_in_uptrend": float((features["structure"] == 1).sum() / n),
        "pct_bars_in_downtrend": float((features["structure"] == -1).sum() / n),
    }
