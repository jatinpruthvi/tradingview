"""Core technical indicators: ATR, EMA, ADX, RSI, Donchian, Supertrend, MACD.

LOOKAHEAD CONTRACT
------------------
Every function here is **causal**: the value at bar *i* depends only on bars
*<= i*. Nothing uses ``.shift(-k)``. :mod:`tvlab.validation.lookahead` enforces
this by truncating the series and asserting the prefix is bit-identical.

The one family that *cannot* be causal by nature is fractal/swing detection,
which needs future bars to confirm a pivot. That lives in :mod:`tvlab.indicators.smc`
and is published with an explicit confirmation lag.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range."""
    prev_close = df["close"].shift(1)
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).to_numpy(),
        (df["high"] - prev_close).abs().to_numpy(),
        (df["low"] - prev_close).abs().to_numpy(),
    ])
    return pd.Series(tr, index=df.index, name="tr").fillna(df["high"] - df["low"])


def atr(df: pd.DataFrame, period: int = 14, method: str = "wilder") -> pd.Series:
    """Average True Range.

    ``wilder`` (default) matches TradingView's ``ta.atr`` - an RMA, i.e. an EMA
    with alpha = 1/period. ``ema`` and ``sma`` are also available.

    That claim is verified, not assumed: ``tests/test_pynecore_reference.py``
    compares this against PyneCore's ``ta.atr(14)`` over 600 bars and agrees to
    ~1e-15 relative error in steady state. The first ~260 bars differ because the
    two seed the recursion differently; see that file for the decay model.
    """
    tr = true_range(df)
    if method == "wilder":
        out = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    elif method == "ema":
        out = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    elif method == "sma":
        out = tr.rolling(period, min_periods=period).mean()
    else:
        raise ValueError(f"unknown atr method {method!r}; use wilder|ema|sma")
    out.name = f"atr_{period}"
    return out


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a fraction of close - the scale-free way to compare across symbols."""
    return (atr(df, period) / df["close"]).rename(f"atr_pct_{period}")


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/period) - what TradingView calls ta.rma."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI, matching TradingView's ta.rsi."""
    delta = df["close"].diff()
    gain = rma(delta.clip(lower=0.0), period)
    loss = rma((-delta).clip(lower=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # all-gain / all-loss edge cases
    out = out.where(loss != 0, 100.0).where(~((gain == 0) & (loss == 0)), 50.0)
    return out.rename(f"rsi_{period}")


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = df["close"]
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig}, index=df.index)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI / -DI. Matches TradingView's ta.dmi.

    Verified, not merely claimed: ``tests/test_pynecore_reference.py`` runs all
    three columns against PyneCore's ``ta.dmi(period, period)`` over 600 bars and
    they converge to ~1e-14 relative error. Pine has no ``ta.adx``, so ``ta.dmi`` -
    which returns ``(+DI, -DI, ADX)`` - is the function this claim is about, and
    using one period for both the DI length and the ADX smoothing is what makes
    ``dmi(p, p)`` the right mapping.

    Warmup is not safe to read here. ``true_range`` fills bar zero with
    ``high - low`` where ``ta.dmi``'s internal ``tr()`` is na, and two chained RMAs
    smooth that seed, so +DI disagrees by up to 28% and -DI by up to 46% relative
    over the first hundred bars before decaying geometrically. This matters more
    than for the other indicators because ``bias_adx_min`` is a *threshold*: inside
    the warmup a different number is being compared against the same cutoff.
    """
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr_smooth = rma(true_range(df), period).replace(0.0, np.nan)
    plus_di = 100.0 * rma(plus_dm, period) / tr_smooth
    minus_di = 100.0 * rma(minus_dm, period) / tr_smooth

    dx_denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / dx_denom
    adx_ = rma(dx, period)
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di}, index=df.index)


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian channel. Bands use ``shift(1)`` so bar i's band excludes bar i -
    a breakout entry must react to the *previous* bars' extreme, not its own."""
    upper = df["high"].rolling(period, min_periods=period).max().shift(1)
    lower = df["low"].rolling(period, min_periods=period).min().shift(1)
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"dc_upper": upper, "dc_lower": lower, "dc_mid": mid}, index=df.index)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend, iteratively computed (the recursion is inherently sequential).

    The ATR warmup leaves the first ``period`` values NaN. Those must not enter
    the recursion: because ``upper[i]`` falls back to ``upper[i-1]`` whenever the
    comparison is False - and every comparison against NaN is False - a single
    NaN seed poisons the band for the entire series and the direction never
    flips. We therefore start the recursion at the first bar with a finite ATR
    and emit NaN before it.

    Note the argument order is the reverse of Pine's ``ta.supertrend(factor,
    atrPeriod)`` - this is ``(period, multiplier)``.

    **``st_direction`` has the opposite sign to Pine's ``direction``.** Here
    ``+1`` means uptrend (the plotted band is ``st_lower``, support under price).
    TradingView's built-in plots ``direction < 0`` as "Up Direction", so Pine uses
    ``+1`` for *down*trend. The band itself matches Pine to machine precision;
    only the flag is negated. Verified against PyneCore's ``ta.supertrend`` in
    ``tests/test_pynecore_reference.py`` - it is the exact negation on every
    settled bar. Porting a Pine strategy that branches on ``direction`` without
    flipping the sign inverts every entry and exit.
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    a = atr(df, period)
    upper_basic = (hl2 + multiplier * a).to_numpy(dtype="float64")
    lower_basic = (hl2 - multiplier * a).to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    n = len(df)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    ready = np.flatnonzero(np.isfinite(upper_basic) & np.isfinite(lower_basic))
    if ready.size == 0:
        return pd.DataFrame(
            {"supertrend": st, "st_direction": direction, "st_upper": upper, "st_lower": lower},
            index=df.index,
        )

    i0 = int(ready[0])
    upper[i0] = upper_basic[i0]
    lower[i0] = lower_basic[i0]
    st[i0] = upper_basic[i0]
    direction[i0] = -1

    for i in range(i0 + 1, n):
        ub, lb = upper_basic[i], lower_basic[i]
        if not (np.isfinite(ub) and np.isfinite(lb)):
            # ATR went non-finite (e.g. a zero-range bar); carry the last band
            upper[i], lower[i] = upper[i - 1], lower[i - 1]
        else:
            upper[i] = ub if (ub < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
            lower[i] = lb if (lb > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]

        if direction[i - 1] == 1:                        # was uptrend
            direction[i] = -1 if close[i] < lower[i] else 1
        else:                                             # was downtrend
            direction[i] = 1 if close[i] > upper[i] else -1

        st[i] = lower[i] if direction[i] == 1 else upper[i]

    return pd.DataFrame(
        {"supertrend": st, "st_direction": direction, "st_upper": upper, "st_lower": lower},
        index=df.index,
    )


def bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(df["close"], period)
    sd = df["close"].rolling(period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + num_std * sd, "bb_lower": mid - num_std * sd,
         "bb_width": (2 * num_std * sd) / mid},
        index=df.index,
    )


def rolling_zscore(series: pd.Series, period: int = 50) -> pd.Series:
    """Z-score of a series vs its own trailing window - regime/vol normalisation."""
    mu = series.rolling(period, min_periods=period).mean()
    sd = series.rolling(period, min_periods=period).std(ddof=0).replace(0.0, np.nan)
    return ((series - mu) / sd).rename(f"z_{period}")


def realised_vol(df: pd.DataFrame, period: int = 20, annualise: float = 252.0) -> pd.Series:
    r = np.log(df["close"]).diff()
    return (r.rolling(period, min_periods=period).std(ddof=0) * np.sqrt(annualise)).rename("rv")


def vwap(df: pd.DataFrame, session: bool = True) -> pd.Series:
    """VWAP, optionally reset each UTC day (session-anchored, as on TradingView)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    if session:
        # .index.shift() raises NullFrequencyError on a sliced index, so diff the
        # normalised dates as a Series instead - it also survives gaps/weekends
        day = pd.Series(df.index.normalize(), index=df.index)
        new_session = (day != day.shift(1)).fillna(True)
        grp = new_session.cumsum()
        cum_pv = pv.groupby(grp).cumsum()
        cum_v = df["volume"].groupby(grp).cumsum()
    else:
        cum_pv, cum_v = pv.cumsum(), df["volume"].cumsum()
    return (cum_pv / cum_v.replace(0.0, np.nan)).rename("vwap")


def higher_highs_lower_lows(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Simple causal market-structure read: where price sits in its own range."""
    hi = df["high"].rolling(period, min_periods=period).max()
    lo = df["low"].rolling(period, min_periods=period).min()
    rng = (hi - lo).replace(0.0, np.nan)
    pos = (df["close"] - lo) / rng
    return pd.DataFrame({"range_pos": pos, "range_high": hi, "range_low": lo}, index=df.index)


def consecutive(series: pd.Series, direction: int = 1) -> pd.Series:
    """Count of consecutive bars where ``series`` moved in ``direction``."""
    sign = np.sign(series.diff().fillna(0.0)) == direction
    grp = (~sign).cumsum()
    return sign.groupby(grp).cumsum().astype(int).rename("consecutive")


def session_of(ts: pd.DatetimeIndex, sessions: Optional[dict] = None) -> pd.Series:
    """Label bars by UTC session - lets the report split P&L by time of day."""
    hours = ts.hour
    out = np.full(len(ts), "off", dtype=object)
    out[(hours >= 0) & (hours < 7)] = "asia"
    out[(hours >= 7) & (hours < 12)] = "london_am"
    out[(hours >= 12) & (hours < 16)] = "london_pm"
    out[(hours >= 13) & (hours < 21)] = "us"
    out[(hours >= 21)] = "late_us"
    if sessions:
        for name, (start, end) in sessions.items():
            out[(hours >= start) & (hours < end)] = name
    return pd.Series(out, index=ts, name="session")


def forward_fill_safe(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """ffill that refuses to fill the leading NaN warmup region."""
    return df.ffill(limit=limit)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Convenience wrapper used by multi-timeframe strategies."""
    from ..data.base import resample

    return resample(df, rule)


def to_arrays(df: pd.DataFrame, *cols: str) -> Tuple[np.ndarray, ...]:
    """Pull float64 numpy views for the hot loops in the backtester."""
    return tuple(np.ascontiguousarray(df[c].to_numpy(dtype="float64")) for c in cols)
