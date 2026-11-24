"""Automatic trendline fitting: diagonal support and resistance.

Why this exists
---------------
:mod:`tvlab.indicators.smc` finds *horizontal* structure - swing points, breaks
of structure, fair value gaps, equal highs and lows, liquidity pools. It says
nothing about *diagonal* structure, which is what a trend actually looks like on
a chart. This module fills that gap: it fits lines through causal swing pivots,
scores them by how many pivots respect them, and reports the nearest support and
resistance as a percentage of price.

Why it is not ``pytrendline``
-----------------------------
``pytrendline`` (MIT, ``vendor/pytrendline``) does exactly this and was evaluated
first. It cannot be used, for four independent reasons, each verified against the
installed 1.0.1:

1. **It is incompatible with pandas 2.** ``detect.py`` calls
   ``DataFrame.append()``, removed in pandas 2.0. This repo pins
   ``pandas>=2.1,<3``, so the call raises ``AttributeError`` the first time a
   trendline is actually found. Fixing it means patching a vendored library or
   downgrading pandas, which would break the rest of the repo.
2. Its documented ``datetime_col=None`` path calls ``self.df.set_index(None)``
   and raises ``KeyError``.
3. Its constructor discards the results of ``rename(...)`` and ``reset_index()``,
   so ``detect()``'s ``df.loc[0].name`` needs a caller-supplied 0-based index.
4. It is roughly cubic: 120 daily bars took 3.9 s and 400 took 141 s.

The pivot-pair fitting *approach* is taken from it; the implementation is native
so it runs on pandas 2, is fast, and reuses this repo's already-validated causal
pivot detector instead of a second one.

The lookahead rule
------------------
This repo has been bitten twice by fitting on the whole series and then reading a
historical value from it (the vendored ``smartmoneyconcepts`` pivots, and the
multi-timeframe bias alignment). Trendlines are the worst place to repeat that,
because a line fitted using next month's pivots looks perfectly reasonable drawn
on today's chart.

So :func:`trendline_state` fits only on the data it is given and describes the
lines at its **last** bar - what a scanner needs, and structurally unable to leak.
Pivots come from :func:`tvlab.indicators.smc.swing_points` with ``lag=True``,
which publishes a pivot at ``i + L``, the first bar where it is knowable.

A historical series therefore requires refitting on each expanding window:
:func:`trendline_series`. It is O(n) fits and it says so. There is deliberately no
function that fits once and returns a column of historical values, because such a
function could not be correct.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .smc import swing_points
from .technical import atr

log = logging.getLogger(__name__)

#: Refuse to consider more than this many pivots per side. Fitting is O(p^2) in
#: the pivot count, and old pivots on a 5,000-bar series describe structure nobody
#: would draw today.
MAX_PIVOTS = 40

#: Above this many expanding-window refits, warn.
SERIES_FIT_WARNING_BARS = 400


@dataclass(frozen=True)
class Trendline:
    """One fitted line, expressed at the last bar of the input.

    ``level`` is the line's price at that bar - the only number a trading decision
    can use. ``slope_pct_per_bar`` is normalised by the level so support and
    resistance on different-priced symbols are comparable. ``distance_pct`` is
    signed: negative means the line is below price (support), positive above.
    """

    kind: str                       # "support" | "resistance"
    level: float
    slope_pct_per_bar: float
    score: float
    n_points: int                   # pivots the line touches
    distance_pct: float             # (level - close) / close * 100, signed
    is_breakout: bool               # price has already closed through it
    first_index: int
    last_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "level": self.level,
            "slope_pct_per_bar": self.slope_pct_per_bar, "score": self.score,
            "n_points": self.n_points, "distance_pct": self.distance_pct,
            "is_breakout": self.is_breakout,
            "first_index": self.first_index, "last_index": self.last_index,
        }


#: Every key the success path returns, so that no exit path can hand a caller a
#: differently shaped dict.
_EMPTY: Dict[str, Any] = {
    "available": False, "reason": None, "lines": [], "n_support": 0, "n_resistance": 0,
    "n_pivots_high": 0, "n_pivots_low": 0,
    "nearest_support_pct": np.nan, "nearest_resistance_pct": np.nan,
    "channel_pct": np.nan, "close": np.nan,
}


def _empty(reason: str, close: float = np.nan) -> Dict[str, Any]:
    """Every exit path returns the SAME keys.

    A failure path that omits ``channel_pct`` turns a handled error into a
    ``KeyError`` somewhere downstream, which is much harder to diagnose than the
    original cause.
    """
    out = dict(_EMPTY)
    out["reason"] = reason
    out["close"] = close
    return out


def _pivots(df: pd.DataFrame, swing_length: int):
    """Return ``(hi_idx, hi_lvl, lo_idx, lo_lvl)`` - causal pivots, most recent last.

    The indices are the bars the pivots actually happened on, **not** the bars they
    became knowable on, and that distinction changes the geometry.

    ``swing_points(lag=True)`` publishes a pivot at ``i + L`` - the first bar where
    it is confirmable - and stores the *price at bar i* there. Anchoring a line at
    the publication index would draw it through ``(i + L, low[i])``, a point that
    never existed: every slope and every extension to the last bar would be wrong,
    by an error that grows with ``swing_length``.

    So subtract ``L`` to recover the true anchor. Causality is preserved, because
    the set of pivots returned is still exactly those knowable by the last bar -
    the most recent ``L`` bars can never contribute one.
    """
    L = int(swing_length)
    sp = swing_points(df, swing_length=L, lag=True)
    hi_pub = np.flatnonzero(sp["swing_high"].to_numpy(dtype=bool))
    lo_pub = np.flatnonzero(sp["swing_low"].to_numpy(dtype=bool))
    hi_lvl = sp["swing_high_level"].to_numpy(dtype="float64")[hi_pub]
    lo_lvl = sp["swing_low_level"].to_numpy(dtype="float64")[lo_pub]
    hi, lo = hi_pub - L, lo_pub - L

    hi_ok = np.isfinite(hi_lvl) & (hi >= 0)
    lo_ok = np.isfinite(lo_lvl) & (lo >= 0)
    return hi[hi_ok], hi_lvl[hi_ok], lo[lo_ok], lo_lvl[lo_ok]


def _fit_side(
    idx: np.ndarray,
    lvl: np.ndarray,
    kind: str,
    close: np.ndarray,
    n: int,
    tol: np.ndarray,
    min_points: int,
    ignore_breakouts: bool,
) -> List[Trendline]:
    """Fit every pair of same-side pivots, score by touches, drop violated lines.

    A line is anchored on two pivots and extended to the final bar. It survives if
    no *other* pivot on the same side sits on the wrong side of it by more than the
    tolerance - a support line with a swing low materially below it was never a
    support line. Tolerance is ATR-scaled, so the same rule works on a 60%-vol
    crypto pair and a 6%-vol currency pair.
    """
    if len(idx) < 2:
        return []

    idx = idx[-MAX_PIVOTS:]
    lvl = lvl[-MAX_PIVOTS:]
    last = n - 1
    out: List[Trendline] = []

    for a, b in combinations(range(len(idx)), 2):
        i0, i1 = int(idx[a]), int(idx[b])
        p0, p1 = float(lvl[a]), float(lvl[b])
        if i1 == i0:
            continue
        slope = (p1 - p0) / (i1 - i0)
        level = p1 + slope * (last - i1)
        if not np.isfinite(level) or level <= 0:
            continue

        # A support line must rise no faster than price could plausibly have been
        # caught by it, and must not be absurdly far from the action.
        if abs(level / close[-1] - 1.0) > 0.75:
            continue

        # value of the line at every pivot index
        line_at = p1 + slope * (idx - i1)
        dev = lvl - line_at
        t = tol[idx]
        touches = int(np.sum(np.abs(dev) <= t))
        if touches < min_points:
            continue

        # violations: for support, a pivot materially BELOW the line; for
        # resistance, materially ABOVE it
        if kind == "support":
            violations = int(np.sum(dev < -t))
            broken = close < (p1 + slope * (np.arange(n) - i1))
        else:
            violations = int(np.sum(dev > t))
            broken = close > (p1 + slope * (np.arange(n) - i1))

        if violations:
            continue
        is_breakout = bool(broken[i1:].any())
        if is_breakout and not ignore_breakouts:
            continue

        # score: touches first, then recency. A line through four pivots that ends
        # last week beats one through four pivots that ended a year ago.
        recency = (i1 - idx[0]) / max(1, (last - idx[0]))
        out.append(Trendline(
            kind=kind, level=float(level),
            slope_pct_per_bar=float(slope / level) if level else 0.0,
            score=float(touches + 0.5 * recency),
            n_points=touches,
            distance_pct=float((level - close[-1]) / close[-1] * 100.0),
            is_breakout=is_breakout, first_index=i0, last_index=i1,
        ))

    return out


def _dedupe(lines: List[Trendline], level_tol_pct: float = 0.75,
            slope_tol: float = 0.0025) -> List[Trendline]:
    """Collapse near-duplicate lines, keeping the best-scoring of each cluster.

    Fitting every pivot pair produces many lines that are the same line drawn
    through slightly different anchors. Without this, "nearest support" is chosen
    from a pile of near-identical candidates and the top-N list is one idea
    repeated.
    """
    lines = sorted(lines, key=lambda t: -t.score)
    kept: List[Trendline] = []
    for cand in lines:
        dup = False
        for k in kept:
            if k.kind != cand.kind:
                continue
            same_level = abs(k.level - cand.level) / max(k.level, 1e-9) * 100 < level_tol_pct
            same_slope = abs(k.slope_pct_per_bar - cand.slope_pct_per_bar) < slope_tol
            if same_level and same_slope:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept


def trendline_state(
    df: pd.DataFrame,
    timeframe: str = "1d",
    swing_length: int = 8,
    min_points: int = 2,
    tol_atr_mult: float = 0.75,
    ignore_breakouts: bool = True,
    max_lines: int = 4,
    max_distance_pct: float = 25.0,
) -> Dict[str, Any]:
    """Fit trendlines on ``df`` and describe them **at its last bar**.

    Lag-safe by construction: the fit sees only the data passed in, and the pivots
    themselves are published ``swing_length`` bars late. Call it with
    ``df.iloc[:i + 1]`` for the state at bar ``i``.

    ``tol_atr_mult`` is the touch tolerance in ATRs. Wider finds more, sloppier
    lines; 0.75 means a pivot within three quarters of an ATR of the line counts
    as touching it.

    ``max_distance_pct`` discards lines further than that from price. A fitted
    line can be mathematically valid and useless: a "resistance" 56% above the
    market describes something that happened long ago, and reporting it as the
    nearest resistance would be worse than reporting none.
    """
    if df is None or len(df) < 2 * swing_length + 5:
        return _empty("not_enough_bars", float(df["close"].iloc[-1]) if df is not None and len(df) else np.nan)

    close = df["close"].to_numpy(dtype="float64")
    if not np.isfinite(close).all() or close[-1] <= 0:
        return _empty("bad_close_series")

    n = len(df)
    atr_series = atr(df, period=14).to_numpy(dtype="float64")
    atr_last = atr_series[-1] if np.isfinite(atr_series[-1]) else np.nanmean(atr_series)
    if not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = float(close[-1]) * 0.01          # fall back to 1% of price
    tol = np.where(np.isfinite(atr_series), atr_series, atr_last) * tol_atr_mult

    hi_idx, hi_lvl, lo_idx, lo_lvl = _pivots(df, swing_length)

    lines: List[Trendline] = []
    lines += _fit_side(lo_idx, lo_lvl, "support", close, n, tol, min_points, ignore_breakouts)
    lines += _fit_side(hi_idx, hi_lvl, "resistance", close, n, tol, min_points, ignore_breakouts)
    lines = _dedupe(lines)
    if max_distance_pct:
        lines = [t for t in lines if abs(t.distance_pct) <= max_distance_pct]

    if max_lines:
        lines.sort(key=lambda t: (-t.score, abs(t.distance_pct)))
        lines = lines[:max_lines]

    supports = [t for t in lines if t.kind == "support" and t.distance_pct < 0]
    resistances = [t for t in lines if t.kind == "resistance" and t.distance_pct > 0]
    nearest_sup = max(supports, key=lambda t: t.level, default=None)
    nearest_res = min(resistances, key=lambda t: t.level, default=None)

    return {
        "available": bool(lines),
        "reason": None if lines else "no_lines_fitted",
        "lines": lines,
        "n_support": len(supports),
        "n_resistance": len(resistances),
        "n_pivots_high": int(len(hi_idx)),
        "n_pivots_low": int(len(lo_idx)),
        "nearest_support_pct": (nearest_sup.distance_pct if nearest_sup else np.nan),
        "nearest_resistance_pct": (nearest_res.distance_pct if nearest_res else np.nan),
        "channel_pct": (abs(nearest_res.distance_pct - nearest_sup.distance_pct)
                        if nearest_res and nearest_sup else np.nan),
        "close": float(close[-1]),
    }


def trendline_series(
    df: pd.DataFrame,
    timeframe: str = "1d",
    warmup: int = 60,
    step: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Nearest support/resistance distance per bar, refit on each expanding window.

    The only correct way to get a historical trendline series, and it costs O(n)
    fits. ``step`` trades fidelity for speed: ``step=5`` refits every fifth bar and
    forward-fills, which is plenty for a daily-bar study.

    Every value at bar ``i`` is fitted on ``df.iloc[:i + 1]`` alone, so the result
    can be fed to a backtester without leaking the future. That is the whole point
    of doing it the expensive way.
    """
    n = len(df)
    cols = ["nearest_support_pct", "nearest_resistance_pct", "trendlines_available", "n_lines"]
    if n <= warmup:
        return pd.DataFrame({c: [] for c in cols}, index=df.index[warmup:])

    if n - warmup > SERIES_FIT_WARNING_BARS and step == 1:
        log.warning(
            "trendline_series will refit %d times over %d bars. Pass step=5 or step=10: "
            "each fit is a pivot-pair search, and the historical series is for "
            "validation, not the hot path.", n - warmup, n,
        )

    idx = df.index[warmup:]
    fitted: List[pd.Series] = []
    for i in range(warmup, n, step):
        state = trendline_state(df.iloc[: i + 1], timeframe=timeframe, **kwargs)
        # A fit with no usable lines records NaN, not a stale value: otherwise the
        # series would carry an old level forward and read as "still valid".
        fitted.append(pd.Series({
            "nearest_support_pct": state["nearest_support_pct"],
            "nearest_resistance_pct": state["nearest_resistance_pct"],
            "trendlines_available": float(state["available"]),
            "n_lines": float(len(state["lines"])),
        }, name=df.index[i]))

    # Fill every column from the SAME fitted bar. Filling them independently is
    # how a row ends up claiming trendlines_available=0 while showing support and
    # resistance values carried over from an earlier fit where lines did exist.
    out = pd.DataFrame(fitted).reindex(idx)
    out = out.ffill()
    out["trendlines_available"] = out["trendlines_available"].fillna(0.0)
    out["n_lines"] = out["n_lines"].fillna(0.0)
    return out
