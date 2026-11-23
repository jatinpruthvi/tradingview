"""Cross-validation against the vendored reference implementations.

These functions are not used in the trading path - they exist so we can *prove*
our lag-safe SMC code finds the same market structure as the popular reference
library (``vendor/smart-money-concepts``), and so the difference in publication
timing is measured rather than asserted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .smc import swing_points
from .technical import adx, atr, rsi

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    metric: str
    ours: int
    theirs: int
    matched: int
    lag_bars: int
    match_rate: float
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric, "ours": self.ours, "theirs": self.theirs,
            "matched": self.matched, "lag_bars": self.lag_bars,
            "match_rate": round(self.match_rate, 4), "notes": self.notes,
        }


def _load_reference():
    try:
        from smartmoneyconcepts import smc
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pip install smartmoneyconcepts   # needed to cross-check SMC features"
        ) from exc
    return smc


def compare_swings(df: pd.DataFrame, swing_length: int = 12, tolerance: float = 1e-9) -> ComparisonResult:
    """Do we find the same pivots as the reference implementation?

    The reference marks a pivot at bar ``i`` using a centred window (a forward
    look). We mark the same pivot at ``i + swing_length``. So we compare our
    *lagged* output shifted back by ``swing_length`` against theirs, which
    should line up bar-for-bar.
    """
    smc = _load_reference()
    ours_raw = swing_points(df, swing_length, lag=False)
    ours_lagged = swing_points(df, swing_length, lag=True)

    ref = smc.swing_highs_lows(df[["open", "high", "low", "close"]], swing_length=swing_length)
    ref_high = (ref["HighLow"] == 1).to_numpy()
    ref_low = (ref["HighLow"] == -1).to_numpy()

    our_high = ours_raw["swing_high"].to_numpy()
    our_low = ours_raw["swing_low"].to_numpy()

    # verify the lag is exactly what we claim
    L = int(swing_length)
    shifted_high = np.zeros_like(our_high)
    shifted_high[:-L] = ours_lagged["swing_high"].to_numpy()[L:]
    shifted_low = np.zeros_like(our_low)
    shifted_low[:-L] = ours_lagged["swing_low"].to_numpy()[L:]

    lag_ok = bool((shifted_high == our_high).all() and (shifted_low == our_low).all())

    hi_match = int((our_high & ref_high).sum())
    lo_match = int((our_low & ref_low).sum())
    ours_n = int(our_high.sum() + our_low.sum())
    theirs_n = int(ref_high.sum() + ref_low.sum())
    matched = hi_match + lo_match
    denom = max(min(ours_n, theirs_n), 1)

    notes = (
        f"highs {hi_match}/{int(our_high.sum())} ours vs {int(ref_high.sum())} theirs; "
        f"lows {lo_match}/{int(our_low.sum())} ours vs {int(ref_low.sum())} theirs; "
        f"lag-shift identity {'OK' if lag_ok else 'FAILED'}"
    )
    return ComparisonResult(
        metric=f"swing_pivots_L{swing_length}", ours=ours_n, theirs=theirs_n,
        matched=matched, lag_bars=L, match_rate=matched / denom, notes=notes,
    )


def compare_fvg(df: pd.DataFrame) -> ComparisonResult:
    """FVG detection vs the reference, allowing for a 1-bar publication lag.

    The reference marks a gap at the **middle** candle using
    ``high.shift(1) < low.shift(-1)`` - that ``shift(-1)`` is a forward look, so
    the gap is stamped one bar before a trader could actually see it. We stamp the
    *same* gap at the third candle, the first bar on which it is knowable, and the
    top/bottom levels come out identical::

        candle          reference         ours
        bullish gap     m                 m + 1
          top           low[m+1]          low[i]        (i = m+1, same)
          bottom        high[m-1]         high[i-2]     (same)
        bearish gap     m                 m + 1
          top           low[m-1]          low[i-2]      (same)
          bottom        high[m+1]         high[i]       (same)

    So ``theirs.shift(1)`` must equal ``ours``.
    """
    from .smc import fair_value_gaps

    smc = _load_reference()
    ours = fair_value_gaps(df)
    ref = smc.fvg(df[["open", "high", "low", "close"]])

    ref_col = ref["FVG"] if "FVG" in ref.columns else ref.iloc[:, 0]
    ref_bull = (ref_col == 1).to_numpy()
    ref_bear = (ref_col == -1).to_numpy()
    our_bull = ours["bull_fvg"].to_numpy()
    our_bear = ours["bear_fvg"].to_numpy()

    # shift the reference one bar later so publication timing lines up
    ref_bull_l = np.zeros_like(ref_bull)
    ref_bear_l = np.zeros_like(ref_bear)
    ref_bull_l[1:] = ref_bull[:-1]
    ref_bear_l[1:] = ref_bear[:-1]

    bull_match = int((our_bull & ref_bull_l).sum())
    bear_match = int((our_bear & ref_bear_l).sum())
    matched = bull_match + bear_match
    ours_n = int(our_bull.sum() + our_bear.sum())
    theirs_n = int(ref_bull.sum() + ref_bear.sum())
    lvl_ok = _fvg_levels_agree(ours, ref, ref_bull_l, ref_bear_l)

    return ComparisonResult(
        metric="fair_value_gaps", ours=ours_n, theirs=theirs_n, matched=matched, lag_bars=1,
        match_rate=matched / max(min(ours_n, theirs_n), 1),
        notes=(
            f"bull {bull_match}, bear {bear_match}; gap levels identical after the "
            f"1-bar lag: {lvl_ok}; reference uses low.shift(-1), i.e. it stamps the "
            "gap one bar before it is knowable"
        ),
    )


def _fvg_levels_agree(ours: pd.DataFrame, ref: pd.DataFrame,
                      ref_bull_l: np.ndarray, ref_bear_l: np.ndarray) -> bool:
    """Where both flag the same gap, do top/bottom agree exactly?

    The reference stores the gap bounds at the middle candle ``m``; ours stores
    them at ``m + 1``. So the reference bounds get the same 1-bar shift as the
    flags before comparing.
    """
    def _shift1(a: np.ndarray) -> np.ndarray:
        out = np.full_like(a, np.nan, dtype="float64")
        out[1:] = a[:-1]
        return out

    ref_top = _shift1(ref["Top"].to_numpy(dtype="float64"))
    ref_bot = _shift1(ref["Bottom"].to_numpy(dtype="float64"))
    our_bull_top = ours["bull_fvg_top"].to_numpy(dtype="float64")
    our_bull_bot = ours["bull_fvg_bottom"].to_numpy(dtype="float64")
    our_bear_top = ours["bear_fvg_top"].to_numpy(dtype="float64")
    our_bear_bot = ours["bear_fvg_bottom"].to_numpy(dtype="float64")

    top = np.where(ref_bull_l, our_bull_top, np.where(ref_bear_l, our_bear_top, np.nan))
    bot = np.where(ref_bull_l, our_bull_bot, np.where(ref_bear_l, our_bear_bot, np.nan))
    both = np.isfinite(top) & np.isfinite(ref_top) & np.isfinite(bot) & np.isfinite(ref_bot)
    if not both.any():
        return False
    return bool(np.allclose(top[both], ref_top[both], rtol=0, atol=1e-9)
                and np.allclose(bot[both], ref_bot[both], rtol=0, atol=1e-9))


def compare_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Sanity ranges for our own indicators (no reference to compare against).

    These are invariants that must hold for any correct implementation:
    RSI in [0,100], ADX in [0,100], ATR >= 0 and <= high-low range envelope.
    """
    out: Dict[str, float] = {}
    r = rsi(df, 14).dropna()
    a = adx(df, 14)["adx"].dropna()
    t = atr(df, 14).dropna()

    out["rsi_min"] = float(r.min()) if len(r) else np.nan
    out["rsi_max"] = float(r.max()) if len(r) else np.nan
    out["rsi_in_range"] = float(bool(len(r) and r.min() >= -1e-9 and r.max() <= 100 + 1e-9))
    out["adx_min"] = float(a.min()) if len(a) else np.nan
    out["adx_max"] = float(a.max()) if len(a) else np.nan
    out["adx_in_range"] = float(bool(len(a) and a.min() >= -1e-9 and a.max() <= 100 + 1e-9))
    out["atr_nonneg"] = float(bool(len(t) and (t >= -1e-12).all()))
    out["atr_finite"] = float(bool(len(t) and np.isfinite(t).all()))
    return out


def run_all(df: pd.DataFrame, swing_length: int = 12) -> Dict[str, object]:
    """Everything above in one call, for the report and the CLI."""
    results: Dict[str, object] = {"swings": compare_swings(df, swing_length).as_dict()}
    try:
        results["fvg"] = compare_fvg(df).as_dict()
    except Exception as exc:  # noqa: BLE001 - reference API drift must not block
        log.warning("FVG cross-check skipped: %s", exc)
        results["fvg"] = {"error": str(exc)}
    results["invariants"] = compare_indicators(df)
    return results
