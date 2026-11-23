"""Ranking candidates by what actually makes a reward:risk ratio worth taking.

The naive ranking - sort by ``reward_risk`` descending - is actively harmful, and
this module exists because the scanner used to do exactly that. Reward:risk is a
ratio of two distances, and you can make it arbitrarily large by shrinking the
denominator. So "highest R:R first" systematically promotes the tightest stops,
which are the stops most likely to be hit by ordinary noise. It selects for the
precise thing that loses money.

What makes an R:R worth taking is **expectancy after costs**, and the honest way
to present that is the *breakeven win rate*: the win rate you must exceed for the
idea to make money at your configured costs. That number requires no forecast and
no assumption about the future - it is arithmetic on the stop distance, the target
distance and the fee schedule. It is a threshold, not a prediction.

The mechanism that kills tight stops, made explicit
---------------------------------------------------
Costs are charged on *notional*, but risk is measured in *R*. Converting between
them divides by the stop distance, so a tight stop magnifies costs in R terms::

    cost_R   = round_trip_cost_fraction / stop_distance_fraction
    p_star   = (1 + cost_R) / (1 + R)

At crypto's configured 10 bp fee + 5 bp slippage (30 bp round trip) with a 0.20%
stop, ``cost_R = 0.003 / 0.002 = 1.5``. You are paying 1.5R to put the trade on,
so an apparently handsome R=5 needs a **41.7%** win rate to break even instead of
the 16.7% the ratio implies. The same 30 bp with a 1.5% stop costs 0.2R and needs
20.0%. Same strategy, same R:R on paper, completely different trade.

So this module never reports an R:R without the breakeven win rate next to it,
and it penalises stops that are tight relative to the instrument's own noise
(ATR) or to the risk mandate's ``min_stop_distance_pct``.

Nothing here predicts anything. It ranks the candidates a strategy has *already*
flagged, explains why each ranks where it does, and says plainly when nothing
qualifies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Reward:risk beyond which extra ratio stops being informative. A 20:1 idea is
#: almost always an arbitrary target or a hair-thin stop, not a better trade, so
#: the score saturates instead of letting one absurd ratio dominate the ranking.
RR_SATURATION = 4.0

#: A stop closer than this fraction of ATR is inside the instrument's own noise.
#: It will be hit by ordinary chopping regardless of whether the thesis is right.
MIN_STOP_ATR = 0.5

#: Weights for the composite score. Deliberately explicit and summing to 1.0 so a
#: reader can see what the ranking actually values. These are judgement, not
#: fitted parameters - fitting them on 13 trades would be pure noise mining.
WEIGHTS: Dict[str, float] = {
    "reward_risk": 0.25,      # the ratio, saturated
    "cost_drag": 0.25,        # how much of an R the costs eat (the killer)
    "stop_quality": 0.20,     # is the stop outside the noise and inside the mandate
    "trend": 0.15,            # ADX-weighted agreement with the higher timeframe
    "location": 0.15,         # zone / fresh sweep / structural invalidation nearby
}

#: Verdict bands for the breakeven win rate, given a win rate you have measured.
VERDICT_BANDS = (
    (0.60, "untradeable_at_these_costs"),
    (0.45, "needs_a_win_rate_you_probably_do_not_have"),
    (0.35, "plausible_only_with_a_real_edge"),
    (0.00, "breakeven_is_achievable"),
)


@dataclass(frozen=True)
class Candidate:
    """One ranked idea, with every number needed to judge it independently.

    The fields are ordered by how much they should change your mind: ``rr`` is
    what people look at, ``breakeven_win_rate`` is what they should look at, and
    ``cost_in_r`` is why the two disagree.
    """

    symbol: str
    side: str                          # "long" | "short"
    last: float
    stop: float
    target: Optional[float]
    rr: float                          # reward:risk as a ratio
    risk_pct: float                    # stop distance as a fraction of price
    reward_pct: Optional[float]
    cost_in_r: float                   # round-trip cost expressed in R
    breakeven_win_rate: float          # win rate needed to make money, 0..1
    implied_win_rate_at_rr: float      # 1/(1+rr) - the number people assume
    quality_score: float               # 0..1 composite, higher is better
    atr_pct: float
    stop_atr_multiple: float           # stop distance / ATR
    adx: float
    trend_aligned: bool
    in_zone: bool
    sweep_age: Optional[float]
    structural_stop: bool              # stop sits beyond a fitted trendline
    bars: int
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["reasons"] = list(self.reasons)
        d["warnings"] = list(self.warnings)
        return d

    @property
    def verdict(self) -> str:
        for floor, label in VERDICT_BANDS:
            if self.breakeven_win_rate >= floor:
                return label
        return "breakeven_is_achievable"


# --------------------------------------------------------------------- costs
def round_trip_cost_fraction(fee_bps: float, slippage_bps: float) -> float:
    """Total cost of entering and exiting, as a fraction of notional.

    Both legs pay both costs. Getting this wrong by 2x is the most common way a
    backtest turns a losing intraday strategy into an apparently profitable one.
    """
    return (float(fee_bps) + float(slippage_bps)) * 2.0 / 1e4


def cost_in_r(stop_distance_pct: float, cost_bps_round_trip: float) -> float:
    """Round-trip cost expressed in units of R (multiples of the amount risked).

    This is where a tight stop becomes expensive: costs scale with notional while
    R scales with the stop distance, so the ratio blows up as the stop narrows.
    Returns ``inf`` for a zero-distance stop, which is the honest answer - such a
    trade costs infinitely many R to place.
    """
    d = float(stop_distance_pct)
    if not np.isfinite(d) or d <= 0:
        return float("inf")
    return float(cost_bps_round_trip) / d


def breakeven_win_rate(rr: float, cost_r: float) -> float:
    """The win rate at which expected R is exactly zero, costs included.

    Solves ``p*rr - (1 - p)*1 - cost_r = 0`` for ``p``::

        p = (1 + cost_r) / (1 + rr)

    With ``cost_r = 0`` this reduces to the familiar ``1 / (1 + rr)``. Costs are
    paid whether the trade wins or loses, which is why they sit outside the
    probability weighting and push the threshold *up*.
    """
    if not np.isfinite(rr) or rr <= 0:
        return 1.0                     # no reward: only a 100% win rate helps
    if not np.isfinite(cost_r):
        return 1.0
    return float(min(1.0, (1.0 + cost_r) / (1.0 + rr)))


def implied_win_rate_at_rr(rr: float) -> float:
    """``1 / (1 + rr)`` - the breakeven win rate people assume when they ignore
    costs. Shown next to the real one so the gap is visible rather than implied."""
    if not np.isfinite(rr) or rr <= 0:
        return 1.0
    return float(1.0 / (1.0 + rr))


# ------------------------------------------------------------------ scoring
def _saturate(rr: float) -> float:
    """Map R:R onto 0..1 with diminishing returns past :data:`RR_SATURATION`."""
    if not np.isfinite(rr) or rr <= 0:
        return 0.0
    return float(min(rr, RR_SATURATION) / RR_SATURATION)


def _cost_score(cost_r: float) -> float:
    """1.0 when costs are negligible in R terms, 0.0 when they eat a whole R."""
    if not np.isfinite(cost_r):
        return 0.0
    return float(np.clip(1.0 - cost_r, 0.0, 1.0))


def _stop_score(stop_atr_multiple: float, risk_pct: float, min_stop_pct: float) -> float:
    """Reward a stop that is outside the noise but inside the mandate.

    Three parts, each 0..1:
    - the stop must clear ``min_stop_distance_pct`` from the risk config, or the
      sizer would reject the trade anyway
    - it should sit at least ``MIN_STOP_ATR`` ATRs away, or ordinary chopping
      hits it regardless of whether the thesis was right
    - it should not be absurdly wide either; a 20% stop on a 4h crypto trade is
      not a stop, it is a hope
    """
    parts: List[float] = []
    parts.append(1.0 if risk_pct >= min_stop_pct else float(np.clip(risk_pct / max(min_stop_pct, 1e-9), 0, 1)))
    parts.append(float(np.clip(stop_atr_multiple / MIN_STOP_ATR, 0.0, 1.0)))
    parts.append(float(np.clip(1.0 - max(0.0, stop_atr_multiple - 4.0) / 4.0, 0.0, 1.0)))
    return float(np.mean(parts))


def _trend_score(adx: float, aligned: bool) -> float:
    """A trade with the higher timeframe, weighted by how strongly it is trending.

    ADX says nothing about direction, only about how directional the move is, so
    it is used as a confidence weight on alignment rather than as a signal itself.
    """
    if not aligned:
        return 0.0
    if not np.isfinite(adx):
        return 0.5
    return float(np.clip(adx / 25.0, 0.0, 1.0))


def _location_score(in_zone: bool, sweep_age: Optional[float], structural_stop: bool) -> float:
    """Is the entry at a level that means something?

    An unmitigated FVG/order block, a fresh liquidity sweep on the side the trend
    wants, and a stop placed beyond a fitted trendline are each evidence that the
    invalidation level is a real level rather than an arbitrary ATR multiple.
    """
    parts = [1.0 if in_zone else 0.0,
             1.0 if (sweep_age is not None and sweep_age <= 6) else 0.0,
             1.0 if structural_stop else 0.0]
    return float(np.mean(parts))


def evaluate(
    row: Any,
    fee_bps: float,
    slippage_bps: float,
    min_stop_distance_pct: float = 0.0015,
    atr_pct: Optional[float] = None,
) -> Optional[Candidate]:
    """Turn one :class:`~tvlab.screener.scanner.ScanRow` into a scored Candidate.

    Returns ``None`` when the row cannot be scored - no signal, no stop, or a
    non-finite price. Callers should report those separately rather than letting
    them sort to the bottom and look like merely weak ideas.
    """
    if getattr(row, "signal", 0) == 0:
        return None
    last = float(getattr(row, "last", np.nan))
    stop = row.stop
    if not np.isfinite(last) or last <= 0 or stop is None or not np.isfinite(stop):
        return None

    side = "long" if row.signal > 0 else "short"
    risk_dist = abs(last - stop)
    risk_pct = risk_dist / last
    if risk_pct <= 0:
        return None

    target = row.target
    if target is not None and np.isfinite(target):
        reward_pct = abs(target - last) / last
        rr = reward_pct / risk_pct
    else:
        reward_pct = None
        rr = float(getattr(row, "reward_risk", np.nan) or np.nan)
        if not np.isfinite(rr):
            return None

    row_atr_pct = atr_pct if atr_pct is not None else float(getattr(row, "atr_pct", np.nan))
    atr_pct_v = float(row_atr_pct) if np.isfinite(row_atr_pct) else np.nan
    # atr_pct is expressed in percent, risk_pct is a fraction - convert before dividing
    stop_atr = (risk_pct * 100.0 / atr_pct_v) if np.isfinite(atr_pct_v) and atr_pct_v > 0 else np.nan

    rt_cost = round_trip_cost_fraction(fee_bps, slippage_bps)
    c_r = cost_in_r(risk_pct, rt_cost)
    be = breakeven_win_rate(rr, c_r)

    trend = int(getattr(row, "trend", 0))
    aligned = (trend == 1 and side == "long") or (trend == -1 and side == "short")

    # A stop beyond the nearest fitted trendline is a structural invalidation;
    # inside it, the line will likely be tested before the stop is.
    structural = False
    sup = getattr(row, "support_pct", None)
    res = getattr(row, "resistance_pct", None)
    if side == "long" and sup is not None and np.isfinite(sup):
        structural = abs(sup) / 100.0 >= risk_pct          # support is further away than the stop
    elif side == "short" and res is not None and np.isfinite(res):
        structural = abs(res) / 100.0 >= risk_pct

    s_rr = _saturate(rr)
    s_cost = _cost_score(c_r)
    s_stop = _stop_score(stop_atr if np.isfinite(stop_atr) else 0.0, risk_pct, min_stop_distance_pct)
    s_trend = _trend_score(float(getattr(row, "adx", np.nan)), aligned)
    s_loc = _location_score(bool(getattr(row, "in_zone", False)) or getattr(row, "zones", "-") != "-",
                            getattr(row, "sweep_age", None), structural)

    score = float(WEIGHTS["reward_risk"] * s_rr
                  + WEIGHTS["cost_drag"] * s_cost
                  + WEIGHTS["stop_quality"] * s_stop
                  + WEIGHTS["trend"] * s_trend
                  + WEIGHTS["location"] * s_loc)

    reasons: List[str] = []
    warnings: List[str] = []

    reasons.append(f"R:R {rr:.2f} (target {reward_pct * 100:.2f}% vs stop {risk_pct * 100:.2f}%)"
                   if reward_pct is not None else f"R:R {rr:.2f} from the configured R multiple")
    reasons.append(f"costs eat {c_r:.2f}R of the {rr:.2f}R reward"
                   if np.isfinite(c_r) else "a zero-distance stop makes costs infinite in R terms")
    reasons.append(f"breakeven win rate {be * 100:.1f}% vs {implied_win_rate_at_rr(rr) * 100:.1f}% if costs were zero")
    if aligned:
        reasons.append(f"with the higher-timeframe trend (ADX {getattr(row, 'adx', float('nan')):.1f})")
    if getattr(row, "zones", "-") != "-":
        reasons.append(f"price inside {row.zones}")
    if structural:
        reasons.append("stop sits beyond the nearest fitted trendline, so the invalidation is structural")

    if risk_pct < min_stop_distance_pct:
        warnings.append(f"stop is {risk_pct * 100:.3f}% away, below the mandate's "
                        f"min_stop_distance_pct {min_stop_distance_pct * 100:.3f}% - the sizer will reject it")
    if np.isfinite(stop_atr) and stop_atr < MIN_STOP_ATR:
        warnings.append(f"stop is only {stop_atr:.2f} ATR away (needs {MIN_STOP_ATR}) - ordinary noise will hit it")
    if not aligned:
        warnings.append("signal is against the higher-timeframe trend")
    if np.isfinite(c_r) and c_r >= 1.0:
        warnings.append(f"you pay {c_r:.2f}R just to place this trade - the R:R on paper is not the R:R you get")
    if be >= 0.60:
        warnings.append(f"needs a {be * 100:.0f}% win rate; almost no cost-bearing strategy sustains that")
    if getattr(row, "bars", 0) < 200:
        warnings.append(f"only {row.bars} bars of history - levels are fitted on very little data")

    return Candidate(
        symbol=row.symbol, side=side, last=last, stop=float(stop), target=target,
        rr=float(rr), risk_pct=float(risk_pct),
        reward_pct=(float(reward_pct) if reward_pct is not None else None),
        cost_in_r=float(c_r) if np.isfinite(c_r) else float("inf"),
        breakeven_win_rate=be, implied_win_rate_at_rr=implied_win_rate_at_rr(rr),
        quality_score=score, atr_pct=atr_pct_v,
        stop_atr_multiple=float(stop_atr) if np.isfinite(stop_atr) else float("nan"),
        adx=float(getattr(row, "adx", np.nan)), trend_aligned=aligned,
        in_zone=getattr(row, "zones", "-") != "-", sweep_age=getattr(row, "sweep_age", None),
        structural_stop=structural, bars=int(getattr(row, "bars", 0)),
        reasons=tuple(reasons), warnings=tuple(warnings),
    )


def suggest(
    rows: Sequence[Any],
    fee_bps: float,
    slippage_bps: float,
    min_stop_distance_pct: float = 0.0015,
    max_breakeven: float = 0.60,
    min_bars: int = 100,
) -> Dict[str, Any]:
    """Rank a scanned universe into an ordered shortlist with reasoning.

    Deliberately returns a *dict with a rejected pile* rather than a bare list.
    A ranking that silently drops candidates hides the reason there is nothing to
    trade; one that shows "7 rejected: stop inside the noise" is informative.

    ``max_breakeven`` is the honesty gate: a candidate needing a higher win rate
    than this to break even is reported as rejected, not ranked below the others.
    Presenting it as merely "lower quality" would imply it is a worse version of
    a good idea, when in fact it is untradeable at these costs.
    """
    candidates: List[Candidate] = []
    unscoreable: List[str] = []
    rejected: List[Dict[str, Any]] = []

    for row in rows:
        c = evaluate(row, fee_bps, slippage_bps, min_stop_distance_pct)
        if c is None:
            if getattr(row, "signal", 0) != 0:
                unscoreable.append(row.symbol)
            continue
        why: List[str] = []
        if c.breakeven_win_rate > max_breakeven:
            why.append(f"needs a {c.breakeven_win_rate * 100:.0f}% win rate to break even")
        if c.bars < min_bars:
            why.append(f"only {c.bars} bars of history")
        if c.risk_pct < min_stop_distance_pct:
            why.append("stop tighter than the risk mandate allows")
        if why:
            rejected.append({"symbol": c.symbol, "side": c.side, "rr": c.rr,
                             "breakeven_win_rate": c.breakeven_win_rate,
                             "reasons": why})
            continue
        candidates.append(c)

    # Highest quality first. Ties break on the lower breakeven win rate, then on
    # symbol so the ordering is deterministic - an unstable shortlist is
    # unreviewable, and two runs on the same data must agree.
    candidates.sort(key=lambda c: (-c.quality_score, c.breakeven_win_rate, c.symbol))

    return {
        "candidates": candidates,
        "rejected": rejected,
        "unscoreable": unscoreable,
        "n_candidates": len(candidates),
        "n_rejected": len(rejected),
        "max_breakeven": max_breakeven,
        "round_trip_cost_bps": (float(fee_bps) + float(slippage_bps)) * 2.0,
        "weights": dict(WEIGHTS),
    }
