"""Pattern-recovery test: does the detector actually find what it claims to find?

The shuffled-null test asks "is there an edge?". This asks a different and more
basic question: "does the signal logic work at all?"

Method
------
:func:`tvlab.data.synthetic.basket` can inject a known number of
liquidity-sweep-then-reversal events at known bars
(``inject_sweeps=N``). The events are recorded in
``df.attrs['injected_sweeps']``. We then run the detector over that series and
score it like any classifier against known ground truth:

* **recall** - of the injected sweeps, how many did we detect?
* **precision** - of the sweeps we flagged, how many were real?
* **side accuracy** - did we get the direction right?
* **latency** - how many bars after the event did the flag appear?

Low recall means the detector is too strict and will miss live setups. Low
precision means it fires on noise and will generate losing trades. Either way
this is knowable *without any market data at all*, which is exactly why it is
worth running before you spend money on a data feed.

The tolerance window matters: a detector that fires within a few bars of the
event is a hit, because the entry does not have to be on the exact bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    injected: int
    detected: int
    false_positives: int
    total_flags: int
    tolerance: int
    precision: float
    recall: float
    f1: float
    side_accuracy: float
    mean_latency: float
    median_latency: float
    eligible: int = 0
    eligible_detected: int = 0
    eligible_recall: float = float("nan")
    blocked_by_filter: int = 0
    per_event: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "injected": self.injected, "detected": self.detected,
            "false_positives": self.false_positives, "total_flags": self.total_flags,
            "tolerance_bars": self.tolerance,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "side_accuracy": self.side_accuracy,
            "mean_latency_bars": self.mean_latency,
            "median_latency_bars": self.median_latency,
            "eligible_events": self.eligible,
            "eligible_detected": self.eligible_detected,
            "eligible_recall": self.eligible_recall,
            "blocked_by_filter": self.blocked_by_filter,
            "per_event": self.per_event[:20],
            "warnings": self.warnings,
        }


def score_recovery(
    df: pd.DataFrame,
    flag_series: pd.Series,
    tolerance: int = 6,
    side_series: Optional[pd.Series] = None,
    eligible: Optional[pd.Series] = None,
) -> RecoveryResult:
    """Score a boolean flag series against ``df.attrs['injected_sweeps']``.

    Parameters
    ----------
    flag_series:
        bool/int series on ``df.index`` - True where the detector fired. May be
        either side; if ``side_series`` is given it is used to check direction.
    tolerance:
        a flag within ``tolerance`` bars *at or after* the injected event counts
        as a detection. Flags before the event are not credited - a detector
        cannot know about an event that has not happened.
    side_series:
        +1/-1 per bar giving the detector's claimed direction, to score side
        accuracy against the injected side.
    eligible:
        bool series marking bars where the strategy was *allowed* to trade at all
        (e.g. the higher-timeframe trend filter agreed). Injected sweeps are
        direction-agnostic, so a trend-following strategy is structurally unable
        to act on the ~half that point against the trend - scoring it against all
        of them measures the filter, not the detector. When given, we also report
        ``eligible_recall``: recall over only the events the strategy could have
        taken. That is the number to tune against.
    """
    events = df.attrs.get("injected_sweeps") or []
    warnings: List[str] = []
    flags = flag_series.reindex(df.index).fillna(False).astype(bool).to_numpy()
    positions = np.flatnonzero(flags)
    total_flags = int(len(positions))

    if not events:
        return RecoveryResult(
            injected=0, detected=0, false_positives=total_flags, total_flags=total_flags,
            tolerance=tolerance, precision=0.0, recall=float("nan"), f1=0.0,
            side_accuracy=float("nan"), mean_latency=float("nan"),
            median_latency=float("nan"),
            warnings=["no injected events in df.attrs['injected_sweeps']; "
                      "generate data with inject_sweeps=N to run this test"],
        )

    idx_of = {ts: i for i, ts in enumerate(df.index)}
    sides = (side_series.reindex(df.index).to_numpy(dtype="float64")
             if side_series is not None else None)
    elig = (eligible.reindex(df.index).fillna(False).astype(bool).to_numpy()
            if eligible is not None else None)

    per_event: List[Dict[str, Any]] = []
    matched_flags: set[int] = set()
    detected = 0
    side_hits = 0
    latencies: List[int] = []
    n_eligible = 0
    n_eligible_hit = 0
    n_blocked = 0

    for ev in events:
        ts = pd.Timestamp(ev["timestamp"])
        want = 1 if ev["side"] == "bullish" else -1
        if ts not in idx_of:
            warnings.append(f"injected event at {ts} is not on the frame index (session filter?)")
            per_event.append({"timestamp": str(ts), "side": ev["side"],
                              "found": False, "latency": None, "reason": "not_on_index"})
            continue

        i0 = idx_of[ts]
        was_eligible = bool(elig[i0]) if elig is not None else True
        if elig is not None:
            if was_eligible:
                n_eligible += 1
            else:
                n_blocked += 1
        window = positions[(positions >= i0) & (positions <= i0 + tolerance)]
        if len(window) == 0:
            reason = "no_flag_in_window" if was_eligible else "blocked_by_filter"
            per_event.append({"timestamp": str(ts), "side": ev["side"], "found": False,
                              "latency": None, "reason": reason, "eligible": was_eligible})
            continue
        if was_eligible:
            n_eligible_hit += 1

        hit = int(window[0])
        matched_flags.add(hit)
        detected += 1
        latencies.append(hit - i0)
        side_ok = None
        if sides is not None:
            claimed = sides[hit]
            side_ok = bool(np.sign(claimed) == want) if claimed != 0 else False
            if side_ok:
                side_hits += 1
        per_event.append({"timestamp": str(ts), "side": ev["side"], "found": True,
                          "latency": int(hit - i0), "side_correct": side_ok,
                          "eligible": was_eligible})

    injected = len(events)
    fp = total_flags - len(matched_flags)
    precision = detected / total_flags if total_flags else float("nan")
    recall = detected / injected if injected else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (total_flags and injected and (precision + recall) > 0) else 0.0)
    side_acc = (side_hits / detected) if (sides is not None and detected) else float("nan")

    elig_recall = (n_eligible_hit / n_eligible) if n_eligible else float("nan")
    if injected and recall < 0.5 and elig is None:
        warnings.append(
            f"recall is only {recall:.0%}: the detector misses more than half of "
            "known sweep events. Loosen swing_length / max_sweep_age before "
            "concluding the setup does not exist in live data."
        )
    if elig is not None and n_blocked:
        warnings.append(
            f"{n_blocked}/{injected} injected events were blocked by the eligibility "
            f"filter (trend/confirmation), not by the detector. Raw recall {recall:.0%} "
            f"understates detection; recall over eligible events is {elig_recall:.0%}."
        )
    if total_flags and precision < 0.3:
        warnings.append(
            f"precision is only {precision:.0%}: most flags are not injected events. "
            "Some of these are genuine patterns the generator did not inject, but "
            "at this level the detector is largely firing on noise."
        )

    return RecoveryResult(
        injected=injected, detected=detected, false_positives=fp, total_flags=total_flags,
        tolerance=tolerance, precision=precision, recall=recall, f1=f1,
        side_accuracy=side_acc,
        mean_latency=float(np.mean(latencies)) if latencies else float("nan"),
        median_latency=float(np.median(latencies)) if latencies else float("nan"),
        eligible=n_eligible, eligible_detected=n_eligible_hit,
        eligible_recall=elig_recall, blocked_by_filter=n_blocked,
        per_event=per_event, warnings=warnings,
    )


def sweep_detector_sides(features: pd.DataFrame) -> pd.Series:
    """+1 where a bullish sweep fired, -1 where a bearish sweep fired, else 0."""
    out = pd.Series(0.0, index=features.index)
    out[features["sweep_low"].astype(bool)] = 1.0
    out[features["sweep_high"].astype(bool)] = -1.0
    return out


def run_smc_recovery(
    df: pd.DataFrame, swing_length: int = 12, tolerance: int = 6,
    use_strategy: bool = False, strategy=None, bias: Optional[pd.DataFrame] = None,
) -> RecoveryResult:
    """Recovery test for the SMC sweep detector.

    ``use_strategy=False`` scores the raw detector (``sweep_low``/``sweep_high``).
    ``use_strategy=True`` scores the *strategy's entries*, which is stricter: the
    trend filter and confirmation rules will reject some genuine sweeps, so
    recall drops. Both numbers are useful - the gap between them is the cost of
    the filters.
    """
    from ..indicators.smc import smc_features

    feats = smc_features(df, swing_length=swing_length)
    if not use_strategy:
        flags = (feats["sweep_low"] | feats["sweep_high"]).astype(bool)
        sides = sweep_detector_sides(feats)
        return score_recovery(df, flags, tolerance=tolerance, side_series=sides)

    if strategy is None:
        raise ValueError("use_strategy=True requires a strategy instance")
    sig = strategy.signals(df, bias)
    flags = (sig["signal"].abs() > 0)

    # Eligibility = the higher-timeframe trend filter pointed the same way as the
    # injected event. Without it, recall mostly measures the trend filter.
    from ..strategy.base import trend_filter

    elig = None
    try:
        p = strategy.params
        trend = trend_filter(df, bias, int(p.get("bias_ema", 50)),
                             int(p.get("bias_adx_period", 14)),
                             float(p.get("bias_adx_min", 0.0))).to_numpy()
        elig = pd.Series(False, index=df.index)
        for ev in (df.attrs.get("injected_sweeps") or []):
            ts = pd.Timestamp(ev["timestamp"])
            if ts not in elig.index:
                continue
            want = 1 if ev["side"] == "bullish" else -1
            elig.loc[ts] = bool(trend[df.index.get_loc(ts)] == want)
        # bars with no injected event stay False; score_recovery only looks at events
    except Exception as exc:  # noqa: BLE001
        log.warning("could not build the eligibility mask: %s", exc)

    return score_recovery(df, flags, tolerance=tolerance, side_series=sig["signal"],
                          eligible=elig)


def sweep_size_sensitivity(
    make_df: Callable[[int], pd.DataFrame],
    swing_lengths: Sequence[int] = (6, 8, 12, 16, 20),
    tolerance: int = 6,
) -> pd.DataFrame:
    """Recall/precision across swing lengths - pick the parameter with evidence.

    ``make_df(n_sweeps)`` should return a freshly generated frame with ``n``
    injected events, so every row of the output is scored on the same truth.
    """
    rows = []
    for L in swing_lengths:
        from ..indicators.smc import smc_features

        df = make_df(L)
        feats = smc_features(df, swing_length=L)
        flags = (feats["sweep_low"] | feats["sweep_high"]).astype(bool)
        r = score_recovery(df, flags, tolerance=tolerance,
                           side_series=sweep_detector_sides(feats))
        rows.append({
            "swing_length": L, "injected": r.injected, "detected": r.detected,
            "flags": r.total_flags, "precision": r.precision, "recall": r.recall,
            "f1": r.f1, "side_accuracy": r.side_accuracy,
            "mean_latency": r.mean_latency,
        })
    return pd.DataFrame(rows)
