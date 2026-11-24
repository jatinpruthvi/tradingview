"""Null-hypothesis testing: does the strategy beat randomness?

THE most important file in this package
---------------------------------------
A backtest that makes money proves nothing on its own, because *something* has
to make money on any given price series. The question is whether the strategy
makes money because it found structure, or because the sample happened to drift
in its favour.

Two nulls are implemented here.

1. :func:`shuffled_null` - **permuted returns**. Take the bar returns, shuffle
   their order, rebuild a price series from them. This preserves the return
   *distribution* exactly (same vol, same fat tails, same total drift) while
   destroying every bit of serial structure: trends, mean reversion, volatility
   clustering, the lot. A strategy with no real edge earns, in expectation,
   exactly minus its transaction costs on this data.

   If your strategy shows a profit factor of 2 on real data and 1.8 on shuffled
   data, the 1.8 is not edge - it is an artefact (usually lookahead, or a
   long-only bias on a series that drifts up).

2. :func:`sign_null` - **random entry**. Keep the real price series but replace
   the signals with random ones at the same trade frequency. This isolates
   *timing* from *direction*: if random entries at the same frequency do as well,
   the entry logic adds nothing and only the exit/risk rules matter.

Both return a p-value: the fraction of null runs that beat the actual result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class NullResult:
    kind: str
    statistic: str
    actual: float
    null_mean: float
    null_std: float
    p_value: float
    n_sims: int
    better_than_null_pct: float
    percentiles: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def verdict(self) -> str:
        if self.p_value < 0.01:
            return "strong"
        if self.p_value < 0.05:
            return "significant"
        if self.p_value < 0.20:
            return "weak"
        return "indistinguishable_from_random"

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "statistic": self.statistic,
            "actual": self.actual, "null_mean": self.null_mean,
            "null_std": self.null_std, "p_value": self.p_value,
            "n_sims": self.n_sims,
            "better_than_null_pct": self.better_than_null_pct,
            "percentiles": self.percentiles,
            "verdict": self.verdict, "significant": self.significant,
            "warnings": self.warnings,
        }


def rebuild_from_returns(returns: np.ndarray, start_price: float,
                         index: pd.DatetimeIndex) -> pd.DataFrame:
    """Rebuild a plausible OHLCV frame from a shuffled return sequence.

    Open is the previous close; the body comes from the return; wicks are drawn
    from the same empirical distribution of wick sizes as the original data would
    have, scaled to the bar. Without wicks the strategy's stop/target logic would
    never trigger and the null would be trivially untradeable.
    """
    n = len(returns)
    close = start_price * np.exp(np.cumsum(returns))
    prev = np.concatenate([[start_price], close[:-1]])
    open_ = prev.copy()

    # wick magnitude scales with |return| so stops behave comparably
    rng = np.random.default_rng(0)
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    scale = np.abs(returns) + 1e-9
    wick = scale * rng.uniform(0.2, 1.2, n)
    high = body_hi * (1.0 + wick)
    low = body_lo * (1.0 - wick)
    volume = np.full(n, 1_000_000.0)

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume}, index=index)


def shuffled_null(
    df: pd.DataFrame,
    run_once: Callable[[pd.DataFrame], Dict[str, float]],
    n_sims: int = 100,
    statistic: str = "total_return_pct",
    seed: int = 101,
    preserve_vol_clustering: bool = False,
) -> NullResult:
    """Permutation test on bar returns.

    ``run_once(df)`` must return a metrics dict (typically
    :func:`tvlab.backtest.metrics.full_report` output for a fixed strategy and
    config). It is called once on the real data and ``n_sims`` times on permuted
    data.

    ``preserve_vol_clustering`` shuffles within volatility blocks instead of
    bar-by-bar. That keeps GARCH-style clustering alive and is the *stricter*
    null: a trend-follower that only profits because volatility clusters will
    fail it. Use the plain shuffle first, this one to be sure.
    """
    if len(df) < 100:
        raise ValueError("need at least 100 bars for a permutation test")

    real = run_once(df)
    actual = float(real.get(statistic, float("nan")))
    if not np.isfinite(actual):
        return NullResult(
            kind="shuffled_returns", statistic=statistic, actual=actual,
            null_mean=float("nan"), null_std=float("nan"), p_value=float("nan"),
            n_sims=0, better_than_null_pct=float("nan"),
            warnings=[f"actual statistic {statistic!r} is not finite: {actual}; "
                      "did the strategy take any trades?"],
        )

    rets = np.log(df["close"] / df["close"].shift(1)).dropna().to_numpy(dtype="float64")
    rets = rets[np.isfinite(rets)]
    start_price = float(df["close"].iloc[0])
    index = df.index

    rng = np.random.default_rng(seed)
    nulls: List[float] = []
    failed = 0
    block = 20 if preserve_vol_clustering else 1

    for _ in range(n_sims):
        if block > 1 and len(rets) > block:
            perm = _block_shuffle(rets, block, rng)
        else:
            perm = rng.permutation(rets)
        # rebuild on the same index (trimmed to the return length)
        sim_df = rebuild_from_returns(perm, start_price, index[-len(perm):])
        try:
            m = run_once(sim_df)
            v = float(m.get(statistic, float("nan")))
            if np.isfinite(v):
                nulls.append(v)
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001 - a failing sim is informative, not fatal
            failed += 1
            log.debug("null sim failed: %s", exc)

    warnings: List[str] = []
    if failed:
        warnings.append(f"{failed}/{n_sims} null simulations failed to produce a result")
    if len(nulls) < 10:
        warnings.append(f"only {len(nulls)} usable null runs; the p-value is unreliable")
    if preserve_vol_clustering:
        warnings.append("volatility clustering was preserved (block shuffle) - stricter null")

    arr = np.array(nulls, dtype="float64")
    if arr.size == 0:
        return NullResult(kind="shuffled_returns", statistic=statistic, actual=actual,
                          null_mean=float("nan"), null_std=float("nan"),
                          p_value=float("nan"), n_sims=0, better_than_null_pct=float("nan"),
                          warnings=warnings + ["all null simulations failed"])

    # one-sided: how often does pure noise do at least as well as the strategy?
    p = float(np.mean(arr >= actual))
    return NullResult(
        kind="shuffled_returns" + ("_block" if block > 1 else ""),
        statistic=statistic, actual=actual,
        null_mean=float(arr.mean()), null_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p_value=p, n_sims=int(arr.size),
        better_than_null_pct=float(np.mean(arr < actual) * 100.0),
        percentiles={f"p{int(q*100)}": float(np.quantile(arr, q))
                     for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        warnings=warnings,
    )


def _block_shuffle(a: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle contiguous blocks - preserves local dependence, destroys long-run order."""
    n = len(a)
    n_blocks = int(np.ceil(n / block))
    pad = n_blocks * block - n
    padded = np.concatenate([a, a[:pad]]) if pad else a
    blocks = padded.reshape(n_blocks, block)
    order = rng.permutation(n_blocks)
    return blocks[order].ravel()[:n]


def sign_null(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    run_on_signals: Callable[[pd.DataFrame, pd.DataFrame], Dict[str, float]],
    n_sims: int = 100,
    statistic: str = "total_return_pct",
    seed: int = 202,
) -> NullResult:
    """Random-entry null at matched trade frequency.

    Keeps the real price path and replaces the signal *direction* with a random
    one on the same bars the strategy chose to trade. This isolates timing from
    direction:

    * actual >> null  -> the direction call carries the edge
    * actual ~= null  -> the entry bars are meaningless; only the exits/risk rules
      are doing work (which is still a valid system, but you should know it)
    """
    real = run_on_signals(df, signals)
    actual = float(real.get(statistic, float("nan")))
    if not np.isfinite(actual):
        return NullResult(kind="random_entry", statistic=statistic, actual=actual,
                          null_mean=float("nan"), null_std=float("nan"),
                          p_value=float("nan"), n_sims=0, better_than_null_pct=float("nan"),
                          warnings=["actual statistic not finite; no trades taken?"])

    sig = signals["signal"].to_numpy(dtype="float64")
    entry_mask = sig != 0
    n_entries = int(entry_mask.sum())
    if n_entries == 0:
        return NullResult(kind="random_entry", statistic=statistic, actual=actual,
                          null_mean=float("nan"), null_std=float("nan"),
                          p_value=float("nan"), n_sims=0, better_than_null_pct=float("nan"),
                          warnings=["strategy took no entries; cannot build a random-entry null"])

    rng = np.random.default_rng(seed)
    nulls: List[float] = []
    failed = 0
    for _ in range(n_sims):
        rand = rng.choice([-1.0, 1.0], size=n_entries)
        sim_sig = signals.copy()
        new = np.zeros_like(sig)
        new[entry_mask] = rand
        sim_sig["signal"] = new
        # random directions invalidate the strategy's own stop/target levels;
        # mirror them to stay consistent with the new direction
        if "stop" in sim_sig and "target" in sim_sig:
            close = df["close"].to_numpy(dtype="float64")
            stop = sim_sig["stop"].to_numpy(dtype="float64").copy()
            tgt = sim_sig["target"].to_numpy(dtype="float64").copy()
            flipped = new < 0
            # a long stop sits below price; a short stop sits above
            stop[flipped] = close[flipped] + np.abs(close[flipped] - stop[flipped])
            tgt[flipped] = close[flipped] - np.abs(tgt[flipped] - close[flipped])
            stop[~flipped] = close[~flipped] - np.abs(close[~flipped] - stop[~flipped])
            tgt[~flipped] = close[~flipped] + np.abs(tgt[~flipped] - close[~flipped])
            sim_sig["stop"], sim_sig["target"] = stop, tgt
        try:
            m = run_on_signals(df, sim_sig)
            v = float(m.get(statistic, float("nan")))
            if np.isfinite(v):
                nulls.append(v)
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.debug("sign-null sim failed: %s", exc)

    warnings: List[str] = [f"{n_entries} entry bars resampled"]
    if failed:
        warnings.append(f"{failed}/{n_sims} simulations failed")
    arr = np.array(nulls, dtype="float64")
    if arr.size == 0:
        return NullResult(kind="random_entry", statistic=statistic, actual=actual,
                          null_mean=float("nan"), null_std=float("nan"), p_value=float("nan"),
                          n_sims=0, better_than_null_pct=float("nan"),
                          warnings=warnings + ["all simulations failed"])
    p = float(np.mean(arr >= actual))
    return NullResult(
        kind="random_entry", statistic=statistic, actual=actual,
        null_mean=float(arr.mean()),
        null_std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p_value=p, n_sims=int(arr.size),
        better_than_null_pct=float(np.mean(arr < actual) * 100.0),
        percentiles={f"p{int(q*100)}": float(np.quantile(arr, q))
                     for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        warnings=warnings,
    )


def interpret(results: Dict[str, NullResult]) -> str:
    """Plain-English reading of the null tests, for the report."""
    lines: List[str] = []
    sh = results.get("shuffled_returns")
    sg = results.get("random_entry")

    if sh:
        if sh.p_value != sh.p_value:  # NaN
            lines.append(
                "- **Shuffled-return null:** could not be evaluated "
                f"({'; '.join(sh.warnings)})."
            )
        elif sh.p_value < 0.05:
            lines.append(
                f"- **Shuffled-return null: PASS** (p={sh.p_value:.3f}). The strategy beat "
                f"{sh.better_than_null_pct:.0f}% of {sh.n_sims} permutations of the same "
                "returns. Serial structure - not just the return distribution - is being used."
            )
        else:
            lines.append(
                f"- **Shuffled-return null: FAIL** (p={sh.p_value:.3f}). Random orderings of "
                f"the same returns did at least as well {sh.p_value*100:.0f}% of the time. "
                "The reported edge is not distinguishable from the sample's drift and "
                "volatility. Do not trade this on the strength of the backtest."
            )

    if sg:
        if sg.p_value == sg.p_value and sg.p_value < 0.05:
            lines.append(
                f"- **Random-entry null: PASS** (p={sg.p_value:.3f}). Entering on the same bars "
                "but with random direction did worse, so the direction call carries information."
            )
        elif sg.p_value == sg.p_value:
            lines.append(
                f"- **Random-entry null: FAIL** (p={sg.p_value:.3f}). Random direction on the "
                "same entry bars performed comparably. The entry *timing* adds nothing; "
                "whatever P&L exists comes from the exit and sizing rules."
            )

    if not lines:
        lines.append("- No null tests were run.")
    return "\n".join(lines)
