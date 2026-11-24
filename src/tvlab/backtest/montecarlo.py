"""Monte Carlo simulation of trade sequences.

A backtest produces ONE path through history. That path's max drawdown is a
sample, not a limit: reshuffling the same trades produces a different - and
often worse - worst case. This module answers the question that actually matters
for position sizing:

    "Given these trades, how bad can the drawdown get, and what is the
     probability of breaching my limit before I reach my target?"

Method
------
Block bootstrap on the per-trade return series. Resampling individual trades
independently destroys the clustering of wins and losses (losing streaks are not
random - they come in regimes), so we resample contiguous blocks of
``block_size`` trades, which preserves local dependence while still permuting
the order at the block level.

Everything is expressed as a fraction of starting equity so the result is
comparable across capital sizes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    n_sims: int
    block_size: int
    confidence: float
    n_trades: int
    final_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))
    ruin_probabilities: Dict[str, float] = field(default_factory=dict)
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        if len(self.final_returns) == 0:
            return {"error": "no trades to simulate", "n_trades": self.n_trades}
        c = self.confidence
        lo, hi = (1 - c) / 2, 1 - (1 - c) / 2
        return {
            "n_sims": self.n_sims,
            "n_trades": self.n_trades,
            "block_size": self.block_size,
            "confidence": c,
            "final_return_pct": {
                "mean": float(np.mean(self.final_returns) * 100),
                "median": float(np.median(self.final_returns) * 100),
                "p5": float(np.quantile(self.final_returns, 0.05) * 100),
                "p25": float(np.quantile(self.final_returns, 0.25) * 100),
                "p75": float(np.quantile(self.final_returns, 0.75) * 100),
                "p95": float(np.quantile(self.final_returns, 0.95) * 100),
                "ci_low": float(np.quantile(self.final_returns, lo) * 100),
                "ci_high": float(np.quantile(self.final_returns, hi) * 100),
            },
            "max_drawdown_pct": {
                "mean": float(np.mean(self.max_drawdowns) * 100),
                "median": float(np.median(self.max_drawdowns) * 100),
                "p95": float(np.quantile(self.max_drawdowns, 0.95) * 100),
                "p99": float(np.quantile(self.max_drawdowns, 0.99) * 100),
                "worst": float(np.max(self.max_drawdowns) * 100),
            },
            "prob_loss": float(np.mean(self.final_returns < 0)),
            "prob_beat_backtest": float(np.mean(
                self.final_returns > np.median(self.final_returns)
            )),
            "ruin_probabilities": self.ruin_probabilities,
            "percentiles": self.percentiles,
            "warnings": self.warnings,
        }


def block_bootstrap(
    returns: np.ndarray, n_sims: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_sims`` sequences of the same length as ``returns``.

    Returns an (n_sims, len(returns)) array. Contiguous blocks are kept intact so
    short-run dependence survives the resampling.
    """
    n = len(returns)
    if n == 0:
        return np.empty((0, 0))
    block_size = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    out = np.empty((n_sims, n), dtype="float64")
    if max_start < 0:
        # fewer trades than one block: resample with replacement
        idx = rng.integers(0, n, size=(n_sims, n))
        return returns[idx]

    for s in range(n_sims):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        seq = np.concatenate([returns[b:b + block_size] for b in starts])
        out[s] = seq[:n]
    return out


def simulate_trades(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
    n_sims: int = 2_000,
    block_size: int = 20,
    confidence: float = 0.95,
    seed: int = 7,
    dd_limits: Optional[List[float]] = None,
) -> MonteCarloResult:
    """Resample the trade sequence and characterise the outcome distribution."""
    res = MonteCarloResult(n_sims=n_sims, block_size=block_size, confidence=confidence,
                           n_trades=0 if trades is None else len(trades))

    if trades is None or len(trades) == 0:
        res.warnings.append("no trades: Monte Carlo cannot run")
        return res

    pnl = trades["pnl"].to_numpy(dtype="float64")
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        res.warnings.append("no finite P&L values")
        return res
    res.n_trades = len(pnl)

    if len(pnl) < 10:
        res.warnings.append(
            f"only {len(pnl)} trades: the bootstrap distribution will be wide and "
            "the percentiles are close to meaningless. Get more trades or trade "
            "more symbols before sizing off this."
        )

    # returns as a fraction of the equity at the time, approximated by the
    # starting capital - exact compounding would need the equity path per trade,
    # which the block bootstrap destroys anyway
    rets = pnl / float(initial_capital)

    rng = np.random.default_rng(seed)
    paths = block_bootstrap(rets, n_sims, block_size, rng)
    if paths.size == 0:
        res.warnings.append("bootstrap produced no paths")
        return res

    equity = np.cumprod(1.0 + paths, axis=1)
    res.final_returns = equity[:, -1] - 1.0

    peak = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - equity / peak
    res.max_drawdowns = np.nanmax(dd, axis=1)

    limits = dd_limits if dd_limits is not None else [0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]
    res.ruin_probabilities = {
        f"dd_exceeds_{int(l * 100)}pct": float(np.mean(res.max_drawdowns >= l)) for l in limits
    }

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    res.percentiles = {
        "final_return_pct": {f"p{int(q * 100)}": float(np.quantile(res.final_returns, q) * 100)
                             for q in qs},
        "max_drawdown_pct": {f"p{int(q * 100)}": float(np.quantile(res.max_drawdowns, q) * 100)
                             for q in qs},
    }
    return res


def simulate_equity(
    equity: pd.Series,
    n_sims: int = 1_000,
    block_size: int = 50,
    confidence: float = 0.95,
    seed: int = 11,
) -> Dict[str, Any]:
    """Bar-return bootstrap - complements the trade-level view.

    Resampling *bar* returns keeps the intra-trade path (so drawdown timing is
    realistic) but breaks the trade boundaries. Running both and comparing is a
    useful consistency check: if the trade-level and bar-level drawdown
    distributions disagree wildly, one of the two views is misleading.
    """
    eq = equity.dropna()
    if len(eq) < 50:
        return {"error": "equity curve too short for a bar-level bootstrap"}

    rets = np.log(eq / eq.shift(1)).dropna().to_numpy(dtype="float64")
    rets = rets[np.isfinite(rets)]
    if len(rets) < 20:
        return {"error": "not enough bar returns"}

    rng = np.random.default_rng(seed)
    paths = block_bootstrap(rets, n_sims, block_size, rng)
    curves = np.exp(np.cumsum(paths, axis=1))
    final = curves[:, -1] - 1.0
    peak = np.maximum.accumulate(curves, axis=1)
    dd = np.nanmax(1.0 - curves / peak, axis=1)

    c = confidence
    lo, hi = (1 - c) / 2, 1 - (1 - c) / 2
    return {
        "n_sims": n_sims, "n_bars": len(rets), "block_size": block_size,
        "final_return_pct": {
            "mean": float(np.mean(final) * 100),
            "median": float(np.median(final) * 100),
            "ci_low": float(np.quantile(final, lo) * 100),
            "ci_high": float(np.quantile(final, hi) * 100),
        },
        "max_drawdown_pct": {
            "mean": float(np.mean(dd) * 100),
            "p95": float(np.quantile(dd, 0.95) * 100),
            "worst": float(np.max(dd) * 100),
        },
        "prob_loss": float(np.mean(final < 0)),
    }


def fan_chart(
    equity: pd.Series, n_sims: int = 500, block_size: int = 50, seed: int = 3,
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """Quantile bands over time, for the report's Monte Carlo fan chart."""
    eq = equity.dropna()
    if len(eq) < 50:
        return pd.DataFrame()
    rets = np.log(eq / eq.shift(1)).dropna().to_numpy(dtype="float64")
    rets = rets[np.isfinite(rets)]
    if len(rets) < 20:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    paths = block_bootstrap(rets, n_sims, block_size, rng)
    curves = np.exp(np.cumsum(paths, axis=1)) * float(eq.iloc[0])

    # curves is (n_sims, n_bars): the quantile must be taken ACROSS simulations
    # (axis=0), giving one value per bar. axis=1 returned n_sims values instead.
    n_bars = curves.shape[1]
    out = pd.DataFrame(index=eq.index[1:1 + n_bars])
    for q in quantiles:
        out[f"p{int(q * 100)}"] = np.quantile(curves, q, axis=0)
    out["actual"] = eq.reindex(out.index).to_numpy()
    return out
