"""Walk-forward optimisation with honest out-of-sample accounting.

Why this exists
---------------
A single in-sample backtest tells you nothing about whether a strategy works. It
tells you that a parameter set fitted this particular history. Walk-forward
splits the sample into train/test windows, fits on train, and *only* scores the
test windows. The stitched out-of-sample curve is the number to believe.

The other half of the job is counting the trials. Searching N parameter sets and
reporting the best Sharpe guarantees an upward bias even on pure noise, so every
window records how many trials it ran and the aggregate is scored with the
**deflated Sharpe ratio** at that trial count.

Objective functions
-------------------
``deflated_sharpe`` (default)
    penalises the search directly; the safest choice
``profit_factor``
    gross win / gross loss, floored by a minimum trade count
``expectancy``
    mean P&L per trade

All objectives return ``-inf`` when the training window produced too few trades,
so degenerate parameter sets cannot win by trading once and getting lucky.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.base import bars_per_year
from ..indicators.technical import atr
from ..pipeline import build_strategy, effective_config
from ..strategy.base import Strategy
from .engine import Backtester
from .metrics import deflated_sharpe_ratio, full_report, sharpe

log = logging.getLogger(__name__)

#: absolute floor on trades in a training window before an objective is scored.
#: Kept low because a selective setup may legitimately trade only a few times per
#: window; ``min_trades_for_window`` scales this with window length.
MIN_TRADES_FOR_OBJECTIVE = 3


@dataclass(frozen=True)
class Window:
    """One train/test split."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class WindowResult:
    index: int
    best_params: Dict[str, Any]
    trials: int
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    train_slice: Tuple[pd.Timestamp, pd.Timestamp]
    test_slice: Tuple[pd.Timestamp, pd.Timestamp]
    objective_train: float
    objective_test: float
    degraded: bool = False
    min_trades_required: int = MIN_TRADES_FOR_OBJECTIVE

    @property
    def efficiency(self) -> float:
        """OOS / IS on the objective. >1 is suspicious, <0 means the fit inverted."""
        if not np.isfinite(self.objective_train) or self.objective_train == 0:
            return float("nan")
        return float(self.objective_test / self.objective_train)


@dataclass
class WalkForwardResult:
    config: AppConfig
    symbol: str
    strategy: str
    windows: List[WindowResult] = field(default_factory=list)
    oos_equity: Optional[pd.Series] = None
    oos_trades: Optional[pd.DataFrame] = None
    oos_metrics: Optional[Dict[str, Any]] = None
    total_trials: int = 0
    is_equity: Optional[pd.Series] = None

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def summary(self) -> Dict[str, Any]:
        effs = [w.efficiency for w in self.windows if np.isfinite(w.efficiency)]
        pos = [w for w in self.windows if (w.test_metrics.get("net_pnl") or 0) > 0]
        m = self.oos_metrics or {}
        return {
            "n_windows": self.n_windows,
            "total_trials": self.total_trials,
            "windows_profitable": len(pos),
            "window_hit_rate": len(pos) / self.n_windows if self.n_windows else float("nan"),
            "mean_oos_efficiency": float(np.mean(effs)) if effs else float("nan"),
            "median_oos_efficiency": float(np.median(effs)) if effs else float("nan"),
            "oos_return_pct": m.get("total_return_pct"),
            "oos_cagr_pct": m.get("cagr_pct"),
            "oos_max_dd_pct": m.get("max_drawdown_pct"),
            "oos_sharpe": m.get("sharpe"),
            "oos_profit_factor": m.get("profit_factor"),
            "oos_trades": m.get("n_trades"),
            "oos_win_rate": m.get("win_rate"),
            "oos_deflated_sharpe": (m.get("deflated_sharpe") or {}).get("dsr"),
            "oos_dsr_verdict": (m.get("deflated_sharpe") or {}).get("verdict"),
            "params_stable": _param_stability(self.windows),
        }


def make_windows(n: int, n_splits: int, train_pct: float, rolling: bool,
                 min_train: int, min_test: int) -> List[Window]:
    """Slice ``n`` bars into train/test windows.

    ``rolling=True`` keeps the training length constant and slides it forward
    (the usual choice: it forgets stale regimes). ``rolling=False`` anchors the
    train start at bar 0 and grows the window, which keeps all history but makes
    early regimes dominate.
    """
    if n < min_train + min_test:
        raise ValueError(
            f"not enough bars for walk-forward: {n} < min_train({min_train}) + min_test({min_test})"
        )
    # `train_pct` is the share of the WHOLE series used for the initial training
    # window, not the share of each split. The out-of-sample remainder is cut
    # into `n_splits` contiguous segments, and the train window slides (rolling)
    # or grows (anchored) by one segment per split. Contiguous OOS segments are
    # what make the stitched curve a real out-of-sample equity line.
    train_len = max(min_train, int(round(n * train_pct)))
    oos_total = n - train_len
    test_len = oos_total / n_splits
    if oos_total < n_splits * max(1, min_test // 4):
        raise ValueError(
            f"walk-forward leaves only {oos_total} out-of-sample bars for "
            f"{n_splits} splits (train_pct={train_pct}, min_test={min_test})"
        )

    windows: List[Window] = []
    for k in range(n_splits):
        e0 = train_len + int(round(k * test_len))
        e1 = train_len + int(round((k + 1) * test_len)) if k < n_splits - 1 else n
        if e1 - e0 < max(1, min_test // 4):
            break
        t1 = e0
        t0 = 0 if not rolling else max(0, t1 - train_len)
        if t1 - t0 < min_train:
            continue
        windows.append(Window(k, t0, t1, e0, e1))

    # de-duplicate and enforce the minimums
    clean = [w for w in windows
             if (w.train_end - w.train_start) >= min_train and (w.test_end - w.test_start) >= 1]
    if not clean:
        raise ValueError(
            f"walk-forward could not build any valid window from {n} bars "
            f"with n_splits={n_splits}, train_pct={train_pct}, min_train={min_train}"
        )
    return clean


def param_grid(strategy_cls: Strategy, max_trials: int) -> List[Dict[str, Any]]:
    """Enumerate the declared grid, capped at ``max_trials`` combinations.

    When the full cross-product exceeds the cap we subsample *deterministically*
    (a strided slice of the shuffled product, seeded) so runs are reproducible.
    The cap matters: trials are the denominator of the multiple-testing penalty.
    """
    grid = {k: tuple(v) for k, v in (strategy_cls.grid or {}).items() if len(tuple(v)) > 0}
    if not grid:
        return [dict(strategy_cls.defaults)]

    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    full = [dict(zip(keys, c)) for c in combos]

    if len(full) <= max_trials:
        return full

    rng = np.random.default_rng(12345)
    order = rng.permutation(len(full))[:max_trials]
    picked = sorted([full[i] for i in order], key=lambda d: tuple(str(v) for v in d.values()))
    log.info("grid has %d combos, capped to %d trials", len(full), max_trials)
    return picked


def min_trades_for_window(n_bars: int, hard_floor: int = MIN_TRADES_FOR_OBJECTIVE,
                          cap: int = 8) -> int:
    """Trades required before a window's fit is scored.

    Scales with window length (roughly one trade per 150 bars) between a hard
    floor and a cap. A fixed threshold of 8 silently rejected *every* window for
    a selective setup that trades ~12 times in 2500 bars, which made
    walk-forward report "no usable fit" instead of a real result.
    """
    return int(max(hard_floor, min(cap, round(n_bars / 150.0))))


def objective_value(metrics: Dict[str, Any], name: str, min_trades: Optional[int] = None) -> float:
    """Score a parameter set. ``-inf`` for degenerate fits."""
    n = metrics.get("n_trades", 0) or 0
    floor = MIN_TRADES_FOR_OBJECTIVE if min_trades is None else int(min_trades)
    if n < floor:
        return -float("inf")

    if name == "deflated_sharpe":
        sr = metrics.get("sharpe", float("nan"))
        if not np.isfinite(sr):
            return -float("inf")
        # penalise by the trial count the caller recorded
        dsr = deflated_sharpe_ratio(
            sr, n_trials=int(metrics.get("_trials", 1)),
            n_obs=int(metrics.get("n_bars", 100)),
            periods_per_year=float(metrics.get("periods_per_year", 252)),
        )
        val = dsr.get("dsr", float("nan"))
        if not np.isfinite(val):
            return -float("inf")
        # tie-break on Sharpe so two sets with equal DSR are still ordered
        return float(val * 10.0 + min(sr, 10.0))

    if name == "profit_factor":
        pf = metrics.get("profit_factor", float("nan"))
        if not np.isfinite(pf):
            return -float("inf")
        return float(min(pf, 20.0) * math.log1p(n))

    if name == "expectancy":
        e = metrics.get("expectancy", float("nan"))
        if not np.isfinite(e):
            return -float("inf")
        return float(e)

    if name == "sharpe":
        sr = metrics.get("sharpe", float("nan"))
        return float(sr) if np.isfinite(sr) else -float("inf")

    raise ValueError(f"unknown objective {name!r}; use deflated_sharpe|profit_factor|expectancy|sharpe")


def _evaluate(cfg: AppConfig, strat_cls, params: Dict[str, Any], df: pd.DataFrame,
              bias: Optional[pd.DataFrame], bpy: float, trials: int) -> Dict[str, Any]:
    """Backtest one parameter set on one slice and return its metrics."""
    strat = strat_cls(params)
    eff = effective_config(cfg, strat)
    sig = strat.signals(df, bias)
    a = atr(df, int(strat.params.get("atr_period", 14)))
    res = Backtester(eff).run(
        df, sig, symbol=cfg.universe.symbols[0] if cfg.universe.symbols else "?",
        timeframe=cfg.timeframe.execution, atr=a,
        max_bars_in_trade=strat.resolve_max_bars(cfg.backtest.max_bars_in_trade),
    )
    m = full_report(res.equity, res.trades, periods_per_year=bpy,
                    n_trials=trials, initial_capital=cfg.backtest.initial_capital)
    m["_trials"] = trials
    m["_equity"] = res.equity
    m["_trades"] = res.trades
    return m


def _slice_bias(bias: Optional[pd.DataFrame], idx: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Restrict a NATIVE higher-timeframe bias frame to one walk-forward window.

    ``bias`` is on its own (coarser) index, so ``bias.reindex(idx)`` - matching
    execution timestamps against daily timestamps - returns an all-NaN frame. The
    trend filter then silently falls back to the execution timeframe, and the
    window is scored on a *different* strategy than the full run. Slicing by time
    range instead of by exact index match avoids that.

    History before the window start is kept deliberately: the EMA/ADX need their
    warmup, and dropping it would leave the window opening with an undefined
    trend state. Bars after ``idx[-1]`` are dropped, which is what keeps the
    window out-of-sample. ``trend_filter`` shifts the aligned state by one HTF
    bar, so an execution bar still only ever sees a completed bias bar.
    """
    if bias is None or len(bias) == 0 or len(idx) == 0:
        return bias
    end = idx[-1]
    try:
        out = bias.loc[:end]
    except (KeyError, TypeError):
        out = bias[bias.index <= end]
    return out if len(out) else bias


def run_walk_forward(
    cfg: AppConfig,
    symbol: str,
    df: pd.DataFrame,
    bias: Optional[pd.DataFrame] = None,
    strategy_name: Optional[str] = None,
    max_trials: Optional[int] = None,
) -> WalkForwardResult:
    """Fit on train, score on test, for every window. Returns the stitched OOS result."""
    from ..data.store import DataStore  # local import avoids a cycle at module load

    strat_name = strategy_name or cfg.strategy.name
    strat_cls = build_strategy(cfg, strat_name).__class__
    trials_cap = max_trials or cfg.walk_forward.max_trials
    grid = param_grid(strat_cls, trials_cap)
    n_trials = len(grid)

    bpy = bars_per_year(cfg.timeframe.execution, cfg.universe.market)
    wf = cfg.walk_forward
    windows = make_windows(
        len(df), wf.n_splits, wf.train_pct, wf.rolling, wf.min_train_bars, wf.min_test_bars,
    )
    log.info("%s: %d walk-forward windows, %d trials each (%s objective)",
             symbol, len(windows), n_trials, wf.objective)

    results: List[WindowResult] = []
    oos_pieces: List[pd.Series] = []
    oos_trade_pieces: List[pd.DataFrame] = []
    is_pieces: List[pd.Series] = []

    for w in windows:
        train_idx = df.index[w.train_start:w.train_end]
        test_idx = df.index[w.test_start:w.test_end]
        train_df, test_df = df.iloc[w.train_start:w.train_end], df.iloc[w.test_start:w.test_end]
        train_bias = _slice_bias(bias, train_idx)
        test_bias = _slice_bias(bias, test_idx)

        # ---- fit on train ----
        min_tr = min_trades_for_window(len(train_df))
        best_score, best_params, best_train = -float("inf"), None, None
        fallback_score, fallback_params, fallback_train = -float("inf"), None, None
        for params in grid:
            try:
                m = _evaluate(cfg, strat_cls, params, train_df, train_bias, bpy, n_trials)
            except Exception as exc:  # noqa: BLE001
                log.debug("trial %s failed on window %d: %s", params, w.index, exc)
                continue
            score = objective_value(m, wf.objective, min_tr)
            if score > best_score:
                best_score, best_params, best_train = score, params, m
            # graceful degradation: if nothing clears the trade floor on the
            # configured objective, fall back to raw expectancy so the window
            # still yields an out-of-sample measurement instead of being dropped
            fb = objective_value(m, "expectancy", 1)
            if fb > fallback_score:
                fallback_score, fallback_params, fallback_train = fb, params, m

        degraded = False
        if best_params is None:
            if fallback_params is None:
                log.warning("window %d: no parameter set produced any trade; skipping", w.index)
                continue
            best_score, best_params, best_train = fallback_score, fallback_params, fallback_train
            degraded = True
            log.info("window %d: objective %r found no fit with >= %d trades; "
                     "fell back to expectancy", w.index, wf.objective, min_tr)

        # ---- score on test with the fitted params ONLY ----
        try:
            test_m = _evaluate(cfg, strat_cls, best_params, test_df, test_bias, bpy, n_trials)
        except Exception as exc:  # noqa: BLE001
            log.warning("window %d: OOS evaluation failed: %s", w.index, exc)
            continue
        test_m["_degraded_objective"] = degraded

        results.append(WindowResult(
            index=w.index, best_params=dict(best_params), trials=n_trials,
            train_metrics={k: v for k, v in best_train.items() if not k.startswith("_")},
            test_metrics={k: v for k, v in test_m.items() if not k.startswith("_")},
            train_slice=(train_idx.min(), train_idx.max()),
            test_slice=(test_idx.min(), test_idx.max()),
            objective_train=best_score,
            objective_test=objective_value(test_m, wf.objective, min_trades_for_window(len(test_df))),
            degraded=degraded,
            min_trades_required=min_tr,
        ))

        eq = test_m.get("_equity")
        if eq is not None and len(eq):
            oos_pieces.append(eq)
        tr = test_m.get("_trades")
        if tr is not None and len(tr):
            oos_trade_pieces.append(tr.assign(window=w.index, params=str(best_params)))
        iseq = best_train.get("_equity")
        if iseq is not None and len(iseq):
            is_pieces.append(iseq)

    if not oos_pieces:
        raise RuntimeError(
            f"walk-forward produced no out-of-sample results for {symbol}. "
            f"Check that the strategy trades at all on this data."
        )

    oos_equity = _stitch(oos_pieces, cfg.backtest.initial_capital)
    oos_trades = pd.concat(oos_trade_pieces, ignore_index=True) if oos_trade_pieces else pd.DataFrame()
    is_equity = _stitch(is_pieces, cfg.backtest.initial_capital) if is_pieces else None

    oos_metrics = full_report(
        oos_equity, oos_trades, periods_per_year=bpy,
        n_trials=max(n_trials * len(results), 1),
        initial_capital=cfg.backtest.initial_capital,
    )
    oos_metrics["_stitched_from"] = len(oos_pieces)

    return WalkForwardResult(
        config=cfg, symbol=symbol, strategy=strat_name, windows=results,
        oos_equity=oos_equity, oos_trades=oos_trades, oos_metrics=oos_metrics,
        total_trials=n_trials * len(results), is_equity=is_equity,
    )


def _stitch(pieces: List[pd.Series], initial_capital: float) -> pd.Series:
    """Chain per-window equity curves into one continuous OOS curve.

    Each window's curve is converted to a return multiple and applied to the
    running equity, so windows of different lengths and capital bases compose
    correctly instead of being naively summed.
    """
    idx = pieces[0].index
    for p in pieces[1:]:
        idx = idx.union(p.index)
    idx = idx.sort_values()

    out = pd.Series(index=idx, dtype="float64")
    equity = float(initial_capital)
    for p in pieces:
        base = float(p.iloc[0]) if p.iloc[0] > 0 else float(initial_capital)
        mult = p / base
        seg = equity * mult
        out.loc[seg.index] = seg.to_numpy()
        equity = float(seg.iloc[-1])
    return out.ffill().bfill().rename("oos_equity")


def _param_stability(windows: List[WindowResult]) -> Dict[str, Any]:
    """How often does the optimiser pick the same value for each parameter?

    A parameter that jumps around between adjacent windows is being fitted to
    noise. A stable one is plausibly structural. This is the cheapest overfitting
    diagnostic there is and it is almost never reported.
    """
    if not windows:
        return {}
    keys = sorted({k for w in windows for k in w.best_params})
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [w.best_params.get(k) for w in windows if k in w.best_params]
        if not vals:
            continue
        uniq = {}
        for v in vals:
            uniq[str(v)] = uniq.get(str(v), 0) + 1
        mode = max(uniq.items(), key=lambda kv: kv[1])
        out[k] = {
            "values": vals,
            "n_unique": len(uniq),
            "mode": mode[0],
            "mode_share": mode[1] / len(vals),
            "stable": mode[1] / len(vals) >= 0.6,
        }
    return out
