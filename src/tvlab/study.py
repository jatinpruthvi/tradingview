"""Full study: run everything and produce the report.

This is what ``tvlab study`` executes. Order matters - the validity checks run
before the report is written, so a leaky strategy never produces a polished
document full of meaningless numbers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest.engine import Backtester
from .backtest.metrics import full_report
from .backtest.montecarlo import fan_chart, simulate_equity, simulate_trades
from .backtest.walkforward import run_walk_forward
from .config import REPO_ROOT, AppConfig
from .data.base import bars_per_year
from .data.store import DataStore
from .indicators.technical import atr
from .pipeline import Study, build_strategy, effective_config, run_study
from .report import write_report
from .validation import lookahead as LA
from .validation import recovery as RC
from .validation.null_test import shuffled_null, sign_null

log = logging.getLogger(__name__)


@dataclass
class FullStudy:
    config: AppConfig
    study: Study
    walk_forward: Any = None
    monte_carlo: Any = None
    monte_carlo_bars: Optional[Dict[str, Any]] = None
    nulls: Dict[str, Any] = field(default_factory=dict)
    cost_sensitivity: Optional[pd.DataFrame] = None
    lookahead: Optional[Dict[str, Any]] = None
    recovery: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    provenance: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, Path] = field(default_factory=dict)

    def verdict(self) -> Dict[str, Any]:
        from .report import _verdict

        wf = self.walk_forward.summary() if self.walk_forward is not None else None
        mc = self.monte_carlo.summary() if self.monte_carlo is not None else None
        return _verdict(self.study, wf, self.nulls, mc)


def run_full_study(
    cfg: AppConfig,
    symbols: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    source: Optional[str] = None,
    do_walk_forward: Optional[bool] = None,
    do_monte_carlo: Optional[bool] = None,
    do_nulls: bool = True,
    do_cost_sensitivity: bool = True,
    do_lookahead: bool = True,
    do_recovery: bool = False,
    do_cross_validation: bool = False,
    n_sims: Optional[int] = None,
    write: bool = True,
    strategy_name: Optional[str] = None,
    allocation_scheme: str = "equal",
) -> FullStudy:
    """Run the complete validation pipeline and (optionally) write the report.

    ``allocation_scheme`` decides how capital is split across symbols before the
    equity curves are combined - see :mod:`tvlab.portfolio.allocate`. "equal" is
    the default; "inverse_vol" weights by underlying volatility. Neither looks at
    past performance, deliberately.
    """
    t_all = time.time()
    timings: Dict[str, float] = {}
    if strategy_name:
        cfg = cfg.merge({"strategy.name": strategy_name})

    store = DataStore(cfg)
    src_name = source or cfg.data.source

    # ---------------------------------------------------------------- study
    t0 = time.time()
    study = run_study(cfg, symbols=symbols, store=store, source=src_name,
                      cross_validate=do_cross_validation,
                      allocation_scheme=allocation_scheme)
    timings["study"] = time.time() - t0
    log.info("study: %d symbols in %.1fs", len(study.symbols), timings["study"])

    primary = study.symbols[0]
    bpy = primary.bars_per_year

    # ---------------------------------------------------------- walk forward
    wf = None
    run_wf = cfg.walk_forward.enabled if do_walk_forward is None else do_walk_forward
    if run_wf:
        t0 = time.time()
        try:
            wf = run_walk_forward(cfg, primary.symbol, primary.ohlcv,
                                  bias=store.bias_frame(primary.symbol, source=src_name))
            log.info("walk-forward: %d windows, %d trials, %.1fs",
                     wf.n_windows, wf.total_trials, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            log.warning("walk-forward failed: %s", exc)
            wf = None
        timings["walk_forward"] = time.time() - t0

    # how many parameter sets did we effectively try? Feeds the deflated Sharpe.
    n_trials = max((wf.total_trials if wf is not None else 1), 1)
    if wf is not None:
        # re-score the headline metrics with the true trial count
        study.portfolio_metrics = full_report(
            study.portfolio_equity, study.portfolio_trades,
            periods_per_year=float(np.median([s.bars_per_year for s in study.symbols])),
            n_trials=n_trials, initial_capital=cfg.backtest.initial_capital,
        )

    # ----------------------------------------------------------- monte carlo
    mc = mc_bars = None
    run_mc = cfg.monte_carlo.enabled if do_monte_carlo is None else do_monte_carlo
    if run_mc:
        t0 = time.time()
        sims = n_sims or cfg.monte_carlo.n_sims
        try:
            mc = simulate_trades(
                study.portfolio_trades, initial_capital=cfg.backtest.initial_capital,
                n_sims=sims, block_size=cfg.monte_carlo.block_size,
                confidence=cfg.monte_carlo.confidence, seed=cfg.monte_carlo.seed,
                dd_limits=sorted({round(cfg.risk.max_drawdown_pct, 3), 0.05, 0.10,
                                  0.15, 0.20, 0.25, 0.33, 0.50}),
            )
            mc_bars = simulate_equity(study.portfolio_equity, n_sims=max(sims // 2, 200),
                                      block_size=50, seed=cfg.monte_carlo.seed + 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("monte carlo failed: %s", exc)
        timings["monte_carlo"] = time.time() - t0

    fan = None
    if mc is not None and study.portfolio_equity is not None:
        try:
            fan = fan_chart(study.portfolio_equity, n_sims=400, block_size=50,
                            seed=cfg.monte_carlo.seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("fan chart failed: %s", exc)

    # ------------------------------------------------------------- null tests
    nulls: Dict[str, Any] = {}
    if do_nulls:
        t0 = time.time()
        strat = build_strategy(cfg)
        eff = effective_config(cfg, strat)

        real_bias = store.bias_frame(primary.symbol, source=src_name)

        def _bias_for(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
            """Give a permuted frame the same *kind* of HTF information.

            The real run sees a native higher-timeframe bias frame. If the
            permuted run saw none, the null would test a different (weaker)
            strategy and the p-value would be meaningless. So resample the
            shuffled bars up to the bias timeframe and hand back NATIVE bars -
            exactly what :func:`tvlab.data.store.DataStore.bias_frame` returns.

            Do NOT pre-align them onto the execution index here: ``trend_filter``
            computes its indicators on native bars and aligns the resulting state
            itself. Handing it a forward-filled frame would quietly revert to the
            broken convention where a 50-period *daily* EMA became a 50-bar (~8
            day) EMA, so the null and the real run would disagree.
            """
            if real_bias is None or not cfg.timeframe.bias:
                return None
            try:
                from .data.base import resample, validate_ohlcv

                return validate_ohlcv(resample(frame, cfg.timeframe.bias), primary.symbol)
            except Exception:  # noqa: BLE001
                return None

        def _run_once(frame: pd.DataFrame) -> Dict[str, float]:
            s = strat.signals(frame, _bias_for(frame))
            a = atr(frame, int(strat.params.get("atr_period", 14)))
            r = Backtester(eff).run(
                frame, s, symbol=primary.symbol, timeframe=cfg.timeframe.execution, atr=a,
                max_bars_in_trade=strat.resolve_max_bars(cfg.backtest.max_bars_in_trade),
            )
            return full_report(r.equity, r.trades, periods_per_year=bpy,
                               initial_capital=cfg.backtest.initial_capital)

        try:
            nulls["shuffled_returns"] = shuffled_null(
                primary.ohlcv, _run_once, n_sims=60, seed=101,
            ).summary()
        except Exception as exc:  # noqa: BLE001
            log.warning("shuffled null failed: %s", exc)

        def _run_on_signals(frame: pd.DataFrame, sig: pd.DataFrame) -> Dict[str, float]:
            a = atr(frame, int(strat.params.get("atr_period", 14)))
            r = Backtester(eff).run(frame, sig, symbol=primary.symbol,
                                    timeframe=cfg.timeframe.execution, atr=a)
            return full_report(r.equity, r.trades, periods_per_year=bpy,
                               initial_capital=cfg.backtest.initial_capital)

        try:
            nulls["random_entry"] = sign_null(
                primary.ohlcv, primary.signals, _run_on_signals, n_sims=60, seed=202,
            ).summary()
        except Exception as exc:  # noqa: BLE001
            log.warning("sign null failed: %s", exc)
        timings["nulls"] = time.time() - t0

    # ------------------------------------------------------ cost sensitivity
    cost_sens = None
    if do_cost_sensitivity:
        t0 = time.time()
        try:
            from .pipeline import cost_sensitivity

            cost_sens = cost_sensitivity(cfg, primary.symbol, store=store, source=src_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("cost sensitivity failed: %s", exc)
        timings["cost_sensitivity"] = time.time() - t0

    # -------------------------------------------------------------- lookahead
    la = None
    if do_lookahead:
        t0 = time.time()
        try:
            strats = [build_strategy(cfg)]
            for extra in ("ema_pullback", "donchian_breakout"):
                if extra != cfg.strategy.name:
                    strats.append(build_strategy(cfg, extra))
            la = LA.run_all(primary.ohlcv,
                            bias=store.bias_frame(primary.symbol, source=src_name),
                            strategies=strats,
                            swing_length=int(strats[0].params.get("swing_length", 12)))
        except Exception as exc:  # noqa: BLE001
            log.warning("lookahead audit failed: %s", exc)
        timings["lookahead"] = time.time() - t0

    # --------------------------------------------------------------- recovery
    rec = None
    if do_recovery:
        t0 = time.time()
        try:
            from .data.synthetic import basket

            injected = 25
            rdf = basket([primary.symbol], n_bars=min(2500, max(len(primary.ohlcv), 800)),
                         timeframe=cfg.timeframe.execution,
                         seed=cfg.data.synthetic_seed + 777,
                         inject_sweeps=injected)[primary.symbol]
            strat = build_strategy(cfg)
            rbias = store.bias_frame(primary.symbol, source="synthetic")
            rec = {
                "smc_detector": RC.run_smc_recovery(
                    rdf, swing_length=int(strat.params.get("swing_length", 12)),
                    tolerance=6,
                ).summary(),
                "strategy_entries": RC.run_smc_recovery(
                    rdf, swing_length=int(strat.params.get("swing_length", 12)),
                    tolerance=6, use_strategy=True, strategy=strat, bias=rbias,
                ).summary(),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("recovery test failed: %s", exc)
        timings["recovery"] = time.time() - t0

    # -------------------------------------------------------- provenance
    provenance = _provenance(cfg, study, src_name)

    timings["total"] = time.time() - t_all

    full = FullStudy(
        config=cfg, study=study, walk_forward=wf, monte_carlo=mc,
        monte_carlo_bars=mc_bars, nulls=nulls, cost_sensitivity=cost_sens,
        lookahead=la, recovery=rec,
        cross_validation=study.cross_validation, provenance=provenance,
        timings=timings,
    )

    if write:
        out = Path(out_dir) if out_dir else (REPO_ROOT / cfg.output.dir /
                                            f"{cfg.strategy.name}_{cfg.universe.market}"
                                            f"_{cfg.timeframe.execution}")
        try:
            full.outputs = write_report(
                study, out, wf=wf, mc=mc, nulls=nulls or None, cost_sens=cost_sens,
                lookahead_report=la, recovery_report=rec,
                cross_validation=study.cross_validation,
                plots=cfg.output.plots, dpi=cfg.output.dpi, fan=fan,
                data_provenance=provenance,
            )
            log.info("report written to %s", out)
        except Exception as exc:  # noqa: BLE001
            log.error("report writing failed: %s", exc)

    return full


def _provenance(cfg: AppConfig, study: Study, src_name: str) -> str:
    """Describe exactly what data produced these numbers."""
    first = study.symbols[0] if study.symbols else None
    if first is None:
        return "no data"
    n = len(first.ohlcv)
    span = f"{first.ohlcv.index.min():%Y-%m-%d} .. {first.ohlcv.index.max():%Y-%m-%d}"
    if src_name == "synthetic":
        return (
            f"**SYNTHETIC** - regime-switching generator, seed {cfg.data.synthetic_seed}. "
            f"{n} bars of `{cfg.timeframe.execution}` spanning {span} for {first.symbol}. "
            "This validates the engine, the risk rules and the signal plumbing. "
            "**It says nothing about any real market**, and a trend-following strategy "
            "will look good on data generated with trending regimes by construction."
        )
    return (f"{src_name} - {n} bars of `{cfg.timeframe.execution}` spanning {span} "
            f"for {first.symbol} ({len(study.symbols)} symbols total)")
