"""The pipeline: config -> data -> signals -> backtest -> metrics -> report.

Everything the CLI and the dashboard call goes through here, so a result produced
by ``tvlab backtest`` and a result shown in the browser are computed identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest.engine import Backtester, BacktestResult
from .backtest.metrics import (
    exit_breakdown,
    full_report,
    monthly_table,
    tag_breakdown,
)
from .config import AppConfig
from .data.base import bars_per_year
from .data.store import DataStore
from .indicators.technical import atr
from .strategy.base import Strategy, get

log = logging.getLogger(__name__)


@dataclass
class SymbolStudy:
    """One symbol's complete result set."""

    symbol: str
    timeframe: str
    result: BacktestResult
    metrics: Dict[str, Any]
    equity: pd.Series
    trades: pd.DataFrame
    ohlcv: pd.DataFrame
    signals: pd.DataFrame
    bars_per_year: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class Study:
    """A whole run: every symbol, plus the portfolio-level roll-up."""

    config: AppConfig
    strategy_name: str
    strategy_params: Dict[str, Any]
    symbols: List[SymbolStudy]
    portfolio_equity: Optional[pd.Series] = None
    portfolio_metrics: Optional[Dict[str, Any]] = None
    portfolio_trades: Optional[pd.DataFrame] = None
    benchmark: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    #: capital allocation actually used to build ``portfolio_equity``
    allocation: Optional[Dict[str, Any]] = None
    #: symbols that were studied but excluded by the risk caps
    unallocated: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    )

    @property
    def per_symbol(self) -> pd.DataFrame:
        rows = []
        for s in self.symbols:
            m = s.metrics
            rows.append({
                "symbol": s.symbol, "bars": s.bars_per_year and len(s.equity),
                "return_pct": m.get("total_return_pct"), "cagr_pct": m.get("cagr_pct"),
                "max_dd_pct": m.get("max_drawdown_pct"), "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"), "calmar": m.get("calmar"),
                "trades": m.get("n_trades"), "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"), "avg_r": m.get("avg_r"),
                "expectancy": m.get("expectancy"), "fees": m.get("total_fees"),
                "dsr": (m.get("deflated_sharpe") or {}).get("dsr"),
            })
        return pd.DataFrame(rows)

    def profitable_symbols(self) -> List[str]:
        return [s.symbol for s in self.symbols
                if (s.metrics.get("net_pnl") or 0) > 0]


def build_strategy(cfg: AppConfig, name: Optional[str] = None,
                   params: Optional[Dict[str, Any]] = None) -> Strategy:
    """Instantiate a strategy from the config.

    ``cfg.strategy.params`` is only used when it was written for the strategy
    being built (``params_for``). Switching ``strategy.name`` - for a benchmark,
    a comparison run, or ``--strategy`` on the CLI - therefore starts from that
    strategy's own defaults instead of inheriting another setup's parameters.
    """
    sname = name or cfg.strategy.name
    cls = get(sname)
    if params is not None:
        payload = params
    elif cfg.strategy.params_for and cfg.strategy.params_for != sname:
        log.debug("strategy.params were written for %r; using %r defaults instead",
                  cfg.strategy.params_for, sname)
        payload = {}
    else:
        payload = cfg.strategy.params
    return cls(payload)


def effective_config(cfg: AppConfig, strat: Strategy) -> AppConfig:
    """Apply a strategy's declared risk overrides (e.g. buy-and-hold's notional cap)."""
    overrides = getattr(strat, "risk_overrides", None) or {}
    if not overrides:
        return cfg
    return cfg.merge({f"risk.{k}": v for k, v in overrides.items()})


def _slice_holdout(frame: Optional[pd.DataFrame], holdout_start: Optional[str]) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty or not holdout_start or not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    cutoff = pd.Timestamp(holdout_start)
    if frame.index.tz is not None and cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(frame.index.tz)
    elif frame.index.tz is None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    return frame[frame.index < cutoff]


def _call_signals(
    strat: Strategy,
    df: pd.DataFrame,
    bias: Optional[pd.DataFrame],
    regime: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if regime is not None:
        try:
            return strat.signals(df, bias, regime=regime)
        except TypeError:
            pass
    return strat.signals(df, bias)


def run_symbol(
    cfg: AppConfig,
    symbol: str,
    store: Optional[DataStore] = None,
    strat: Optional[Strategy] = None,
    timeframe: Optional[str] = None,
    n_trials: int = 1,
    source: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    bias: Optional[pd.DataFrame] = None,
    regime: Optional[pd.DataFrame] = None,
    rs_mask: Optional[pd.DataFrame] = None,
) -> SymbolStudy:
    """Full single-symbol study."""
    store = store or DataStore(cfg)
    strat = strat or build_strategy(cfg)
    eff = effective_config(cfg, strat)
    timeframe = timeframe or cfg.timeframe.execution
    holdout = getattr(cfg.data, "holdout_start", None)

    df = df if df is not None else store.ohlcv(symbol, timeframe, source=source)
    df = _slice_holdout(df, holdout)
    if bias is None:
        bias = store.bias_frame(symbol, source=source)
    bias = _slice_holdout(bias, holdout)
    if regime is None and getattr(cfg.data, "regime_symbol", None):
        reg_sym = cfg.data.regime_symbol
        if reg_sym and reg_sym != symbol:
            try:
                regime = store.ohlcv(reg_sym, timeframe, source=source)
            except Exception as exc:  # noqa: BLE001
                log.warning("regime_symbol %s fetch failed (%s); continuing without regime", reg_sym, exc)
    regime = _slice_holdout(regime, holdout)

    sig = _call_signals(strat, df, bias, regime=regime)
    if rs_mask is not None and not rs_mask.empty:
        aligned_mask = rs_mask.reindex(sig.index).fillna(True)
        drop_long = (sig["signal"] > 0) & (~aligned_mask["allow_long"].to_numpy(dtype=bool))
        drop_short = (sig["signal"] < 0) & (~aligned_mask["allow_short"].to_numpy(dtype=bool))
        drop = drop_long | drop_short
        if drop.any():
            sig = sig.copy()
            sig.loc[drop, "signal"] = 0.0

    a = atr(df, int(strat.params.get("atr_period", 14)))
    bt = Backtester(eff)
    res = bt.run(
        df, sig, symbol=symbol, timeframe=timeframe, atr=a,
        max_bars_in_trade=strat.resolve_max_bars(cfg.backtest.max_bars_in_trade),
    )

    bpy = bars_per_year(timeframe, cfg.universe.market)
    metrics = full_report(
        res.equity, res.trades, periods_per_year=bpy,
        n_trials=n_trials, initial_capital=cfg.backtest.initial_capital,
    )
    metrics["exit_breakdown"] = exit_breakdown(res.trades)
    metrics["tag_breakdown"] = tag_breakdown(res.trades)
    metrics["monthly"] = monthly_table(res.equity).to_dict()

    warnings = list(res.warnings)
    if bias is None and cfg.timeframe.bias:
        warnings.append(
            f"higher-timeframe bias ({cfg.timeframe.bias}) unavailable for {symbol}; "
            "the trend filter fell back to the execution timeframe"
        )

    return SymbolStudy(
        symbol=symbol, timeframe=timeframe, result=res, metrics=metrics,
        equity=res.equity, trades=res.trades, ohlcv=df, signals=sig,
        bars_per_year=bpy, warnings=warnings,
    )


def combine_equity(studies: List[SymbolStudy], initial_capital: float,
                   method: str = "equal_weight",
                   weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Roll per-symbol equity curves into one portfolio curve.

    Each symbol is funded with a slice of capital and traded independently, which
    is what the per-symbol backtests actually simulated. The combined curve is
    the sum of the slices - so it inherits every symbol's drawdown timing and
    shows the diversification benefit (or its absence) directly.

    ``weights`` overrides the equal split and must come from
    :func:`tvlab.portfolio.allocate`, which enforces ``max_position_pct``,
    ``max_open_positions`` and ``max_correlated_positions``. Passing them is not
    cosmetic: splitting capital equally across ten 90%-correlated alt pairs
    reports a portfolio that the risk mandate would never allow, and understates
    the drawdown the account would actually have taken.

    Symbols with a zero weight are excluded from the curve rather than included
    at zero size, so the result is the equity of the book you would really hold.
    """
    if not studies:
        return pd.Series(dtype="float64")

    if weights is not None:
        selected = [s for s in studies if float(weights.get(s.symbol, 0.0)) > 0]
        if not selected:
            log.warning("allocation gave every symbol zero weight; "
                        "falling back to an equal split so the curve is not empty")
            selected, weights = studies, None
    else:
        selected = studies

    n = len(selected)
    frames = []
    for s in selected:
        eq = s.equity.copy()
        if weights is not None:
            per_symbol_capital = initial_capital * float(weights[s.symbol])
        elif method == "equal_weight":
            per_symbol_capital = initial_capital / n
        else:
            per_symbol_capital = initial_capital
        # rescale each curve to its own capital slice
        base = float(eq.iloc[0]) if len(eq) and eq.iloc[0] > 0 else initial_capital
        frames.append((eq / base) * per_symbol_capital)

    if len(frames) == 1:
        combined = frames[0]
    else:
        # union of timestamps, forward-filled: symbols may have different sessions
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.union(f.index)
        aligned = [f.reindex(idx).ffill() for f in frames]
        combined = aligned[0]
        for f in aligned[1:]:
            combined = combined.add(f, fill_value=np.nan)
        combined = combined.bfill()

    # Unallocated capital is CASH, not nothing. When the risk caps fund only part
    # of the book - which is exactly what max_correlated_positions does to a
    # single-market universe - the funded slices sum to less than
    # initial_capital, so a curve built from them alone starts below the account
    # balance. Every metric that compares against initial_capital then disagrees
    # with every metric computed from the curve's own start: this produced a
    # report showing "Net P&L -49,916" next to "Total return +0.17%" for the same
    # run. Adding the idle cash back makes the curve start at the real account
    # balance and makes return, drawdown and Sharpe account-level, which is what
    # the mandate is written against.
    if weights is not None:
        deployed = float(sum(weights.get(s.symbol, 0.0) for s in selected))
        idle = initial_capital * max(0.0, 1.0 - deployed)
        if idle > 1e-9:
            combined = combined + idle

    return combined.rename("portfolio_equity")


def run_study(
    cfg: AppConfig,
    symbols: Optional[List[str]] = None,
    store: Optional[DataStore] = None,
    n_trials: int = 1,
    with_benchmark: bool = True,
    source: Optional[str] = None,
    cross_validate: bool = False,
    allocation_scheme: str = "equal",
    corr_threshold: Optional[float] = None,
) -> Study:
    """Run the strategy across the universe and roll it up.

    ``n_trials`` feeds the deflated Sharpe: pass the number of parameter sets you
    searched, not 1, or the multiple-testing correction is meaningless.

    ``allocation_scheme`` decides how capital is split between symbols before the
    curves are combined - see :mod:`tvlab.portfolio.allocate`. "equal" is the
    default and the only scheme that does not look at performance.
    """
    store = store or DataStore(cfg)
    syms = list(symbols or cfg.universe.symbols)
    if not syms:
        syms = ["BTC/USDT"]
    syms = syms[: cfg.universe.max_symbols] if cfg.universe.max_symbols else syms

    strat = build_strategy(cfg)
    holdout = getattr(cfg.data, "holdout_start", None)
    timeframe = cfg.timeframe.execution

    regime_df: Optional[pd.DataFrame] = None
    reg_sym = getattr(cfg.data, "regime_symbol", None)
    if reg_sym:
        try:
            regime_df = _slice_holdout(store.ohlcv(reg_sym, timeframe, source=source), holdout)
        except Exception as exc:  # noqa: BLE001
            log.warning("regime_symbol %s fetch failed (%s)", reg_sym, exc)

    rs_masks: Dict[str, pd.DataFrame] = {}
    preloaded: Dict[str, pd.DataFrame] = {}
    rs_pct = getattr(cfg.universe, "rs_top_pct", None)
    if rs_pct is not None and 0.0 < float(rs_pct) < 1.0 and len(syms) >= 2:
        from .portfolio import causal_rs_mask

        for sym in syms:
            try:
                preloaded[sym] = _slice_holdout(store.ohlcv(sym, timeframe, source=source), holdout)
            except Exception:  # noqa: BLE001
                continue
        rs_masks = causal_rs_mask(
            preloaded,
            lookback_sessions=int(getattr(cfg.universe, "rs_lookback_sessions", 20)),
            top_pct=float(rs_pct),
        )

    studies: List[SymbolStudy] = []
    for sym in syms:
        try:
            studies.append(run_symbol(
                cfg, sym, store=store, strat=strat,
                n_trials=n_trials, source=source,
                df=preloaded.get(sym), regime=regime_df,
                rs_mask=rs_masks.get(sym),
            ))
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the study
            log.warning("%s: study failed (%s: %s)", sym, type(exc).__name__, exc)

    if not studies:
        raise RuntimeError(
            f"no symbol produced a result for strategy {cfg.strategy.name!r} on "
            f"{cfg.universe.market}; check the data source and symbol list"
        )

    # ---- capital allocation, capped by the risk mandate ---------------------
    from .portfolio import (allocate, allocation_report, asset_returns_from_ohlcv,
                            liquidity_from_ohlcv)

    curves = {s.symbol: s.equity for s in studies}
    # Correlation is measured on the UNDERLYING assets, not on the equity curves:
    # a selective strategy is flat most of the time, and flat curves correlate at
    # ~0 no matter how correlated the markets are. See portfolio.allocate.
    asset_rets = asset_returns_from_ohlcv({s.symbol: s.ohlcv for s in studies})
    # corr_threshold=None means "use risk.correlation_threshold from the config"
    ohlcvs = {s.symbol: s.ohlcv for s in studies}
    alloc = allocate(curves, cfg, scheme=allocation_scheme, corr_threshold=corr_threshold,
                     asset_returns=asset_rets, liquidity=liquidity_from_ohlcv(ohlcvs))
    alloc_info = allocation_report(alloc, cfg.backtest.initial_capital,
                                   cfg.risk.risk_per_trade)
    unallocated = sorted(s.symbol for s in studies if s.symbol not in alloc.weights)
    if unallocated:
        log.info("allocation excluded %d symbol(s) under the risk caps: %s",
                 len(unallocated), ", ".join(unallocated))
    for w in alloc.warnings:
        log.info("allocation: %s", w)

    funded = [s for s in studies if s.symbol in alloc.weights] or studies
    port_eq = combine_equity(studies, cfg.backtest.initial_capital, weights=alloc.weights)
    port_trades = pd.concat([s.trades.assign(symbol=s.symbol) for s in funded
                             if not s.trades.empty], ignore_index=True) \
        if any(not s.trades.empty for s in funded) else pd.DataFrame()
    port_metrics = full_report(
        port_eq, port_trades,
        periods_per_year=float(np.median([s.bars_per_year for s in studies])),
        n_trials=n_trials, initial_capital=cfg.backtest.initial_capital,
    )
    port_metrics["exit_breakdown"] = exit_breakdown(port_trades)
    port_metrics["tag_breakdown"] = tag_breakdown(port_trades)
    port_metrics["monthly"] = monthly_table(port_eq).to_dict()

    benchmark = None
    if with_benchmark:
        benchmark = _run_benchmark(cfg, studies[0].symbol, store, source)

    xv = None
    if cross_validate:
        from .indicators.reference import run_all

        try:
            xv = run_all(studies[0].ohlcv, int(strat.params.get("swing_length", 12)))
        except Exception as exc:  # noqa: BLE001
            log.warning("cross-validation skipped: %s", exc)
            xv = {"error": str(exc)}

    return Study(
        config=cfg, strategy_name=cfg.strategy.name,
        strategy_params=dict(strat.params), symbols=studies,
        portfolio_equity=port_eq, portfolio_metrics=port_metrics,
        portfolio_trades=port_trades, benchmark=benchmark, cross_validation=xv,
        allocation=alloc_info, unallocated=unallocated,
    )


def _run_benchmark(cfg: AppConfig, symbol: str, store: DataStore,
                   source: Optional[str]) -> Optional[Dict[str, Any]]:
    """Buy-and-hold on the first symbol, under the same capital and costs."""
    try:
        bench_cfg = cfg.merge({"strategy.name": "buy_and_hold"})
        s = run_symbol(bench_cfg, symbol, store=store, source=source)
        return {"symbol": symbol, "metrics": s.metrics}
    except Exception as exc:  # noqa: BLE001
        log.warning("benchmark failed: %s", exc)
        return None


def correlation_matrix(studies: List[SymbolStudy]) -> pd.DataFrame:
    """Return correlation of per-symbol bar returns - the diversification check.

    A "diversified" basket of crypto alts routinely correlates 0.8+; if that is
    the case here, ``max_open_positions`` is not really limiting exposure.
    """
    if len(studies) < 2:
        return pd.DataFrame()
    rets = {}
    for s in studies:
        rets[s.symbol] = np.log(s.equity / s.equity.shift(1))
    df = pd.DataFrame(rets)
    return df.corr(min_periods=20)


def cost_sensitivity(cfg: AppConfig, symbol: str, store: Optional[DataStore] = None,
                     multipliers: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
                     source: Optional[str] = None) -> pd.DataFrame:
    """Re-run the study at multiples of the configured transaction cost.

    The single most useful robustness check for a retail strategy: if the edge
    dies at 2x cost, it was never an edge, it was a fee subsidy.
    """
    store = store or DataStore(cfg)
    strat = build_strategy(cfg)
    eff = effective_config(cfg, strat)
    rows = []
    for mult in multipliers:
        c = eff.merge({
            "costs.fee_bps": cfg.costs.fee_bps * mult,
            "costs.slippage_bps": cfg.costs.slippage_bps * mult,
        })
        s = run_symbol(c, symbol, store=store, strat=strat, source=source)
        m = s.metrics
        rows.append({
            "cost_multiplier": mult,
            "fee_bps": c.costs.fee_bps,
            "slippage_bps": c.costs.slippage_bps,
            "round_trip_bps": (c.costs.fee_bps + c.costs.slippage_bps) * 2,
            "return_pct": m.get("total_return_pct"),
            "sharpe": m.get("sharpe"),
            "profit_factor": m.get("profit_factor"),
            "trades": m.get("n_trades"),
            "fees_paid": m.get("total_fees"),
        })
    return pd.DataFrame(rows)
