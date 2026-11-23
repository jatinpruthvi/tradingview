"""tvlab command line interface.

    tvlab markets                       list configured markets and styles
    tvlab data                          fetch/preview data for a universe
    tvlab scan                          screen a universe for setups right now
    tvlab backtest                      single-pass backtest
    tvlab study                         backtest + walk-forward + MC + nulls + report
    tvlab compare                       run every strategy side by side
    tvlab validate                      lookahead audit + pattern recovery only
    tvlab serve                         dashboard (FastAPI + lightweight-charts)

Every command takes ``--market`` and ``--style`` to pick a config layer, and
``--set key.path=value`` for one-off overrides.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import CONFIG_DIR, REPO_ROOT, AppConfig, load_config

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def _parse_set(pairs: tuple) -> Dict[str, Any]:
    """Parse ``--set key=value`` pairs, coercing JSON-ish scalars."""
    out: Dict[str, Any] = {}
    for item in pairs or ():
        if "=" not in item:
            raise click.BadParameter(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _cfg(market, style, config, set_pairs, source=None) -> AppConfig:
    cfg = load_config(market=market, style=style, config_path=config,
                      overrides=_parse_set(set_pairs))
    if source:
        cfg = cfg.merge({"data.source": source})
    return cfg


def _symbols_arg(cfg: AppConfig, symbols: Optional[str]) -> List[str]:
    if symbols:
        return [s.strip() for s in symbols.split(",") if s.strip()]
    return list(cfg.universe.symbols)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="debug logging")
@click.version_option(package_name=None, version="tvlab 0.1.0")
def cli(verbose: bool) -> None:
    """tvlab - multi-market backtest, validation and screening workbench."""
    _setup_logging(verbose)


# --------------------------------------------------------------------- markets
@cli.command()
def markets() -> None:
    """List the configured markets and trading styles."""
    mk = sorted(p.stem for p in (CONFIG_DIR / "markets").glob("*.yaml"))
    st = sorted(p.stem for p in (CONFIG_DIR / "styles").glob("*.yaml"))

    t = Table(title="Markets", show_lines=False)
    for col in ("market", "symbols", "exchange", "quote", "fee bp", "slip bp", "shorts"):
        t.add_column(col)
    for m in mk:
        c = load_config(market=m)
        t.add_row(m, str(len(c.universe.symbols)), c.universe.exchange,
                  c.universe.quote_currency, f"{c.costs.fee_bps:.1f}",
                  f"{c.costs.slippage_bps:.1f}",
                  "yes" if c.risk.allow_short else "no")
    console.print(t)

    t2 = Table(title="Styles")
    for col in ("style", "execution", "bias", "strategy", "max hold", "trials"):
        t2.add_column(col)
    for s in st:
        c = load_config(market=mk[0] if mk else None, style=s)
        t2.add_row(s, c.timeframe.execution, str(c.timeframe.bias or "-"),
                   c.strategy.name, str(c.backtest.max_bars_in_trade or "-"),
                   str(c.walk_forward.max_trials))
    console.print(t2)
    console.print(f"\n{len(mk)} markets x {len(st)} styles = {len(mk)*len(st)} combinations.")
    console.print("Usage: tvlab study --market crypto --style swing")


# ------------------------------------------------------------------------ data
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None, help="comma-separated, overrides the config")
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener"]))
@click.option("--timeframe", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--export", type=click.Path(path_type=Path), default=None,
              help="write the fetched bars to a parquet/csv directory")
def data(market, style, config, set_pairs, symbols, data_source, timeframe, limit, export) -> None:
    """Fetch and summarise data for a universe."""
    from .data.store import DataStore

    cfg = _cfg(market, style, config, set_pairs, data_source)
    store = DataStore(cfg)
    syms = _symbols_arg(cfg, symbols)
    tf = timeframe or cfg.timeframe.execution

    t = Table(title=f"Data - source={cfg.data.source} timeframe={tf}")
    for col in ("symbol", "bars", "from", "to", "first", "last", "return %",
                "ann vol %", "max DD %", "median vol"):
        t.add_column(col, justify="right" if col not in ("symbol", "from", "to") else "left")

    frames = {}
    for s in syms:
        try:
            df = store.ohlcv(s, tf, limit=limit)
            frames[s] = df
            r = np.log(df["close"]).diff().dropna()
            from .data.base import bars_per_year
            bpy = bars_per_year(tf, cfg.universe.market)
            eq = df["close"] / df["close"].iloc[0]
            dd = float((eq / eq.cummax() - 1).min())
            t.add_row(
                s, str(len(df)), f"{df.index.min():%Y-%m-%d}", f"{df.index.max():%Y-%m-%d}",
                f"{df['close'].iloc[0]:,.4g}", f"{df['close'].iloc[-1]:,.4g}",
                f"{(eq.iloc[-1]-1)*100:,.1f}", f"{r.std()*np.sqrt(bpy)*100:,.1f}",
                f"{dd*100:,.1f}", f"{df['volume'].median():,.3g}",
            )
        except Exception as exc:  # noqa: BLE001
            t.add_row(s, "FAILED", str(exc)[:60], "", "", "", "", "", "", "")
    console.print(t)

    if export and frames:
        export = Path(export)
        export.mkdir(parents=True, exist_ok=True)
        for s, df in frames.items():
            safe = "".join(ch if ch.isalnum() else "_" for ch in s)
            df.to_parquet(export / f"{safe}_{tf}.parquet")
        console.print(f"\nexported {len(frames)} frames to {export}")


# ------------------------------------------------------------------------ scan
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener"]))
@click.option("--screener", is_flag=True,
              help="build the universe from the TradingView screener first")
@click.option("--timeframe", default=None)
@click.option("--top", type=int, default=20)
def scan(market, style, config, set_pairs, symbols, data_source, screener, timeframe, top) -> None:
    """Screen a universe and report the current setup state per symbol.

    For each symbol this reports the higher-timeframe trend, whether a liquidity
    sweep fired recently, whether price is inside an unmitigated FVG or order
    block, and the resulting signal. It is a *read-out of now*, not a forecast:
    a shortlist to look at, not a list of orders to place.

    The scanning itself lives in :mod:`tvlab.screener`, shared with the
    dashboard's ``/api/scan`` so the two cannot drift apart.
    """
    from .data.store import DataStore
    from .pipeline import build_strategy
    from .screener import rank, scan_universe, summarize

    cfg = _cfg(market, style, config, set_pairs, data_source)
    tf = timeframe or cfg.timeframe.execution

    syms = _symbols_arg(cfg, symbols)
    if screener:
        try:
            from .data.tv_screener_source import TvScreenerSource

            src = TvScreenerSource(market=cfg.universe.market)
            syms = src.symbols(min_volume=cfg.universe.min_quote_volume,
                               max_symbols=cfg.universe.max_symbols)
            console.print(f"screener returned {len(syms)} symbols")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]screener unavailable ({exc}); using config symbols[/yellow]")

    store = DataStore(cfg)
    strat = build_strategy(cfg)
    rows, errors = scan_universe(store, cfg, syms[:top * 2], strat=strat, timeframe=tf)

    t = Table(title=f"Scan - {cfg.strategy.name} on {tf} ({cfg.universe.market})")
    for col in ("symbol", "last", "chg %", "trend", "ADX", "RSI", "ATR %",
                "sweep", "sweep age", "FVG/OB", "signal", "stop", "target", "R:R"):
        t.add_column(col, justify="left" if col == "symbol" else "right")

    _trend_markup = {1: "[green]UP[/green]", -1: "[red]DOWN[/red]", 0: "flat"}
    _signal_markup = {1: "[green]LONG[/green]", -1: "[red]SHORT[/red]", 0: "-"}

    for r in rows:
        if r.bars < 2:
            t.add_row(r.symbol, "NO DATA", (r.warnings or [""])[0][:70], *[""] * 11)
            continue
        t.add_row(
            r.symbol, f"{r.last:,.4g}", f"{r.change_pct:+.2f}",
            _trend_markup.get(r.trend, "flat"),
            f"{r.adx:.1f}" if np.isfinite(r.adx) else "-",
            f"{r.rsi:.1f}" if np.isfinite(r.rsi) else "-",
            f"{r.atr_pct:.2f}" if np.isfinite(r.atr_pct) else "-",
            r.sweep or "-",
            f"{r.sweep_age:.0f}" if r.sweep_age is not None else "-",
            r.zones,
            _signal_markup.get(r.signal, "-"),
            f"{r.stop:,.4g}" if r.stop is not None else "-",
            f"{r.target:,.4g}" if r.target is not None else "-",
            f"{r.reward_risk:.2f}" if r.reward_risk is not None else "-",
        )
    for sym, err in errors.items():
        t.add_row(sym, "FAILED", err[:70], *[""] * 11)

    console.print(t)

    s = summarize(rows, errors)
    console.print(f"\n[bold]{s['live']} live signal(s)[/bold] of {s['scanned']} symbols scanned"
                  + (f" ({s['failed']} failed)" if s["failed"] else "") + ".")
    for r in rank(rows):
        side = "LONG " if r.signal > 0 else "SHORT"
        rr = f"  R:R {r.reward_risk:.2f}" if r.reward_risk is not None else ""
        console.print(f"  {side} {r.symbol}{rr}")
    if not s["live"]:
        console.print("  No symbol satisfies bias + sweep + confirmation right now. "
                      "That is the normal state - this setup should not fire often.")
    if s["failed"]:
        console.print(f"[yellow]  {s['failed']} symbol(s) could not be scanned; "
                      f"see the FAILED rows above.[/yellow]")


# -------------------------------------------------------------------- backtest
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--strategy", "strategy_name", default=None)
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener"]))
@click.option("--timeframe", default=None)
@click.option("--limit", type=int, default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def backtest(market, style, config, set_pairs, symbols, strategy_name, data_source,
             timeframe, limit, out) -> None:
    """Single-pass backtest across the universe (no walk-forward, no nulls)."""
    from .backtest.metrics import format_metrics
    from .data.store import DataStore
    from .pipeline import run_study
    from .report import write_report

    cfg = _cfg(market, style, config, set_pairs, data_source)
    if strategy_name:
        cfg = cfg.merge({"strategy.name": strategy_name})
    if timeframe:
        cfg = cfg.merge({"timeframe.execution": timeframe})
    if limit:
        cfg = cfg.merge({"data.synthetic_bars": limit})

    store = DataStore(cfg)
    study = run_study(cfg, symbols=_symbols_arg(cfg, symbols), store=store)

    console.print(f"\n[bold]{study.strategy_name}[/bold] on {cfg.universe.market} "
                  f"{cfg.timeframe.execution} - {len(study.symbols)} symbols\n")
    console.print(format_metrics(study.portfolio_metrics or {}))
    console.print()

    t = Table(title="Per symbol")
    for col in ("symbol", "return %", "maxDD %", "sharpe", "trades", "win %", "PF", "avg R"):
        t.add_column(col, justify="left" if col == "symbol" else "right")
    for _, row in study.per_symbol.iterrows():
        t.add_row(str(row["symbol"]), _f(row["return_pct"]), _f(row["max_dd_pct"]),
                  _f(row["sharpe"]), str(int(row["trades"] or 0)),
                  _f((row["win_rate"] or 0) * 100), _f(row["profit_factor"]), _f(row["avg_r"], 3))
    console.print(t)

    if study.benchmark:
        bm = study.benchmark["metrics"]
        console.print(f"\nbenchmark buy&hold {study.benchmark['symbol']}: "
                      f"{bm.get('total_return_pct', float('nan')):.2f}% return, "
                      f"{bm.get('max_drawdown_pct', float('nan')):.2f}% DD")

    if out:
        paths = write_report(study, Path(out), data_provenance="backtest command (no validation)")
        console.print(f"\nreport -> {paths['markdown']}")


# ----------------------------------------------------------------------- study
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--strategy", "strategy_name", default=None)
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener"]))
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--no-walk-forward", is_flag=True)
@click.option("--no-monte-carlo", is_flag=True)
@click.option("--no-nulls", is_flag=True, help="skip the permutation/random-entry tests")
@click.option("--no-costs", is_flag=True, help="skip the cost-sensitivity sweep")
@click.option("--recovery", is_flag=True, help="also run the pattern-recovery test")
@click.option("--xval", is_flag=True, help="cross-check SMC vs the vendored reference")
@click.option("--sims", type=int, default=None, help="Monte Carlo simulations")
@click.option("--allocation", type=click.Choice(["equal", "inverse_vol"]), default="equal",
              help="how capital is split across symbols before the curves are combined")
def study(market, style, config, set_pairs, symbols, strategy_name, data_source, out,
          no_walk_forward, no_monte_carlo, no_nulls, no_costs, recovery, xval, sims,
          allocation) -> None:
    """The full run: backtest + walk-forward + Monte Carlo + validity checks + report."""
    from .backtest.metrics import format_metrics
    from .study import run_full_study

    cfg = _cfg(market, style, config, set_pairs, data_source)
    if out is None:
        out = Path(cfg.output.dir) / f"{cfg.strategy.name}_{cfg.universe.market}_{cfg.timeframe.execution}"

    with console.status("running full study (backtest, walk-forward, Monte Carlo, null tests)..."):
        full = run_full_study(
            cfg, symbols=_symbols_arg(cfg, symbols), out_dir=Path(out),
            source=data_source,
            do_walk_forward=False if no_walk_forward else None,
            do_monte_carlo=False if no_monte_carlo else None,
            do_nulls=not no_nulls, do_cost_sensitivity=not no_costs,
            do_recovery=recovery, do_cross_validation=xval, n_sims=sims,
            allocation_scheme=allocation,
        )

    v = full.verdict()
    colour = {"PASSES ON THIS DATA": "green", "PARTIAL": "yellow",
              "NOT VALIDATED": "red"}.get(v["verdict"], "white")
    console.print(f"\n[bold {colour}]VERDICT: {v['verdict']}[/bold {colour}]")
    console.print(v["headline"])
    console.print()

    t = Table(title="Validity checks")
    t.add_column("#", justify="right")
    t.add_column("Check")
    t.add_column("Result", justify="center")
    t.add_column("Detail")
    for i, c in enumerate(v["checks"], 1):
        mark = "[green]PASS[/green]" if c["passed"] else (
            "[red]FAIL[/red]" if c["passed"] is False else "n/a")
        t.add_row(str(i), c["check"], mark, c["detail"])
    console.print(t)

    console.print("\n[bold]Portfolio results[/bold]")
    console.print(format_metrics(full.study.portfolio_metrics or {}))

    if full.walk_forward is not None:
        ws = full.walk_forward.summary()
        console.print("\n[bold]Walk-forward (out of sample)[/bold]")
        console.print(f"  windows profitable   {ws['windows_profitable']}/{ws['n_windows']}")
        console.print(f"  OOS return           {ws['oos_return_pct']:.2f}%")
        console.print(f"  OOS max drawdown     {ws['oos_max_dd_pct']:.2f}%")
        console.print(f"  OOS Sharpe           {ws['oos_sharpe']:.2f}")
        console.print(f"  OOS profit factor    {ws['oos_profit_factor']:.2f}")
        console.print(f"  total trials         {ws['total_trials']}")

    if full.monte_carlo is not None:
        ms = full.monte_carlo.summary()
        if "error" not in ms:
            console.print("\n[bold]Monte Carlo[/bold]")
            console.print(f"  P(final return < 0)  {ms['prob_loss']:.1%}")
            console.print(f"  median final return  {ms['final_return_pct']['median']:.2f}%")
            console.print(f"  95th pct max DD      {ms['max_drawdown_pct']['p95']:.2f}%")
            for k, val in ms["ruin_probabilities"].items():
                console.print(f"  P({k.replace('dd_exceeds_','DD >= ').replace('pct','%')})"
                              f"{'':<6}{val:.1%}")

    if full.nulls:
        console.print("\n[bold]Null tests[/bold]")
        for k, d in full.nulls.items():
            if not isinstance(d, dict):
                continue
            pv = d.get("p_value", float("nan"))
            console.print(f"  {k:<20} p={pv:.3f}  actual={d.get('actual', float('nan')):.2f} "
                          f"null_mean={d.get('null_mean', float('nan')):.2f}  -> {d.get('verdict')}")

    console.print(f"\n[bold]Report:[/bold] {full.outputs.get('markdown', '(not written)')}")
    for k, p in full.outputs.items():
        if k != "markdown":
            console.print(f"  {k:<12} {p}")
    console.print(f"\ntimings: " + ", ".join(f"{k}={v_:.1f}s" for k, v_ in full.timings.items()))


# --------------------------------------------------------------------- compare
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener"]))
@click.option("--strategies", default=None,
              help="comma-separated; default is all registered")
def compare(market, style, config, set_pairs, symbols, data_source, strategies) -> None:
    """Run every strategy on the same data and rank them.

    Always include the baselines. If your new setup cannot beat buy-and-hold and
    a Donchian breakout on the same sample, it is not a setup.
    """
    from .data.store import DataStore
    from .pipeline import run_study
    from .strategy import available

    cfg = _cfg(market, style, config, set_pairs, data_source)
    names = ([s.strip() for s in strategies.split(",")] if strategies
             else available())
    store = DataStore(cfg)
    syms = _symbols_arg(cfg, symbols)

    rows = []
    for name in names:
        try:
            c = cfg.merge({"strategy.name": name})
            st = run_study(c, symbols=syms, store=store, with_benchmark=False)
            m = st.portfolio_metrics or {}
            rows.append({
                "strategy": name,
                "return_pct": m.get("total_return_pct"),
                "cagr_pct": m.get("cagr_pct"),
                "max_dd_pct": m.get("max_drawdown_pct"),
                "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"),
                "calmar": m.get("calmar"),
                "trades": m.get("n_trades"),
                "win_rate": m.get("win_rate"),
                "profit_factor": m.get("profit_factor"),
                "avg_r": m.get("avg_r"),
                "dsr": (m.get("deflated_sharpe") or {}).get("dsr"),
                "long/short": f"{m.get('long_trades', 0)}/{m.get('short_trades', 0)}",
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"strategy": name, "return_pct": None,
                         "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame(rows)
    if "sharpe" in df:
        df = df.sort_values("sharpe", ascending=False, na_position="last")

    t = Table(title=f"Strategy comparison - {cfg.universe.market} {cfg.timeframe.execution}, "
                    f"{len(syms)} symbols")
    for col in df.columns:
        t.add_column(str(col), justify="left" if col in ("strategy", "long/short") else "right")
    for _, row in df.iterrows():
        t.add_row(*[_cell(row[c]) for c in df.columns])
    console.print(t)
    console.print("\nRead this table with the cost-sensitivity and null tests, not alone. "
                  "On synthetic data a trend-follower wins by construction.")


# -------------------------------------------------------------------- validate
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--data-source", default=None)
@click.option("--sweeps", type=int, default=25, help="injected events for the recovery test")
def validate(market, style, config, set_pairs, symbols, data_source, sweeps) -> None:
    """Run the validity checks only: lookahead audit + pattern recovery + xval."""
    from .data.store import DataStore
    from .data.synthetic import basket
    from .indicators import reference
    from .pipeline import build_strategy
    from .validation import lookahead as LA
    from .validation import recovery as RC

    cfg = _cfg(market, style, config, set_pairs, data_source)
    store = DataStore(cfg)
    syms = _symbols_arg(cfg, symbols) or ["BTC/USDT"]
    sym = syms[0]
    df = store.ohlcv(sym, cfg.timeframe.execution, limit=2000)
    bias = store.bias_frame(sym)
    strat = build_strategy(cfg)
    L = int(strat.params.get("swing_length", 12))

    console.print(f"\n[bold]Lookahead audit[/bold] - {sym}, {len(df)} bars")
    strats = [strat]
    for extra in ("ema_pullback", "donchian_breakout"):
        if extra != cfg.strategy.name:
            strats.append(build_strategy(cfg, extra))
    la = LA.run_all(df, bias=bias, strategies=strats, swing_length=L)
    t = Table()
    for col in ("target", "columns", "cuts", "leaks", "result"):
        t.add_column(col, justify="left" if col in ("target", "result") else "right")
    for name, d in la.items():
        if not isinstance(d, dict) or "columns_checked" not in d:
            continue
        t.add_row(name, str(d.get("columns_checked")), str(len(d.get("cuts_tested", []))),
                  str(d.get("n_leaks")),
                  "[green]OK[/green]" if d.get("ok") else "[red]LEAK[/red]")
    console.print(t)
    console.print(f"overall: {'[green]PASS[/green]' if la.get('ok') else '[red]FAIL[/red]'}")

    console.print(f"\n[bold]Cross-validation vs vendored smartmoneyconcepts[/bold]")
    try:
        xv = reference.run_all(df, swing_length=L)
        for key in ("swings", "fvg"):
            d = xv.get(key)
            if isinstance(d, dict) and "match_rate" in d:
                console.print(f"  {key:<8} ours={d['ours']:<5} theirs={d['theirs']:<5} "
                              f"matched={d['matched']:<5} ({d['match_rate']:.1%}, "
                              f"lag {d['lag_bars']} bars)")
        inv = xv.get("invariants", {})
        console.print(f"  invariants: rsi in range={inv.get('rsi_in_range')}, "
                      f"adx in range={inv.get('adx_in_range')}, "
                      f"atr>=0={inv.get('atr_nonneg')}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]skipped: {exc}[/yellow]")

    console.print(f"\n[bold]Pattern recovery[/bold] - {sweeps} injected liquidity sweeps")
    rdf = basket([sym], n_bars=min(2500, max(len(df), 800)),
                 timeframe=cfg.timeframe.execution,
                 seed=cfg.data.synthetic_seed + 777, inject_sweeps=sweeps)[sym]
    rbias = store.bias_frame(sym, source="synthetic")
    r1 = RC.run_smc_recovery(rdf, swing_length=L, tolerance=6)
    r2 = RC.run_smc_recovery(rdf, swing_length=L, tolerance=6, use_strategy=True,
                             strategy=strat, bias=rbias)
    t2 = Table()
    for col in ("detector", "injected", "detected", "recall", "precision", "F1",
                "side acc", "latency", "eligible recall", "blocked"):
        t2.add_column(col, justify="left" if col == "detector" else "right")
    for label, r in (("raw SMC", r1), (f"{strat.name} entries", r2)):
        t2.add_row(label, str(r.injected), str(r.detected), _p(r.recall), _p(r.precision),
                   _p(r.f1), _p(r.side_accuracy), _f(r.mean_latency, 2),
                   _p(r.eligible_recall) if np.isfinite(r.eligible_recall) else "-",
                   str(r.blocked_by_filter))
    console.print(t2)
    for r in (r1, r2):
        for w in r.warnings:
            console.print(f"  [yellow]![/yellow] {w}")

    console.print(f"\n[bold]Swing-length sensitivity[/bold] (same truth, different parameter)")
    sens = RC.sweep_size_sensitivity(
        lambda _L: basket([sym], n_bars=min(2500, max(len(df), 800)),
                          timeframe=cfg.timeframe.execution,
                          seed=cfg.data.synthetic_seed + 777, inject_sweeps=sweeps)[sym],
        swing_lengths=(6, 8, 12, 16, 20),
    )
    console.print(sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    best = sens.loc[sens["f1"].idxmax()] if len(sens) else None
    if best is not None:
        console.print(f"\nbest F1 at swing_length={int(best['swing_length'])} "
                      f"(F1={best['f1']:.3f}, recall={best['recall']:.2f}, "
                      f"precision={best['precision']:.2f})")


# ----------------------------------------------------------------------- serve
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--data-source", default=None)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--symbols", default=None)
def serve(market, style, config, set_pairs, data_source, host, port, symbols) -> None:
    """Serve the dashboard: interactive lightweight-charts + backtest results."""
    import uvicorn

    from .dashboard.app import create_app

    cfg = _cfg(market, style, config, set_pairs, data_source)
    if symbols:
        cfg = cfg.merge({"universe.symbols": [s.strip() for s in symbols.split(",")]})
    app = create_app(cfg)
    console.print(f"\n[bold]dashboard[/bold] -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ----------------------------------------------------------------------- utils
def _f(v: Any, dp: int = 2) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def _p(v: Any) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(v)


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if not np.isfinite(v) else f"{v:,.3f}"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (np.floating,)):
        return "-" if not np.isfinite(v) else f"{float(v):,.3f}"
    return str(v)


def main() -> None:
    cli(prog_name="tvlab")


if __name__ == "__main__":
    main()
