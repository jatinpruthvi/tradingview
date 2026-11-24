"""tvlab command line interface.

    tvlab markets                       list configured markets and styles
    tvlab data                          fetch/preview data for a universe
    tvlab scan                          screen a universe for setups right now
    tvlab suggest                       the best idea per symbol, after costs
    tvlab daily                         scan + rank + fundamental gates, journaled
    tvlab review                        score that journal against what happened
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
from datetime import datetime, timedelta, timezone
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
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                  "nse", "yahoo_chart"]))
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


# -------------------------------------------------------------------- suggest
def _measured_win_rate(cfg: AppConfig, strategy_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read the win rate this strategy actually achieved, from the last study.

    This is what turns a breakeven threshold into a decision. On its own, "you
    need a 26.7% win rate" is just arithmetic; next to "you measured 38.5%, but
    on 13 trades" it is an answer - and the sample size is the part that stops it
    being over-read.

    Returns ``None`` only when no report file exists at all. A report that ran but
    produced no trades returns ``status="no_trades"`` - collapsing that into "no
    previous study" tells someone to re-run a study they already ran, and hides the
    one thing that run actually learned.

    ``status`` is one of ``ok`` | ``no_trades`` | ``corrupt``.
    """
    import json

    strat = strategy_name or cfg.strategy.name
    tf = cfg.timeframe.execution
    candidates = [
        REPO_ROOT / cfg.output.dir / f"{strat}_{cfg.universe.market}_{tf}" / f"{strat}_{cfg.universe.market}_report.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            d = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - a corrupt report must not kill the scan
            return {"status": "corrupt", "win_rate": None, "n_trades": 0, "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}"}
        pm = d.get("portfolio_metrics") or {}
        wr, n = pm.get("win_rate"), pm.get("n_trades")
        # The win rate alone is not enough to act on. It comes from a study that
        # also ran eight validity checks, and quoting the number without the
        # verdict is how a 38.5% win rate measured on 13 trades of an unvalidated
        # strategy starts to look like a green light.
        v = d.get("verdict") or {}
        base = {
            "payoff_ratio": pm.get("payoff_ratio"), "expectancy": pm.get("expectancy"),
            "profit_factor": pm.get("profit_factor"),
            "path": str(path), "created_at": d.get("created_at"),
            # None means "this report predates the verdict being written to JSON",
            # which must not be rendered as either validated or unvalidated.
            "verdict": v.get("verdict"),
            "checks_passed": v.get("n_passed"),
            "checks_total": v.get("n_checks"),
            "failed_checks": [c.get("check") for c in (v.get("checks") or [])
                              if c.get("passed") is False],
        }
        if wr is None or not np.isfinite(wr):
            # A real study that took no trades. `smc_trend` is deliberately rare, so
            # on a short sample or a small universe this is a normal outcome - and
            # an informative one.
            return {**base, "status": "no_trades", "win_rate": None,
                    "n_trades": int(n or 0)}
        return {**base, "status": "ok", "win_rate": float(wr), "n_trades": int(n or 0)}
    return None


@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                  "nse", "yahoo_chart"]))
@click.option("--timeframe", default=None)
@click.option("--top", type=int, default=10, help="how many candidates to show")
@click.option("--max-breakeven", type=float, default=0.60, show_default=True,
              help="reject anything needing a higher win rate than this to break even")
@click.option("--trendlines", is_flag=True, default=False,
              help="fit diagonal support/resistance and prefer structural stops")
def suggest(market, style, config, set_pairs, symbols, data_source, timeframe, top,
            max_breakeven, trendlines) -> None:
    """Rank the universe into a shortlist, best risk:reward after costs first.

    This is the command to run when you want "what is the best idea right now".
    It answers that question honestly, which means it ranks by **expectancy after
    costs** rather than by the reward:risk ratio on paper.

    Why that distinction is the whole point: reward:risk is a ratio of two
    distances, so you can make it arbitrarily large by shrinking the stop. Sorting
    by R:R therefore promotes the tightest stops - the ones ordinary noise hits -
    and demotes the ideas that are actually tradeable. Every candidate below is
    shown with its **breakeven win rate**: the win rate you must exceed to make
    money at your configured fees and slippage. That number is arithmetic, not a
    forecast, and it needs no assumption about the future.

    Where a previous ``tvlab study`` exists for this market/style, its *measured*
    win rate is printed alongside so you can compare the threshold against what
    the strategy has actually done - with the trade count, because 38% on 13
    trades is not 38%.

    A shortlist is not a recommendation. Nothing here has been validated as
    profitable; ``tvlab study`` is what decides that, and today it says NOT
    VALIDATED.
    """
    from .data.store import DataStore
    from .pipeline import build_strategy
    from .screener import SCAN_BARS, scan_universe, suggest as build_suggestions, summarize

    cfg = _cfg(market, style, config, set_pairs, data_source)
    tf = timeframe or cfg.timeframe.execution
    syms = _symbols_arg(cfg, symbols)

    store = DataStore(cfg)
    strat = build_strategy(cfg)
    rows, errors = scan_universe(store, cfg, syms, strat=strat, timeframe=tf,
                                 bars=SCAN_BARS, with_trendlines=trendlines)

    res = build_suggestions(rows, fee_bps=cfg.costs.fee_bps, slippage_bps=cfg.costs.slippage_bps,
                            min_stop_distance_pct=cfg.risk.min_stop_distance_pct,
                            max_breakeven=max_breakeven)

    rt = res["round_trip_cost_bps"]
    console.print(f"\n[bold]Suggestions[/bold] - {cfg.strategy.name} on {tf} "
                  f"({cfg.universe.market}/{cfg.style or style or 'default'})")
    console.print(f"  round-trip cost {rt:.1f} bp  ·  scanned {len(rows)} symbols  ·  "
                  f"{res['n_candidates']} candidate(s), {res['n_rejected']} rejected  ·  "
                  f"max breakeven {max_breakeven * 100:.0f}%")

    measured = _measured_win_rate(cfg, strat.name)
    status = (measured or {}).get("status")
    if status is None and measured is not None:
        # Defensive: the dashboard round-trips this dict through JSON and tests
        # build it by hand, so a missing status must not silently demote a real
        # measurement to "no previous study". Infer it from the win rate.
        status = "ok" if measured.get("win_rate") is not None else "no_trades"
    # A non-empty `measured` dict is NOT the same as having a win rate: the
    # no_trades and corrupt statuses both carry a path and a message but
    # win_rate=None. Every comparison against it has to go through this flag,
    # or `None > float` raises a TypeError on exactly the path that only appears
    # when there are candidates AND the last study took no trades.
    has_measured = status == "ok"
    if status == "ok":
        n = measured["n_trades"]
        caveat = ("[red]far too few trades to trust[/red]" if n < 30
                  else "[yellow]thin sample[/yellow]" if n < 100 else "adequate sample")
        pf = measured.get("profit_factor")
        console.print(f"  measured on the last study: win rate "
                      f"[bold]{measured['win_rate'] * 100:.1f}%[/bold] over {n} trades ({caveat})"
                      + (f", profit factor {pf:.2f}" if pf else ""))
        # A win rate quoted without the verdict it came from reads as an
        # endorsement. The study that produced this number also ran eight validity
        # checks, and if it failed them the number describes one sample rather than
        # an edge - which is the difference between a threshold comparison and a
        # reason to trade.
        verdict = measured.get("verdict")
        cp, ct = measured.get("checks_passed"), measured.get("checks_total")
        score = f" ({cp}/{ct} checks passed)" if cp is not None and ct else ""
        if verdict == "PASSES ON THIS DATA":
            console.print(f"  [dim]that study cleared its validity checks on this sample - "
                          f"necessary, not sufficient; confirm out of sample.[/dim]")
        elif verdict:
            console.print(f"  [red]that study was [bold]{verdict}[/bold]{score}: this win rate "
                          f"describes one historical sample, not an edge.[/red]")
            failed = measured.get("failed_checks") or []
            if failed:
                shown = ", ".join(str(f) for f in failed[:3])
                console.print(f"  [dim]failed: {shown}"
                              f"{' …and ' + str(len(failed) - 3) + ' more' if len(failed) > 3 else ''}[/dim]")
        else:
            console.print(f"  [dim]that report predates the verdict being written to JSON, so "
                          f"its validity status is unknown - re-run `tvlab study`.[/dim]")
        console.print(f"  [dim]{measured['path']}[/dim]")
    elif status == "no_trades":
        # A study ran and took no trades. That is an answer, not an absence of one:
        # there is no win rate to compare the breakeven thresholds against, and
        # saying "no previous study" would send the user to re-run what they ran.
        console.print("  [yellow]the last study took [bold]no trades[/bold], so there is no "
                      "measured win rate to compare these breakeven thresholds "
                      "against.[/yellow]")
        console.print("  [dim]Widen the universe, lengthen the history, or accept that this "
                      "setup does not fire here. `smc_trend` is deliberately rare.[/dim]")
        console.print(f"  [dim]{measured['path']}[/dim]")
    elif status == "corrupt":
        console.print(f"  [red]the last study report is unreadable "
                      f"({measured.get('error', 'unknown error')}); re-run "
                      f"`tvlab study`.[/red]")
        console.print(f"  [dim]{measured['path']}[/dim]")
    else:
        console.print("  [yellow]no previous study for this market/style - run "
                      "`tvlab study` so the breakeven thresholds have something to be "
                      "compared against[/yellow]")

    why = summarize(rows, errors)

    # "Nothing qualifies" is a statement about the market. It is only true if the
    # market was actually observed. When every symbol failed to load, scanned is 0
    # and the honest answer is "we could not look", not "we looked and found
    # nothing" - the second reads as a legitimate quiet market and sends someone
    # away believing a data outage was a trading signal.
    if why["scanned"] == 0 and errors:
        console.print("\n[bold red]Nothing was scanned - this is a data failure, not a "
                      "market verdict.[/bold red]")
        console.print(f"  All {len(errors)} symbol(s) failed to load. First few:")
        for sym, err in list(errors.items())[:5]:
            console.print(f"    [red]{sym:<14}[/red] [dim]{err}[/dim]")
        if len(errors) > 5:
            console.print(f"    [dim]... and {len(errors) - 5} more[/dim]")
        console.print("\n  [yellow]No suggestion is possible until bars load. Check the "
                      "data source and network route, then retry:[/yellow]")
        src_flag = f"--data-source {data_source} " if data_source else ""
        console.print(f"    [dim]tvlab data --market {cfg.universe.market} "
                      f"--style {cfg.style or style or 'default'} {src_flag}--limit 50[/dim]")
        console.print("  [dim]For Indian equities the NSE route needs no key but does need "
                      "query1.finance.yahoo.com reachable for history; `tvlab nse session` "
                      "and `tvlab nse breadth` test the NSE endpoints on their own.[/dim]")
        raise SystemExit(2)

    if not res["candidates"]:
        console.print("\n[bold yellow]Nothing qualifies.[/bold yellow]")
        console.print(f"  {why['live']} live signal(s) of {why['scanned']} scanned; "
                      f"{res['n_rejected']} rejected by the cost/breakeven gate.")
        for rej in res["rejected"][:top]:
            console.print(f"    [dim]{rej['symbol']:<12} {rej['side']:<5} R:R {rej['rr']:.2f}"
                          f"  - {'; '.join(rej['reasons'])}[/dim]")
        console.print("\n  [dim]This is the normal state. A setup that requires bias + sweep +"
                      " confirmation should not fire often, and an idea whose stop is so tight"
                      " that costs eat the reward is not a worse version of a good idea - it is"
                      " untradeable at these costs.[/dim]")
        return

    t = Table(title=f"Ranked by quality after {rt:.0f} bp round-trip cost")
    for col, just in (("#", "right"), ("symbol", "left"), ("side", "left"), ("R:R", "right"),
                      ("stop %", "right"), ("reward %", "right"), ("cost R", "right"),
                      ("breakeven", "right"), ("no-cost", "right"), ("score", "right"),
                      ("stop ATR", "right"), ("ADX", "right"), ("zone", "left")):
        t.add_column(col, justify=just)

    for i, c in enumerate(res["candidates"][:top], 1):
        be = c.breakeven_win_rate
        be_colour = "green" if be < 0.30 else ("yellow" if be < 0.45 else "red")
        marker = ""
        if has_measured:
            marker = " [green]PASS[/green]" if measured["win_rate"] > be else " [red]FAIL[/red]"
        t.add_row(
            str(i), c.symbol, c.side, f"{c.rr:.2f}",
            f"{c.risk_pct * 100:.2f}",
            f"{c.reward_pct * 100:.2f}" if c.reward_pct is not None else "-",
            f"{c.cost_in_r:.2f}" if np.isfinite(c.cost_in_r) else "inf",
            f"[{be_colour}]{be * 100:.1f}%[/]{marker}",
            f"{c.implied_win_rate_at_rr * 100:.1f}%",
            f"{c.quality_score:.3f}",
            f"{c.stop_atr_multiple:.2f}" if np.isfinite(c.stop_atr_multiple) else "-",
            f"{c.adx:.1f}" if np.isfinite(c.adx) else "-",
            "yes" if c.in_zone else "-",
        )
    console.print(t)

    best = res["candidates"][0]
    console.print(f"\n[bold]Top candidate: {best.symbol} {best.side}[/bold]  "
                  f"entry ~{best.last:,.4g}  stop {best.stop:,.4g} ({best.risk_pct * 100:.2f}%)"
                  + (f"  target {best.target:,.4g}" if best.target else ""))
    for r in best.reasons:
        console.print(f"  [green]+[/green] {r}")
    for w in best.warnings:
        console.print(f"  [yellow]![/yellow] {w}")
    if has_measured:
        gap = (measured["win_rate"] - best.breakeven_win_rate) * 100
        verdict = ("clears it" if gap > 0 else "does NOT clear it")
        console.print(f"  [bold]measured {measured['win_rate'] * 100:.1f}% vs breakeven "
                      f"{best.breakeven_win_rate * 100:.1f}% -> {verdict} by {abs(gap):.1f} pts[/bold]")
        if measured["n_trades"] < 30:
            console.print(f"  [red]but that win rate comes from {measured['n_trades']} trades. "
                          f"Below 30 the confidence interval is wide enough that this comparison "
                          f"is not evidence - it is a prompt to gather more data.[/red]")

    if res["rejected"]:
        console.print(f"\n[dim]Rejected ({res['n_rejected']}):[/dim]")
        for rej in res["rejected"][:top]:
            console.print(f"  [dim]{rej['symbol']:<12} {rej['side']:<5} R:R {rej['rr']:.2f}"
                          f"  breakeven {rej['breakeven_win_rate'] * 100:.0f}%"
                          f"  - {'; '.join(rej['reasons'])}[/dim]")
    if errors:
        console.print(f"\n[yellow]{len(errors)} symbol(s) failed to load[/yellow]")

    console.print("\n[dim]Position size, if you take it: risk_per_trade "
                  f"{cfg.risk.risk_per_trade * 100:.2f}% of equity / stop distance "
                  f"{best.risk_pct * 100:.2f}% = {cfg.risk.risk_per_trade / best.risk_pct * 100:.1f}% "
                  f"of equity notional, capped at {cfg.risk.max_position_pct * 100:.0f}%.[/dim]")


# ----------------------------------------------------------------------- daily
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--data-source", default=None,
              help="synthetic | csv | ccxt | yfinance | tvscreener | nse | yahoo_chart")
@click.option("--timeframe", default=None)
@click.option("--top", type=int, default=10, help="how many ideas to show")
@click.option("--max-breakeven", type=float, default=0.60, show_default=True,
              help="reject ideas needing a higher win rate than this to break even")
@click.option("--no-fundamentals", is_flag=True, default=False,
              help="skip the fundamental gates (equity markets only)")
@click.option("--no-journal", is_flag=True, default=False,
              help="do not append this run to output/daily/journal.jsonl")
@click.option("--trendlines", is_flag=True, default=False,
              help="also fit diagonal support/resistance (~27 ms per symbol)")
def daily(market, style, config, set_pairs, symbols, data_source, timeframe, top,
          max_breakeven, no_fundamentals, no_journal, trendlines) -> None:
    """The daily run: what to look at today, and why not to trust it.

    This is the command to schedule. It composes three things that already exist
    and adds a record:

    1. **Scan** - is there a setup on the last bar of any symbol?
    2. **Rank** - of those, which can actually be taken at your configured fees and
       slippage? Ranked by cost-adjusted expectancy, never by reward:risk, because
       R:R can be inflated by shrinking the stop into the noise.
    3. **Gate on fundamentals** - is this a company worth owning while the setup
       plays out, and does it report earnings before the stop would matter?

    Then it appends one line to ``output/daily/journal.jsonl``. That is the point of
    running it daily rather than once: a recommendation you cannot look back at is
    unfalsifiable, so nothing is ever checked and nothing is ever learned. Come back
    later and score it with ``tvlab review``, which replays each journaled suggestion
    against the bars that followed and reports what it did in R.

    Writing the journal and scoring it are deliberately separate commands. This one
    runs every morning whether or not there is anything to score; ``review`` only
    becomes useful once enough bars have printed for the ideas to have resolved, and
    it says so rather than reporting a rate over one suggestion.

    \b
    What this is NOT
    ----------------
    Not advice, and not a claim that these names will go up. The strategy behind
    this ranking is NOT VALIDATED on its default configuration - it was not
    distinguishable from random entry in the study that produced that verdict - and
    that fact is printed above every shortlist rather than buried below it. The
    fundamental thresholds are risk controls that have never been backtested; they
    veto, they do not score.

    \b
    Exit codes
    ----------
    0  a shortlist was produced, or the market was observed and nothing qualified
    2  every symbol failed to load - a data failure, not a market verdict

    The distinction matters because the two have opposite remedies. "Nothing
    qualifies today" means wait; "we could not look" means fix the data route. An
    outage rendered as a quiet market sends you away believing a trading signal
    that never existed.
    """
    from .data.fundamentals import FUNDAMENTAL_MARKETS, fetch_fundamentals
    from .data.store import DataStore
    from .pipeline import build_strategy
    from .screener import SCAN_BARS, scan_universe
    from .screener.daily import append_journal, build_briefing

    cfg = _cfg(market, style, config, set_pairs, data_source)
    tf = timeframe or cfg.timeframe.execution
    syms = _symbols_arg(cfg, symbols)
    mkt = cfg.universe.market

    store = DataStore(cfg)
    strat = build_strategy(cfg)
    rows, errors = scan_universe(store, cfg, syms, strat=strat, timeframe=tf,
                                 bars=SCAN_BARS, with_trendlines=trendlines)

    # ---- fundamentals: guarded, because this is the only step that needs a route
    # to TradingView, and its absence must be visible rather than silent.
    snaps: Optional[Dict[str, Any]] = None
    fund_error: Optional[str] = None
    want_fundamentals = (cfg.fundamentals.enabled and not no_fundamentals
                         and mkt in FUNDAMENTAL_MARKETS)
    if cfg.fundamentals.enabled and mkt not in FUNDAMENTAL_MARKETS:
        fund_error = (f"market '{mkt}' has no earnings, book value or analyst "
                      f"coverage, so there is nothing to gate on")
    elif want_fundamentals and rows:
        try:
            snaps = fetch_fundamentals([r.symbol for r in rows], market=mkt,
                                       max_symbols=max(len(rows), 200),
                                       extra_fields=cfg.fundamentals.extra_fields)
        except Exception as exc:  # noqa: BLE001 - any failure is a coverage fact
            fund_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    b = build_briefing(rows, errors, cfg, snaps=snaps,
                       study=_measured_win_rate(cfg, strat.name),
                       max_breakeven=max_breakeven, top=top,
                       fundamentals_error=fund_error)

    console.print(f"\n[bold]Daily briefing[/bold] - {b.as_of}  "
                  f"[dim]{mkt} / {b.style} / {b.strategy} / {tf}[/dim]")

    # The advisory precedes the shortlist, always. Ranking candidates without
    # carrying the validation verdict forward would launder an unvalidated
    # strategy into something that reads as advice.
    colour = "green" if b.is_validated else ("red" if b.study_verdict else "yellow")
    console.print(f"  [{colour}]{b.advisory_line}[/{colour}]")

    console.print(f"  scanned {b.symbols_scanned} symbol(s)  ·  {b.signals_found} live "
                  f"signal(s)  ·  {len(b.rows)} on the shortlist  ·  "
                  f"{len(b.rejected_technical)} rejected on cost  ·  "
                  f"{len(b.rejected_fundamental)} rejected or unrated on fundamentals")
    if b.data_errors:
        console.print(f"  [yellow]{len(b.data_errors)} symbol(s) failed to load - the "
                      f"shortlist covers the rest[/yellow]")

    # ---- fundamental coverage, stated whether it ran or not
    if want_fundamentals and snaps is not None:
        cov = b.fundamental_coverage or {}
        mf = cov.get("missing_fields") or {}
        console.print(f"  fundamentals: {cov.get('symbols', 0)} symbol(s) returned, "
                      f"{cov.get('complete', 0)} complete, {cov.get('partial', 0)} partial")
        if mf:
            worst = ", ".join(f"{k} ({v})" for k, v in list(mf.items())[:4])
            console.print(f"  [dim]fields the endpoint did not report: {worst}"
                          f"{' …' if len(mf) > 4 else ''}[/dim]")
    elif fund_error:
        console.print(f"  [yellow]fundamentals NOT applied: {fund_error}[/yellow]")
        if b.rows:
            console.print("  [yellow]every name below is therefore NOT fundamentally "
                          "screened - including for an earnings release.[/yellow]")

    # ---- the three-way outcome
    if b.outcome == "data_failure":
        console.print("\n[bold red]Nothing was scanned - this is a data failure, not a "
                      "market verdict.[/bold red]")
        for sym, err in list(b.data_errors.items())[:5]:
            console.print(f"    [red]{sym:<14}[/red] [dim]{err}[/dim]")
        if len(b.data_errors) > 5:
            console.print(f"    [dim]... and {len(b.data_errors) - 5} more[/dim]")
        src_flag = f"--data-source {data_source} " if data_source else ""
        console.print("\n  [yellow]No briefing is possible until bars load. Check the data "
                      "route, then retry:[/yellow]")
        console.print(f"    [dim]tvlab data --market {mkt} --style {b.style} "
                      f"{src_flag}--limit 50[/dim]")
        raise SystemExit(2)

    if not b.rows:
        console.print("\n[bold yellow]Nothing qualifies today.[/bold yellow]")
        console.print(f"  {b.outcome_detail}")
        for rej in b.rejected_technical[:top]:
            console.print(f"    [dim]{rej['symbol']:<12} {rej['side']:<5} R:R {rej['rr']:.2f}"
                          f"  - {'; '.join(rej['reasons'])}[/dim]")
        for rej in b.rejected_fundamental[:top]:
            console.print(f"    [dim]{rej['symbol']:<12} {rej['status']:<8}"
                          f"  - {rej['detail']}[/dim]")
        console.print("\n  [dim]This is the normal state. A setup requiring bias + sweep +"
                      " confirmation should not fire often, and an idea whose stop is so"
                      " tight that costs eat the reward is untradeable, not merely worse."
                      " Come back tomorrow - that is what the journal is for.[/dim]")
        if not no_journal:
            jp = append_journal(b, REPO_ROOT / cfg.output.dir)
            console.print(f"  [dim]journaled: {jp}[/dim]")
        return

    # ---- the shortlist
    t = Table(title=f"{b.as_of} shortlist - ranked by quality after costs")
    for col, just in (("#", "right"), ("symbol", "left"), ("side", "left"), ("R:R", "right"),
                      ("stop %", "right"), ("cost R", "right"), ("breakeven", "right"),
                      ("score", "right"), ("sector", "left"), ("mkt cap", "right"),
                      ("P/E", "right"), ("ROE", "right"), ("earnings", "right"),
                      ("fundamentals", "left")):
        t.add_column(col, justify=just)

    for i, r in enumerate(b.rows, 1):
        be = r.breakeven_win_rate
        be_colour = "green" if be < 0.30 else ("yellow" if be < 0.45 else "red")
        fstat = r.fundamental_status or ("not screened" if fund_error else "-")
        fcolour = {"approved": "green", "unrated": "yellow"}.get(fstat, "dim")
        d2e = "-" if r.days_to_earnings is None else f"{r.days_to_earnings}d"
        if r.days_to_earnings is not None and r.days_to_earnings <= 14:
            d2e = f"[yellow]{d2e}[/yellow]"
        t.add_row(
            str(i), r.symbol, r.side, f"{r.rr:.2f}", f"{r.risk_pct * 100:.2f}",
            f"{r.cost_in_r:.2f}" if np.isfinite(r.cost_in_r) else "inf",
            f"[{be_colour}]{be * 100:.1f}%[/]", f"{r.quality_score:.3f}",
            (r.sector or "-")[:14],
            _mcap(r.market_cap, r.symbol),
            "-" if r.pe_ttm is None else f"{r.pe_ttm:.1f}",
            "-" if r.roe is None else f"{r.roe * 100:.1f}%",
            d2e, f"[{fcolour}]{fstat}[/]",
        )
    console.print(t)

    best = b.rows[0]
    console.print(f"\n[bold]Top of the list: {best.symbol} {best.side}[/bold]  "
                  f"entry ~{best.last:,.4g}  stop {best.stop:,.4g} "
                  f"({best.risk_pct * 100:.2f}%)"
                  + (f"  target {best.target:,.4g}" if best.target else ""))
    for r_ in best.reasons:
        console.print(f"  [green]+[/green] {r_}")
    for w in best.warnings:
        console.print(f"  [yellow]![/yellow] {w}")
    if best.fundamental_detail:
        console.print(f"  [dim]fundamentals: {best.fundamental_detail}[/dim]")
    console.print(f"  [dim]position size if taken: risk_per_trade "
                  f"{cfg.risk.risk_per_trade * 100:.2f}% of equity / stop "
                  f"{best.risk_pct * 100:.2f}% = "
                  f"{cfg.risk.risk_per_trade / best.risk_pct * 100:.1f}% of equity notional, "
                  f"capped at {cfg.risk.max_position_pct * 100:.0f}%.[/dim]")

    if b.rejected_fundamental:
        console.print(f"\n[dim]Screened out on fundamentals "
                      f"({len(b.rejected_fundamental)}):[/dim]")
        for rej in b.rejected_fundamental[:top]:
            console.print(f"  [dim]{rej['symbol']:<14} {rej['status']:<8} - "
                          f"{rej['detail']}[/dim]")
    if b.rejected_technical:
        console.print(f"\n[dim]Rejected on cost ({len(b.rejected_technical)}):[/dim]")
        for rej in b.rejected_technical[:top]:
            console.print(f"  [dim]{rej['symbol']:<14} {rej['side']:<5} R:R "
                          f"{rej['rr']:.2f}  breakeven "
                          f"{rej['breakeven_win_rate'] * 100:.0f}%  - "
                          f"{'; '.join(rej['reasons'])}[/dim]")

    if not no_journal:
        jp = append_journal(b, REPO_ROOT / cfg.output.dir)
        console.print(f"\n[dim]journaled: {jp}[/dim]")
        console.print("  [dim]a recommendation you cannot look back at is unfalsifiable. "
                      "Once enough bars have printed, score it:[/dim]")
        console.print(f"    [dim]tvlab review --market {mkt} --style {b.style}[/dim]")


def _mcap(v: Optional[float], symbol: str) -> str:
    """Market cap in the listing currency, abbreviated.

    A bare number here is unreadable (5000000000000) and a USD-normalised one would
    be a claim about an exchange rate this function has not seen, so it is scaled
    and left in the currency the scanner reported.
    """
    if v is None or not np.isfinite(v):
        return "-"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= div:
            return f"{v / div:,.1f}{suf}"
    return f"{v:,.0f}"


# ---------------------------------------------------------------------- review
@cli.command(name="review")
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--data-source", default=None,
              help="synthetic | csv | ccxt | yfinance | tvscreener | nse | yahoo_chart")
@click.option("--timeframe", default=None,
              help="replay every run at this timeframe instead of the one it recorded")
@click.option("--horizon", type=int, default=90, show_default=True,
              help="bars to give each idea before marking it to market")
@click.option("--entry-tolerance", type=float, default=0.005, show_default=True,
              metavar="FRACTION",
              help="how far the fetched bars may miss the journaled entry price "
                   "before the idea is unscorable rather than scored")
@click.option("--days", type=int, default=None,
              help="review only the most recent N journal runs")
@click.option("--top", type=int, default=40, show_default=True,
              help="how many scored rows to print")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="emit the summary as JSON instead of a table")
def review_journal(market, style, config, set_pairs, data_source, timeframe,
                   horizon, entry_tolerance, days, top, as_json) -> None:
    """Score the journal: what the daily shortlists actually did.

    ``tvlab daily`` appends every shortlist to ``output/daily/journal.jsonl``. This
    command replays each of those suggestions against the bars that followed it and
    reports what happened, in R - the same unit the suggestion was quoted in.

    Each idea becomes one of five things, and the distinction is the point:

    \b
    take_profit   the target was reached
    stop_loss     the stop was reached
    time_exit     the horizon elapsed with neither level touched, marked to market
    open          not enough bars yet - still running, counted as neither win nor loss
    unscorable    no prices, no stop, or the prices fetched are not the ones this idea
                  was based on

    Counting an idea that is still running as a loss, or one that could not be priced
    as a failure, is how a review turns into a number that flatters or damns the
    strategy for reasons that have nothing to do with it.

    \b
    What the numbers mean
    ---------------------
    ``win rate`` is reported against the breakeven implied by the *realised* payoff,
    not the one quoted when the idea was suggested. A shortlist promising 3:1 needs a
    28% win rate; the same shortlist whose winners time-exit at +0.5R needs 55%. If
    winners stopped paying what was promised, the summary says so explicitly rather
    than passing a losing book.

    ``expectancy`` is the bottom line: mean R per suggestion after costs. It assumes
    nothing about the shape of the payoff, so it stays meaningful when the win rate
    does not.

    \b
    On synthetic data
    -----------------
    The default source generates a fixed 4000-bar window ending years in the past.
    Every suggestion is made on its final bar, so there is no future to replay and
    every idea comes back ``open`` or ``unscorable``. That is the honest answer: a
    generator with no bars after today cannot be reviewed, and a win rate computed
    against it would measure the generator, not a market. Run this against a live or
    CSV source that has actually accumulated history since you started journaling.

    \b
    Exit codes
    ----------
    0  the journal was read and scored, including when nothing could be scored
    2  no prices could be fetched for any symbol - a data failure, not a verdict
    """
    from .data.store import DataStore
    from .screener import SCAN_BARS
    from .screener.daily import journal_path, read_journal
    from .screener.review import review as score_journal, store_fetcher

    cfg = _cfg(market, style, config, set_pairs, data_source)
    out_dir = REPO_ROOT / cfg.output.dir
    jp = journal_path(out_dir)
    # `cfg.style` is None unless a style was named, and printing a suggested command
    # containing "--style None" hands the user something they cannot paste. Same
    # convention as `daily.py` and the other commands.
    style_name = cfg.style or style or "default"

    entries = list(reversed(read_journal(out_dir, limit=days or 10_000_000)))
    if not entries:
        console.print(f"\n[bold]Nothing to review yet.[/bold]")
        console.print(f"  [dim]no journal at {jp}[/dim]")
        console.print("\n  Run the daily briefing first - it is what writes the journal:")
        console.print(f"    [dim]tvlab daily --market {cfg.universe.market} "
                      f"--style {style_name}[/dim]")
        console.print("  [dim]Then come back after enough bars have printed for the ideas "
                      "to have resolved. A review run the same morning it was suggested "
                      "has nothing to say.[/dim]")
        return

    # Each run carries the timeframe it was made under. Honouring it matters because
    # the journal accumulates across config changes, and silently changing the
    # timeframe would change what "90 bars" means.
    for run in entries:
        if timeframe:
            run["timeframe"] = timeframe
        elif not run.get("timeframe"):
            run["timeframe"] = cfg.timeframe.execution

    store = DataStore(cfg)
    fetch_errors: List[str] = []
    no_future: Dict[str, str] = {}

    # The anchoring lives in screener.review.store_fetcher rather than here, because
    # GET /api/review needs identical logic and two copies of price anchoring is how a
    # terminal and a browser end up reporting different outcomes for the same journal.
    fetch = store_fetcher(store, scan_bars=SCAN_BARS, errors=fetch_errors,
                          no_future=no_future)

    summary = score_journal(entries, fetch, horizon_bars=horizon,
                            provenance=cfg.data.source,
                            entry_tolerance=entry_tolerance,
                            as_of=datetime.now(timezone.utc).date().isoformat())

    if as_json:
        console.print_json(json.dumps(summary.to_dict(), default=str))
        return

    n_runs = len(entries)
    console.print(f"\n[bold]Journal review[/bold] - {summary.as_of}  "
                  f"[dim]{cfg.universe.market} / {style_name} / source "
                  f"{cfg.data.source}[/dim]")
    console.print(f"  {n_runs} run(s) since {entries[0].get('as_of')}  ·  "
                  f"{summary.n_suggestions} suggestion(s)  ·  horizon {horizon} bars  ·  "
                  f"entry tolerance {entry_tolerance * 100:.2f}%")
    console.print(f"  [bold]{summary.coverage}[/bold]")

    if cfg.data.source == "synthetic":
        # Two different facts, and only the second is always true. Whether the
        # generated window happens to contain bars after a given suggestion depends on
        # the date the suggestion was journaled under, so asserting "there are no bars
        # after any suggestion" would be a claim this code has not checked - the same
        # class of unbacked assertion the rest of this repo keeps retracting.
        nothing_scored = summary.n_scored == 0
        if nothing_scored:
            console.print("\n  [yellow]Synthetic source: the generated window has no bars "
                          "after these suggestions, so nothing could be scored. A "
                          "generator that ends before the journal does cannot be "
                          "reviewed.[/yellow]")
            if no_future:
                sym, why = next(iter(no_future.items()))
                console.print(f"  [dim]e.g. {sym}: {why}[/dim]")
        console.print("  [yellow]Synthetic source: any win rate computed here measures "
                      "the data generator, not a market. The bars are reproducible from "
                      "a seed and contain no information about real order flow.[/yellow]")

    if fetch_errors:
        console.print(f"\n  [red]{len(fetch_errors)} fetch failure(s) - these symbols "
                      f"are unscorable, not losing:[/red]")
        for e in fetch_errors[:5]:
            console.print(f"    [red]{e}[/red]")
        if len(fetch_errors) > 5:
            console.print(f"    [dim]... and {len(fetch_errors) - 5} more[/dim]")

    scored = [r for r in summary.rows if r.scored]
    if scored:
        t = Table(title=f"scored suggestions - {len(scored)} of {summary.n_suggestions}")
        for col, just in (("date", "left"), ("symbol", "left"), ("side", "left"),
                          ("entry", "right"), ("exit", "right"), ("outcome", "left"),
                          ("R", "right"), ("net R", "right"), ("bars", "right"),
                          ("max adv", "right"), ("max fav", "right")):
            t.add_column(col, justify=just)
        for r in scored[:top]:
            colour = "green" if r.is_win else "red"
            t.add_row(
                r.as_of, r.symbol, r.side,
                f"{r.entry:,.4g}", f"{r.exit_price:,.4g}",
                f"[{colour}]{r.outcome}[/]",
                f"[{colour}]{r.r_multiple:+.2f}[/]",
                f"[{colour}]{r.net_r:+.2f}[/]",
                str(r.bars_observed),
                f"{r.max_adverse_r:+.2f}", f"{r.max_favourable_r:+.2f}",
            )
        console.print(t)
        if len(scored) > top:
            console.print(f"  [dim]... and {len(scored) - top} more scored rows[/dim]")

    unscored = [r for r in summary.rows if not r.scored and r.outcome != "open"]
    if unscored:
        console.print(f"\n[dim]Unscorable ({len(unscored)}) - excluded from every rate "
                      f"below:[/dim]")
        for r in unscored[:top]:
            console.print(f"  [dim]{r.as_of}  {r.symbol:<14} {r.note}[/dim]")

    if summary.n_open:
        console.print(f"\n[dim]Still open ({summary.n_open}) - not enough bars have "
                      f"printed to judge these yet:[/dim]")
        for r in [r for r in summary.rows if r.outcome == "open"][:top]:
            console.print(f"  [dim]{r.as_of}  {r.symbol:<14} {r.note}[/dim]")

    # ---- the verdict, and the reasons not to read too much into it
    if summary.n_scored == 0:
        console.print("\n[bold yellow]Nothing could be scored.[/bold yellow]")
        console.print("  [dim]No win rate, no expectancy - because there is no outcome "
                      "to measure yet. This is not a verdict on the strategy.[/dim]")
    else:
        console.print(f"\n[bold]Outcome over {summary.n_scored} scored suggestion(s)"
                      f"[/bold]  [dim](of {summary.n_suggestions} journaled)[/dim]")
        for k in ("take_profit", "stop_loss", "time_exit"):
            if summary.by_outcome.get(k):
                console.print(f"  {k:<12} {summary.by_outcome[k]}")

        wr = summary.win_rate or 0.0
        be = summary.realised_breakeven
        if summary.n_wins == 0:
            # A required win rate above 100% is arithmetically what a book with no
            # winners implies, but printing it as a threshold invites the reader to
            # think a slightly better win rate would have passed. Say the fact instead.
            console.print(f"  win rate     [red]0.0%[/] [dim](n={summary.n_scored})"
                          f"[/dim]  [red]not one suggestion reached its target "
                          f"or closed in profit[/red]")
        elif summary.clears_breakeven is None:
            console.print(f"  win rate     [yellow]{wr * 100:.1f}%[/yellow] "
                          f"[dim]- no comparison possible[/dim]")
        else:
            c = "green" if summary.clears_breakeven else "red"
            verb = "clears" if summary.clears_breakeven else "does NOT clear"
            console.print(f"  win rate     [{c}]{wr * 100:.1f}%[/] "
                          f"[dim](n={summary.n_scored})[/]  [{c}]{verb}[/] the "
                          f"{be * 100:.1f}% this payoff requires")

        exp_c = "green" if (summary.expectancy_r or 0) > 0 else "red"
        console.print(f"  expectancy   [{exp_c}]{summary.expectancy_r:+.3f} R[/] per "
                      f"suggestion, after costs")
        console.print(f"  payoff       promised "
                      f"{(summary.quoted_rr or 0):.2f}R:1, realised "
                      f"{(summary.realised_payoff_r or 0):.2f}R:1")
        if summary.payoff_diverged:
            # Divergence has two causes with different meanings, and only one of them
            # is about winners underpaying.
            if summary.n_wins == 0:
                console.print("  [yellow]! there were no winners at all, so the quoted "
                              "breakeven rate has nothing to apply to[/yellow]")
            else:
                console.print("  [yellow]! winners did not pay what was promised, so the "
                              "quoted breakeven rate does not apply to this book[/yellow]")

        console.print(f"\n  [dim]n={summary.n_scored}. A win rate on a handful of "
                      f"suggestions is a description of those suggestions, not an "
                      f"estimate of an edge: the default study behind this ranking "
                      f"measured 31 trades before it was declared NOT VALIDATED. "
                      f"Score this again in a few months.[/dim]")

    if summary.provenance == "synthetic":
        console.print("  [dim]provenance: SYNTHETIC - any number above measures the "
                      "data generator, not a market.[/dim]")

    # A total fetch failure is a data outage and gets the same exit code `daily` uses,
    # because the remedy is the opposite of "wait and see".
    if summary.n_suggestions and summary.n_scored == 0 and summary.n_open == 0 \
            and fetch_errors and not scored:
        raise SystemExit(2)


# ------------------------------------------------------------------------ scan
@cli.command()
@click.option("--market", default=None)
@click.option("--style", default=None)
@click.option("--config", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE")
@click.option("--symbols", default=None)
@click.option("--data-source", default=None,
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                  "nse", "yahoo_chart"]))
@click.option("--screener", is_flag=True,
              help="build the universe from the TradingView screener first")
@click.option("--timeframe", default=None)
@click.option("--top", type=int, default=20)
@click.option("--trendlines", is_flag=True, default=False,
              help="also fit diagonal support/resistance (~27 ms per symbol)")
def scan(market, style, config, set_pairs, symbols, data_source, screener, timeframe, top,
         trendlines) -> None:
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
    rows, errors = scan_universe(store, cfg, syms[:top * 2], strat=strat, timeframe=tf,
                                 with_trendlines=trendlines)

    t = Table(title=f"Scan - {cfg.strategy.name} on {tf} ({cfg.universe.market})")
    cols = ["symbol", "last", "chg %", "trend", "ADX", "RSI", "ATR %",
            "sweep", "sweep age", "FVG/OB", "signal", "stop", "target", "R:R"]
    if trendlines:
        cols += ["support %", "resist %", "channel %"]
    for col in cols:
        t.add_column(col, justify="left" if col == "symbol" else "right")
    #: padding for the short NO DATA / FAILED rows. Derived from the column count
    #: rather than hardcoded: it was a literal 11, which silently misaligns the
    #: table the moment a column is added.
    pad = len(cols) - 3

    _trend_markup = {1: "[green]UP[/green]", -1: "[red]DOWN[/red]", 0: "flat"}
    _signal_markup = {1: "[green]LONG[/green]", -1: "[red]SHORT[/red]", 0: "-"}

    for r in rows:
        if r.bars < 2:
            t.add_row(r.symbol, "NO DATA", (r.warnings or [""])[0][:70], *[""] * pad)
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
            *([_pct(r.support_pct), _pct(r.resistance_pct), _pct(r.channel_pct)]
              if trendlines else []),
        )
    for sym, err in errors.items():
        t.add_row(sym, "FAILED", err[:70], *[""] * pad)

    console.print(t)

    s = summarize(rows, errors)
    console.print(f"\n[bold]{s['live']} live signal(s)[/bold] of {s['scanned']} symbols scanned"
                  + (f" ({s['failed']} failed)" if s["failed"] else "") + ".")
    for r in rank(rows, fee_bps=cfg.costs.fee_bps, slippage_bps=cfg.costs.slippage_bps,
                  min_stop_distance_pct=cfg.risk.min_stop_distance_pct):
        side = "LONG " if r.signal > 0 else "SHORT"
        rr = f"  R:R {r.reward_risk:.2f}" if r.reward_risk is not None else ""
        console.print(f"  {side} {r.symbol}{rr}")
    if s["scanned"] == 0 and s["failed"]:
        # Same trap as `suggest`: "nothing fired" is a claim about the market, and
        # it is only licensed if the market was observed. With every symbol failing
        # there is no observation to report, and the reassuring line below would
        # turn a data outage into a quiet-market verdict.
        console.print("[bold red]  Nothing was scanned - this is a data failure, not a "
                      "market verdict.[/bold red]")
        console.print(f"[red]  All {s['failed']} symbol(s) failed to load; see the FAILED "
                      f"rows above.[/red]")
        console.print(f"  [yellow]Retry once bars load:[/yellow] [dim]tvlab data "
                      f"--market {cfg.universe.market} --data-source {cfg.data.source} "
                      f"--limit 50[/dim]")
        raise SystemExit(2)
    if not s["live"]:
        console.print("  No symbol satisfies bias + sweep + confirmation right now. "
                      "That is the normal state - this setup should not fire often."
                      + (f" (based on {s['scanned']} of {s['scanned'] + s['failed']} "
                         f"symbols - the rest failed to load)" if s["failed"] else ""))
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
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                  "nse", "yahoo_chart"]))
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
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                  "nse", "yahoo_chart"]))
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
              type=click.Choice(["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                  "nse", "yahoo_chart"]))
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


def _catch_data_error(fn):
    """Turn a DataError into a readable one-line failure.

    NSE serves 403s, HTML interstitials and changed field names as a matter of
    course, and Yahoo rate-limits. Those are expected operating conditions for
    this command group, so they should read as a message with a hint, not as a
    traceback that buries the cause.
    """
    import functools

    from .data.base import DataError

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except DataError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print("[dim]NSE and Yahoo are undocumented public endpoints: they rate-limit, "
                          "serve an HTML interstitial when they want a browser cookie, and change "
                          "field names without notice. Retry, run from a residential connection, "
                          "or fall back to --data-source synthetic.[/dim]")
            sys.exit(1)

    return wrapper


def _nse_src(cfg: AppConfig, index: Optional[str] = None):
    """Build an :class:`~tvlab.data.nse.NseSource`, failing with guidance."""
    from .data.nse import NseSource

    return NseSource(index=index or cfg.universe.nse_index, market=cfg.universe.market)


@cli.group()
def nse() -> None:
    """India: NSE session clock, index breadth, liquidity and universe.

    Reads NSE's own public JSON APIs plus Yahoo's public chart endpoint. No
    vendor, no API key and no ``yfinance`` install - only the standard library.
    The machine running this must be able to reach ``nseindia.com`` and
    ``query1.finance.yahoo.com``.

    These commands describe the market as it is. None of them is a signal, a
    tip or a prediction, and nothing here has been validated as profitable:
    ``tvlab study --market indian_equities`` is the command that decides whether
    a rule set deserves money, and today it says NOT VALIDATED.
    """


@nse.command("session")
@click.option("--market", default="indian_equities", show_default=True)
@click.option("--style", default="intraday", show_default=True)
@click.option("--config", default=None, type=click.Path(exists=True))
@click.option("--index", default=None, help="NSE index to read the universe from")
@_catch_data_error
def nse_session(market, style, config, index) -> None:
    """Is the NSE cash market open right now, and for how much longer?

    Answers from NSE's published holiday calendar, not just the weekday - so a
    market holiday is reported as closed instead of looking like a quiet session.
    """
    from .data.nse import IST, SESSION_END_MIN, SESSION_START_MIN

    cfg = _cfg(market, style, config, ())
    src = _nse_src(cfg, index)
    now = datetime.now(IST)
    open_now = src.is_session_open(now)
    left = src.minutes_to_close(now)

    console.print(f"[bold]NSE cash session[/bold]  {now:%A %d %b %Y %H:%M:%S} IST")
    console.print(f"  regular hours   {SESSION_START_MIN // 60:02d}:{SESSION_START_MIN % 60:02d}"
                  f" - {SESSION_END_MIN // 60:02d}:{SESSION_END_MIN % 60:02d} IST")
    console.print(f"  market open     [bold {'green' if open_now else 'red'}]{'YES' if open_now else 'NO'}[/]")
    if open_now and left is not None:
        note = ""
        if left <= 15:
            note = "  [red]- closing auction territory; a same-day trade here has almost no time to work[/]"
        elif left <= 60:
            note = "  [yellow]- under an hour left[/]"
        console.print(f"  minutes to close  {left}{note}")
    elif now.weekday() >= 5:
        console.print("  [dim]weekend[/dim]")
    else:
        try:
            hol = src.trading_holidays()
            if now.date() in hol:
                console.print(f"  [yellow]market holiday[/yellow]")
        except Exception:  # noqa: BLE001
            pass


@nse.command("breadth")
@click.option("--market", default="indian_equities", show_default=True)
@click.option("--style", default="intraday", show_default=True)
@click.option("--config", default=None, type=click.Path(exists=True))
@click.option("--index", default=None)
@_catch_data_error
def nse_breadth(market, style, config, index) -> None:
    """Every NSE index: level, change, advances/declines and PE.

    Breadth is the read that matters most for an intraday long-only mandate.
    ``allow_short`` is false for Indian equities, so on a day when declines
    outnumber advances there is no trade to take - and that is knowable before
    the open rather than after the loss.
    """
    cfg = _cfg(market, style, config, ())
    df = _nse_src(cfg, index).indices()

    t = Table(title="NSE index breadth", show_lines=False)
    for col, justify in (("index", "left"), ("last", "right"), ("pct_change", "right"),
                         ("advances", "right"), ("declines", "right"), ("breadth", "right"),
                         ("pe", "right")):
        t.add_column(col, justify=justify)
    for _, r in df.iterrows():
        b = r.get("breadth")
        colour = "green" if (b or 0) > 0.1 else ("red" if (b or 0) < -0.1 else "yellow")
        t.add_row(str(r["index"]), _cell(r["last"]), _cell(r["pct_change"]),
                  str(int(r["advances"])), str(int(r["declines"])),
                  f"[{colour}]{_cell(b)}[/]", _cell(r["pe"]))
    console.print(t)

    nifty = df[df["index"].str.upper().str.startswith("NIFTY 50")]
    if not nifty.empty:
        b = nifty.iloc[0].get("breadth")
        if b is not None and not pd.isna(b):
            if b > 0.2:
                read = "advances dominate - a long-only intraday mandate has room to work"
            elif b < -0.2:
                read = "[red]declines dominate - with allow_short=false there is no intraday trade here[/red]"
            else:
                read = "mixed breadth - index moves are being driven by a few weights, not the market"
            console.print(f"\nNIFTY 50 breadth {b:+.2f}: {read}")


@nse.command("liquidity")
@click.option("--market", default="indian_equities", show_default=True)
@click.option("--style", default="intraday", show_default=True)
@click.option("--config", default=None, type=click.Path(exists=True))
@click.option("--index", default="NIFTY 500", show_default=True,
              help='NSE index, or "SECURITIES IN F&O"')
@click.option("--top", default=25, show_default=True)
@_catch_data_error
def nse_liquidity(market, style, config, index, top) -> None:
    """The most tradeable names right now, ranked by traded value.

    Ranked by rupees traded, not by share volume. A ₹21 stock with 90 million
    shares traded is a smaller position than a ₹2,950 stock with 8 million -
    ranking on volume alone sends you into the name you cannot exit.
    """
    cfg = _cfg(market, style, config, ())
    snap = _nse_src(cfg, index).snapshot(index)
    if snap.empty:
        console.print("[yellow]NSE returned no EQ-series rows for that index[/yellow]")
        return

    snap = snap.head(top)
    t = Table(title=f"NSE liquidity - top {len(snap)} of {index} by traded value")
    for col, justify in (("symbol", "left"), ("ltp", "right"), ("pct_change", "right"),
                         ("day_low", "right"), ("day_high", "right"),
                         ("traded_value_cr", "right"), ("range_pct", "right")):
        t.add_column(col, justify=justify)
    for _, r in snap.iterrows():
        lo, hi, ltp = r["low"], r["high"], r["ltp"]
        rng = (hi - lo) / lo * 100 if lo and hi and lo > 0 else float("nan")
        pct = r["pct_change"]
        colour = "green" if (pct or 0) > 0 else ("red" if (pct or 0) < 0 else "white")
        t.add_row(str(r["symbol"]), _cell(ltp), f"[{colour}]{_cell(pct)}[/]",
                  _cell(lo), _cell(hi), f"{(r['traded_value'] or 0) / 1e7:,.1f}", _cell(rng))
    console.print(t)
    console.print("\n[dim]traded_value_cr is in crore rupees. range_pct is today's high-low as a "
                  "percent of the low - the room a same-day trade actually has.[/dim]")


@nse.command("universe")
@click.option("--market", default="indian_equities", show_default=True)
@click.option("--style", default="intraday", show_default=True)
@click.option("--config", default=None, type=click.Path(exists=True))
@click.option("--top", default=0, help="keep only the N most liquid names (0 = all)")
@_catch_data_error
def nse_universe(market, style, config, top) -> None:
    """List the live NSE universe: Nifty 500 members plus every F&O security.

    This replaces a hand-typed symbol list with whatever NSE publishes today.
    Pass ``--top N`` to rank by traded value first, which is what you want
    before feeding names into ``tvlab scan``.
    """
    cfg = _cfg(market, style, config, ())
    src = _nse_src(cfg)
    syms = src.list_symbols()
    console.print(f"{len(syms)} symbols (Nifty 500 CSV + NSE F&O list)")

    if top:
        from .data.nse import top_by_liquidity

        try:
            syms = top_by_liquidity(syms, src.snapshot("NIFTY 500"), top)
            console.print(f"narrowed to the {len(syms)} most liquid by traded value")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]could not rank by liquidity ({exc}); showing the first {top}[/yellow]")
            syms = syms[:top]

    console.print(", ".join(syms))
    console.print("\n[dim]To use these: tvlab scan --market indian_equities --style intraday "
                  "--data-source nse --symbols " + ",".join(syms[:6]) + ",...[/dim]")


@nse.command("holidays")
@click.option("--market", default="indian_equities", show_default=True)
@click.option("--style", default="intraday", show_default=True)
@click.option("--config", default=None, type=click.Path(exists=True))
@click.option("--index", default=None)
@click.option("--days", default=180, show_default=True, help="look ahead this many days")
@_catch_data_error
def nse_holidays(market, style, config, index, days) -> None:
    """Upcoming NSE cash-market trading holidays.

    Matters for backtests as much as for planning: a bar series with holiday
    gaps will show ATR and ADX artefacts at the seams if the calendar is ignored.
    """
    from .data.nse import IST

    cfg = _cfg(market, style, config, ())
    hol = sorted(_nse_src(cfg, index).trading_holidays())
    today = datetime.now(IST).date()
    upcoming = [d for d in hol if today <= d <= today + timedelta(days=days)]
    if not upcoming:
        console.print(f"[dim]no NSE trading holidays in the next {days} days[/dim]")
        return
    t = Table(title=f"NSE trading holidays - next {days} days")
    t.add_column("date", justify="left")
    t.add_column("weekday", justify="left")
    for d in upcoming:
        t.add_row(d.isoformat(), d.strftime("%A"))
    console.print(t)


def _pct(v: Optional[float]) -> str:
    """Signed percentage, or '-' when the value is missing.

    Missing is common and meaningful here: a symbol can have a fitted support and
    no resistance above price, and rendering that as 0.00 would claim the market
    is sitting exactly on a line it is nowhere near.
    """
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:+.2f}"


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
