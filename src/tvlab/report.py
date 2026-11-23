"""Report generation: markdown, JSON and matplotlib charts.

The markdown report is deliberately opinionated. It leads with the verdict and
the validity checks rather than the profit number, because a profit figure from
an unvalidated backtest is not information.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .backtest.metrics import drawdown_series, format_metrics, monthly_table
from .config import REPO_ROOT, AppConfig

log = logging.getLogger(__name__)

UP = "#16a34a"
DOWN = "#dc2626"
NEUTRAL = "#64748b"


# ---------------------------------------------------------------------- charts
def plot_equity(equity: pd.Series, benchmark: Optional[pd.Series] = None,
                title: str = "Equity", path: Optional[Path] = None, dpi: int = 130) -> Optional[Path]:
    """Equity curve with its drawdown underneath - the two panels that matter."""
    if equity is None or len(equity) < 2:
        return None
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax1.plot(equity.index, equity.to_numpy(), color="#2563eb", lw=1.3, label="strategy")
    if benchmark is not None and len(benchmark):
        b = benchmark.reindex(equity.index).ffill()
        ax1.plot(b.index, b.to_numpy(), color=NEUTRAL, lw=1.0, ls="--", alpha=0.85,
                 label="buy & hold")
    ax1.set_ylabel("Equity")
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    dd = drawdown_series(equity) * 100.0
    ax2.fill_between(dd.index, dd.to_numpy(), 0, color=DOWN, alpha=0.35, lw=0)
    ax2.plot(dd.index, dd.to_numpy(), color=DOWN, lw=0.8)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_trades(equity: pd.Series, trades: pd.DataFrame, path: Optional[Path] = None,
                dpi: int = 130) -> Optional[Path]:
    """Cumulative P&L per trade plus the R-multiple distribution."""
    if trades is None or trades.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    cum = trades["pnl"].cumsum()
    colours = [UP if p > 0 else DOWN for p in trades["pnl"]]
    ax1.bar(range(len(trades)), trades["pnl"], color=colours, width=0.8, alpha=0.7)
    ax1.plot(range(len(trades)), cum, color="#1e293b", lw=1.4, label="cumulative P&L")
    ax1.axhline(0, color=NEUTRAL, lw=0.7)
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("P&L")
    ax1.set_title("Per-trade P&L and cumulative", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25, lw=0.5)

    r = trades["r_multiple"].dropna().to_numpy()
    if len(r):
        r = r[np.isfinite(r)]
        ax2.hist(r, bins=min(40, max(8, len(r) // 3)), color="#2563eb", alpha=0.75)
        ax2.axvline(0, color=NEUTRAL, lw=0.8)
        ax2.axvline(np.mean(r), color=UP, lw=1.6, ls="--", label=f"mean R = {np.mean(r):.2f}")
        ax2.set_xlabel("R multiple")
        ax2.set_ylabel("Trades")
        ax2.set_title("R-multiple distribution", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_chart_with_trades(ohlcv: pd.DataFrame, trades: pd.DataFrame,
                           title: str = "", path: Optional[Path] = None,
                           dpi: int = 130, max_bars: int = 600) -> Optional[Path]:
    """Candlestick-style chart with entry/exit markers (matplotlib fallback).

    The interactive version lives in the dashboard (TradingView lightweight-charts);
    this is the static image that goes into the markdown report.
    """
    if ohlcv is None or len(ohlcv) < 5:
        return None
    df = ohlcv.iloc[-max_bars:]
    fig, ax = plt.subplots(figsize=(14, 6.5))

    x = np.arange(len(df))
    w = 0.62
    up = df["close"] >= df["open"]
    ax.bar(x[up.to_numpy()], (df["high"] - df["low"])[up], width=0.12, color=UP, alpha=0.9)
    ax.bar(x[~up.to_numpy()], (df["high"] - df["low"])[~up], width=0.12, color=DOWN, alpha=0.9)
    ax.bar(x[up.to_numpy()], (df["close"] - df["open"])[up], width=w, bottom=df["open"][up],
           color=UP, alpha=0.85)
    ax.bar(x[~up.to_numpy()], (df["open"] - df["close"])[~up], width=w, bottom=df["close"][~up],
           color=DOWN, alpha=0.85)

    if trades is not None and not trades.empty:
        idx = {ts: i for i, ts in enumerate(df.index)}
        for _, t in trades.iterrows():
            ei, xi = t.get("entry_time"), t.get("exit_time")
            if ei in idx:
                ax.scatter(idx[ei], t["entry_price"], marker="^" if t["side"] > 0 else "v",
                           s=70, color="#2563eb", zorder=5, edgecolors="white", linewidths=0.6)
            if xi in idx:
                # no edgecolors: 'x' is an unfilled marker and matplotlib warns
                ax.scatter(idx[xi], t["exit_price"], marker="x", s=55,
                           color=UP if t["pnl"] > 0 else DOWN, zorder=5, linewidths=1.4)

    step = max(1, len(df) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%y-%m-%d") for d in df.index[::step]], rotation=30, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_walk_forward(wf, path: Optional[Path] = None, dpi: int = 130) -> Optional[Path]:
    """Stitched out-of-sample equity with the window boundaries marked."""
    if wf is None or wf.oos_equity is None or len(wf.oos_equity) < 2:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.35})
    eq = wf.oos_equity
    ax1.plot(eq.index, eq.to_numpy(), color="#2563eb", lw=1.4, label="stitched OOS")
    for w in wf.windows:
        ax1.axvline(w.test_slice[0], color=NEUTRAL, lw=0.7, ls=":", alpha=0.8)
    ax1.axhline(float(eq.iloc[0]), color=NEUTRAL, lw=0.7)
    ax1.set_ylabel("Equity (out of sample)")
    ax1.set_title("Walk-forward out-of-sample equity", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25, lw=0.5)

    labels = [f"W{w.index}" for w in wf.windows]
    is_r = [(w.train_metrics.get("total_return_pct") or 0) for w in wf.windows]
    oos_r = [(w.test_metrics.get("total_return_pct") or 0) for w in wf.windows]
    xx = np.arange(len(labels))
    ax2.bar(xx - 0.2, is_r, width=0.4, color=NEUTRAL, alpha=0.75, label="in-sample (train)")
    ax2.bar(xx + 0.2, oos_r, width=0.4, color="#2563eb", alpha=0.85, label="out-of-sample (test)")
    ax2.axhline(0, color="#1e293b", lw=0.7)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Return %")
    ax2.set_title("In-sample vs out-of-sample per window", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, lw=0.5)

    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_monte_carlo(fan: pd.DataFrame, path: Optional[Path] = None,
                     dpi: int = 130) -> Optional[Path]:
    """Bootstrap fan chart: the range of futures consistent with these bar returns."""
    if fan is None or fan.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(fan.index, fan["p5"], fan["p95"], color="#2563eb", alpha=0.12,
                    label="5-95%")
    ax.fill_between(fan.index, fan["p25"], fan["p75"], color="#2563eb", alpha=0.22,
                    label="25-75%")
    ax.plot(fan.index, fan["p50"], color="#2563eb", lw=1.1, label="median")
    if "actual" in fan:
        ax.plot(fan.index, fan["actual"], color="#1e293b", lw=1.4, label="actual path")
    ax.set_ylabel("Equity")
    ax.set_title("Monte Carlo fan (block bootstrap on bar returns)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.autofmt_xdate()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return None


def plot_cost_sensitivity(cs: pd.DataFrame, path: Optional[Path] = None,
                          dpi: int = 130) -> Optional[Path]:
    """Return and Sharpe vs transaction-cost multiple."""
    if cs is None or cs.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = cs["cost_multiplier"].to_numpy()
    bars = ax1.bar(x, cs["return_pct"], width=0.55,
                   color=[UP if v > 0 else DOWN for v in cs["return_pct"]], alpha=0.8)
    ax1.axhline(0, color="#1e293b", lw=0.8)
    ax1.set_xlabel("Multiple of configured transaction cost")
    ax1.set_ylabel("Total return %")
    ax1.set_title("Cost sensitivity - where does the edge die?", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.25, lw=0.5, axis="y")
    for b, v in zip(bars, cs["return_pct"]):
        ax1.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, cs["sharpe"], color="#1e293b", marker="o", lw=1.4, ms=4, label="Sharpe")
    ax2.set_ylabel("Sharpe")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.close(fig)
    return None


# -------------------------------------------------------------------- markdown
def _df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}", max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    d = df.head(max_rows).copy()
    try:
        return d.to_markdown(floatfmt=floatfmt) + "\n"
    except Exception:  # noqa: BLE001 - tabulate is optional
        return "```\n" + d.to_string() + "\n```\n"


def _verdict(study, wf_summary: Optional[Dict[str, Any]],
             nulls: Optional[Dict[str, Any]], mc_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble an honest go/no-go verdict from the validity checks."""
    m = study.portfolio_metrics or {}
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: Optional[bool], detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    n_trades = int(m.get("n_trades") or 0)
    add("Enough trades to measure",
        n_trades >= 30,
        f"{n_trades} trades" + ("" if n_trades >= 30 else
                                " - below 30 the win rate and profit factor have huge "
                                "confidence intervals; treat every ratio as noise"))

    dsr = (m.get("deflated_sharpe") or {})
    add("Deflated Sharpe (multiple testing)",
        (dsr.get("dsr") or 0) >= 0.95,
        f"DSR={dsr.get('dsr', float('nan')):.3f} ({dsr.get('verdict', '-')}) "
        f"over {dsr.get('n_trials', 1)} trials")

    mdd = float(m.get("max_drawdown_pct") or 0)
    limit = study.config.risk.max_drawdown_pct * 100
    add("Drawdown within mandate",
        mdd <= limit,
        f"max DD {mdd:.2f}% vs limit {limit:.1f}%")

    pf = float(m.get("profit_factor") or float("nan"))
    add("Profit factor > 1 after costs",
        np.isfinite(pf) and pf > 1.0,
        f"PF={pf:.2f}" if np.isfinite(pf) else "PF undefined (no losing trades or no trades)")

    if wf_summary:
        hr = wf_summary.get("window_hit_rate", float("nan"))
        add("Walk-forward windows profitable",
            np.isfinite(hr) and hr >= 0.6,
            f"{wf_summary.get('windows_profitable', 0)}/{wf_summary.get('n_windows', 0)} "
            f"OOS windows profitable (hit rate {hr:.0%})" if np.isfinite(hr) else "n/a")
        eff = wf_summary.get("mean_oos_efficiency", float("nan"))
        add("OOS retains IS performance",
            np.isfinite(eff) and eff >= 0.3,
            f"mean OOS/IS objective ratio = {eff:.2f}" if np.isfinite(eff) else "n/a")

    if nulls:
        sh = nulls.get("shuffled_returns")
        if sh:
            pv = sh.get("p_value", float("nan"))
            add("Beats shuffled-return null",
                np.isfinite(pv) and pv < 0.05,
                f"p={pv:.3f} - beat {sh.get('better_than_null_pct', float('nan')):.0f}% "
                f"of {sh.get('n_sims', 0)} permutations" if np.isfinite(pv) else "not evaluated")

    if mc_summary:
        p10 = (mc_summary.get("ruin_probabilities") or {}).get("dd_exceeds_10pct")
        if p10 is not None:
            add("Monte Carlo DD-breach risk acceptable",
                p10 <= 0.35,
                f"P(max DD >= 10%) = {p10:.1%} across {mc_summary.get('n_sims', 0)} "
                "resampled trade sequences")

    hard_fail = any(c["passed"] is False and c["check"] in {
        "Beats shuffled-return null", "Deflated Sharpe (multiple testing)",
        "Walk-forward windows profitable", "Enough trades to measure",
    } for c in checks)
    n_pass = sum(1 for c in checks if c["passed"])
    n_tot = len(checks)

    if hard_fail:
        verdict = "NOT VALIDATED"
        headline = ("The strategy did not clear its validity checks. The P&L below is a "
                    "description of one historical sample, not evidence of an edge.")
    elif n_pass == n_tot:
        verdict = "PASSES ON THIS DATA"
        headline = ("All validity checks passed on this sample. That is a necessary "
                    "condition, not a sufficient one - confirm on out-of-sample real "
                    "data and in paper trading before risking capital.")
    else:
        verdict = "PARTIAL"
        headline = (f"{n_pass}/{n_tot} validity checks passed. Read the failures before "
                    "drawing any conclusion.")

    return {"verdict": verdict, "headline": headline, "checks": checks,
            "n_passed": n_pass, "n_checks": n_tot}


def build_markdown(
    study,
    wf=None,
    mc=None,
    nulls: Optional[Dict[str, Any]] = None,
    cost_sens: Optional[pd.DataFrame] = None,
    lookahead_report: Optional[Dict[str, Any]] = None,
    recovery_report: Optional[Dict[str, Any]] = None,
    cross_validation: Optional[Dict[str, Any]] = None,
    images: Optional[Dict[str, Path]] = None,
    data_provenance: str = "",
) -> str:
    """Render the full markdown report."""
    cfg = study.config
    m = study.portfolio_metrics or {}
    images = images or {}
    wf_summary = wf.summary() if wf is not None else None
    mc_summary = mc.summary() if mc is not None else None
    v = _verdict(study, wf_summary, nulls, mc_summary)

    L: List[str] = []
    A = L.append

    A(f"# Trading study: `{study.strategy_name}` on {cfg.universe.market}")
    A("")
    A(f"_Generated {study.created_at} · tvlab_")
    A("")
    A(f"## Verdict: **{v['verdict']}**")
    A("")
    A(v["headline"])
    A("")
    if data_provenance:
        A(f"> **Data provenance:** {data_provenance}")
        A("")

    A("### Validity checks")
    A("")
    A("| # | Check | Result | Detail |")
    A("|--:|:------|:------:|:-------|")
    for i, c in enumerate(v["checks"], 1):
        mark = "PASS" if c["passed"] else ("FAIL" if c["passed"] is False else "n/a")
        A(f"| {i} | {c['check']} | **{mark}** | {c['detail']} |")
    A("")
    A(f"Score: {v['n_passed']}/{v['n_checks']} checks passed.")
    A("")

    alloc = getattr(study, "allocation", None)
    if alloc:
        A("## Capital allocation")
        A("")
        A(f"Scheme `{alloc.get('scheme')}` · {alloc.get('n_symbols')} of "
          f"{len(study.symbols)} symbols funded · "
          f"{alloc.get('total_weight_pct', 0):.1f}% of capital deployed")
        A("")
        A("| Symbol | Weight | Capital | Risk if stopped out |")
        A("|:-------|-------:|--------:|--------------------:|")
        for r in alloc.get("rows", []):
            A(f"| `{r['symbol']}` | {r['weight_pct']:.2f}% | {r['capital']:,.0f} | "
              f"{r['risk_at_mandate']:,.0f} |")
        A("")
        agg = alloc.get("aggregate_risk_pct")
        if agg is not None:
            A(f"**Aggregate risk if every position stops out at once: "
              f"{alloc.get('aggregate_risk_if_all_stop_out'):,.0f} "
              f"{cfg.universe.quote_currency} ({agg:.2f}% of equity).** This is the number the "
              f"per-trade mandate does not show you: {cfg.risk.risk_per_trade:.2%} per trade "
              f"across several concurrent positions is not "
              f"{cfg.risk.risk_per_trade:.2%} of account risk.")
            A("")
        groups = [g for g in (alloc.get("groups") or []) if len(g) > 1]
        if groups:
            A("Correlated groups (treated as one bet for "
              f"`max_correlated_positions={cfg.risk.max_correlated_positions}`):")
            A("")
            for g in groups:
                A(f"- {', '.join(f'`{x}`' for x in g)}")
            A("")
        for cap in alloc.get("caps_applied") or []:
            A(f"- cap applied: {cap}")
        if alloc.get("caps_applied"):
            A("")
        unalloc = getattr(study, "unallocated", None) or []
        if unalloc:
            A(f"Excluded by the risk caps ({len(unalloc)}): "
              + ", ".join(f"`{s}`" for s in unalloc))
            A("")
            A("_Excluded symbols are still backtested and reported per-symbol below; they are "
              "simply not funded in the portfolio curve, because the mandate does not allow "
              "holding them alongside the ones that were._")
            A("")
        idle = 100.0 - float(alloc.get("total_weight_pct", 0.0))
        if idle > 1.0:
            A(f"> **{idle:.1f}% of capital is left idle.** That is the mandate working, not a "
              f"bug: with `max_correlated_positions={cfg.risk.max_correlated_positions}` and "
              f"`max_position_pct={cfg.risk.max_position_pct:.0%}`, a universe whose members "
              f"all move together cannot absorb more capital without breaking the risk limits. "
              f"To deploy it, add genuinely uncorrelated markets rather than more of the same.")
            A("")

    A("## Portfolio results")
    A("")
    A("```")
    A(format_metrics(m))
    A("```")
    A("")
    A(f"Universe: {len(study.symbols)} symbols · timeframe `{cfg.timeframe.execution}`"
      f"{f' with `{cfg.timeframe.bias}` bias' if cfg.timeframe.bias else ''}"
      f" · capital {cfg.backtest.initial_capital:,.0f} {cfg.universe.quote_currency}")
    A("")
    A(f"Risk: {cfg.risk.risk_per_trade:.2%} per trade · max DD halt "
      f"{cfg.risk.max_drawdown_pct:.0%} · daily loss limit "
      f"{cfg.risk.daily_loss_limit_pct:.1%} · max {cfg.risk.max_open_positions} positions"
      f" · shorts {'on' if cfg.risk.allow_short else 'off'}")
    A("")
    A(f"Costs: {cfg.costs.fee_bps:.1f} bp fee + {cfg.costs.slippage_bps:.1f} bp slippage "
      f"per side = {(cfg.costs.fee_bps + cfg.costs.slippage_bps) * 2:.1f} bp round trip")
    A("")
    if "equity" in images:
        A(f"![Equity curve]({images['equity']})")
        A("")
    if "trades" in images:
        A(f"![Trade distribution]({images['trades']})")
        A("")

    if study.benchmark:
        bm = study.benchmark["metrics"]
        A("### Benchmark")
        A("")
        A(f"Buy & hold {study.benchmark['symbol']}: "
          f"{bm.get('total_return_pct', float('nan')):.2f}% return, "
          f"{bm.get('max_drawdown_pct', float('nan')):.2f}% max DD, "
          f"Sharpe {bm.get('sharpe', float('nan')):.2f}")
        A("")

    A("### Per symbol")
    A("")
    A(_df_to_md(study.per_symbol.set_index("symbol"), floatfmt="{:.2f}"))
    A("")

    A("### Exit reasons")
    A("")
    eb = m.get("exit_breakdown") or {}
    if eb:
        A(_df_to_md(pd.DataFrame(eb).T[["n", "pnl", "win_rate", "avg_r"]], floatfmt="{:.2f}"))
    else:
        A("_no trades_")
    A("")
    A("### By setup tag")
    A("")
    tb = m.get("tag_breakdown") or {}
    if tb:
        A(_df_to_md(pd.DataFrame(tb).T[["n", "pnl", "win_rate", "avg_r", "expectancy"]],
                    floatfmt="{:.2f}"))
    else:
        A("_no trades_")
    A("")

    # ---------------------------------------------------------- walk-forward
    if wf is not None:
        A("## Walk-forward validation")
        A("")
        A(f"{wf.n_windows} windows · {wf.total_trials} total trials · "
          f"objective `{cfg.walk_forward.objective}` · "
          f"{'rolling' if cfg.walk_forward.rolling else 'anchored'}")
        A("")
        if wf_summary:
            A("| Metric | Value |")
            A("|:-------|------:|")
            A(f"| OOS windows profitable | {wf_summary['windows_profitable']}/{wf_summary['n_windows']} |")
            A(f"| OOS window hit rate | {wf_summary['window_hit_rate']:.0%} |")
            A(f"| Mean OOS/IS efficiency | {wf_summary['mean_oos_efficiency']:.2f} |")
            A(f"| OOS return | {wf_summary['oos_return_pct']:.2f}% |")
            A(f"| OOS max DD | {wf_summary['oos_max_dd_pct']:.2f}% |")
            A(f"| OOS Sharpe | {wf_summary['oos_sharpe']:.2f} |")
            A(f"| OOS profit factor | {wf_summary['oos_profit_factor']:.2f} |")
            A(f"| OOS trades | {wf_summary['oos_trades']} |")
            dsr = wf_summary.get('oos_deflated_sharpe')
            A(f"| OOS deflated Sharpe | {dsr:.3f} ({wf_summary.get('oos_dsr_verdict', '-')}) |"
              if dsr is not None and np.isfinite(dsr) else "| OOS deflated Sharpe | n/a |")
            A("")
        if "walk_forward" in images:
            A(f"![Walk forward]({images['walk_forward']})")
            A("")
        A("### Per window")
        A("")
        rows = []
        for w in wf.windows:
            rows.append({
                "window": w.index,
                "train": f"{w.train_slice[0]:%Y-%m-%d}..{w.train_slice[1]:%Y-%m-%d}",
                "test": f"{w.test_slice[0]:%Y-%m-%d}..{w.test_slice[1]:%Y-%m-%d}",
                "IS_ret%": w.train_metrics.get("total_return_pct"),
                "OOS_ret%": w.test_metrics.get("total_return_pct"),
                "IS_trades": w.train_metrics.get("n_trades"),
                "OOS_trades": w.test_metrics.get("n_trades"),
                "OOS_PF": w.test_metrics.get("profit_factor"),
                "OOS_DD%": w.test_metrics.get("max_drawdown_pct"),
                "efficiency": w.efficiency,
                "best_params": ", ".join(f"{k}={v_}" for k, v_ in sorted(w.best_params.items())),
            })
        A(_df_to_md(pd.DataFrame(rows).set_index("window"), floatfmt="{:.2f}", max_rows=60))
        A("")
        A("### Parameter stability")
        A("")
        A("A parameter that changes value between adjacent windows is being fitted to "
          "noise. `mode_share` is how often the optimiser picked the same value.")
        A("")
        ps = (wf_summary or {}).get("params_stable") or {}
        if ps:
            rows = [{"param": k, "mode": d["mode"], "mode_share": d["mode_share"],
                     "n_unique": d["n_unique"], "stable": "yes" if d["stable"] else "NO",
                     "values": ", ".join(str(x) for x in d["values"])}
                    for k, d in ps.items()]
            A(_df_to_md(pd.DataFrame(rows).set_index("param"), floatfmt="{:.2f}"))
        else:
            A("_no walk-forward windows_")
        A("")

    # ----------------------------------------------------------- monte carlo
    if mc is not None:
        A("## Monte Carlo")
        A("")
        ms = mc_summary or {}
        if "error" in ms:
            A(f"_{ms['error']}_")
        else:
            A(f"{ms.get('n_sims', 0)} block-bootstrap resamples of "
              f"{ms.get('n_trades', 0)} trades (block size {ms.get('block_size')})")
            A("")
            fr = ms.get("final_return_pct", {})
            dd = ms.get("max_drawdown_pct", {})
            A("| Statistic | Mean | Median | 5th pct | 95th pct |")
            A("|:----------|-----:|-------:|--------:|---------:|")
            A(f"| Final return % | {fr.get('mean', float('nan')):.2f} | "
              f"{fr.get('median', float('nan')):.2f} | {fr.get('p5', float('nan')):.2f} | "
              f"{fr.get('p95', float('nan')):.2f} |")
            A(f"| Max drawdown % | {dd.get('mean', float('nan')):.2f} | "
              f"{dd.get('median', float('nan')):.2f} | - | "
              f"{dd.get('p95', float('nan')):.2f} |")
            A("")
            A(f"P(final return < 0) = **{ms.get('prob_loss', float('nan')):.1%}**")
            A("")
            A("Probability of breaching a drawdown threshold at any point:")
            A("")
            A("| Threshold | P(breach) |")
            A("|:----------|----------:|")
            for k, val in (ms.get("ruin_probabilities") or {}).items():
                lbl = k.replace("dd_exceeds_", "").replace("pct", "%")
                A(f"| {lbl} | {val:.1%} |")
            A("")
            if mc.warnings:
                for w in mc.warnings:
                    A(f"> **Warning:** {w}")
                A("")
            if "monte_carlo" in images:
                A(f"![Monte Carlo fan]({images['monte_carlo']})")
                A("")

    # ------------------------------------------------------------ null tests
    if nulls:
        A("## Null-hypothesis tests")
        A("")
        A("A backtest that makes money proves nothing by itself. These tests ask whether "
          "the result is distinguishable from randomness.")
        A("")
        from .validation.null_test import interpret
        from .validation.null_test import NullResult

        rebuilt = {k: NullResult(**{kk: vv for kk, vv in d.items()
                                    if kk in NullResult.__dataclass_fields__})
                   for k, d in nulls.items() if isinstance(d, dict)}
        A(interpret(rebuilt))
        A("")
        for k, d in nulls.items():
            if not isinstance(d, dict):
                continue
            A(f"**{k}** — statistic `{d.get('statistic')}`, "
              f"actual {d.get('actual', float('nan')):.2f}, "
              f"null mean {d.get('null_mean', float('nan')):.2f} ± "
              f"{d.get('null_std', float('nan')):.2f}, "
              f"p = {d.get('p_value', float('nan')):.3f} "
              f"({d.get('n_sims', 0)} sims)")
            A("")
            for w in d.get("warnings", []):
                A(f"> {w}")
                A("")

    # ------------------------------------------------------- cost sensitivity
    if cost_sens is not None and not cost_sens.empty:
        A("## Cost sensitivity")
        A("")
        A("The most useful robustness check for a retail strategy. If the edge dies at "
          "2x cost, it was never an edge.")
        A("")
        A(_df_to_md(cost_sens.set_index("cost_multiplier"), floatfmt="{:.2f}"))
        A("")
        if "cost_sensitivity" in images:
            A(f"![Cost sensitivity]({images['cost_sensitivity']})")
            A("")

    # ------------------------------------------------------------- validation
    if lookahead_report:
        A("## Lookahead-bias audit")
        A("")
        A("Every series is recomputed on truncated data and compared to the full-sample "
          "prefix. Any difference means the calculation used future bars.")
        A("")
        A("| Target | Columns | Cuts | Leaks | Result |")
        A("|:-------|--------:|:-----|------:|:------:|")
        for name, d in lookahead_report.items():
            if not isinstance(d, dict) or "columns_checked" not in d:
                continue
            A(f"| `{name}` | {d.get('columns_checked', 0)} | "
              f"{len(d.get('cuts_tested', []))} | {d.get('n_leaks', 0)} | "
              f"{'**OK**' if d.get('ok') else '**LEAK**'} |")
        A("")
        if not lookahead_report.get("ok", True):
            A("> **A lookahead leak was detected. Every number in this report is invalid "
              "until it is fixed.**")
            A("")

    if recovery_report:
        A("## Pattern-recovery test")
        A("")
        A("Known liquidity-sweep events are injected into synthetic data, then the "
          "detector is scored against that ground truth.")
        A("")
        for name, d in recovery_report.items():
            if not isinstance(d, dict):
                continue
            A(f"**{name}**")
            A("")
            A("| Metric | Value |")
            A("|:-------|------:|")
            for k in ("injected", "detected", "total_flags", "false_positives",
                      "tolerance_bars", "precision", "recall", "f1", "side_accuracy",
                      "mean_latency_bars", "eligible_events", "eligible_recall",
                      "blocked_by_filter"):
                if k in d:
                    val = d[k]
                    A(f"| {k.replace('_', ' ')} | "
                      f"{val:.3f} |" if isinstance(val, float) else f"| {k.replace('_', ' ')} | {val} |")
            A("")
            for w in d.get("warnings", []):
                A(f"> {w}")
                A("")

    if cross_validation:
        A("## Cross-validation vs vendored reference implementations")
        A("")
        A("Our SMC code is compared against `vendor/smart-money-concepts` "
          "(`joshyattridge/smart-money-concepts`).")
        A("")
        for key in ("swings", "fvg"):
            d = cross_validation.get(key)
            if not isinstance(d, dict):
                continue
            A(f"**{key}** — ours {d.get('ours')} vs reference {d.get('theirs')}, "
              f"matched {d.get('matched')} ({d.get('match_rate', 0):.1%}), "
              f"publication lag {d.get('lag_bars')} bars")
            A("")
            A(f"> {d.get('notes', '')}")
            A("")
        inv = cross_validation.get("invariants")
        if isinstance(inv, dict):
            A("Indicator invariants: " +
              ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in inv.items()))
            A("")
        A("**Why the lag matters:** the reference marks a swing pivot at bar `i` using "
          "`high.shift(-L)` — a forward look. Fed straight into a backtester that is a "
          "perfect preview of where the pivots were, which is why naive SMC backtests "
          "look spectacular and then fail live. We publish the same pivot at `i + L`, "
          "the first bar a trader could know it.")
        A("")

    # ---------------------------------------------------------------- appendix
    A("## Configuration")
    A("")
    A("```yaml")
    A(json.dumps(cfg.to_dict(), indent=2, default=str))
    A("```")
    A("")
    A("### Strategy parameters")
    A("")
    A("```json")
    A(json.dumps(study.strategy_params, indent=2, default=str))
    A("```")
    A("")
    A("## Limitations")
    A("")
    A("Stated plainly, because these are the things that turn a good backtest into a "
      "bad live account:")
    A("")
    A("- **Not modelled:** partial fills, order-book depth, funding/swap, short borrow "
      "cost, dividends, corporate actions, taxes, and network/order latency.")
    A("- **Survivorship:** the symbol list is whatever you configured. If it only "
      "contains names that still trade today, the result is biased upward.")
    A("- **Synthetic data:** if this run used the synthetic source, the P&L describes "
      "the generator, not any market. It validates the plumbing only.")
    A("- **Regime dependence:** every sample is dominated by the regimes it happened to "
      "contain. The walk-forward and Monte Carlo sections bound this; they do not "
      "eliminate it.")
    A("- **One path:** the backtest is a single draw from history. The Monte Carlo fan "
      "is the honest range.")
    A("")
    return "\n".join(L)


def write_report(
    study,
    out_dir: Path,
    wf=None,
    mc=None,
    nulls: Optional[Dict[str, Any]] = None,
    cost_sens: Optional[pd.DataFrame] = None,
    lookahead_report: Optional[Dict[str, Any]] = None,
    recovery_report: Optional[Dict[str, Any]] = None,
    cross_validation: Optional[Dict[str, Any]] = None,
    plots: bool = True,
    dpi: int = 130,
    fan: Optional[pd.DataFrame] = None,
    data_provenance: str = "",
    name: Optional[str] = None,
) -> Dict[str, Path]:
    """Write report.md, report.json and every chart into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or f"{study.strategy_name}_{study.config.universe.market}"
    images: Dict[str, Path] = {}

    if plots:
        try:
            p = plot_equity(
                study.portfolio_equity, title=f"{study.strategy_name} portfolio equity",
                path=out_dir / f"{stem}_equity.png", dpi=dpi,
            )
            if p:
                images["equity"] = p
        except Exception as exc:  # noqa: BLE001
            log.warning("equity plot failed: %s", exc)

        first = study.symbols[0] if study.symbols else None
        if first is not None:
            try:
                p = plot_trades(study.portfolio_equity, study.portfolio_trades,
                                path=out_dir / f"{stem}_trades.png", dpi=dpi)
                if p:
                    images["trades"] = p
            except Exception as exc:  # noqa: BLE001
                log.warning("trade plot failed: %s", exc)
            try:
                p = plot_chart_with_trades(
                    first.ohlcv, first.trades,
                    title=f"{first.symbol} {first.timeframe} - {study.strategy_name}",
                    path=out_dir / f"{stem}_chart.png", dpi=dpi,
                )
                if p:
                    images["chart"] = p
            except Exception as exc:  # noqa: BLE001
                log.warning("chart plot failed: %s", exc)

        if wf is not None:
            try:
                p = plot_walk_forward(wf, path=out_dir / f"{stem}_walkforward.png", dpi=dpi)
                if p:
                    images["walk_forward"] = p
            except Exception as exc:  # noqa: BLE001
                log.warning("walk-forward plot failed: %s", exc)

        if fan is not None:
            try:
                p = plot_monte_carlo(fan, path=out_dir / f"{stem}_montecarlo.png", dpi=dpi)
                if p:
                    images["monte_carlo"] = p
            except Exception as exc:  # noqa: BLE001
                log.warning("monte carlo plot failed: %s", exc)

        if cost_sens is not None:
            try:
                p = plot_cost_sensitivity(cost_sens,
                                          path=out_dir / f"{stem}_costs.png", dpi=dpi)
                if p:
                    images["cost_sensitivity"] = p
            except Exception as exc:  # noqa: BLE001
                log.warning("cost plot failed: %s", exc)

    md = build_markdown(
        study, wf, mc, nulls, cost_sens, lookahead_report, recovery_report,
        cross_validation,
        images={k: Path(v).name for k, v in images.items()},
        data_provenance=data_provenance,
    )
    md_path = out_dir / f"{stem}_report.md"
    md_path.write_text(md, encoding="utf-8")

    payload: Dict[str, Any] = {
        "created_at": study.created_at,
        "strategy": study.strategy_name,
        "strategy_params": study.strategy_params,
        "config": study.config.to_dict(),
        "config_fingerprint": study.config.fingerprint(),
        "data_provenance": data_provenance,
        "portfolio_metrics": _jsonable(study.portfolio_metrics or {}),
        "allocation": _jsonable(getattr(study, "allocation", None)),
        "unallocated": _jsonable(getattr(study, "unallocated", []) or []),
        "per_symbol": _jsonable(study.per_symbol.to_dict(orient="records")),
        "benchmark": _jsonable(study.benchmark) if study.benchmark else None,
        "walk_forward": _jsonable(wf.summary()) if wf is not None else None,
        "monte_carlo": _jsonable(mc.summary()) if mc is not None else None,
        "null_tests": _jsonable(nulls),
        "cost_sensitivity": _jsonable(cost_sens.to_dict(orient="records"))
                          if cost_sens is not None else None,
        "lookahead": _jsonable(lookahead_report),
        "recovery": _jsonable(recovery_report),
        "cross_validation": _jsonable(cross_validation),
        "trades": _jsonable(study.portfolio_trades.to_dict(orient="records"))
                  if study.portfolio_trades is not None and not study.portfolio_trades.empty
                  else [],
        "images": {k: str(Path(v).name) for k, v in images.items()},
    }
    json_path = out_dir / f"{stem}_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if study.portfolio_trades is not None and not study.portfolio_trades.empty:
        study.portfolio_trades.to_csv(out_dir / f"{stem}_trades.csv", index=False)
    if study.portfolio_equity is not None and len(study.portfolio_equity):
        study.portfolio_equity.to_frame().to_csv(out_dir / f"{stem}_equity.csv")

    return {"markdown": md_path, "json": json_path, **images}


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars so json.dumps does not choke."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj
