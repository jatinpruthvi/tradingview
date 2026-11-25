"""Dashboard: FastAPI backend + TradingView lightweight-charts frontend.

The browser talks only to this server (relative URLs), and the server does all
data fetching and backtesting. That keeps the app working behind a proxy on a
different origin, and keeps exchange credentials out of the browser entirely.

Endpoints
---------
``GET  /api/config``          current config + available markets/styles/strategies
``GET  /api/symbols``         the configured universe
``GET  /api/ohlcv``           bars for a symbol/timeframe
``GET  /api/indicators``      SMC + trend overlay series for charting
``GET  /api/backtest``        run a backtest, return equity + trades + metrics
``GET  /api/scan``            live setup read-out for the whole universe
``GET  /api/suggest``         cost-adjusted ranked ideas for one symbol
``GET  /api/daily``           the daily briefing (scan + rank + fundamental gates)
``GET  /api/review``          score the journal: what past shortlists actually did
``GET  /api/report``          the most recent study's report.json, if present
``POST /api/config``          change market/style/strategy/params for the session

Only ``POST /api/config`` mutates anything. ``/api/daily`` can append to the journal
but only when asked explicitly, and ``/api/review`` is read-only by construction - a
GET that writes to the record it is measuring would be a surprise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import CONFIG_DIR, REPO_ROOT, AppConfig, load_config
from ..data.fundamentals import FUNDAMENTAL_MARKETS, fetch_fundamentals
from ..data.store import DataStore
from ..indicators.smc import smc_features
from ..indicators.technical import adx, atr, ema, rsi, supertrend
from ..pipeline import build_strategy, effective_config, run_study
from ..screener.daily import journal_path, read_journal
from ..screener.universe import MODES, resolve_universe
from ..strategy import available as available_strategies
from ..strategy.base import trend_filter
from .jobs import JobRunner, Progress

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():                       # running from src/ layout
    STATIC_DIR = REPO_ROOT / "dashboard" / "static"


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return int(o.timestamp() * 1000)
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (pd.DataFrame,)):
        return _json_safe(o.to_dict(orient="records"))
    if isinstance(o, (pd.Series,)):
        return _json_safe(o.to_dict())
    return o


#: How many concurrent symbol fetches a dashboard scan uses when the request does
#: not say otherwise. The CLI's default, because the dashboard's "full NSE scan"
#: is the CLI's command with a table around it: serially that is ~20 minutes and
#: the button would look broken.
DEFAULT_WORKERS = 8


def _opt_int(v: Any) -> Optional[int]:
    """Query-string ints arrive as text, JSON ones as numbers, absent as None."""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _scan_progress(progress: Optional[Progress]) -> Optional[Any]:
    """Adapt a job's Progress handle to ``scan_universe``'s callback.

    The bar comes from ``advance``; the sentence beside it is set by the caller
    once ("scanning 502 symbol(s)"). Repeating the counts in both places printed
    ``355/502 · 355/502`` beside the bar, which reads like two different numbers.
    """
    return None if progress is None else progress.advance


def _measured(cfg: AppConfig) -> Optional[Dict[str, Any]]:
    """The last study's measured win rate, or None.

    A missing report is not an API failure and not a zero: :mod:`tvlab.cli` is the
    one place that decides what "measured" means, including the difference between
    a study that took no trades and a report that cannot be parsed.
    """
    try:
        from ..cli import _measured_win_rate

        return _measured_win_rate(cfg, cfg.strategy.name)
    except Exception as exc:  # noqa: BLE001 - a missing report is not an API failure
        log.debug("no measured win rate available: %s", exc)
        return None


def _scan_payload(cfg: AppConfig, *, symbols=None, universe="auto", max_symbols=None,
                  timeframe=None, top=None, trendlines=False,
                  workers=DEFAULT_WORKERS,
                  progress: Optional[Progress] = None) -> Dict[str, Any]:
    """Current setup state across the universe - the ``tvlab scan`` read-out.

    ``top`` truncates the *table*, never the scan. It used to slice ``syms``, so
    asking for 20 rows quietly turned a 502-name screen into a 40-name one - the
    exact silent-coverage bug the CLI had, reproduced in the dashboard.
    """
    from ..screener import SCAN_BARS, rank, scan_universe, summarize

    c = cfg
    if timeframe:
        c = c.merge({"timeframe.execution": timeframe})
    syms, universe_note = resolve_universe(c, symbols, universe, max_symbols)
    if progress is not None:
        progress.set_total(len(syms))
        progress.note(f"scanning {len(syms)} symbol(s)")

    strat = build_strategy(c)
    store = DataStore(c)
    rows, errors = scan_universe(store, c, syms, strat=strat,
                                 timeframe=c.timeframe.execution, bars=SCAN_BARS,
                                 with_trendlines=trendlines, workers=workers,
                                 progress=_scan_progress(progress))

    payload = []
    for r in rank(rows, live_only=False, fee_bps=c.costs.fee_bps,
                  slippage_bps=c.costs.slippage_bps,
                  min_stop_distance_pct=c.risk.min_stop_distance_pct):
        d = r.to_dict()
        # `rr` is the key the frontend has always read; keep it as an alias
        # rather than forcing a coordinated change in two places.
        d["rr"] = d.pop("reward_risk", None)
        payload.append(d)

    s = summarize(rows, errors)
    shown = payload[:top] if top else payload
    return _json_safe({
        "rows": shown,
        "n_rows": len(payload),
        "n_shown": len(shown),
        "errors": [{"symbol": k, "error": v} for k, v in errors.items()],
        "strategy": c.strategy.name,
        "timeframe": c.timeframe.execution,
        "live_signals": s["live"],
        "summary": s,
        "universe": universe_note,
    })


def _suggest_payload(cfg: AppConfig, *, symbols=None, universe="auto", max_symbols=None,
                     timeframe=None, top=None, trendlines=False,
                     max_breakeven=0.60, workers=DEFAULT_WORKERS,
                     progress: Optional[Progress] = None) -> Dict[str, Any]:
    """Ranked shortlist, best risk:reward *after costs* first - ``tvlab suggest``."""
    from ..screener import SCAN_BARS, scan_universe
    from ..screener.quality import suggest as build_suggestions

    c = cfg
    if timeframe:
        c = c.merge({"timeframe.execution": timeframe})
    syms, universe_note = resolve_universe(c, symbols, universe, max_symbols)
    if progress is not None:
        progress.set_total(len(syms))
        progress.note(f"scanning {len(syms)} symbol(s)")

    store = DataStore(c)
    rows, errors = scan_universe(store, c, syms, strat=build_strategy(c),
                                 timeframe=c.timeframe.execution, bars=SCAN_BARS,
                                 with_trendlines=trendlines, workers=workers,
                                 progress=_scan_progress(progress))
    if progress is not None:
        progress.note(f"ranking {len(rows)} row(s) after costs")
    res = build_suggestions(rows, fee_bps=c.costs.fee_bps, slippage_bps=c.costs.slippage_bps,
                            min_stop_distance_pct=c.risk.min_stop_distance_pct,
                            max_breakeven=max_breakeven,
                            allow_short=c.risk.allow_short)

    candidates = res["candidates"][:top] if top else res["candidates"]
    return _json_safe({
        "candidates": [x.to_dict() for x in candidates],
        "rejected": res["rejected"],
        "unscoreable": res["unscoreable"],
        "shorts_skipped": res["shorts_skipped"],
        "n_shorts_skipped": res["n_shorts_skipped"],
        "n_candidates": res["n_candidates"],
        "n_rejected": res["n_rejected"],
        "n_shown": len(candidates),
        "round_trip_cost_bps": res["round_trip_cost_bps"],
        "max_breakeven": res["max_breakeven"],
        "weights": res["weights"],
        "measured": _measured(c),
        "strategy": c.strategy.name,
        "timeframe": c.timeframe.execution,
        "market": c.universe.market,
        "scanned": len(rows),
        "errors": [{"symbol": k, "error": v} for k, v in errors.items()],
        "universe": universe_note,
        "note": ("A ranked shortlist is not a recommendation. Nothing here has been "
                 "validated as profitable; `tvlab study` decides that, and its current "
                 "verdict is NOT VALIDATED."),
    })


def _briefing_payload(cfg: AppConfig, *, symbols=None, universe="auto", max_symbols=None,
                      timeframe=None, top=10, trendlines=False,
                      max_breakeven=0.60, fundamentals=True, journal=False,
                      workers=DEFAULT_WORKERS,
                      progress: Optional[Progress] = None) -> Dict[str, Any]:
    """The daily briefing: the same composition ``tvlab daily`` runs."""
    from ..screener import SCAN_BARS, scan_universe
    from ..screener.daily import append_journal, build_briefing

    c = cfg
    if timeframe:
        c = c.merge({"timeframe.execution": timeframe})
    syms, universe_note = resolve_universe(c, symbols, universe, max_symbols)
    if progress is not None:
        progress.set_total(len(syms))
        progress.note(f"scanning {len(syms)} symbol(s)")

    store = DataStore(c)
    rows, errors = scan_universe(store, c, syms, strat=build_strategy(c),
                                 timeframe=c.timeframe.execution, bars=SCAN_BARS,
                                 with_trendlines=trendlines, workers=workers,
                                 progress=_scan_progress(progress))

    snaps, fund_error = None, None
    want = bool(fundamentals and c.fundamentals.enabled)
    if want and c.universe.market not in FUNDAMENTAL_MARKETS:
        fund_error = (f"market '{c.universe.market}' has no earnings, book value or "
                      f"analyst coverage, so there is nothing to gate on")
    elif want and rows:
        if progress is not None:
            progress.note(f"fetching fundamentals for {len(rows)} symbol(s)")
        try:
            snaps = fetch_fundamentals([r.symbol for r in rows],
                                       market=c.universe.market,
                                       max_symbols=max(len(rows), 200),
                                       extra_fields=c.fundamentals.extra_fields)
        except Exception as exc:  # noqa: BLE001 - an outage is a coverage fact
            fund_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    if progress is not None:
        progress.note("ranking and gating")
    b = build_briefing(rows, errors, c, snaps=snaps, study=_measured(c),
                       max_breakeven=max_breakeven, top=top,
                       fundamentals_error=fund_error)
    if journal:
        try:
            append_journal(b, REPO_ROOT / c.output.dir)
        except Exception as exc:  # noqa: BLE001 - never fail the briefing on this
            log.warning("could not journal the daily run: %s", exc)

    payload = b.to_dict()
    payload["fundamentals_error"] = fund_error
    payload["journal_path"] = str(journal_path(REPO_ROOT / c.output.dir))
    payload["recent_runs"] = read_journal(REPO_ROOT / c.output.dir, limit=10)
    payload["universe"] = universe_note
    return _json_safe(payload)


class AppState:
    """Mutable session state, so the UI can switch markets without a restart."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.store = DataStore(cfg)
        self._last_study = None

    def reconfigure(self, market=None, style=None, overrides=None) -> AppConfig:
        self.cfg = load_config(market=market or self.cfg.universe.market,
                               style=style, config_path=None, overrides=overrides or {})
        self.store = DataStore(self.cfg)
        self._last_study = None
        return self.cfg


def create_app(cfg: Optional[AppConfig] = None) -> FastAPI:
    cfg = cfg or load_config()
    state = AppState(cfg)

    app = FastAPI(
        title="tvlab dashboard",
        version="0.1.0",
        description="Backtest, screening and validation workbench. All computation "
                    "happens server-side; the browser only renders.",
    )
    # The preview host is arbitrary and not known ahead of time, so allow any
    # origin. The API is read-only over public/synthetic data and holds no
    # credentials - do not reuse this CORS policy on a service that does.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    # ------------------------------------------------------------------ config
    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        c = state.cfg
        d = c.to_dict()

        # Report the params that will ACTUALLY be used, not merely the ones
        # written in the YAML. `build_strategy` discards `strategy.params` when
        # they were declared for a different strategy (`params_for`), so a user
        # who switches strategy in the UI would otherwise be shown - and would
        # edit - a set of parameters that have no effect on the run.
        try:
            effective = dict(build_strategy(c).params)
        except Exception:  # noqa: BLE001 - never let the config panel 500
            effective = dict(c.strategy.params)
        inherited = bool(c.strategy.params_for and c.strategy.params_for != c.strategy.name)

        return _json_safe({
            "market": c.universe.market,
            "timeframe": {"execution": c.timeframe.execution, "bias": c.timeframe.bias,
                          "entry": c.timeframe.entry},
            "strategy": {
                "name": c.strategy.name,
                "params": effective,
                "declared_params": c.strategy.params,
                "params_for": c.strategy.params_for,
                "params_ignored": inherited,
            },
            "style": c.style,
            "risk": d["risk"],
            "costs": d["costs"],
            "backtest": d["backtest"],
            "data_source": c.data.source,
            "quote_currency": c.universe.quote_currency,
            "symbols": list(c.universe.symbols),
            # The configured list is a fallback for offline work, not a statement
            # about the exchange: NSE lists 502 names while the config ships 12.
            "universe": {"modes": list(MODES),
                         "config_symbols": len(c.universe.symbols),
                         "max_symbols": c.universe.max_symbols},
            "workers": DEFAULT_WORKERS,
            "available": {
                "markets": sorted(p.stem for p in (CONFIG_DIR / "markets").glob("*.yaml")),
                "styles": sorted(p.stem for p in (CONFIG_DIR / "styles").glob("*.yaml")),
                "strategies": available_strategies(),
                # Every source the CLI accepts. The dashboard used to omit `nse`
                # and `yahoo_chart`, which is the difference between screening 12
                # typed-in names and the exchange's own listing.
                "data_sources": ["synthetic", "csv", "ccxt", "yfinance", "tvscreener",
                                 "nse", "yahoo_chart"],
                "timeframes": ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"],
            },
            "fingerprint": c.fingerprint(),
        })

    @app.post("/api/config")
    def set_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        market = payload.get("market")
        style = payload.get("style")
        overrides = payload.get("overrides") or {}
        if "strategy" in payload:
            overrides["strategy.name"] = payload["strategy"]
        if "timeframe" in payload:
            overrides["timeframe.execution"] = payload["timeframe"]
        if "data_source" in payload:
            overrides["data.source"] = payload["data_source"]
        for k, v in (payload.get("params") or {}).items():
            overrides[f"strategy.params.{k}"] = v
        for k, v in (payload.get("risk") or {}).items():
            overrides[f"risk.{k}"] = v
        try:
            state.reconfigure(market=market, style=style, overrides=overrides)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
        return get_config()

    @app.get("/api/symbols")
    def get_symbols() -> Dict[str, Any]:
        return _json_safe({"symbols": list(state.cfg.universe.symbols),
                           "market": state.cfg.universe.market})

    # -------------------------------------------------------------------- data
    @app.get("/api/ohlcv")
    def get_ohlcv(
        symbol: str = Query(..., description="e.g. BTC/USDT or RELIANCE.NS"),
        timeframe: Optional[str] = None,
        limit: int = Query(1000, ge=50, le=20000),
    ) -> Dict[str, Any]:
        tf = timeframe or state.cfg.timeframe.execution
        try:
            df = state.store.ohlcv(symbol, tf, limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

        candles = [
            {"time": int(ts.timestamp()), "open": float(r[0]), "high": float(r[1]),
             "low": float(r[2]), "close": float(r[3])}
            for ts, r in zip(df.index, df[["open", "high", "low", "close"]].to_numpy())
        ]
        volumes = [
            {"time": int(ts.timestamp()), "value": float(v),
             "color": "rgba(38,166,154,0.45)" if c >= o else "rgba(239,83,80,0.45)"}
            for ts, v, o, c in zip(df.index, df["volume"], df["open"], df["close"])
        ]
        return _json_safe({
            "symbol": symbol, "timeframe": tf, "bars": len(df),
            "candles": candles, "volume": volumes,
            "source": df.attrs.get("source", state.cfg.data.source),
        })

    @app.get("/api/indicators")
    def get_indicators(
        symbol: str, timeframe: Optional[str] = None,
        limit: int = Query(1000, ge=50, le=20000),
        swing_length: int = Query(12, ge=2, le=100),
    ) -> Dict[str, Any]:
        """Overlay series for the chart: EMAs, swings, sweeps, FVGs, order blocks."""
        tf = timeframe or state.cfg.timeframe.execution
        try:
            df = state.store.ohlcv(symbol, tf, limit=limit)
            bias = state.store.bias_frame(symbol)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

        strat = build_strategy(state.cfg)
        L = int(strat.params.get("swing_length", swing_length))
        feats = smc_features(df, swing_length=L)
        trend = trend_filter(df, bias, int(strat.params.get("bias_ema", 50)),
                             int(strat.params.get("bias_adx_period", 14)),
                             float(strat.params.get("bias_adx_min", 0.0)))

        def ms(ts) -> int:
            return int(ts.timestamp())

        def markers(flags: pd.Series, side: int, text: str, color: str) -> List[Dict[str, Any]]:
            out = []
            idx = np.flatnonzero(flags.to_numpy(dtype=bool))
            for i in idx:
                out.append({"time": ms(df.index[i]),
                            "position": "belowBar" if side > 0 else "aboveBar",
                            "color": color, "shape": "arrowUp" if side > 0 else "arrowDown",
                            "text": text})
            return out

        def levels(col: str) -> List[Dict[str, Any]]:
            s = feats[col].dropna()
            return [{"time": ms(ts), "value": float(v)} for ts, v in s.items()]

        zones: List[Dict[str, Any]] = []
        fvg_top = feats["active_fvg_top"].to_numpy()
        fvg_bot = feats["active_fvg_bottom"].to_numpy()
        ob_top = feats["ob_top"].to_numpy()
        ob_bot = feats["ob_bottom"].to_numpy()
        for i in range(len(df)):
            if np.isfinite(fvg_top[i]) and np.isfinite(fvg_bot[i]):
                zones.append({"time": ms(df.index[i]), "top": float(fvg_top[i]),
                              "bottom": float(fvg_bot[i]), "kind": "fvg"})
            if np.isfinite(ob_top[i]) and np.isfinite(ob_bot[i]):
                zones.append({"time": ms(df.index[i]), "top": float(ob_top[i]),
                              "bottom": float(ob_bot[i]), "kind": "ob"})

        return _json_safe({
            "symbol": symbol, "timeframe": tf, "swing_length": L,
            "ema21": [{"time": ms(ts), "value": float(v)}
                      for ts, v in ema(df["close"], 21).dropna().items()],
            "ema50": [{"time": ms(ts), "value": float(v)}
                      for ts, v in ema(df["close"], 50).dropna().items()],
            "ema200": [{"time": ms(ts), "value": float(v)}
                       for ts, v in ema(df["close"], 200).dropna().items()],
            "supertrend": [{"time": ms(ts), "value": float(v)}
                           for ts, v in supertrend(df, 10, 3.0)["supertrend"].dropna().items()],
            "swing_highs": [{"time": ms(df.index[i]), "value": float(feats["swing_high_level"].iloc[i])}
                            for i in np.flatnonzero(feats["swing_high"].to_numpy())],
            "swing_lows": [{"time": ms(df.index[i]), "value": float(feats["swing_low_level"].iloc[i])}
                           for i in np.flatnonzero(feats["swing_low"].to_numpy())],
            "sweep_markers": (markers(feats["sweep_low"], 1, "sweep", "#2563eb")
                              + markers(feats["sweep_high"], -1, "sweep", "#dc2626")),
            "bos_markers": (markers(feats["bos"] == 1, 1, "BOS", "#16a34a")
                            + markers(feats["bos"] == -1, -1, "BOS", "#f97316")),
            "choch_markers": (markers(feats["choch"] == 1, 1, "CHoCH", "#0891b2")
                              + markers(feats["choch"] == -1, -1, "CHoCH", "#a21caf")),
            "zones": zones,
            "trend": [{"time": ms(ts), "value": int(v)} for ts, v in trend.items()],
            "rsi": [{"time": ms(ts), "value": float(v)}
                    for ts, v in rsi(df, 14).dropna().items()],
            "adx": [{"time": ms(ts), "value": float(v)}
                    for ts, v in adx(df, 14)["adx"].dropna().items()],
            "atr_pct": [{"time": ms(ts), "value": float(v)}
                        for ts, v in (atr(df, 14) / df["close"]).dropna().items()],
            "counts": {
                "bos_up": int((feats["bos"] == 1).sum()),
                "bos_down": int((feats["bos"] == -1).sum()),
                "choch_up": int((feats["choch"] == 1).sum()),
                "choch_down": int((feats["choch"] == -1).sum()),
                "bull_sweeps": int(feats["sweep_low"].sum()),
                "bear_sweeps": int(feats["sweep_high"].sum()),
                "bull_fvgs": int(feats["bull_fvg"].sum()),
                "bear_fvgs": int(feats["bear_fvg"].sum()),
            },
        })

    # ---------------------------------------------------------------- backtest
    @app.get("/api/backtest")
    def run_backtest(
        symbols: Optional[str] = Query(None, description="comma-separated; default = config"),
        strategy: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = Query(None, ge=100, le=20000),
    ) -> Dict[str, Any]:
        c = state.cfg
        overrides: Dict[str, Any] = {}
        if strategy:
            overrides["strategy.name"] = strategy
        if timeframe:
            overrides["timeframe.execution"] = timeframe
        if limit:
            overrides["data.synthetic_bars"] = limit
        if overrides:
            c = c.merge(overrides)

        syms = ([s.strip() for s in symbols.split(",")] if symbols
                else list(c.universe.symbols))
        try:
            store = DataStore(c)
            st = run_study(c, symbols=syms, store=store, with_benchmark=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400,
                                detail=f"{type(exc).__name__}: {exc}") from exc

        state._last_study = st
        m = st.portfolio_metrics or {}
        eq = st.portfolio_equity
        per_symbol = st.per_symbol

        first = st.symbols[0] if st.symbols else None
        chart_trades: List[Dict[str, Any]] = []
        if first is not None and not first.trades.empty:
            for _, t in first.trades.iterrows():
                chart_trades.append({
                    "entry_time": int(pd.Timestamp(t["entry_time"]).timestamp()),
                    "exit_time": int(pd.Timestamp(t["exit_time"]).timestamp()),
                    "entry_price": float(t["entry_price"]),
                    "exit_price": float(t["exit_price"]),
                    "side": int(t["side"]),
                    "pnl": float(t["pnl"]),
                    "r_multiple": None if not np.isfinite(t["r_multiple"]) else float(t["r_multiple"]),
                    "exit_reason": str(t["exit_reason"]),
                    "tag": str(t.get("tag", "")),
                    "bars_held": int(t["bars_held"]),
                })

        return _json_safe({
            "strategy": st.strategy_name,
            "strategy_params": st.strategy_params,
            "timeframe": c.timeframe.execution,
            "market": c.universe.market,
            "metrics": m,
            "equity": [{"time": int(ts.timestamp()), "value": float(v)}
                       for ts, v in eq.items()] if eq is not None else [],
            "per_symbol": per_symbol.to_dict(orient="records"),
            "benchmark": st.benchmark,
            "trades_first_symbol": chart_trades,
            "first_symbol": first.symbol if first else None,
            "all_trades": st.portfolio_trades.to_dict(orient="records")
                          if st.portfolio_trades is not None else [],
            "warnings": [w for s in st.symbols for w in s.warnings][:20],
        })

    # -------------------------------------------------------------------- scan
    @app.get("/api/scan")
    def scan(
        symbols: Optional[str] = None,
        timeframe: Optional[str] = None,
        top: Optional[int] = Query(None, ge=1, le=500),
        trendlines: bool = Query(False, description="also fit diagonal support/resistance"),
        universe: str = Query("auto", description="auto | config | source"),
        max_symbols: Optional[int] = Query(None, ge=0),
        workers: int = Query(DEFAULT_WORKERS, ge=1, le=32),
    ) -> Dict[str, Any]:
        """Current setup state across the universe.

        Delegates to :mod:`tvlab.screener` - the same code path the CLI's
        ``tvlab scan`` uses - so the dashboard and the terminal cannot disagree
        about what a symbol's state is. They had already drifted once: this
        endpoint hardcoded ADX/RSI period 14 and called ``trend_filter`` without
        the strategy's ``bias_mode``, so it applied a stricter trend gate than
        the backtest did.

        ``universe`` resolves the symbol list exactly as ``tvlab scan --universe``
        does, so "scan" here means the exchange's listing when the source can
        enumerate it (NSE: 502 names), not the hand-typed config list.
        """
        try:
            return _scan_payload(state.cfg, symbols=symbols, universe=universe,
                                 max_symbols=max_symbols, timeframe=timeframe, top=top,
                                 trendlines=trendlines, workers=workers)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ----------------------------------------------------------------- suggest
    @app.get("/api/suggest")
    def api_suggest(
        symbols: Optional[str] = Query(None),
        timeframe: Optional[str] = Query(None),
        top: Optional[int] = Query(None, ge=1, le=500),
        # Off by default, matching `tvlab suggest`: no rendered column shows a
        # trendline, and fitting one per symbol is pure cost across 500 names.
        trendlines: bool = Query(False, description="fit diagonal support/resistance"),
        max_breakeven: float = Query(0.60, ge=0.05, le=1.0),
        universe: str = Query("auto", description="auto | config | source"),
        max_symbols: Optional[int] = Query(None, ge=0),
        workers: int = Query(DEFAULT_WORKERS, ge=1, le=32),
    ) -> Dict[str, Any]:
        """Ranked shortlist, best risk:reward *after costs* first.

        Same code path as ``tvlab suggest`` - :mod:`tvlab.screener.quality` - so the
        dashboard and the terminal cannot disagree about which idea is best. Both
        delegate to ``scan_universe``, which is the arrangement that stopped the two
        scan implementations from drifting apart.

        Every candidate carries its ``breakeven_win_rate``: the win rate needed to
        make money at the configured fee and slippage. That number is arithmetic on
        the stop distance and the fee schedule, not a forecast. ``measured`` is the
        win rate this strategy actually achieved in the most recent study, with its
        trade count, so the two can be compared - and so the sample size stops the
        comparison being over-read.
        """
        try:
            return _suggest_payload(state.cfg, symbols=symbols, universe=universe,
                                    max_symbols=max_symbols, timeframe=timeframe, top=top,
                                    trendlines=trendlines, max_breakeven=max_breakeven,
                                    workers=workers)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------- daily
    @app.get("/api/daily")
    def api_daily(
        symbols: Optional[str] = Query(None),
        timeframe: Optional[str] = Query(None),
        top: int = Query(10, ge=1, le=500),
        trendlines: bool = Query(False, description="fit diagonal support/resistance"),
        max_breakeven: float = Query(0.60, ge=0.05, le=1.0),
        fundamentals: bool = Query(True, description="apply the fundamental gates"),
        journal: bool = Query(False, description="append this run to the daily journal"),
        universe: str = Query("auto", description="auto | config | source"),
        max_symbols: Optional[int] = Query(None, ge=0),
        workers: int = Query(DEFAULT_WORKERS, ge=1, le=32),
    ) -> Dict[str, Any]:
        """The daily briefing: the same composition ``tvlab daily`` runs.

        Both delegate to :mod:`tvlab.screener.daily`, so the browser and the terminal
        cannot disagree about today's list - the same arrangement that keeps
        ``/api/suggest`` and ``tvlab suggest`` from drifting.

        The payload leads with ``advisory`` and ``is_validated`` rather than with the
        shortlist. A UI that renders ranked candidates first and mentions their
        validity status in a footnote has already framed an unvalidated strategy as
        advice, so the verdict is a top-level field the panel is expected to show
        above the table.

        ``journal=true`` appends to ``output/daily/journal.jsonl``. It defaults to
        false because a GET that writes is a surprise, and a dashboard that polls
        would otherwise fill the journal with duplicates of the same morning.
        """
        try:
            return _briefing_payload(state.cfg, symbols=symbols, universe=universe,
                                     max_symbols=max_symbols, timeframe=timeframe, top=top,
                                     trendlines=trendlines, max_breakeven=max_breakeven,
                                     fundamentals=fundamentals, journal=journal,
                                     workers=workers)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -------------------------------------------------------------------- jobs
    # A 500-name NSE scan is ~1,500 HTTP requests: 80-140 seconds. Inside the GET
    # that renders the panel that is a blank browser, and a reader cannot tell a
    # slow scan from a hung one. So the long reads run as jobs: POST starts one,
    # GET polls it with progress, and the page stays responsive while it works.
    runner = JobRunner()

    def _job_body(kind: str, cfg: AppConfig, params: Dict[str, Any]):
        """Bind one job kind to the right builder, with the submit-time config.

        The config is captured when the job is submitted, not read when it runs:
        a scan that starts under NSE and finishes after the user switched to
        crypto would otherwise report bars from one market against the fees of
        another.
        """
        common = dict(
            symbols=params.get("symbols") or None,
            universe=str(params.get("universe") or "auto"),
            max_symbols=_opt_int(params.get("max_symbols")),
            timeframe=params.get("timeframe") or None,
            workers=int(params.get("workers") or DEFAULT_WORKERS),
        )
        if kind == "daily":
            return lambda prog: _briefing_payload(
                cfg, **common,
                top=int(params.get("top") or 10),
                trendlines=bool(params.get("trendlines", False)),
                max_breakeven=float(params.get("max_breakeven") or 0.60),
                fundamentals=bool(params.get("fundamentals", True)),
                journal=bool(params.get("journal", False)),
                progress=prog)
        if kind == "suggest":
            return lambda prog: _suggest_payload(
                cfg, **common,
                top=_opt_int(params.get("top")),
                trendlines=bool(params.get("trendlines", False)),
                max_breakeven=float(params.get("max_breakeven") or 0.60),
                progress=prog)
        return lambda prog: _scan_payload(
            cfg, **common,
            top=_opt_int(params.get("top")),
            trendlines=bool(params.get("trendlines", False)),
            progress=prog)

    @app.post("/api/jobs")
    def start_job(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Start a scan; returns immediately with an id to poll.

        A second request for a kind that is already running joins that job rather
        than queueing another 500-name scan behind it. Pressing the button twice
        should not make the answer fifteen minutes staler.
        """
        kind = str(payload.get("kind") or "").strip()
        if kind not in ("daily", "suggest", "scan"):
            raise HTTPException(status_code=400,
                                detail=f"kind must be daily, suggest or scan; got {kind!r}")
        params = dict(payload.get("params") or {})
        running = runner.active(kind)
        # Only an identical request joins: a scan that started under NSE must not
        # answer a request that arrived after the user switched to the 12-name
        # config list, because the symbols are the whole question.
        if running is not None and running.params == params:
            joined = running.to_dict(include_result=False)
            joined["joined"] = True
            return _json_safe(joined)
        job = runner.submit(kind, _job_body(kind, state.cfg, params), params)
        return _json_safe(job.to_dict(include_result=False))

    @app.get("/api/jobs")
    def list_jobs(limit: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
        return _json_safe({"jobs": [j.to_dict(include_result=False)
                                     for j in runner.recent(limit)]})

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404,
                                detail=f"no job {job_id!r}; it may have been trimmed "
                                       f"from the last {len(runner.recent(50))} jobs")
        return _json_safe(job.to_dict())

    # ------------------------------------------------------------------ review
    @app.get("/api/review")
    def api_review(
        horizon: int = Query(90, ge=1, le=2000,
                             description="bars to give each idea before marking it "
                                         "to market"),
        entry_tolerance: float = Query(0.005, ge=0.0, le=0.5,
                                       description="how far the fetched bars may miss "
                                                   "the journaled entry price before "
                                                   "the idea is unscorable"),
        days: Optional[int] = Query(None, ge=1,
                                    description="review only the most recent N runs"),
        timeframe: Optional[str] = Query(None,
                                         description="replay every run at this "
                                                     "timeframe instead of its own"),
    ) -> Dict[str, Any]:
        """Score the journal: what past daily shortlists actually did.

        The same composition ``tvlab review`` runs, delegating to
        :mod:`tvlab.screener.review` - and to the same ``store_fetcher`` - so the
        browser and the terminal cannot report different outcomes for one journal
        line. Price anchoring is the part most likely to drift between two
        implementations, which is why there is only one.

        Unlike ``/api/daily`` this endpoint never writes. Reading a journal is all it
        does, so there is no ``journal=true`` flag and no risk of a polling dashboard
        mutating the record it is supposed to be measuring.

        The payload leads with ``coverage`` rather than ``win_rate``. A rate without
        its denominator reads as a verdict, and the honest first fact about a review
        is how much of the journal could actually be scored: ideas still running are
        ``open`` and ideas that could not be priced are ``unscorable``, and neither
        belongs in a win rate.

        ``realised_breakeven`` and ``payoff_diverged`` travel with ``win_rate`` on
        purpose. The breakeven quoted when a suggestion was made assumes winners pay
        the promised R:R; when they time-exit short of it, the required win rate is
        much higher, and comparing the realised rate to the promised one would pass a
        losing book. ``expectancy_r`` is the number that needs no such assumption.
        """
        from ..screener import SCAN_BARS
        from ..screener.review import review as score_journal, store_fetcher

        c = state.cfg
        out_dir = REPO_ROOT / c.output.dir
        jp = journal_path(out_dir)

        entries = list(reversed(read_journal(out_dir, limit=days or 10_000_000)))
        if not entries:
            # Not an error and not an empty table: the journal simply does not exist
            # yet. A UI that rendered this as "0% win rate" would be reporting a
            # verdict about a strategy that has never been asked for one.
            return _json_safe({
                "available": False,
                "detail": f"no journal at {jp}",
                "journal_path": str(jp),
                "next_step": (f"run the daily briefing with journal=true, or "
                              f"`tvlab daily --market {c.universe.market} "
                              f"--style {c.style}`, then return once enough bars "
                              f"have printed for the ideas to resolve"),
                "as_of": pd.Timestamp.now(tz="UTC").date().isoformat(),
                "provenance": c.data.source,
                "horizon_bars": horizon,
                "n_suggestions": 0,
                "n_scored": 0,
                "coverage": "no suggestions in the journal",
                "rows": [],
            })

        for run in entries:
            if timeframe:
                run["timeframe"] = timeframe
            elif not run.get("timeframe"):
                run["timeframe"] = c.timeframe.execution

        fetch_errors: List[str] = []
        no_future: Dict[str, str] = {}
        fetch = store_fetcher(DataStore(c), scan_bars=SCAN_BARS,
                              errors=fetch_errors, no_future=no_future)

        summary = score_journal(entries, fetch, horizon_bars=horizon,
                                provenance=c.data.source,
                                entry_tolerance=entry_tolerance,
                                as_of=pd.Timestamp.now(tz="UTC").date().isoformat())

        payload = summary.to_dict()
        payload["available"] = True
        payload["journal_path"] = str(jp)
        payload["n_runs"] = len(entries)
        payload["first_run"] = entries[0].get("as_of")
        payload["last_run"] = entries[-1].get("as_of")
        payload["fetch_errors"] = fetch_errors
        payload["no_future"] = no_future
        # Stated at the top level, not left for the client to infer from provenance:
        # on synthetic data every rate here measures the generator.
        payload["measures_generator_not_market"] = bool(c.data.source == "synthetic")
        return _json_safe(payload)

    # ------------------------------------------------------------------ report
    @app.get("/api/report")
    def latest_report() -> Dict[str, Any]:
        out = REPO_ROOT / state.cfg.output.dir
        if not out.exists():
            return _json_safe({"available": False, "detail": f"no output directory at {out}"})
        candidates = sorted(out.rglob("*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return _json_safe({"available": False, "detail": "no report.json written yet; run `tvlab study`"})
        newest = candidates[0]
        try:
            payload = json.loads(newest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        payload["_path"] = str(newest.relative_to(REPO_ROOT))
        payload["_mtime"] = pd.Timestamp(newest.stat().st_mtime, unit="s", tz="UTC").isoformat()
        payload["available"] = True
        return _json_safe(payload)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "market": state.cfg.universe.market,
                "strategy": state.cfg.strategy.name,
                "data_source": state.cfg.data.source,
                "symbols": len(state.cfg.universe.symbols)}

    # ------------------------------------------------------------------ static
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> Any:
            idx = STATIC_DIR / "index.html"
            if idx.exists():
                return FileResponse(str(idx))
            return JSONResponse({"error": f"no index.html in {STATIC_DIR}"})
    else:  # pragma: no cover
        @app.get("/")
        def index_missing() -> Dict[str, Any]:
            return {"error": "dashboard static assets not found",
                    "expected": str(STATIC_DIR)}

    return app


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
