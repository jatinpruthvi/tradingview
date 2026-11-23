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
``GET  /api/report``          the most recent study's report.json, if present
``POST /api/config``          change market/style/strategy/params for the session
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
from ..data.store import DataStore
from ..indicators.smc import smc_features
from ..indicators.technical import adx, atr, ema, rsi, supertrend
from ..pipeline import build_strategy, effective_config, run_study
from ..strategy import available as available_strategies
from ..strategy.base import trend_filter

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
            "available": {
                "markets": sorted(p.stem for p in (CONFIG_DIR / "markets").glob("*.yaml")),
                "styles": sorted(p.stem for p in (CONFIG_DIR / "styles").glob("*.yaml")),
                "strategies": available_strategies(),
                "data_sources": ["synthetic", "csv", "ccxt", "yfinance", "tvscreener"],
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
    ) -> Dict[str, Any]:
        """Current setup state across the universe.

        Delegates to :mod:`tvlab.screener` - the same code path the CLI's
        ``tvlab scan`` uses - so the dashboard and the terminal cannot disagree
        about what a symbol's state is. They had already drifted once: this
        endpoint hardcoded ADX/RSI period 14 and called ``trend_filter`` without
        the strategy's ``bias_mode``, so it applied a stricter trend gate than
        the backtest did.
        """
        from ..screener import SCAN_BARS, rank, scan_universe, summarize

        c = state.cfg
        if timeframe:
            c = c.merge({"timeframe.execution": timeframe})
        syms = ([s.strip() for s in symbols.split(",")] if symbols
                else list(c.universe.symbols))
        if top:
            syms = syms[:top]

        strat = build_strategy(c)
        store = DataStore(c)
        rows, errors = scan_universe(store, c, syms, strat=strat,
                                     timeframe=c.timeframe.execution, bars=SCAN_BARS,
                                     with_trendlines=trendlines)

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
        return _json_safe({
            "rows": payload,
            "errors": [{"symbol": k, "error": v} for k, v in errors.items()],
            "strategy": c.strategy.name,
            "timeframe": c.timeframe.execution,
            "live_signals": s["live"],
            "summary": s,
        })

    # ----------------------------------------------------------------- suggest
    @app.get("/api/suggest")
    def api_suggest(
        symbols: Optional[str] = Query(None),
        timeframe: Optional[str] = Query(None),
        top: Optional[int] = Query(None, ge=1, le=500),
        trendlines: bool = Query(True, description="fit diagonal support/resistance"),
        max_breakeven: float = Query(0.60, ge=0.05, le=1.0),
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
        from ..screener import SCAN_BARS, scan_universe
        from ..screener.quality import suggest as build_suggestions

        c = state.cfg
        if timeframe:
            c = c.merge({"timeframe.execution": timeframe})
        syms = ([s.strip() for s in symbols.split(",")] if symbols
                else list(c.universe.symbols))
        if top:
            syms = syms[:top]

        store = DataStore(c)
        rows, errors = scan_universe(store, c, syms, strat=build_strategy(c),
                                     timeframe=c.timeframe.execution, bars=SCAN_BARS,
                                     with_trendlines=trendlines)
        res = build_suggestions(rows, fee_bps=c.costs.fee_bps, slippage_bps=c.costs.slippage_bps,
                                min_stop_distance_pct=c.risk.min_stop_distance_pct,
                                max_breakeven=max_breakeven)

        measured = None
        try:
            from ..cli import _measured_win_rate

            measured = _measured_win_rate(c, c.strategy.name)
        except Exception as exc:  # noqa: BLE001 - a missing report is not an API failure
            log.debug("no measured win rate available: %s", exc)

        return _json_safe({
            "candidates": [x.to_dict() for x in res["candidates"]],
            "rejected": res["rejected"],
            "unscoreable": res["unscoreable"],
            "n_candidates": res["n_candidates"],
            "n_rejected": res["n_rejected"],
            "round_trip_cost_bps": res["round_trip_cost_bps"],
            "max_breakeven": res["max_breakeven"],
            "weights": res["weights"],
            "measured": measured,
            "strategy": c.strategy.name,
            "timeframe": c.timeframe.execution,
            "market": c.universe.market,
            "scanned": len(rows),
            "errors": [{"symbol": k, "error": v} for k, v in errors.items()],
            "note": ("A ranked shortlist is not a recommendation. Nothing here has been "
                     "validated as profitable; `tvlab study` decides that, and its current "
                     "verdict is NOT VALIDATED."),
        })

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
