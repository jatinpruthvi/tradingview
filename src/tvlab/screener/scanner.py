"""Universe scanner: a read-out of *now*, shared by the CLI and the dashboard.

This module exists because the same per-symbol read-out was being computed twice
- once in ``cli.py`` for the ``tvlab scan`` table and once in
``dashboard/app.py`` for ``/api/scan`` - and the two copies had already started
to drift. One implementation, two renderers.

    from tvlab.screener import scan_universe
    rows, errors = scan_universe(store, cfg, symbols)

Nothing here forecasts. A row says: this is the higher-timeframe trend, this is
whether a liquidity sweep fired recently, this is whether price sits inside an
unmitigated FVG or order block, and this is the signal the strategy would take
on the next bar. Whether that signal is *worth* taking is the backtest's job,
and the backtest's answer on this data is in the study report - not here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..indicators.smc import smc_features
from ..indicators.technical import adx, atr, rsi
from ..pipeline import build_strategy
from ..strategy.base import Strategy, blank_signals, trend_filter

log = logging.getLogger(__name__)

#: Bars of history a scan needs. Enough for a 200-period EMA plus SMC swings to
#: settle; more only slows the scan down without changing the last row.
SCAN_BARS = 1200

_TREND_LABEL = {1: "UP", -1: "DOWN", 0: "flat"}
_SIGNAL_LABEL = {1: "LONG", -1: "SHORT", 0: "-"}


@dataclass
class ScanRow:
    """Everything the CLI table and the dashboard card need about one symbol."""

    symbol: str
    last: float = float("nan")
    change_pct: float = 0.0
    trend: int = 0
    trend_label: str = "flat"
    adx: float = float("nan")
    rsi: float = float("nan")
    atr: float = float("nan")
    atr_pct: float = float("nan")
    #: "bull", "bear" or None - which side swept on the latest bar
    sweep: Optional[str] = None
    #: bars since the last sweep *on the side the trend filter wants*
    sweep_age: Optional[float] = None
    bull_sweep_age: Optional[float] = None
    bear_sweep_age: Optional[float] = None
    in_fvg: bool = False
    in_ob: bool = False
    zones: str = "-"
    structure: int = 0
    signal: int = 0
    signal_label: str = "-"
    stop: Optional[float] = None
    target: Optional[float] = None
    reward_risk: Optional[float] = None
    bars: int = 0
    as_of: Optional[str] = None
    #: Nearest fitted trendline as a signed percentage of last price: support is
    #: negative (below), resistance positive (above). Populated only when the scan
    #: is asked for trendlines - fitting is ~27 ms per symbol on 1,200 bars, which
    #: is free for a watchlist and 25 s across a 500-name universe.
    support_pct: Optional[float] = None
    resistance_pct: Optional[float] = None
    channel_pct: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return self.signal != 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def scan_symbol(store, cfg: AppConfig, symbol: str, strat: Optional[Strategy] = None,
                timeframe: Optional[str] = None, bars: int = SCAN_BARS,
                bias: Optional[pd.DataFrame] = None,
                with_trendlines: bool = False) -> ScanRow:
    """Read the current setup state of one symbol.

    ``bias`` may be supplied to avoid a second fetch when scanning a universe;
    it must be NATIVE higher-timeframe bars (what ``store.bias_frame`` returns),
    not bars pre-aligned onto the execution index.

    ``with_trendlines`` adds the nearest diagonal support and resistance. It is
    opt-in because it costs a pivot-pair fit per symbol.
    """
    strat = strat or build_strategy(cfg)
    p = strat.params
    tf = timeframe or cfg.timeframe.execution

    df = store.ohlcv(symbol, tf, limit=bars)
    if bias is None:
        try:
            bias = store.bias_frame(symbol)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s: no bias frame (%s); scanning without the HTF filter", symbol, exc)
            bias = None

    row = ScanRow(symbol=symbol, bars=len(df))
    if len(df) < 2:
        row.warnings.append("not enough bars")
        return row

    row.as_of = str(df.index[-1])
    row.last = float(df["close"].iloc[-1])
    row.change_pct = float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100)

    atr_period = int(p.get("atr_period", 14))
    a = atr(df, atr_period)
    row.atr = _finite(a.iloc[-1]) or float("nan")
    row.atr_pct = float(row.atr / row.last * 100) if row.last else float("nan")

    adx_period = int(p.get("bias_adx_period", 14))
    row.adx = _finite(adx(df, adx_period)["adx"].iloc[-1]) or float("nan")
    rsi_period = int(p.get("rsi_period", 14))
    row.rsi = _finite(rsi(df, rsi_period).iloc[-1]) or float("nan")

    feats = smc_features(df, swing_length=int(p.get("swing_length", 12)))
    trend = trend_filter(df, bias, int(p.get("bias_ema", 50)), adx_period,
                         float(p.get("bias_adx_min", 0.0)),
                         mode=str(p.get("bias_mode", "ema_adx")),
                         slope_lookback=int(p.get("bias_slope_lookback", 5)))

    row.trend = int(trend.iloc[-1])
    row.trend_label = _TREND_LABEL.get(row.trend, "flat")
    row.structure = int(feats["structure"].iloc[-1])

    if feats["sweep_low"].iloc[-1]:
        row.sweep = "bull"
    elif feats["sweep_high"].iloc[-1]:
        row.sweep = "bear"

    bull_age = _finite(feats["bull_sweep_age"].iloc[-1])
    bear_age = _finite(feats["bear_sweep_age"].iloc[-1])
    row.bull_sweep_age = bull_age
    row.bear_sweep_age = bear_age
    # report the age on the side the trend filter actually wants; a bearish sweep
    # is irrelevant to a long, and showing it would read as a false confirmation
    row.sweep_age = bull_age if row.trend == 1 else (bear_age if row.trend == -1 else None)

    row.in_fvg = bool(feats["fvg_in_zone"].iloc[-1])
    row.in_ob = bool(feats["ob_in_zone"].iloc[-1])
    row.zones = "+".join([z for z, on in (("FVG", row.in_fvg), ("OB", row.in_ob)) if on]) or "-"

    try:
        sig = strat.signals(df, bias)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: strategy failed (%s); reporting state without a signal", symbol, exc)
        sig = blank_signals(df)
        row.warnings.append(f"strategy error: {exc}")

    row.signal = int(sig["signal"].iloc[-1])
    row.signal_label = _SIGNAL_LABEL.get(row.signal, "-")
    row.stop = _finite(sig["stop"].iloc[-1])
    row.target = _finite(sig["target"].iloc[-1])

    if row.signal != 0 and row.stop is not None and abs(row.last - row.stop) > 0:
        if row.target is not None:
            row.reward_risk = float(abs(row.target - row.last) / abs(row.last - row.stop))
        else:
            # no fixed target: express the reward as the configured R multiple
            r_mult = _finite(p.get("r_multiple"))
            if r_mult:
                row.reward_risk = float(r_mult)

    if with_trendlines:
        # Fitted on df alone, so it describes the last bar and cannot see the
        # future. A fit failure must not lose the rest of the row.
        try:
            from ..indicators.trendlines import trendline_state

            tl = trendline_state(df, tf, swing_length=max(3, int(p.get("swing_length", 8))))
            row.support_pct = _finite(tl["nearest_support_pct"])
            row.resistance_pct = _finite(tl["nearest_resistance_pct"])
            row.channel_pct = _finite(tl["channel_pct"])
            if not tl["available"]:
                row.warnings.append(f"no trendlines fitted ({tl['reason']})")
        except Exception as exc:  # noqa: BLE001
            log.debug("%s: trendline fit failed (%s)", symbol, exc)
            row.warnings.append(f"trendline error: {exc}")

    if bias is None:
        row.warnings.append("no higher-timeframe bias; trend filter ran on the execution frame")
    return row


def scan_universe(store, cfg: AppConfig, symbols: Sequence[str],
                  strat: Optional[Strategy] = None, timeframe: Optional[str] = None,
                  bars: int = SCAN_BARS, with_bias: bool = True,
                  with_trendlines: bool = False,
                  ) -> Tuple[List[ScanRow], Dict[str, str]]:
    """Scan every symbol, collecting per-symbol errors instead of aborting.

    A universe scan that dies on the first delisted ticker is useless, so errors
    are returned alongside the rows. The bias frame is fetched once per symbol
    and reused for both the trend filter and the signal, which halves the work.
    """
    strat = strat or build_strategy(cfg)
    rows: List[ScanRow] = []
    errors: Dict[str, str] = {}

    for sym in symbols:
        try:
            bias = None
            if with_bias:
                try:
                    bias = store.bias_frame(sym)
                except Exception as exc:  # noqa: BLE001
                    log.debug("%s: bias unavailable (%s)", sym, exc)
            rows.append(scan_symbol(store, cfg, sym, strat=strat, timeframe=timeframe,
                                    bars=bars, bias=bias, with_trendlines=with_trendlines))
        except Exception as exc:  # noqa: BLE001
            errors[sym] = f"{type(exc).__name__}: {exc}"
            log.warning("scan failed for %s: %s", sym, exc)

    return rows, errors


def rank(rows: Sequence[ScanRow], live_only: bool = True) -> List[ScanRow]:
    """Order rows by how actionable they are.

    Live signals first, best reward:risk first. Symbols with no signal sort by
    proximity to one - a fresh sweep on the trend side with price inside an
    unmitigated zone is the nearest thing to a setup, and that is what a scanner
    is for: a shortlist to look at, not a list of orders to place.
    """
    def key(r: ScanRow) -> Tuple[int, float, float]:
        if r.signal != 0:
            return (0, -(r.reward_risk or 0.0), 0.0)
        # not live: rank by setup proximity
        age = r.sweep_age if r.sweep_age is not None else 999.0
        proximity = (1.0 if r.zones != "-" else 0.0) + (1.0 if age <= 6 else 0.0)
        return (1, -proximity, age)

    ordered = sorted(rows, key=key)
    return [r for r in ordered if r.signal != 0] if live_only else ordered


def summarize(rows: Sequence[ScanRow], errors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Counts for a one-line status read-out."""
    live = [r for r in rows if r.is_live]
    out = {
        "scanned": len(rows),
        "failed": len(errors or {}),
        "live": len(live),
        "long": sum(1 for r in live if r.signal > 0),
        "short": sum(1 for r in live if r.signal < 0),
        "uptrend": sum(1 for r in rows if r.trend == 1),
        "downtrend": sum(1 for r in rows if r.trend == -1),
        "in_zone": sum(1 for r in rows if r.zones != "-"),
        "median_atr_pct": float(np.nanmedian([r.atr_pct for r in rows])) if rows else float("nan"),
    }
    if errors:
        out["errors"] = dict(errors)
    return out
