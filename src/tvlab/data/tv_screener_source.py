"""TradingView screener adapter (universe selection) via ``tvscreener``.

This does NOT provide OHLCV history - TradingView's public scanner endpoint
returns point-in-time fundamentals + technicals for a whole market. Its job in
this stack is to answer *"which symbols are worth backtesting / trading today?"*
and hand that list to a price adapter (ccxt or yfinance).

Requires network access to scanner.tradingview.com. In a sandbox with no route
to TradingView it raises :class:`DataError` with the offline fallback spelled out.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataError

log = logging.getLogger(__name__)

#: tvlab market name -> tvscreener Market enum name
MARKET_MAP: Dict[str, str] = {
    "us_equities": "AMERICA",
    "indian_equities": "INDIA",
    "uk_equities": "UK",
    "crypto": "ALL",
    "forex": "ALL",
}


class TvScreenerSource:
    """Rank a market's symbols by liquidity / momentum / volatility."""

    name = "tvscreener"

    def __init__(self, market: str = "crypto"):
        self.market = market
        self._tvscreener = _import()

    # ------------------------------------------------------------------ public
    def screen(
        self,
        symbols: Optional[List[str]] = None,
        min_volume: float = 1_000_000.0,
        max_symbols: int = 20,
        sort_by: str = "volume",
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Return a ranked universe table.

        Columns are normalised to: ``symbol, last, change_pct, volume,
        quote_volume, adx, rsi, atr_pct, market_cap`` (whatever the endpoint
        actually returned - missing columns are dropped, not faked).
        """
        tvs = self._tvscreener
        is_crypto = self.market == "crypto"
        try:
            screener = (tvs.CryptoScreener() if is_crypto else tvs.StockScreener())
        except Exception as exc:  # pragma: no cover - version drift
            raise DataError(f"could not construct tvscreener screener: {exc}") from exc

        Market = tvs.Market
        enum_name = MARKET_MAP.get(self.market, "AMERICA")
        if hasattr(Market, enum_name):
            try:
                screener.set_markets(getattr(Market, enum_name))
            except Exception as exc:
                log.warning("set_markets(%s) failed: %s", enum_name, exc)

        field_enum = tvs.CryptoField if is_crypto else tvs.StockField
        op = tvs.FilterOperator

        vol_field = _pick(field_enum, "VOLUME", "TOTAL_VOLUME", "AVERAGE_VOLUME_30_DAY")
        if vol_field is not None and min_volume:
            try:
                screener.add_filter(vol_field, op.GREATER, min_volume)
            except Exception as exc:
                log.warning("volume filter failed: %s", exc)

        for name, (fld_name, oper, value) in (extra_filters or {}).items():
            fld = _pick(field_enum, fld_name)
            if fld is None:
                log.warning("field %s not available on this screener; skipping filter %s", fld_name, name)
                continue
            try:
                screener.add_filter(fld, getattr(op, oper.upper()), value)
            except Exception as exc:
                log.warning("filter %s failed: %s", name, exc)

        sort_field = _pick(field_enum, {"volume": "VOLUME", "quote_volume": "TURNOVER",
                                        "change": "CHANGE", "adx": "AVERAGE_DIRECTIONAL_INDEX_14",
                                        "rsi": "RELATIVE_STRENGTH_INDEX_14"}.get(sort_by, "VOLUME"))
        try:
            screener.select_all()
            if sort_field is not None:
                screener.sort_by(sort_field, ascending=False)
            df = screener.get()
        except Exception as exc:
            raise DataError(
                "TradingView screener request failed: "
                f"{type(exc).__name__}: {exc}\n"
                "  This needs outbound access to scanner.tradingview.com.\n"
                "  Offline fallback: name your symbols explicitly in "
                "config/markets/<market>.yaml, or run with --data-source csv."
            ) from exc

        if df is None or len(df) == 0:
            raise DataError("screener returned 0 rows - your filters may be too strict")

        df = _normalise_columns(df)
        if symbols:
            wanted = {s.upper().replace("/", "").replace(":", "") for s in symbols}
            key = "symbol" if "symbol" in df.columns else df.columns[0]
            mask = df[key].astype(str).str.upper().str.replace("/", "", regex=False).str.replace(":", "", regex=False)
            df = df[mask.isin(wanted)]
            if df.empty:
                log.warning("none of %s matched the screener output", symbols[:5])

        if "symbol" in df.columns:
            df = df.drop_duplicates(subset=["symbol"])
        return df.head(max_symbols).reset_index(drop=True)

    def symbols(self, **kwargs) -> List[str]:
        df = self.screen(**kwargs)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        return [str(s) for s in df[col].tolist()]

    def list_symbols(self) -> List[str]:
        return self.symbols()

    def fetch_ohlcv(self, *args, **kwargs) -> pd.DataFrame:
        raise DataError(
            "tvscreener cannot serve OHLCV history. Use it for universe selection "
            "(--screener tv) and pair it with --data-source ccxt|yfinance|csv for prices."
        )


# --------------------------------------------------------------------- helpers
def _import():
    try:
        import tvscreener  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise DataError("pip install tvscreener   # required for the TradingView screener source") from exc
    return tvscreener


def _pick(enum, *names: str):
    for n in names:
        if n is None:
            continue
        if hasattr(enum, n):
            return getattr(enum, n)
    return None


_COLUMN_MAP = {
    "name": "symbol", "symbol": "symbol", "ticker": "symbol", "s": "symbol",
    "close": "last", "price": "last", "last": "last",
    "change": "change_pct", "change_abs": "change_pct", "perf.1m": "change_pct",
    "volume": "volume", "total_volume": "quote_volume", "turnover": "quote_volume",
    "market_cap_basic": "market_cap", "market_cap": "market_cap",
    "ADX": "adx", "RSI": "rsi", "RSI14": "rsi", "ADX14": "adx",
    "ATR": "atr", "ATR14": "atr",
    "description": "description", "exchange": "exchange", "type": "type",
    "recommendations": "recommendations",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        key = str(c).strip()
        rename[key] = _COLUMN_MAP.get(key, _COLUMN_MAP.get(key.lower(), key.lower()))
    out = out.rename(columns=rename)
    if "symbol" not in out.columns:
        out = out.rename(columns={out.columns[0]: "symbol"})
    if "atr" in out.columns and "last" in out.columns:
        out["atr_pct"] = (out["atr"] / out["last"]).astype("float64")
    return out
