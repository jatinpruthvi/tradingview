"""CSV / Parquet / JSONL OHLCV loader.

This is the adapter to use when you have data from your broker, a paid vendor,
or an export from TradingView. It is deliberately permissive about column names
because every vendor spells them differently.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .base import DataError, slice_range, validate_ohlcv

log = logging.getLogger(__name__)

#: vendor column -> canonical column
ALIASES: Dict[str, str] = {
    "date": "timestamp", "datetime": "timestamp", "time": "timestamp",
    "ts": "timestamp", "timestamp_ms": "timestamp", "epoch": "timestamp",
    "o": "open", "open_price": "open", "price_open": "open",
    "h": "high", "high_price": "high", "price_high": "high",
    "l": "low", "low_price": "low", "price_low": "low",
    "c": "close", "close_price": "close", "price_close": "close", "last": "close",
    "v": "volume", "vol": "volume", "base_volume": "volume", "volume_base": "volume",
    "quote_volume": "quote_volume", "turnover": "quote_volume", "value_traded": "quote_volume",
}


class CsvSource:
    """Load OHLCV from a directory of per-symbol files.

    Layout::

        <root>/<SYMBOL>.csv      (symbol separators / . : get flattened)

    or a single long file with a ``symbol`` column.
    """

    name = "csv"

    def __init__(self, root: str | Path, symbol_column: Optional[str] = None):
        self.root = Path(root)
        self.symbol_column = symbol_column
        if not self.root.exists():
            raise DataError(
                f"CSV data root does not exist: {self.root}. Export bars there with "
                f"`tvlab data --export {self.root}`, or point data.csv_root at the "
                f"directory that holds them."
            )
        self._long: Optional[pd.DataFrame] = None
        if self.root.is_file():
            self._long = self._read_one(self.root)

    # ------------------------------------------------------------------ public
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self._frame_for(symbol, timeframe)
        df = validate_ohlcv(df, symbol)
        return slice_range(df, start, end, limit)

    def list_symbols(self) -> List[str]:
        if self._long is not None:
            col = self.symbol_column or _guess_symbol_column(self._long)
            if col:
                return sorted(map(str, self._long[col].dropna().unique()))
        out = set()
        for p in self.root.glob("*"):
            if p.suffix.lower() not in {".csv", ".parquet", ".pq", ".jsonl", ".json"}:
                continue
            out.add(_TF_SUFFIX.sub("", p.stem))
        return sorted(out)

    # ----------------------------------------------------------------- private
    def _frame_for(self, symbol: str, timeframe: Optional[str] = None) -> pd.DataFrame:
        if self._long is not None:
            col = self.symbol_column or _guess_symbol_column(self._long)
            if not col:
                raise ValueError("long-format CSV has no symbol column; pass symbol_column=")
            sub = self._long[self._long[col].astype(str) == str(symbol)]
            if sub.empty:
                raise DataError(f"symbol {symbol!r} not present in {self.root}")
            return sub.drop(columns=[c for c in (col,) if c != "timestamp"])

        cands = _symbol_filenames(symbol)
        if timeframe:
            # Prefer the exact timeframe, then fall back to any file for the symbol.
            cands = [f"{c}_{timeframe}" for c in cands] + cands
        for cand in cands:
            for ext in (".csv", ".parquet", ".pq", ".jsonl", ".json"):
                p = self.root / f"{cand}{ext}"
                if p.exists():
                    return self._read_one(p)
        raise DataError(
            f"no {timeframe or ''} bars for symbol {symbol!r} under {self.root} "
            f"(tried {[f'{c}{e}' for c in cands[:5] for e in ('.csv', '.parquet')][:6]}); "
            f"available: {self.list_symbols()[:20]}"
        )

    @staticmethod
    def _read_one(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if suffix in (".jsonl", ".json"):
            return pd.read_json(path, lines=(suffix == ".jsonl"))
        # CSV: sniff the separator, tolerate epoch-ms and ISO timestamps
        sep = _sniff_sep(path)
        raw = pd.read_csv(path, sep=sep)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        raw = raw.rename(columns={k: v for k, v in ALIASES.items() if k in raw.columns})
        for col in ("timestamp", "time", "date", "datetime", "ts"):
            if col in raw.columns:
                ts = raw[col]
                if pd.api.types.is_numeric_dtype(ts):
                    unit = "ms" if float(ts.dropna().median() or 0) > 1e11 else "s"
                    raw[col] = pd.to_datetime(ts.astype("float64"), unit=unit, utc=True)
                else:
                    raw[col] = pd.to_datetime(ts, utc=True, errors="coerce")
                break
        if "quote_volume" in raw.columns and "volume" not in raw.columns:
            raw["volume"] = raw["quote_volume"]
        return raw


def _sniff_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        head = fh.readline()
    for sep in (",", ";", "\t", "|"):
        if sep in head:
            return sep
    return ","


def _guess_symbol_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ("symbol", "ticker", "instrument", "pair", "code", "name"):
        if cand in df.columns:
            return cand
    return None


#: A trailing ``_4h`` / ``_1d`` / ``_15m`` on a filename. `tvlab data --export`
#: appends the timeframe so one directory can hold several; matching has to allow
#: for it, and list_symbols() has to strip it or the universe reports symbols that
#: no config will ever name.
_TF_SUFFIX = re.compile(r"_\d+[smhdwSMHDW]$", re.IGNORECASE)


def _symbol_filenames(symbol: str) -> List[str]:
    """Candidate filenames for a symbol, handling / : and . separators.

    Includes the *underscore*-flattened form (``BTC/USDT`` -> ``BTC_USDT``), which
    is what ``tvlab data --export`` writes: it replaces every non-alphanumeric
    character with ``_``. Without it the export -> ``--data-source csv`` round trip
    cannot find its own files, and the failure surfaces as
    "no symbol produced a result" far from the cause.
    """
    flat = symbol.replace("/", "-").replace(":", "-")
    nosuffix = symbol.split(":")[0]
    dotted = symbol.replace("/", ".")
    plain = symbol.replace("/", "")
    base = nosuffix.replace("/", "-")
    underscore = "".join(ch if ch.isalnum() else "_" for ch in symbol)
    underscore_base = "".join(ch if ch.isalnum() else "_" for ch in nosuffix)
    seen, out = set(), []
    for cand in (symbol, flat, dotted, plain, base, nosuffix, base.split(".")[0],
                 underscore, underscore_base):
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out
