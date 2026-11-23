"""Data source protocol, OHLCV validation and timeframe resampling.

Every adapter in this package must return a frame that passes
:func:`validate_ohlcv`. The backtester trusts that contract, so being strict
here is what keeps garbage out of the results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: pandas resample codes, keyed by the timeframe strings used in configs
RESAMPLE_RULE: Dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "1D": "1D",
    "1w": "1W",
}


class DataError(Exception):
    """Raised when a data source returns something unusable."""


@dataclass(frozen=True)
class Bar:
    """A single validated OHLCV bar."""

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataSource(Protocol):
    """Anything that can produce OHLCV bars for a symbol."""

    name: str

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:  # pragma: no cover - protocol
        ...

    def list_symbols(self) -> List[str]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------- validate
def validate_ohlcv(df: pd.DataFrame, symbol: str = "?", fix: bool = True) -> pd.DataFrame:
    """Normalise and sanity-check an OHLCV frame.

    With ``fix=True`` repairable problems (duplicate timestamps, unsorted index,
    high/low envelopes that do not contain the body) are corrected and logged.
    Anything unrepairable raises :class:`DataError`.
    """
    if df is None or len(df) == 0:
        raise DataError(f"{symbol}: empty frame from data source")

    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        for cand in ("timestamp", "time", "date", "datetime", "ts"):
            if cand in out.columns:
                out[cand] = pd.to_datetime(out[cand], utc=True, errors="coerce")
                out = out.set_index(cand)
                break
        else:
            raise DataError(f"{symbol}: no DatetimeIndex and no timestamp column")

    out.index = pd.to_datetime(out.index, utc=True)
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        if set(missing) == {"volume"}:
            out["volume"] = np.nan
        else:
            raise DataError(f"{symbol}: missing required columns {missing}")

    # columns starting with "_" are ground-truth metadata (synthetic regime,
    # injected pattern flags). They are carried through untouched so the
    # validation suite can score detectors against known truth.
    meta = {c: out[c] for c in out.columns if str(c).startswith("_")}

    out = out[OHLCV_COLUMNS].astype("float64")

    if not out.index.is_monotonic_increasing:
        if fix:
            log.warning("%s: index not sorted - sorting", symbol)
            out = out.sort_index()
        else:
            raise DataError(f"{symbol}: index is not sorted")

    dupes = out.index.duplicated(keep="first")
    if dupes.any():
        if fix:
            log.warning("%s: dropping %d duplicate timestamps", symbol, int(dupes.sum()))
            out = out[~dupes]
        else:
            raise DataError(f"{symbol}: duplicate timestamps")

    nan_ohlc = out[["open", "high", "low", "close"]].isna().any(axis=1)
    if nan_ohlc.any():
        n = int(nan_ohlc.sum())
        if fix and n < len(out):
            log.warning("%s: dropping %d rows with NaN in OHLC", symbol, n)
            out = out[~nan_ohlc]
        else:
            raise DataError(f"{symbol}: {n} rows with NaN in OHLC")

    if not np.isfinite(out[["open", "high", "low", "close"]].to_numpy()).all():
        bad = ~np.isfinite(out[["open", "high", "low", "close"]].to_numpy()).all(axis=1)
        if fix and bad.sum() < len(out):
            log.warning("%s: dropping %d non-finite rows", symbol, int(bad.sum()))
            out = out[~bad]
        else:
            raise DataError(f"{symbol}: non-finite values in OHLC")

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataError(f"{symbol}: non-positive prices present")

    # high must contain the body, low must contain the body
    bad_high = out["high"] < out[["open", "close"]].max(axis=1)
    bad_low = out["low"] > out[["open", "close"]].min(axis=1)
    bad_hl = out["high"] < out["low"]
    if (bad_high | bad_low | bad_hl).any():
        n = int((bad_high | bad_low | bad_hl).sum())
        if fix:
            log.warning("%s: repairing %d bars with inconsistent high/low envelope", symbol, n)
            out["high"] = np.maximum.reduce(
                [out["high"].to_numpy(), out["open"].to_numpy(), out["close"].to_numpy()]
            )
            out["low"] = np.minimum.reduce(
                [out["low"].to_numpy(), out["open"].to_numpy(), out["close"].to_numpy()]
            )
        else:
            raise DataError(f"{symbol}: {n} bars where high/low do not contain open/close")

    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)

    for col, series in meta.items():
        out[col] = series.reindex(out.index).to_numpy()

    out.attrs.update(df.attrs or {})
    out.index.name = "timestamp"
    return out


# -------------------------------------------------------------------- resample
def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate a lower-timeframe OHLCV frame up to ``timeframe``."""
    rule = RESAMPLE_RULE.get(timeframe)
    if rule is None:
        raise DataError(f"unsupported timeframe for resampling: {timeframe!r}")
    agg = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    # drop empty buckets (weekends/out-of-session) rather than carrying NaN bars
    return agg.dropna(subset=["open", "high", "low", "close"])


def align_to_execution(bias_df: pd.DataFrame, exec_df: pd.DataFrame) -> pd.DataFrame:
    """Project a higher-timeframe frame onto the execution index without lookahead.

    A bias bar closing at 00:00 on day D is only knowable to execution bars
    stamped >= that close. ``ffill`` after ``reindex`` gives exactly that: each
    execution bar sees the most recent *completed* bias bar.
    """
    rebased = bias_df[["open", "high", "low", "close", "volume"]].reindex(
        exec_df.index.union(bias_df.index)
    )
    filled = rebased.ffill().reindex(exec_df.index)
    return filled


def align_indicator(native: pd.Series, exec_index: pd.DatetimeIndex) -> pd.Series:
    """Align a higher-timeframe INDICATOR onto the execution index, leak-free.

    Compute indicators on native HTF bars, then use this - do NOT ffill HTF
    *prices* onto the execution index and compute indicators there. A daily close
    forward-filled across six 4h bars turns ``ema(close, 50)`` into an 8-day EMA
    rather than a 50-day one, which silently changes what the trend filter means.

    A bias bar stamped ``T`` covers ``[T, T + period)`` and is only complete at
    ``T + period``. So the value is shifted by one HTF bar before alignment:
    execution bars inside day D see day D-1's *completed* value. Skipping that
    shift is the classic multi-timeframe leak.
    """
    if native is None or len(native) == 0:
        return pd.Series(np.nan, index=exec_index)
    shifted = native.shift(1)
    return (shifted.reindex(exec_index.union(shifted.index))
                  .ffill()
                  .reindex(exec_index))


def timeframe_seconds(timeframe: str) -> float:
    rule = RESAMPLE_RULE.get(timeframe, timeframe)
    try:
        return pd.Timedelta(rule).total_seconds()
    except ValueError:
        if rule.endswith("W"):
            return 7 * 86400.0
        if rule.endswith("D"):
            return 86400.0
        raise DataError(f"cannot interpret timeframe {timeframe!r}")


def detect_timeframe(df: pd.DataFrame) -> Optional[str]:
    """Infer the timeframe string from a frame's median bar spacing."""
    if len(df) < 3:
        return None
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return None
    med = diffs.median().total_seconds()
    best, best_err = None, float("inf")
    for tf in RESAMPLE_RULE:
        try:
            secs = timeframe_seconds(tf)
        except DataError:
            continue
        err = abs(np.log(secs / med)) if med > 0 else float("inf")
        if err < best_err:
            best, best_err = tf, err
    return best if best_err < 0.5 else None


def split_ranges(
    start: Optional[str], end: Optional[str], limit: Optional[int], timeframe: str
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[int]]:
    """Normalise user-supplied range arguments."""
    s = pd.Timestamp(start, tz="UTC") if start else None
    e = pd.Timestamp(end, tz="UTC") if end else None
    if s and e and e <= s:
        raise DataError(f"end ({e}) must be after start ({s})")
    if limit and limit <= 0:
        raise DataError("limit must be positive")
    return s, e, limit


def slice_range(
    df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, limit: Optional[int] = None
) -> pd.DataFrame:
    s, e, lim = split_ranges(start, end, limit, detect_timeframe(df) or "1d")
    out = df
    if s is not None:
        out = out[out.index >= s]
    if e is not None:
        out = out[out.index <= e]
    if lim is not None:
        out = out.iloc[-lim:]
    return out


def to_bars(df: pd.DataFrame) -> Iterable[Bar]:
    """Iterate validated frames as :class:`Bar` objects (used by the engine)."""
    for ts, row in zip(df.index, df.itertuples(index=False, name=None)):
        yield Bar(ts, row[0], row[1], row[2], row[3], row[4])


#: Trading hours per year, by market. This is the only thing standing between a
#: correct and a nonsense Sharpe for intraday data: annualising a 4h forex bar
#: with a 6.5h US equity session gives 409 bars/year instead of 1560, which
#: understates annualised volatility - and therefore Sharpe, Sortino and CAGR -
#: by a factor of sqrt(3.8) ~= 1.95.
MARKET_HOURS_PER_YEAR: Dict[str, float] = {
    "crypto": 24.0 * 365,        # 8760 - trades continuously
    "us": 6.5 * 252,             # 1638 - 09:30-16:00 ET
    "india": 6.25 * 252,         # 1575 - 09:15-15:30 IST
    "forex": 24.0 * 5 * 52,      # 6240 - 24h Sunday open to Friday close
}

#: Trading days per year, used for daily-and-coarser bars where the session
#: length is irrelevant.
MARKET_DAYS_PER_YEAR: Dict[str, float] = {
    "crypto": 365.0, "us": 252.0, "india": 252.0, "forex": 260.0,
}

#: config/market names -> the short keys above. The universe config uses long
#: names ("us_equities") while some callers pass short ones ("us").
_MARKET_ALIASES: Dict[str, str] = {
    "crypto": "crypto",
    "us": "us", "us_equities": "us", "america": "us", "us_stock": "us",
    "india": "india", "indian_equities": "india", "india_equities": "india",
    "forex": "forex", "fx": "forex", "metals": "forex", "commodities": "forex",
}


def normalise_market(market: Optional[str]) -> str:
    """Map any market spelling used in this repo onto a canonical key."""
    key = str(market or "crypto").strip().lower()
    return _MARKET_ALIASES.get(key, "crypto" if "crypto" in key else "us")


def bars_per_year(timeframe: str, market: str = "crypto") -> float:
    """Annualisation factor: how many bars of this timeframe occur in a year.

    Session length is market-specific - see :data:`MARKET_HOURS_PER_YEAR`. Using
    one equity session for every non-crypto market is a silent ~2x error on every
    risk-adjusted metric for FX and metals.
    """
    m = normalise_market(market)
    secs = timeframe_seconds(timeframe)
    if secs >= 86400:
        return MARKET_DAYS_PER_YEAR[m]
    hours_per_year = MARKET_HOURS_PER_YEAR[m]
    return hours_per_year * 3600.0 / secs
