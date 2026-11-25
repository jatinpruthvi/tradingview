"""NSE delivery data, *joined onto* prices that come from somewhere else.

Why this is a join and not a data source
----------------------------------------
``jugaad_data`` publishes daily NSE rows with two columns no other free source
gives you, and no intraday bars at all. Swapping it in for prices would replace
Yahoo's split/dividend-adjusted series with NSE's unadjusted official close, and
every corporate action would then look like a cliff in the chart. So prices stay
where they are and only these two columns are attached, keyed by session date:

``DELIVERY %``
    The share of the day's traded volume that actually took delivery. In India
    this is the closest free proxy for whether a move was position-taking or
    intraday churn.
``VALUE / NO OF TRADES``
    Average trade size in rupees. A better liquidity gate than share volume: at
    Rs 50 a share, 1M shares is Rs 5 crore; at Rs 3,000 it is Rs 300 crore.

Both are published in the evening bhavcopy, so for session ``D`` they are knowable
before ``D+1`` opens and **not** during ``D``. A strategy reading them must shift
by one session; this module deliberately attaches them to their own session's row
and stops there, because hiding the shift behind a forward fill is how a
backtest ends up reading tonight's bhavcopy at 09:30 this morning.

Two measured traps in jugaad-data 0.37.4, both pinned by tests:

* ``DATE`` arrives as ``datetime64`` stamped **18:30 UTC**, i.e. IST midnight
  shifted back by the offset. Taking ``.dt.date`` puts every row on the previous
  calendar day, which silently shifts the join by one session - the difference
  between a causal filter and a lookahead one.
* ``series="EQ"`` returns the ``T0`` (T+0 settlement) rows as well, duplicating
  each date.

The join leaves unmatched sessions as ``NaN``. It never forward-fills: a missing
bhavcopy is missing information, not yesterday's number.

Columns are ``_``-prefixed because ``validate_ohlcv`` carries only OHLCV plus
``_``-prefixed metadata through a resample/store round trip; a bare
``delivery_pct`` column would be silently dropped by the store.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .base import DataError, detect_timeframe

log = logging.getLogger(__name__)

IST = "Asia/Kolkata"

#: A day is a day: frames whose median bar spacing is under this are not daily
#: and delivery data cannot be joined to them.
_DAILY_SPACING = pd.Timedelta(hours=20)

#: Below this share of matched sessions the join is reported as unreliable.
MIN_COVERAGE = 0.5

#: ``fetch(symbol, start, end, series) -> DataFrame`` mirroring
#: ``jugaad_data.nse.stock_df``. Injectable so the join is testable offline.
DeliveryFetch = Callable[[str, date, date, str], pd.DataFrame]


def _stock_df():
    """Import ``jugaad_data.nse.stock_df`` or explain how to install it."""
    try:
        from jugaad_data.nse import stock_df
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise DataError(
            "the NSE delivery source needs the optional jugaad-data package: "
            'pip install "tvlab[delivery]"  (or: pip install jugaad-data). '
            "The rest of tvlab runs without it."
        ) from exc
    return stock_df


def _default_fetch(symbol: str, start: date, end: date, series: str) -> pd.DataFrame:
    return _stock_df()(symbol=symbol, from_date=start, to_date=end, series=series)


def _as_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, pd.Timestamp):
        return v
    ts = pd.Timestamp(v)
    return ts.tz_convert(IST).date() if ts.tz is not None else ts.date()


def session_dates(stamps: Any) -> pd.DatetimeIndex:
    """NSE session dates from jugaad's ``DATE`` column, timezone-proof.

    Measured on 0.37.4: ``DATE`` is a ``dict``-free numpy timestamp at 18:30 UTC
    representing IST midnight, so it has to be shifted forward to the IST date
    before it is normalised. A plain naive midnight (another version, or the raw
    JSON before conversion) is left alone - shifting that one would move every
    row a day forward and invert the join.
    """
    s = pd.to_datetime(pd.Series(stamps).reset_index(drop=True))
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(IST).dt.tz_localize(None)
    elif float((s.dt.hour == 18).mean()) > 0.5:
        s = s.dt.tz_localize("UTC").dt.tz_convert(IST).dt.tz_localize(None)
    return pd.DatetimeIndex(s.dt.normalize().to_numpy(), name="date")


def fetch_delivery(
    symbol: str,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    *,
    series: str = "EQ",
    fetch: Optional[DeliveryFetch] = None,
) -> pd.DataFrame:
    """Per-session delivery percentage and average trade size, indexed by date.

    Returns a frame with ``delivery_pct`` (0-100) and ``avg_trade_value`` (rupees)
    indexed by a tz-naive session date. Raises :class:`DataError` rather than
    returning an empty frame: an empty frame would silently become an all-NaN
    column, which reads as "this symbol has no delivery data" when in fact the
    fetch failed.
    """
    s = _as_date(start) or date(2000, 1, 1)
    e = _as_date(end) or date.today()
    if e < s:
        raise DataError(f"{symbol}: delivery window ends before it starts ({s}..{e})")

    raw = (fetch or _default_fetch)(symbol, s, e, series)
    df = pd.DataFrame(raw)
    if df.empty:
        raise DataError(f"{symbol}: NSE returned no delivery rows for {s}..{e}")

    for col in ("DATE", "DELIVERY %", "VALUE", "NO OF TRADES"):
        if col not in df.columns:
            raise DataError(
                f"{symbol}: NSE delivery payload has no {col!r} column "
                f"(got {sorted(df.columns)}). The endpoint's shape changed."
            )

    # `series="EQ"` comes back as EQ *and* T0 (T+0 settlement) rows for the same
    # session. Keeping both doubles the history and gives every date two rows.
    if "SERIES" in df.columns:
        keep = df["SERIES"].astype(str).str.upper().str.strip() == series.upper()
        df = df[keep]
    if df.empty:
        raise DataError(f"{symbol}: no {series} rows in the delivery payload for {s}..{e}")

    days = session_dates(df["DATE"])
    out = pd.DataFrame(
        {
            "delivery_pct": pd.to_numeric(df["DELIVERY %"], errors="coerce").to_numpy(),
            "avg_trade_value": _avg_trade_value(df),
        },
        index=days,
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["delivery_pct"] = out["delivery_pct"].clip(lower=0.0, upper=100.0)
    return out


def _avg_trade_value(df: pd.DataFrame) -> np.ndarray:
    """VALUE / NO OF TRADES, with a zero trade count left as NaN."""
    value = pd.to_numeric(df["VALUE"], errors="coerce").to_numpy(dtype="float64")
    trades = pd.to_numeric(df["NO OF TRADES"], errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(value) & np.isfinite(trades) & (trades > 0)
    out = np.full(len(df), np.nan)
    out[ok] = value[ok] / trades[ok]
    return out


def frame_session_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Session dates for a price frame's index, in IST."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    return idx.normalize()


def is_daily(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return True
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return True
    return bool(diffs.median() >= _DAILY_SPACING)


def attach_delivery(
    df: pd.DataFrame,
    symbol: str,
    *,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    fetch: Optional[DeliveryFetch] = None,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """Join delivery columns onto ``df`` without touching any price column.

    Adds ``_delivery_pct`` and ``_avg_trade_value`` on each session's own row and
    records ``attrs["delivery_coverage"]`` (matched sessions / total). Unmatched
    sessions stay ``NaN`` - they are not filled, because the bhavcopy for a
    holiday or a suspended symbol genuinely has no number.

    Non-daily frames are returned unchanged: delivery is per session, and there
    is no honest way to put one number on 75 five-minute bars without deciding
    when it became known.
    """
    if df is None or len(df) == 0:
        return df
    tf = detect_timeframe(df)
    if not is_daily(df):
        log.warning(
            "%s: delivery data is per session; %s bars are returned without it",
            symbol, tf or "sub-daily",
        )
        return df

    key = frame_session_dates(df.index)
    d = fetch_delivery(
        symbol,
        start or key[0].date(),
        end or key[-1].date(),
        fetch=fetch,
    )
    aligned = d.reindex(key)

    out = df.copy()
    out["_delivery_pct"] = aligned["delivery_pct"].to_numpy()
    out["_avg_trade_value"] = aligned["avg_trade_value"].to_numpy()

    coverage = float(out["_delivery_pct"].notna().mean())
    out.attrs.update(df.attrs or {})
    out.attrs["delivery_coverage"] = coverage
    if coverage == 0.0:
        log.warning(
            "%s: no session in %s..%s matched an NSE delivery row - check the "
            "symbol's series and the date range", symbol, key[0].date(), key[-1].date(),
        )
    elif coverage < min_coverage:
        log.warning(
            "%s: only %.0f%% of sessions matched a delivery row; the column is "
            "mostly empty and any filter built on it is mostly missing data",
            symbol, coverage * 100.0,
        )
    return out


class DeliveryAugmentedSource:
    """A ``DataSource`` that serves another source's prices, plus delivery data.

    Wrap whatever already fetches prices (typically
    :class:`tvlab.data.nse.NseSource`, i.e. Yahoo's adjusted series)::

        src = DeliveryAugmentedSource(NseSource())
        df = src.fetch_ohlcv("RELIANCE", "1d", limit=500)
        df["_delivery_pct"]      # same session
        df["_delivery_pct"].shift(1)   # the only causal form for a same-day trade

    Every other attribute is delegated to the inner source, so universe and
    snapshot calls keep working.
    """

    name = "nse_delivery"

    def __init__(
        self,
        inner: Any,
        *,
        market: str = "indian_equities",
        fetch: Optional[DeliveryFetch] = None,
    ):
        if inner is None:
            raise DataError("DeliveryAugmentedSource needs an inner data source")
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "fetch", fetch)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.inner.fetch_ohlcv(symbol, timeframe, start=start, end=end, limit=limit)
        return attach_delivery(df, symbol, start=start, end=end, fetch=self.fetch)

    def list_symbols(self):
        return self.inner.list_symbols()

    def __getattr__(self, item: str) -> Any:
        # `inner` is set through object.__setattr__; guarding the lookup keeps a
        # missing attribute from recursing through this method forever.
        inner = object.__getattribute__(self, "inner")
        return getattr(inner, item)

    def __repr__(self) -> str:
        return f"DeliveryAugmentedSource(inner={self.inner!r})"
