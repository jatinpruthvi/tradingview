"""Network data adapters: ccxt (crypto), yfinance (equities/FX/indices).

Both are *optional* imports. They are not in requirements.txt as hard deps and
this sandbox has no route to exchanges, so the adapters fail loudly with an
actionable message instead of silently returning empty frames.

Install with::

    pip install ccxt          # crypto, 100+ exchanges
    pip install yfinance      # US/Indian equities, FX, indices
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from .base import DataError, resample, slice_range, validate_ohlcv

log = logging.getLogger(__name__)

#: ccxt uses raw timeframe codes already ("1m","4h","1d")
CCXT_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}

YF_INTERVALS = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "1h", "1h": "1h", "1d": "1d", "1D": "1d", "1w": "1wk",
}

#: Period to request when the caller passes neither dates nor a bar count.
#:
#: yfinance's own default is ``period="1mo"``, which for daily bars is ~22 rows -
#: below the store's 60-bar floor and far short of a 1200-bar scan window, so every
#: caller that did not pin dates silently lost its higher-timeframe bias history
#: (the trend filter then disabled itself instead of complaining). Each value is the
#: longest history the interval is actually allowed to serve.
YF_DEFAULT_PERIOD = {
    "1m": "5d", "2m": "1mo", "5m": "1mo", "15m": "2mo", "30m": "2mo",
    "1h": "2y", "1d": "5y", "1wk": "10y",
}


def _require(module: str, pip_name: Optional[str] = None):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            f"the '{module}' package is required for this data source but is not installed.\n"
            f"    pip install {pip_name or module}\n"
            f"  (original error: {exc})"
        ) from exc


class CcxtSource:
    """OHLCV from any ccxt-supported exchange.

    Paginates automatically and respects the exchange's rate limiter. Higher
    timeframes that the exchange does not serve natively are resampled up from
    the highest native one.
    """

    name = "ccxt"

    def __init__(
        self,
        exchange: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        sandbox: bool = False,
        max_retries: int = 4,
    ):
        ccxt = _require("ccxt")
        klass = getattr(ccxt, exchange, None)
        if klass is None:
            raise DataError(f"ccxt has no exchange named {exchange!r}")
        self._ccxt = ccxt
        self.exchange_id = exchange
        config: Dict[str, object] = {"enableRateLimit": True, "timeout": 30_000}
        if api_key:
            config["apiKey"] = api_key
        if secret:
            config["secret"] = secret
        self.client = klass(config)
        if sandbox:
            self.client.set_sandbox_mode(True)
        self.max_retries = max_retries
        self._markets: Optional[Dict] = None

    # ------------------------------------------------------------------ public
    def load_markets(self) -> Dict:
        if self._markets is None:
            self._markets = self._retry(self.client.load_markets)
        return self._markets

    def list_symbols(self, quote: str = "USDT", spot: bool = True) -> List[str]:
        markets = self.load_markets()
        out = []
        for sym, m in markets.items():
            if spot and not m.get("spot"):
                continue
            if quote and m.get("quote") != quote:
                continue
            if not m.get("active", True):
                continue
            out.append(sym)
        return sorted(out)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        self.load_markets()
        if symbol not in self._markets:
            raise DataError(
                f"{self.exchange_id} does not list {symbol!r}. "
                f"Did you mean one of {[s for s in self.list_symbols() if symbol.split('/')[0] in s][:8]}?"
            )

        native_tf, needs_resample = self._resolve_timeframe(timeframe)
        since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else None
        end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
        if since is None:
            want = limit or 2_000
            since = int((end_ts - pd.Timedelta(seconds=want * _tf_seconds(native_tf))).timestamp() * 1000)

        rows = self._paginate(symbol, native_tf, since, int(end_ts.timestamp() * 1000), limit)
        if not rows:
            raise DataError(f"{self.exchange_id} returned no bars for {symbol} {native_tf}")

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = validate_ohlcv(df, symbol)
        df = df[~df.index.duplicated(keep="last")]

        if needs_resample:
            df = resample(df, timeframe)
            df = validate_ohlcv(df, symbol)
        return slice_range(df, start, end, limit)

    # ----------------------------------------------------------------- private
    def _resolve_timeframe(self, timeframe: str) -> tuple[str, bool]:
        """Return (native timeframe to fetch, whether resampling is needed)."""
        markets = self.load_markets()
        supported = set()
        for m in markets.values():
            supported |= set((m.get("timeframes") or {}).keys())
        if not supported:
            supported = CCXT_TIMEFRAMES
        if timeframe in supported:
            return timeframe, False
        # pick the largest supported timeframe below the requested one
        target = _tf_seconds(timeframe)
        candidates = sorted((t for t in supported if _tf_seconds(t) <= target),
                            key=_tf_seconds, reverse=True)
        if not candidates:
            raise DataError(
                f"{self.exchange_id} cannot serve {timeframe}: smallest supported is "
                f"{min(supported, key=_tf_seconds)}"
            )
        base = candidates[0]
        if target % _tf_seconds(base) != 0:
            log.warning("resampling %s -> %s is not an exact multiple; bars may be uneven", base, timeframe)
        return base, True

    def _paginate(self, symbol: str, timeframe: str, since: int, end_ms: int,
                  limit: Optional[int]) -> List[list]:
        batch = 1000 if limit is None else min(1000, max(2, limit))
        out: List[list] = []
        cursor = since
        for _ in range(400):  # hard cap: ~400k bars
            chunk = self._retry(self.client.fetch_ohlcv, symbol, timeframe, cursor, batch)
            if not chunk:
                break
            out.extend(chunk)
            last_ts = chunk[-1][0]
            if last_ts >= end_ms or len(chunk) < 2:
                break
            nxt = last_ts + int(_tf_seconds(timeframe) * 1000)
            if nxt <= cursor:
                break
            cursor = nxt
            if limit and len(out) >= limit:
                out = out[-limit:]
                break
            time.sleep(0.0)  # rate limiter lives inside ccxt
        return [r for r in out if r[0] <= end_ms]

    def _retry(self, fn, *args, **kwargs):
        last: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - ccxt raises many types
                last = exc
                transient = any(k in type(exc).__name__ for k in
                                ("Network", "Timeout", "RateLimit", "ExchangeNotAvailable", "DDoS"))
                wait = min(2 ** attempt * 1.5, 30)
                if not transient and attempt > 0:
                    break
                log.warning("%s: %s - retrying in %.1fs", self.exchange_id, exc, wait)
                time.sleep(wait)
        raise DataError(f"{self.exchange_id} request failed after {self.max_retries} tries: {last}")


class YFinanceSource:
    """OHLCV via yfinance: US equities/ETFs, NSE (``.NS``), FX, indices."""

    name = "yfinance"

    #: yfinance refuses 1m data older than ~30 days and >7 days per request
    INTRADAY_LIMIT_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730}

    def __init__(self, session=None, adjust: str = "auto"):
        self._yf = _require("yfinance")
        self.session = session
        self.adjust = adjust

    def list_symbols(self) -> List[str]:
        raise DataError("yfinance has no symbol listing endpoint; name your symbols in the market config")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        interval = YF_INTERVALS.get(timeframe)
        if interval is None:
            # resample up from the closest supported interval. The base fetch has to
            # ask for proportionally more bars than `limit`: the limit counts bars of
            # the *requested* timeframe, and without this the request fell through to
            # yfinance's 1-month default, handing a 1200-bar 4h scan ~130 bars.
            tf_secs = _tf_seconds(timeframe)
            base = "1d" if tf_secs >= 86400 else "1h"
            extra: Dict[str, object] = {}
            if not start and not end:
                if limit:
                    ratio = max(tf_secs / _tf_seconds(base), 1.0)
                    extra["period"] = _period_for(max(int(round(limit * ratio)), limit), base)
                else:
                    extra["period"] = YF_DEFAULT_PERIOD.get(YF_INTERVALS[base], "1y")
            df = self._download(symbol, base, start, end, **extra)
            return validate_ohlcv(resample(df, timeframe), symbol)

        # Only `interval` is positional in the _download call below. `start`/`end`
        # belong in kwargs: they used to be passed both ways, which raised TypeError
        # before any HTTP request, so every dated fetch failed.
        kwargs: Dict[str, object] = {"auto_adjust": self.adjust != "raw"}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        if not start:
            if limit:
                kwargs["period"] = _period_for(limit, timeframe)
            elif not end:
                kwargs["period"] = YF_DEFAULT_PERIOD.get(interval, "1y")
        cap_days = self.INTRADAY_LIMIT_DAYS.get(interval)
        if cap_days and start:
            s = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start)
            e = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
            if (e - s).days > cap_days:
                log.warning(
                    "yfinance caps %s history at ~%d days per request; %s..%s will be truncated. "
                    "Use a paid vendor or broker export for longer intraday history.",
                    interval, cap_days, s.date(), e.date(),
                )
        df = self._download(symbol, interval, **kwargs)
        df = validate_ohlcv(df, symbol)
        return slice_range(df, start, end, limit)

    def _download(self, symbol, interval, start=None, end=None, **kwargs) -> pd.DataFrame:
        import yfinance as yf  # noqa: F401  (ensures import error surfaces with guidance)

        args: Dict[str, object] = {"tickers": symbol, "progress": False, "threads": False}
        if interval:
            args["interval"] = interval
        if start:
            args["start"] = start
        if end:
            args["end"] = end
        args.update(kwargs)
        raw = self._yf.download(**args)
        if raw is None or raw.empty:
            raise DataError(
                f"yfinance returned nothing for {symbol!r}. Check the ticker "
                f"(Indian equities need the .NS/.BO suffix, e.g. RELIANCE.NS)."
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        rename = {"adj close": "close", "adj_close": "close", "stock": "volume"}
        raw = raw.rename(columns=rename)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")
        return raw


def _period_for(limit: int, timeframe: str) -> str:
    secs = _tf_seconds(timeframe) * limit
    days = secs / 86400
    for period, d in (("1d", 1), ("5d", 5), ("1mo", 30), ("3mo", 90), ("6mo", 180),
                      ("1y", 365), ("2y", 730), ("5y", 1825), ("10y", 3650), ("max", 1e9)):
        if days <= d:
            return period
    return "max"


def _tf_seconds(timeframe: str) -> float:
    from .base import timeframe_seconds

    return timeframe_seconds(timeframe)
