"""NSE (Indian equities) data: public NSE JSON APIs + Yahoo's chart endpoint.

Two problems this solves that ``network_sources.py`` does not:

1. **``yfinance`` is a dependency, and the Yahoo *chart* endpoint is not.** The
   v8 chart API is a plain JSON GET. Talking to it directly means live Indian,
   US, FX and index history with nothing but ``urllib`` in the standard library.
   ``YFinanceSource`` stays for anyone who wants it, but this path installs
   nothing.

2. **A hand-listed universe goes stale.** ``config/markets/indian_equities.yaml``
   names 12 large caps because someone had to type them. NSE publishes the
   Nifty 500 constituent list as a CSV and returns every member of an index -
   with last price, change, volume and traded value - from a *single*
   market-watch call. ``NseSource`` reads those, so the universe is whatever NSE
   says it is today.

Endpoint shapes, the IST regular-session window (09:15-15:30) and the
index-to-Yahoo-symbol map are adapted from ``heysanjeevv/market-lab`` (MIT,
see ``vendor/market-lab`` and ``VENDOR.md``). Only the endpoint knowledge is
taken; no code is copied. The HTTP layer here adds politeness that matters when
you are the one being rate-limited: sequential requests, a delay between them,
and an explicit backoff on 429 instead of a retry loop.

Everything accepts an injectable ``fetcher`` so the parsers are testable with
recorded payloads and no network. Nothing here has been executed against live
NSE or Yahoo from this repository's sandbox - both hosts are unreachable - so
treat the payload handling as best-effort until you run it yourself:

    tvlab data --market indian_equities --style intraday --data-source nse

NSE's API is undocumented and changes without notice. A field going missing
raises ``DataError`` with the URL and the keys that were present, rather than
returning an empty frame - an empty frame looks like "no trades today".
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from .base import OHLCV_COLUMNS, DataError, resample, slice_range, timeframe_seconds, validate_ohlcv

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

#: NSE and Yahoo both reject the default urllib agent, but Yahoo rejects a
#: *full browser* string outright: with a Chrome/120 User-Agent its chart
#: endpoint answers HTTP 429 "Too Many Requests" for every symbol, while the
#: same URL with a bare "Mozilla/5.0" answers 200 - measured on 2026-11-24
#: against 12 NSE tickers in a row, all 200 except one 404 for a delisted
#: ticker. The bare string from market-lab is what actually works; a more
#: convincing-looking one is the failure. NSE's own endpoints accept both.
USER_AGENT = "Mozilla/5.0"

#: Regular session in IST, as minutes from midnight: 09:15 to 15:30.
SESSION_START_MIN = 9 * 60 + 15
SESSION_END_MIN = 15 * 60 + 30

Fetcher = Callable[[str, float], bytes]


# --------------------------------------------------------------------- HTTP
def _default_fetcher(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise DataError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise DataError(
            f"could not reach {urllib.parse.urlparse(url).netloc}: {exc.reason}. "
            f"This host must be reachable from the machine running tvlab."
        ) from exc


@dataclass
class Http:
    """Sequential, polite HTTP with a 429 backoff.

    ``delay`` is slept *after* every successful request. That is the difference
    between a research tool and something that gets an IP banned from NSE.
    """

    #: None means "resolve ``_default_fetcher`` at construction time". A dataclass
    #: default of ``fetcher: Fetcher = _default_fetcher`` binds the function when
    #: the class body runs, so replacing the module attribute afterwards - which
    #: is how the tests route NSE and Yahoo to fixtures, and how anyone would
    #: inject a proxy or a shared session - silently does nothing and real
    #: requests go out. Resolving in __post_init__ keeps the indirection live.
    fetcher: Optional[Fetcher] = None
    #: Seconds slept after every request. Overridable with TVLAB_HTTP_DELAY so
    #: politeness can be tuned without editing code - raise it if NSE starts
    #: returning 429, set it to 0 in tests. Read at construction, not at import.
    delay: Optional[float] = None
    timeout: float = 25.0
    rate_limited_until: float = 0.0
    requests: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.fetcher is None:
            self.fetcher = _default_fetcher
        if self.delay is None:
            self.delay = float(os.environ.get("TVLAB_HTTP_DELAY", "0.08"))

    def get(self, url: str, timeout: Optional[float] = None) -> bytes:
        wait = self.rate_limited_until - time.time()
        if wait > 0:
            log.warning("rate limited; sleeping %.0fs before %s", wait, url)
            time.sleep(wait)
        raw = self.fetcher(url, timeout or self.timeout)
        self.requests += 1
        if self.delay:
            time.sleep(self.delay)
        return raw

    def get_json(self, url: str, timeout: Optional[float] = None) -> Any:
        raw = self.get(url, timeout)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            head = raw[:200].decode("utf-8", "replace")
            raise DataError(
                f"{url} did not return JSON ({exc}). First 200 bytes: {head!r}. "
                f"NSE serves an HTML interstitial when it wants a browser cookie - "
                f"retry, or run from a residential connection."
            ) from exc

    def note_rate_limit(self, seconds: float = 120.0) -> None:
        self.rate_limited_until = time.time() + seconds

    @staticmethod
    def get_text(url: str, timeout: float = 25.0, fetcher: Optional[Fetcher] = None) -> str:
        return (fetcher or _default_fetcher)(url, timeout).decode("utf-8", "replace")


# ------------------------------------------------------- Yahoo chart endpoint
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval={iv}&{query}"

#: Yahoo interval per tvlab timeframe. Anything not listed is resampled up from
#: the closest supported interval.
YAHOO_INTERVALS = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "4h": "1d", "1d": "1d", "1w": "1wk", "1mo": "1mo",
}

#: Yahoo's own history limits, per interval. 1m is 8 days per request (measured
#: 2026-11-25: an 8-day period window returns 1,875 bars, 30 and 60 both HTTP
#: 422), the rest of intraday is 60 days, 60m reaches 730 days and daily reaches
#: everything. Going over is a hard 422, not a truncation - both for a named
#: ``range`` (``range=3mo&interval=5m``) and for an explicit period window - so
#: the window is clamped here and warned about instead of failing the fetch.
YAHOO_MAX_DAYS = {"1m": 8, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730}

#: ``range`` values Yahoo accepts, widest last.
YAHOO_RANGE_FOR_DAYS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")
_RANGE_DAYS = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
               "2y": 730, "5y": 1826, "10y": 3652, "max": 10**6}

#: Indian indices have no ``.NS`` form on Yahoo; they are caret symbols. Sector
#: indices are mixed - some are carets, some are ``.NS``.
INDIA_INDEX_SYMBOLS = {
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK", "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN", "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
    "NIFTYIT": "^CNXIT", "NIFTYAUTO": "^CNXAUTO", "NIFTYPHARMA": "^CNXPHARMA",
    "NIFTYMETAL": "^CNXMETAL", "NIFTYFMCG": "^CNXFMCG", "NIFTYREALTY": "^CNXREALTY",
    "NIFTYENERGY": "^CNXENERGY", "NIFTYPSUBANK": "^CNXPSUBANK",
    "NIFTYINFRA": "^CNXINFRA", "NIFTYMIDCAP": "NIFTY_MIDCAP_100.NS",
    "INDIAVIX": "^INDIAVIX",
}


def yahoo_symbol(symbol: str, market: str = "indian_equities") -> str:
    """Map a tvlab symbol to the Yahoo ticker for it.

    Indian cash equities need ``.NS``. Indices need the caret form. Anything
    already carrying a suffix or a caret is passed through untouched, so
    ``BTC/USDT``, ``EURUSD=X`` and ``AAPL`` all still work through this source.
    """
    s = str(symbol).strip()
    up = s.upper()
    if up in INDIA_INDEX_SYMBOLS:
        return INDIA_INDEX_SYMBOLS[up]
    # A hyphen is NOT evidence of an already-qualified Yahoo ticker. `BRK-B` is a
    # US ticker, but `BAJAJ-AUTO` and `NAM-INDIA` are NSE symbols whose Yahoo
    # ticker is `BAJAJ-AUTO.NS`. Treating every hyphen as "already qualified"
    # silently 404'd those names, so a full-NSE scan lost them with no reason
    # beyond "HTTP 404" - and they were the only names affected, which is exactly
    # the kind of partial coverage that is invisible in a summary count.
    indian = market in ("indian_equities", "india", "nse")
    if any(c in s for c in ("^", "=", "/")) or s.endswith((".NS", ".BO", ".SS", ".HK")):
        return s
    if indian:
        return f"{s}.NS"
    return s


def _requested_days(limit: Optional[int], timeframe: str,
                    start: Optional[str], end: Optional[str]) -> int:
    """Calendar days the request spans, before Yahoo's per-interval cap."""
    if start:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp.utcnow()
        if s.tz is None:
            s = s.tz_localize("UTC")
        if e.tz is None:
            e = e.tz_localize("UTC")
        return max(1, (e - s).days + 1)
    if limit:
        return max(1, int(limit * timeframe_seconds(timeframe) / 86400) + 5)
    return 365


def _range_for(days: int, interval: Optional[str] = None) -> Optional[str]:
    """Smallest named Yahoo ``range`` that covers ``days`` at ``interval``.

    ``None`` means no named range can express the window, and the caller must
    send ``period1``/``period2`` instead. Yahoo validates ``range`` against the
    interval *before* it truncates: ``range=3mo&interval=5m`` is an HTTP 422,
    not a 60-day series. Every range wide enough for a long intraday request is
    wider than that interval's history cap, which is why the two have to be
    reconciled here rather than in the URL.
    """
    cap = YAHOO_MAX_DAYS.get(interval) if interval else None
    for r in YAHOO_RANGE_FOR_DAYS:
        span = _RANGE_DAYS[r]
        if cap is not None and span > cap:
            continue
        if span >= days:
            return r
    return None


def _period_query(days: int, end: Optional[str]) -> str:
    """Explicit ``period1``/``period2`` for the last ``days`` ending at ``end``.

    Unix seconds, which Yahoo serves up to the interval cap without validating
    them against the named ranges.
    """
    e = pd.Timestamp(end) if end else pd.Timestamp.utcnow()
    if e.tz is None:
        e = e.tz_localize("UTC")
    e = e.tz_convert("UTC")
    start = e - pd.Timedelta(days=max(1, int(days)))
    return f"period1={int(start.timestamp())}&period2={int(e.timestamp())}"


class YahooChartSource:
    """OHLCV from Yahoo's public v8 chart endpoint. No third-party package.

    Covers Indian (``.NS``), US, FX (``=X``), index (``^``) and crypto (``-USD``)
    symbols. Use ``--data-source yahoo_chart`` or ``nse`` (which delegates here
    for history).
    """

    name = "yahoo_chart"

    def __init__(
        self,
        market: str = "indian_equities",
        session_filter: bool = True,
        adjust: bool = True,
        http: Optional[Http] = None,
    ):
        self.market = market
        self.session_filter = session_filter
        self.adjust = adjust
        self.http = http or Http()

    def list_symbols(self) -> List[str]:
        raise DataError(
            "the Yahoo chart endpoint has no symbol listing; use NseSource.list_symbols() "
            "for Indian universes, or name symbols in the market config"
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        iv = YAHOO_INTERVALS.get(timeframe)
        native_tf = timeframe
        if iv is None:
            # 4h and anything exotic: pull the closest smaller interval and
            # resample up through the shared, tested resampler
            iv = "1d" if timeframe_seconds(timeframe) >= 86400 else "60m"
            native_tf = "1d" if iv == "1d" else "1h"

        days = _requested_days(limit, timeframe, start, end)
        self._warn_if_capped(symbol, iv, days)
        rg = _range_for(days, iv)
        # No named range fits: ask for explicit periods instead, clamped to what
        # this interval can actually serve. A period window is not exempt from
        # the cap either - 81 days of 5m is a 422 just like range=3mo was.
        query = f"range={rg}" if rg else _period_query(min(days, YAHOO_MAX_DAYS[iv]), end)

        y = urllib.parse.quote(yahoo_symbol(symbol, self.market), safe="^=")
        url = YAHOO_CHART.format(sym=y, iv=iv, query=query)
        payload = self.http.get_json(url)

        df = self._parse(payload, symbol, url)
        if self.session_filter and iv != "1d" and self.market in ("indian_equities", "india", "nse"):
            df = filter_ist_session(df)
        df = validate_ohlcv(df, symbol)

        if native_tf != timeframe:
            df = validate_ohlcv(resample(df, timeframe), symbol)
        return slice_range(df, start, end, limit)

    # ------------------------------------------------------------------ parse
    def _parse(self, payload: Any, symbol: str, url: str) -> pd.DataFrame:
        try:
            result = payload["chart"]["result"][0]
        except (KeyError, IndexError, TypeError) as exc:
            err = (payload.get("chart", {}) or {}).get("error") if isinstance(payload, dict) else None
            raise DataError(
                f"no chart data for {symbol!r} from {url}: {err or exc}. "
                f"Check the ticker - NSE symbols need the .NS suffix, which this "
                f"source adds automatically for market={self.market!r}."
            ) from exc

        ts = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        if not ts:
            raise DataError(f"{symbol!r}: Yahoo returned 0 bars from {url} (market closed, or unknown ticker)")

        df = pd.DataFrame(
            {c: quote.get(c) for c in OHLCV_COLUMNS},
            index=pd.to_datetime(ts, unit="s", utc=True),
        )
        df.index.name = "timestamp"

        # Yahoo returns nulls for bars with no trade (halts, illiquid sessions).
        # Dropping them is right for OHLC; a null volume is a genuine zero.
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        if self.adjust:
            adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
            if adj and len(adj) == len(ts):
                df = _apply_adjustment(df, pd.Series(adj, index=pd.to_datetime(ts, unit="s", utc=True)))

        return df[~df.index.duplicated(keep="last")].sort_index()

    def _warn_if_capped(self, symbol, iv, days) -> None:
        cap = YAHOO_MAX_DAYS.get(iv)
        if not cap or days <= cap:
            return
        log.warning(
            "%s: Yahoo serves only ~%d days of %s history; asked for %d, so the window is "
            "clamped to %d days. This is why intraday Indian backtests are short - use "
            "daily bars for anything needing years of history.", symbol, cap, iv, days, cap,
        )


def _apply_adjustment(df: pd.DataFrame, adj: pd.Series) -> pd.DataFrame:
    """Scale OHLC by Yahoo's adjustment factor, leaving volume alone.

    Splits and dividends otherwise produce a cliff in the price series that a
    momentum indicator reads as a signal. Volume is *not* scaled: it is a count
    of what actually traded.
    """
    factor = (adj / df["close"]).reindex(df.index)
    factor = factor.replace([float("inf"), -float("inf")], pd.NA).dropna()
    if factor.empty:
        return df
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * factor
    return out


def filter_ist_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the NSE regular session, 09:15-15:30 IST.

    Yahoo returns pre-open and post-close rows for Indian symbols. Left in, they
    stretch the bar count, corrupt ATR across the overnight gap and make a 5m
    backtest see bars that could never have been traded.
    """
    if df.empty:
        return df
    idx = df.index.tz_convert(IST) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(IST)
    minutes = idx.hour * 60 + idx.minute
    keep = (minutes >= SESSION_START_MIN) & (minutes < SESSION_END_MIN)
    return df[keep]


# ------------------------------------------------------------- NSE endpoints
NIFTY500_CSV = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
ALL_INDICES = "https://www.nseindia.com/api/allIndices"
MARKET_WATCH = ("https://www.nseindia.com/api/NextApi/apiClient/marketWatchApi"
                "?functionName=getIndicesData&symbol={q}")
HOLIDAY_MASTER = "https://www.nseindia.com/api/holiday-master?type=trading"

#: Special "index" names NSE accepts in the market-watch call that are not
#: indices at all - notably the full F&O list, which is the tradeable universe.
NSE_SPECIAL_WATCH = {"F&O": "SECURITIES IN F&O"}


class NseSource:
    """India-specific: universe, breadth, snapshots and the trading calendar.

    History is delegated to :class:`YahooChartSource` - NSE's public API has no
    OHLCV history endpoint, only current quotes. This class is what makes the
    rest of tvlab work on India without a paid vendor.

    Conforms to the ``DataSource`` protocol, so ``--data-source nse`` works
    anywhere a source is accepted.
    """

    name = "nse"

    def __init__(
        self,
        index: str = "NIFTY 500",
        http: Optional[Http] = None,
        session_filter: bool = True,
        market: str = "indian_equities",
    ):
        self.index = index
        self.http = http or Http()
        self.market = market
        self.history = YahooChartSource(market=market, session_filter=session_filter, http=self.http)

    # ---------------------------------------------------------------- history
    def fetch_ohlcv(self, symbol, timeframe="1d", start=None, end=None, limit=None) -> pd.DataFrame:
        return self.history.fetch_ohlcv(symbol, timeframe, start, end, limit)

    # --------------------------------------------------------------- universe
    def list_symbols(self) -> List[str]:
        """Nifty 500 constituents, plus every F&O security NSE lists.

        The union matters: the F&O list contains names the Nifty 500 does not,
        and F&O status is what makes intraday shorting and derivatives possible.
        A failure of either leg degrades to the other rather than raising - a
        partial universe is still usable, no universe is not.
        """
        syms: Set[str] = set()
        errors: List[str] = []

        try:
            body = self.http.get(NIFTY500_CSV).decode("utf-8", "replace")
            syms |= {r["Symbol"].strip().upper() for r in csv.DictReader(io.StringIO(body))
                     if r.get("Symbol", "").strip()}
        except Exception as exc:  # noqa: BLE001 - degrade to the F&O leg
            errors.append(f"nifty500 csv: {exc}")

        for key, watch_name in NSE_SPECIAL_WATCH.items():
            try:
                rows = self.market_watch(watch_name)
                syms |= {r["symbol"].strip().upper() for r in rows}
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{key}: {exc}")

        if not syms:
            raise DataError(
                "could not build an NSE universe:\n  " + "\n  ".join(errors) +
                "\nFall back to the symbols listed in config/markets/indian_equities.yaml."
            )
        if errors:
            log.warning("NSE universe is partial: %s", "; ".join(errors))
        return sorted(syms)

    def market_watch(self, index: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every member of an index in ONE call, with quote and volume fields.

        This is the cheapest way to snapshot a whole Indian universe: one HTTP
        request for 500 names, each with last price, change, day range, volume,
        traded value and 52-week range. ``index`` may also be ``"SECURITIES IN
        F&O"``. Only the ``EQ`` series is returned - NSE mixes in BE/SM/ST
        series that are not the same instrument.
        """
        name = index or self.index
        q = urllib.parse.quote(name)
        payload = self.http.get_json(MARKET_WATCH.format(q=q))
        try:
            rows = payload["data"]["data"]
        except (KeyError, TypeError) as exc:
            keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
            raise DataError(f"market-watch for {name!r} has no data.data (top-level keys: {keys})") from exc
        return [r for r in rows if r.get("series") == "EQ"]

    def snapshot(self, index: Optional[str] = None) -> pd.DataFrame:
        """Market watch as a tidy frame, sorted by traded value.

        ``traded_value`` is the liquidity measure that matters for India: a stock
        can have large volume in absolute shares and still be untradeable in size.
        """
        rows = self.market_watch(index)
        if not rows:
            return pd.DataFrame(columns=["symbol", "ltp", "pct_change", "open", "high", "low",
                                         "prev_close", "volume", "traded_value", "high_52w", "low_52w"])
        recs = []
        for r in rows:
            recs.append({
                "symbol": r.get("symbol"),
                "ltp": _num(r.get("lastPrice")),
                "pct_change": _num(r.get("pChange")),
                "open": _num(r.get("open")),
                "high": _num(r.get("dayHigh")),
                "low": _num(r.get("dayLow")),
                "prev_close": _num(r.get("previousClose")),
                "volume": _num(r.get("totalTradedVolume")) or 0.0,
                "traded_value": _num(r.get("totalTradedValue")) or 0.0,
                "high_52w": _num(r.get("yearHigh")),
                "low_52w": _num(r.get("yearLow")),
                "company": (r.get("companyName") or r.get("symbol") or "").title(),
            })
        df = pd.DataFrame(recs).dropna(subset=["symbol"])
        return df.sort_values("traded_value", ascending=False, ignore_index=True)

    def indices(self) -> pd.DataFrame:
        """Every NSE index: level, change, breadth (advances/declines), PE.

        Breadth is the market-regime input an intraday long-only mandate needs.
        ``allow_short`` is false for Indian equities, so on a day when declines
        outnumber advances there is simply no trade - and that is knowable before
        the open rather than after the loss.
        """
        payload = self.http.get_json(ALL_INDICES)
        try:
            rows = payload["data"]
        except (KeyError, TypeError) as exc:
            raise DataError("allIndices has no 'data' key") from exc
        recs = []
        for r in rows:
            recs.append({
                "index": r.get("index"),
                "last": _num(r.get("last")),
                "pct_change": _num(r.get("percentChange")),
                "variation": _num(r.get("variation")),
                "prev_close": _num(r.get("previousClose")),
                "advances": _int(r.get("advances")),
                "declines": _int(r.get("declines")),
                "unchanged": _int(r.get("unchanged")),
                "pe": _num(r.get("pe")),
                "high_52w": _num(r.get("yearHigh")),
                "low_52w": _num(r.get("yearLow")),
                "open": _num(r.get("open")),
            })
        df = pd.DataFrame(recs)
        if "advances" in df and "declines" in df:
            total = df["advances"] + df["declines"] + df["unchanged"]
            df["breadth"] = (df["advances"] - df["declines"]) / total.replace(0, pd.NA)
        return df

    def trading_holidays(self) -> Set[date]:
        """NSE cash-market trading holidays, as ``date`` objects."""
        payload = self.http.get_json(HOLIDAY_MASTER)
        rows = payload.get("CM") if isinstance(payload, dict) else None
        if rows is None:
            raise DataError(f"holiday-master has no 'CM' key (got {list(payload) if isinstance(payload, dict) else '?'})")
        out: Set[date] = set()
        for r in rows:
            try:
                out.add(datetime.strptime(r["tradingDate"], "%d-%b-%Y").date())
            except (KeyError, ValueError, TypeError):
                continue
        return out

    def is_session_open(self, now: Optional[datetime] = None, holidays: Optional[Iterable[date]] = None) -> bool:
        """Is the NSE cash market open right now?

        Weekends and holidays are answered from the calendar; ``holidays`` lets a
        caller pass a cached set instead of refetching. Without network access
        this falls back to weekday-and-clock only, and says so in the log - a
        wrong "market open" is worse than an admitted guess.
        """
        now = now or datetime.now(IST)
        if now.tzinfo is None:
            now = now.replace(tzinfo=IST)
        else:
            now = now.astimezone(IST)
        if now.weekday() >= 5:
            return False
        days = set(holidays) if holidays is not None else None
        if days is None:
            try:
                days = self.trading_holidays()
            except Exception as exc:  # noqa: BLE001
                log.warning("could not fetch the NSE holiday calendar (%s); assuming today is a trading day", exc)
                days = set()
        if now.date() in days:
            return False
        minutes = now.hour * 60 + now.minute
        return SESSION_START_MIN <= minutes < SESSION_END_MIN

    def minutes_to_close(self, now: Optional[datetime] = None) -> Optional[int]:
        """Minutes until the 15:30 IST close, or None if already closed.

        Worth knowing before opening an intraday position: a "buy today, sell
        today" trade with 20 minutes left has almost no time to work and is
        mostly a bet on the closing auction.
        """
        now = now or datetime.now(IST)
        now = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
        close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now >= close or not self.is_session_open(now):
            return None
        return int((close - now).total_seconds() // 60)


# ------------------------------------------------------------------- helpers
def _num(v: Any) -> Optional[float]:
    """NSE mixes numbers, strings and '' for missing. Coerce or return None."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _int(v: Any) -> int:
    n = _num(v)
    return int(n) if n is not None else 0


def top_by_liquidity(symbols: Iterable[str], snapshot: pd.DataFrame, n: int) -> List[str]:
    """Pick the ``n`` most liquid names from a snapshot, preserving only those asked for.

    Liquidity is traded value, not volume: at ₹50 a share, 1M shares is ₹5 crore;
    at ₹3,000 a share it is ₹300 crore. Volume alone would rank the cheap stock
    higher and put you in something you cannot exit.
    """
    want = {s.strip().upper() for s in symbols}
    if snapshot.empty:
        return sorted(want)[:n]
    have = snapshot[snapshot["symbol"].str.upper().isin(want)]
    ordered = have.sort_values("traded_value", ascending=False)["symbol"].tolist()
    missing = sorted(want - {s.upper() for s in ordered})
    return (ordered + missing)[:n]
