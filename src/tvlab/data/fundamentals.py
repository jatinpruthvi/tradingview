"""Fundamentals from TradingView's scanner, via ``tvscreener``.

The technical side of this repo answers *"is there a setup right now?"*. It cannot
answer *"is this a company worth owning while the setup plays out?"*, and for
equities that second question is not optional: a textbook liquidity sweep into a
name reporting earnings tomorrow is a coin flip on a gap, not a trade.

This module adds the missing half. It pulls valuation, quality, balance-sheet,
dividend and analyst fields for a symbol list, plus the two dates that matter most
to a daily workflow - the next and most recent earnings release.

What it is built on
-------------------
``tvscreener`` (Apache-2.0), which was already a dependency for universe selection
and whose fundamental surface was going almost entirely unused: ``StockField`` has
3,666 members, and the package also ships ``fundamentals.get_financials`` (income
statement, balance sheet, cash flow and statistics over ~20 fiscal years),
``get_corporate_events``, ``get_seasonals`` and earnings/dividend/IPO/economic
calendars. See ``VENDOR.md`` section 4.

Honest status
-------------
The parsing is tested against fixtures, and the request path was verified against
``scanner.tradingview.com`` by hand (see the notes on currency and on scoping the
scan below) rather than captured in CI, which has no route to it - the same caveat
``data/nse.py`` carries. Two consequences are designed in rather than hoped away:

* Column identity is resolved by *label and by field name*, normalised, because
  ``tvscreener`` builds its request as ``{field_name: label}`` and different
  versions have returned one or the other. Matching only one convention would
  silently produce an all-missing table.
* A field the endpoint does not return is recorded in ``snapshot.missing`` and
  reported, never defaulted to zero. A missing P/E is not a P/E of 0, and treating
  it as one would pass a ``max_pe_ttm`` gate that the data cannot actually support.

Requires network access. Offline, ``fetch_fundamentals`` raises :class:`DataError`
with the fallback spelled out - it does not return an empty table, because an empty
table is indistinguishable from "every symbol failed the filter".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .base import DataError

log = logging.getLogger(__name__)

#: Canonical name -> the ``StockField`` member names that could carry it, in order
#: of preference.
#:
#: **Every name listed here was verified to exist** on the installed enum, and
#: ``tests/test_fundamentals.py`` re-verifies the whole table on every run. An
#: earlier draft also listed speculative alternates for names TradingView might use
#: one day (``CLOSE``, ``MARKET_CAP``, ``DEBT_TO_EQUITY_MRQ``, ``BETA_1_YEAR``, …).
#: They were removed: a fallback that has never existed cannot be tested, so it is
#: not a safeguard, it is an unverified claim about an API. When a real version
#: drift is observed, add the name that the new version actually uses and pin it
#: with a test.
#:
#: The list form is still worth keeping where TradingView genuinely ships more than
#: one spelling for the same concept - ``PRICE_TO_BOOK_MRQ`` and ``_FY``,
#: ``RETURN_ON_EQUITY_TTM`` and ``_FY`` - because those differ in meaning and the
#: first present is the better one.
FIELD_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "name": ("NAME", "DESCRIPTION"),
    "sector": ("SECTOR",),
    "industry": ("INDUSTRY",),
    "currency": ("CURRENCY", "CURRENCY_ID"),
    "exchange": ("EXCHANGE", "SUBMARKET"),
    "last": ("PRICE",),
    "market_cap": ("MARKET_CAPITALIZATION",),
    "pe_ttm": ("PRICE_TO_EARNINGS_RATIO_TTM",),
    "price_to_book": ("PRICE_TO_BOOK_MRQ", "PRICE_TO_BOOK_FY"),
    "price_to_sales": ("PRICE_TO_SALES_FY",),
    # the missing slash is TradingView's own spelling, not a typo here
    "ev_to_ebitda": ("ENTERPRISE_VALUEEBITDA_TTM",),
    "roe": ("RETURN_ON_EQUITY_TTM", "RETURN_ON_EQUITY_FY"),
    "roa": ("RETURN_ON_ASSETS_TTM",),
    "roic": ("RETURN_ON_INVESTED_CAPITAL_TTM",),
    "gross_margin": ("GROSS_MARGIN_TTM",),
    "operating_margin": ("OPERATING_MARGIN_TTM",),
    "net_margin": ("NET_MARGIN_TTM",),
    "revenue_growth_yoy": ("REVENUE_TTM_YOY_GROWTH", "REVENUE_ANNUAL_YOY_GROWTH"),
    "eps_growth_yoy": ("EPS_DILUTED_TTM_YOY_GROWTH", "EPS_DILUTED_ANNUAL_YOY_GROWTH"),
    "eps_ttm": ("EARNINGS_PER_SHARE_BASIC_TTM", "BASIC_EPS_TTM", "EPS_DILUTED_TTM"),
    "debt_to_equity": ("DEBT_TO_EQUITY_FQ",),
    "current_ratio": ("CURRENT_RATIO_FQ", "CURRENT_RATIO_MRQ"),
    "free_cash_flow": ("FREE_CASH_FLOW_TTM", "FREE_CASH_FLOW_FY"),
    "dividend_yield": ("DIVIDENDS_YIELD_CURRENT", "DIVIDENDS_YIELD", "DIVIDENDS_YIELD_FY"),
    "dividend_payout": ("DIVIDEND_PAYOUT_RATIO_TTM",),
    "beta": ("BETA_3_YEAR", "BETA_5_YEAR"),
    "analyst_rating": ("RECOMMENDATION_MARK",),
    "next_earnings": ("UPCOMING_EARNINGS_DATE",),
    "last_earnings": ("RECENT_EARNINGS_DATE",),
}

#: Fields that must be dates rather than numbers.
_DATE_FIELDS = {"next_earnings", "last_earnings"}

#: Fields the scanner reports as a *percentage* rather than a fraction.
#: ``RETURN_ON_EQUITY_TTM`` is 9.0147 for RELIANCE, i.e. 9.01% - not 901%.
#: Everything downstream (the gates, ``min_roe: 0.15`` in the market yaml, the
#: ``roe * 100`` in the briefing table) treats these as fractions, so the divide
#: happens here, once, at the boundary. Without it a 9% ROE clears a 15% gate
#: and prints as 901.5%: both the screen and the risk control silently invert.
#: Measured against scanner.tradingview.com on 2026-11-24. Ratios that happen
#: to be small - P/E, P/B, debt/equity, current ratio - are NOT in this set.
_PERCENT_FIELDS = frozenset({
    "roe", "roa", "roic", "gross_margin", "operating_margin", "net_margin",
    "revenue_growth_yoy", "eps_growth_yoy", "dividend_yield", "dividend_payout",
})

#: tvlab market name -> tvscreener ``Market`` enum member name.
MARKET_MAP: Dict[str, str] = {
    "us_equities": "AMERICA",
    "indian_equities": "INDIA",
    "uk_equities": "UK",
    "crypto": "ALL",
    "forex": "ALL",
}

#: Currency each market's fundamentals must be reported in. The global scanner
#: (``StockScreener()`` with no market) returns fundamentals in USD whatever the
#: listing currency is, and the frame's ``currency`` column still shows the
#: *listing* currency - so the mismatch is invisible. RELIANCE came back as
#: 1.97e11 labelled "INR" while scanner.tradingview.com returns 1.83e13 for the
#: same moment: dollars against rupees at ~93/USD. The caps in
#: ``config/markets/*_equities.yaml`` are stated in the local currency ("INR,
#: not USD: 5.0e10 is ~USD 6bn"), so without the conversion a 6bn-rupee floor is
#: applied to a dollar figure and every Indian name fails it by ~93x.
MARKET_CURRENCY: Dict[str, str] = {
    "us_equities": "usd",
    "indian_equities": "inr",
    "uk_equities": "gbp",
}

#: Markets where fundamentals are meaningful at all. Crypto and forex have no
#: earnings, no book value and no analyst coverage, so a gate on them would filter
#: on columns that are structurally absent - which is how a data gap turns into a
#: market verdict.
FUNDAMENTAL_MARKETS = frozenset({"us_equities", "indian_equities", "uk_equities"})

#: Punctuation and case are the only differences between the scanner's own column
#: names and the labels it reports them under. This must be case-insensitive: an
#: uppercase-only class strips *every* letter from a lowercased label, collapsing
#: "Price to Earnings Ratio (TTM)" to "", which then matches every lookup key and
#: silently fills unrelated fields with one column's values.
_SYMBOL_CLEAN = re.compile(r"[^A-Za-z0-9]")


@dataclass(frozen=True)
class FundamentalSnapshot:
    """One symbol's fundamentals, with its gaps named rather than filled in.

    Every numeric field is ``Optional``. ``None`` means the endpoint did not
    report it, which is a different statement from ``0.0`` and is kept distinct all
    the way to the gates in ``screener/fundamental.py``.
    """

    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    exchange: Optional[str] = None
    last: Optional[float] = None

    market_cap: Optional[float] = None
    pe_ttm: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None

    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    eps_growth_yoy: Optional[float] = None
    eps_ttm: Optional[float] = None

    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None

    dividend_yield: Optional[float] = None
    dividend_payout: Optional[float] = None
    beta: Optional[float] = None
    analyst_rating: Optional[float] = None

    next_earnings: Optional[pd.Timestamp] = None
    last_earnings: Optional[pd.Timestamp] = None

    #: canonical names the endpoint did not provide
    missing: Tuple[str, ...] = ()
    #: anything requested via ``extra_fields`` that has no canonical home
    extra: Dict[str, Any] = field(default_factory=dict)
    #: when this snapshot was taken; fundamentals are point-in-time, not timeless
    as_of: Optional[str] = None

    # ------------------------------------------------------------------ derived
    def days_to_earnings(self, as_of: Optional[pd.Timestamp] = None) -> Optional[int]:
        """Calendar days until the next earnings release, or None if unknown.

        Negative means the scheduled date has already passed without the feed
        updating, which happens; callers should treat that as "unknown", not as
        "safe".
        """
        if self.next_earnings is None or pd.isna(self.next_earnings):
            return None
        ref = as_of or pd.Timestamp.now(tz="UTC")
        a = self.next_earnings.tz_localize(None) if self.next_earnings.tzinfo else self.next_earnings
        b = ref.tz_localize(None) if getattr(ref, "tzinfo", None) else ref
        return int((a.normalize() - b.normalize()).days)

    def days_since_earnings(self, as_of: Optional[pd.Timestamp] = None) -> Optional[int]:
        if self.last_earnings is None or pd.isna(self.last_earnings):
            return None
        ref = as_of or pd.Timestamp.now(tz="UTC")
        a = self.last_earnings.tz_localize(None) if self.last_earnings.tzinfo else self.last_earnings
        b = ref.tz_localize(None) if getattr(ref, "tzinfo", None) else ref
        return int((b.normalize() - a.normalize()).days)

    def get(self, canonical: str) -> Any:
        """Read a canonical field by name, so gates need no attribute switch."""
        if hasattr(self, canonical):
            return getattr(self, canonical)
        return self.extra.get(canonical)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "extra":
                continue
            if isinstance(v, pd.Timestamp):
                out[k] = v.isoformat()
            else:
                out[k] = v
        out.update(self.extra)
        return out


# --------------------------------------------------------------------- fetching


def normalise_symbol(symbol: str) -> str:
    """Reduce a symbol to the alphanumerics it can be matched on.

    The same company arrives as ``RELIANCE``, ``RELIANCE.NS`` and ``NSE:RELIANCE``
    from the NSE list, the Yahoo adapter and the TradingView scanner respectively.
    Matching has to survive all three, and it has to *not* collapse
    ``BTC/USDT:USDT`` style quotes into something that collides.
    """
    s = str(symbol).upper()
    s = s.split(":")[-1] if ":" in s and not s.endswith(".NS") else s
    s = s.replace(".NS", "").replace(".BO", "")
    return _SYMBOL_CLEAN.sub("", s)


def _resolve_fields(tvscreener, extra: Sequence[str] = ()) -> Tuple[Dict[str, Any], List[str]]:
    """Map canonical names to real ``StockField`` members.

    Returns the mapping and the list of canonical names this version of
    ``tvscreener`` cannot serve. Dropping those with a warning is deliberate: an
    upstream rename should cost one column, not the whole request.
    """
    enum = tvscreener.StockField
    resolved: Dict[str, Any] = {}
    unresolved: List[str] = []
    candidates = dict(FIELD_CANDIDATES)
    for name in extra or ():
        candidates.setdefault(str(name).lower(), (str(name).upper(), str(name)))
    for canonical, names in candidates.items():
        for n in names:
            if hasattr(enum, n):
                resolved[canonical] = getattr(enum, n)
                break
        else:
            unresolved.append(canonical)
    if unresolved:
        log.warning("tvscreener has no field for: %s", ", ".join(sorted(unresolved)))
    return resolved, unresolved


def _column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    """Build a normalised index of the frame's columns.

    ``tvscreener`` assembles its request as ``{field_name: label}`` and has
    returned results keyed by either, depending on version. Indexing on both,
    case- and punctuation-insensitively, is the difference between a working
    adapter and a table of Nones that looks like a market with no fundamentals.
    """
    out: Dict[str, str] = {}
    for c in df.columns:
        raw = str(c)
        for key in (raw, raw.strip().lower(), _SYMBOL_CLEAN.sub("", raw.strip().lower())):
            # An empty key would match every lookup and return this column for
            # every field, so it is never indexed.
            if key:
                out.setdefault(key, raw)
    return out


def _read(row: pd.Series, lookup: Dict[str, str], keys: Iterable[str]) -> Any:
    for k in keys:
        cleaned = _SYMBOL_CLEAN.sub("", str(k).strip().lower())
        col = lookup.get(k) or lookup.get(str(k).strip().lower()) \
            or (lookup.get(cleaned) if cleaned else None)
        if col is not None and col in row.index:
            v = row[col]
            if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
                continue
            return v
    return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if not v or v in {"-", "n/a", "N/A", "None", "nan"}:
            return None
        mult = 1.0
        if v[-1] in "KMBkmb" and len(v) > 1:
            mult = {"K": 1e3, "M": 1e6, "B": 1e9}[v[-1].upper()]
            v = v[:-1]
        try:
            return float(v) * mult
        except ValueError:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if pd.notna(f) else None


def _to_date(v: Any) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # TradingView dates arrive as epoch seconds or milliseconds
            v = float(v)
            if v > 1e11:
                v /= 1000.0
            return pd.Timestamp(datetime.fromtimestamp(v, tz=timezone.utc))
        ts = pd.Timestamp(v)
        return ts if pd.notna(ts) else None
    except (ValueError, TypeError, OSError, OverflowError):
        log.debug("unparseable fundamental date: %r", v)
        return None


#: Quote suffixes this repo's market files use. They are stripped, not translated to
#: an exchange: the scanner filters on ``name``, which is the bare ticker.
_TICKER_SUFFIXES = frozenset({"NS", "BO", "L"})


def _tv_names(symbols: Sequence[str]) -> Optional[List[str]]:
    """Configured symbols -> the scanner's own ``name`` values, or ``None`` if unsure.

    ``RELIANCE.NS``, ``RELIANCE.BO`` and ``NSE:RELIANCE`` are all ``RELIANCE`` to
    TradingView. Only the suffixes this repo's market files actually use are
    stripped; an unrecognised one returns ``None`` so the caller filters on nothing
    rather than on a guess - and a name filter that matches nothing would read as a
    market with no fundamentals.
    """
    out: List[str] = []
    for symbol in symbols:
        base, dot, suffix = str(symbol).rpartition(".")
        if dot:
            if suffix.upper() not in _TICKER_SUFFIXES:
                return None
            symbol = base
        out.append(str(symbol).split(":")[-1].upper())
    return out or None


def fetch_fundamentals(
    symbols: Optional[Sequence[str]] = None,
    market: str = "us_equities",
    max_symbols: int = 200,
    extra_fields: Sequence[str] = (),
    tvscreener_module: Optional[Any] = None,
) -> Dict[str, FundamentalSnapshot]:
    """Pull fundamentals for ``symbols`` (or the whole market if None).

    Returns ``{original_symbol: FundamentalSnapshot}``. Symbols the endpoint did
    not return are absent from the mapping, which the caller must report rather
    than treat as a pass.

    Raises :class:`DataError` if ``tvscreener`` is missing, the market has no
    fundamentals, or the request fails. Never returns an empty dict for a failure.
    """
    if market not in FUNDAMENTAL_MARKETS:
        raise DataError(
            f"fundamentals are not available for market '{market}'. "
            f"{', '.join(sorted(FUNDAMENTAL_MARKETS))} have earnings, book value and "
            f"analyst coverage; crypto and forex do not, so there is nothing to gate on."
        )

    tvs = tvscreener_module
    if tvs is None:
        try:
            import tvscreener as tvs  # noqa: PLC0415
        except ImportError as exc:
            raise DataError(
                "pip install tvscreener   # required for fundamentals"
            ) from exc

    resolved, unresolved = _resolve_fields(tvs, extra_fields)
    if not resolved:
        raise DataError(
            "this version of tvscreener exposes none of the fundamental fields "
            "tvlab asks for; the enum has probably been renamed upstream"
        )

    try:
        screener = tvs.StockScreener()
    except Exception as exc:
        raise DataError(f"could not construct tvscreener StockScreener: {exc}") from exc

    enum_name = MARKET_MAP.get(market, "AMERICA")
    Market = getattr(tvs, "Market", None)
    if Market is not None and hasattr(Market, enum_name):
        try:
            screener.set_markets(getattr(Market, enum_name))
        except Exception as exc:
            log.warning("set_markets(%s) failed: %s", enum_name, exc)

    # Ask for the market's own currency before the single request goes out. The
    # parser is not the problem here - it reads whatever number the frame holds -
    # so fixing this in ``build_snapshots`` would mean guessing an FX rate.
    ccy = MARKET_CURRENCY.get(market)
    set_currency = getattr(screener, "set_currency", None)
    if ccy and callable(set_currency):
        try:
            set_currency(ccy)
        except Exception as exc:  # noqa: BLE001 - a cosmetic setting, not the data
            log.warning("set_currency(%r) failed: %s; market cap is in the "
                        "scanner's default currency (USD)", ccy, exc)
    elif ccy:
        log.warning("this tvscreener build has no set_currency(); %s market cap "
                    "arrives in USD and min_market_cap must be read as USD", market)

    # Scope the scan to the asked-for names. Without this every request returns the
    # default 150-row page, which holds only ~75 Indian companies (each is listed on
    # both NSE and BSE), and a symbol outside that page is absent from the result
    # while looking exactly like a symbol that failed a filter. TMPV - renamed from
    # TATAMOTORS on 2025-10-24 and 243rd of 7,393 Indian tickers - was not in it.
    #
    # This is a ``where()`` filter rather than a ticker list assigned to
    # ``screener.symbols``: filtering happens inside the scanner, needs no exchange
    # prefix (so the suffix-less US config is scoped too), and goes through the
    # documented API instead of an attribute that only exists to hold the request
    # body. A name that matches nothing simply does not come back - the caller
    # reports it as no data, which is the truth.
    names = _tv_names(symbols) if symbols else None

    request_failed = (
        "  This needs outbound access to scanner.tradingview.com.\n"
        "  Offline fallback: run with --data-source csv|synthetic and leave\n"
        "  fundamentals.enabled false - the technical ranking still works,\n"
        "  it just cannot tell you whether the company is worth owning."
    )

    try:
        screener.select(*resolved.values())
        if names:
            screener.where(tvs.StockField.NAME, tvs.FilterOperator.IN_RANGE, names)
        df = screener.get()
    except Exception as exc:
        raise DataError(
            f"TradingView fundamentals request failed: {type(exc).__name__}: {exc}\n"
            + request_failed
        ) from exc

    if df is None or len(df) == 0:
        raise DataError("fundamentals request returned 0 rows")

    return build_snapshots(df, resolved, symbols=symbols, max_symbols=max_symbols,
                           unresolved=unresolved)


def build_snapshots(
    df: pd.DataFrame,
    resolved: Dict[str, Any],
    symbols: Optional[Sequence[str]] = None,
    max_symbols: int = 200,
    unresolved: Sequence[str] = (),
) -> Dict[str, FundamentalSnapshot]:
    """Turn a raw scanner frame into snapshots.

    Split out from ``fetch_fundamentals`` because this is the half that can be
    tested: the request needs a network, the parsing does not. Every fixture in
    ``tests/test_fundamentals.py`` drives this directly.
    """
    lookup = _column_lookup(df)
    as_of = pd.Timestamp.now(tz="UTC").isoformat()

    # The symbol column: NAME is requested first and is the scanner's own key.
    sym_keys = [getattr(resolved["name"], "field_name", "name"),
                getattr(resolved["name"], "label", "Name"),
                "symbol", "name", "ticker", "s"]
    sym_col = None
    for k in sym_keys:
        if k in lookup:
            sym_col = lookup[k]
            break
    if sym_col is None:
        sym_col = df.columns[0]

    wanted: Optional[Dict[str, str]] = None
    if symbols:
        wanted = {}
        for s in symbols:
            wanted[normalise_symbol(s)] = s

    out: Dict[str, FundamentalSnapshot] = {}
    unresolved_set = set(unresolved)
    for _, row in df.iterrows():
        raw_sym = str(row[sym_col]) if sym_col in row.index else None
        if not raw_sym:
            continue
        key = normalise_symbol(raw_sym)
        original = raw_sym
        if wanted is not None:
            if key not in wanted:
                continue
            original = wanted[key]

        values: Dict[str, Any] = {}
        missing: List[str] = []
        for canonical, fld in resolved.items():
            keys = [getattr(fld, "field_name", canonical), getattr(fld, "label", canonical)]
            v = _read(row, lookup, keys)
            if canonical in _DATE_FIELDS:
                parsed = _to_date(v)
            elif canonical in {"name", "sector", "industry", "currency", "exchange"}:
                parsed = None if v is None else str(v)
            else:
                parsed = _to_float(v)
                if parsed is not None and canonical in _PERCENT_FIELDS:
                    parsed /= 100.0
            if parsed is None:
                missing.append(canonical)
            else:
                values[canonical] = parsed

        known = {k: v for k, v in values.items() if k in FIELD_CANDIDATES}
        extra = {k: v for k, v in values.items() if k not in FIELD_CANDIDATES}
        # A field this tvscreener version cannot even request is missing too, and
        # saying so is what stops a gate from being believed when it never ran.
        missing.extend(m for m in unresolved_set if m not in missing)

        snap = FundamentalSnapshot(symbol=original, missing=tuple(sorted(set(missing))),
                                   extra=extra, as_of=as_of, **known)
        out[original] = snap
        if len(out) >= max_symbols:
            break
    return out


def fundamentals_frame(snaps: Iterable[FundamentalSnapshot]) -> pd.DataFrame:
    """Snapshots as a table, for the CLI, the dashboard and the journal."""
    rows = [s.to_dict() for s in snaps]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["missing_count"] = [len(s.missing) for s in snaps]
    return df


def coverage(snaps: Iterable[FundamentalSnapshot]) -> Dict[str, Any]:
    """How much of the requested data actually arrived.

    Reported alongside every gate decision, because "passed the fundamental
    filters" means something very different when half the fields were absent.
    """
    snaps = list(snaps)
    if not snaps:
        return {"symbols": 0, "complete": 0, "partial": 0, "missing_fields": {}}
    counts: Dict[str, int] = {}
    for s in snaps:
        for m in s.missing:
            counts[m] = counts.get(m, 0) + 1
    complete = sum(1 for s in snaps if not s.missing)
    return {
        "symbols": len(snaps),
        "complete": complete,
        "partial": len(snaps) - complete,
        "missing_fields": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
