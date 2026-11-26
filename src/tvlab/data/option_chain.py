"""NSE option chain: read it, summarise it, and snapshot it daily.

Why this exists at all
----------------------
The chain was measured against the screener before any of this was built
(``output/option_chain_study.py`` in the working tree, 2026-11-25). Three
findings shaped this module:

* **The popular readings do not discriminate.** "Do not buy into a call wall
  below your target" vetoed 15 of 15 ranked longs, because the strategy targets
  +25-33% and every "wall" sits 0.4-7.8% above spot. 12 of those 15 walls were
  just the next strike on a 50-point grid, holding a few hundred lots against
  five figures of traded volume - rounding dust, not dealer positioning.
* **The one useful number is volatility, not open interest.** Eight of those 15
  candidates had a stop inside one sigma of the market's own 28-day expected
  move. That is a real critique of stop placement - and IV/RV20 was 0.91, i.e.
  the chain was repeating what 20 days of bars already said.
* **It cannot be backfilled.** NSE publishes a live chain and no history, so the
  only way to ever test a chain filter is to record today's chain today.

So this module is deliberately a *recorder*, not a filter. Nothing here is
wired into scoring, and it should not be until the panel it produces has been
tested against forward returns - the honest test, because it is dense (218 names
x every trading day) where "does it lift the win rate" would need ~290 trades
per arm and this strategy produces ~19 a year.

Undocumented endpoints, politely
--------------------------------
Both endpoints are public, undocumented and change without notice. A missing
field is reported as an error row rather than a zero, requests are serial, and
``TVLAB_HTTP_DELAY`` still governs the pause between them. The fetcher is
injectable so the tests never touch the network - same contract as
:mod:`tvlab.data.nse`.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .base import DataError
from .nse import IST, USER_AGENT

log = logging.getLogger(__name__)

NSE_BASE = "https://www.nseindia.com"
CONTRACT_INFO_URL = NSE_BASE + "/api/option-chain-contract-info?symbol={symbol}"
CHAIN_URL = (NSE_BASE + "/api/option-chain-v3?type={inst_type}&symbol={symbol}"
             "&expiry={expiry}")
INDEX_SYMBOLS = frozenset({"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"})

#: A swing hold is not a weekly expiry: (nearly) nothing in the 0-20 DTE window
#: describes a position held for weeks, and its OI is dominated by expiry-day
#: mechanics rather than positioning.
DEFAULT_MIN_DTE = 21

#: Only positioning within this fraction of spot is about the trade in hand. A
#: 40% out-of-the-money strike with large OI says something about next quarter,
#: not about the next stop.
DEFAULT_BAND = 0.12

Fetcher = Callable[[str, float], bytes]

#: The snapshot panel. ``as_of`` is the IST trading date, as an ISO string: it
#: sorts, it groups, and it survives parquet round-trips without pandas dtype
#: surprises when frames written on different days are concatenated. ``ok`` and
#: ``error`` are kept because a panel of only successful measurements cannot
#: tell "no positioning" from "the request failed" - and coverage is a finding,
#: not a nuisance.
SNAPSHOT_COLUMNS: Tuple[str, ...] = (
    "as_of", "symbol", "ok", "error", "fetched_at", "expiry", "dte",
    "spot", "strikes_in_band", "live_strikes", "ce_oi", "pe_oi", "pcr",
    "traded_volume", "call_wall", "call_wall_dist_pct", "call_wall_oi",
    "put_wall", "put_wall_dist_pct", "atm_iv", "skew_put_minus_call",
)

FEATURE_COLUMNS: Tuple[str, ...] = (
    "spot", "strikes_in_band", "live_strikes", "ce_oi", "pe_oi", "pcr",
    "traded_volume", "call_wall", "call_wall_dist_pct", "call_wall_oi",
    "put_wall", "put_wall_dist_pct", "atm_iv", "skew_put_minus_call",
)

#: Record-only LTP Calculator features (kept in a separate panel so existing
#: ``option_chain_panel.parquet`` files keep their exact schema).
LTP_SNAPSHOT_COLUMNS: Tuple[str, ...] = (
    "as_of", "symbol", "ok", "error", "fetched_at", "expiry", "dte",
    "spot", "res_strike", "sup_strike", "ce_vol_wall", "pe_vol_wall",
    "res_wtt_ratio", "res_wtb_ratio", "sup_wtt_ratio", "sup_wtb_ratio",
    "res_state", "sup_state", "coa_regime", "eor", "eos",
    "eor_dist_pct", "eos_dist_pct",
)


# ----------------------------------------------------------------------- HTTP


class _CookieSession:
    """A stdlib session: NSE hands out cookies and then wants them back.

    A plain ``urlopen`` per call gets an HTML interstitial from the chain
    endpoints because every request arrives without the cookie the previous one
    was given. ``requests`` solves this with a Session, but ``requests`` is not a
    declared dependency of this project (only of the ad-hoc study script), and
    pulling it in for one endpoint would make ``pip install -r
    requirements.txt`` insufficient for a command that is otherwise stdlib-only.
    """

    def __init__(self, timeout: float = 25.0) -> None:
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))
        self.timeout = timeout
        self.primed = False

    def _open(self, url: str, timeout: float) -> bytes:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{NSE_BASE}/option-chain",
        })
        try:
            with self.opener.open(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise DataError(f"HTTP {exc.code} from {url}") from exc
        except urllib.error.URLError as exc:
            raise DataError(
                f"could not reach {urllib.parse.urlparse(url).netloc}: {exc.reason}. "
                f"This host must be reachable from the machine running tvlab."
            ) from exc

    def __call__(self, url: str, timeout: float) -> bytes:
        if not self.primed:
            # The cookie is issued by the page, not by the API. Failure here is
            # not fatal: some sessions answer the API without it.
            try:
                self._open(f"{NSE_BASE}/option-chain", self.timeout)
            except DataError as exc:
                log.debug("chain cookie warm-up failed (%s); continuing", exc)
            self.primed = True
        return self._open(url, timeout)


class ChainHttp:
    """Sequential, polite JSON access to the chain endpoints.

    ``fetcher`` is resolved lazily for the same reason :class:`tvlab.data.nse.Http`
    resolves it in ``__post_init__``: a dataclass default binds at class-body
    time, so replacing the module attribute afterwards silently does nothing and
    the tests quietly go to the network.

    ``delay`` is slept *after* every successful request, read from
    ``TVLAB_HTTP_DELAY`` at construction (0.08s default). These are undocumented
    public endpoints; hammering them is how the IP gets banned.
    """

    def __init__(
        self,
        fetcher: Optional[Fetcher] = None,
        delay: Optional[float] = None,
        timeout: float = 25.0,
    ) -> None:
        self._session = _CookieSession(timeout) if fetcher is None else None
        self.fetcher = fetcher or self._session
        self.timeout = timeout
        self.delay = (float(os.environ.get("TVLAB_HTTP_DELAY", "0.08"))
                      if delay is None else delay)
        self.requests = 0

    def get_json(self, url: str, timeout: Optional[float] = None) -> Any:
        raw = self.fetcher(url, timeout or self.timeout)  # type: ignore[misc]
        self.requests += 1
        if self.delay:
            time.sleep(self.delay)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            head = raw[:200].decode("utf-8", "replace")
            raise DataError(
                f"{url} did not return JSON ({exc}). First 200 bytes: {head!r}. "
                f"NSE serves an HTML interstitial when it wants a browser cookie - "
                f"retry, or run from a residential connection."
            ) from exc


# -------------------------------------------------------------- chain parsing


def contract_expiries(http: ChainHttp, symbol: str) -> List[str]:
    """Expiry strings NSE lists for ``symbol``, e.g. ``['29-Dec-2026', ...]``.

    An empty list is the honest answer for a name with no derivatives: NSE
    returns ``{}`` rather than an error, so this must not be treated as a
    transport failure.
    """
    payload = http.get_json(CONTRACT_INFO_URL.format(symbol=urllib.parse.quote(symbol)))
    if not isinstance(payload, dict):
        raise DataError(f"contract-info for {symbol} is not a JSON object")
    expiries = payload.get("expiryDates") or []
    return [str(e) for e in expiries if str(e).strip()]


def chain_record(http: ChainHttp, symbol: str, expiry: str,
                 instrument_type: Optional[str] = None) -> Dict[str, Any]:
    """The raw ``records`` block of a chain call."""
    inst = instrument_type or ("Indices" if symbol.upper() in INDEX_SYMBOLS else "Equity")
    url = CHAIN_URL.format(inst_type=inst,
                           symbol=urllib.parse.quote(symbol),
                           expiry=urllib.parse.quote(expiry))
    payload = http.get_json(url)
    if not isinstance(payload, dict):
        raise DataError(f"chain for {symbol} {expiry} is not a JSON object")
    rec = payload.get("records")
    if not isinstance(rec, dict):
        raise DataError(f"chain for {symbol} {expiry} has no 'records' block")
    return rec


def parse_expiry(raw: str) -> Optional[date]:
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def pick_expiry(expiries: Sequence[str], today: Optional[date] = None,
                min_dte: int = DEFAULT_MIN_DTE
                ) -> Tuple[Optional[str], Optional[int]]:
    """The nearest expiry that is at least ``min_dte`` away, and its DTE.

    When every listed expiry is nearer than ``min_dte`` (an unusual calendar, or
    a stale caller-supplied date) the *farthest* one is returned with its true
    DTE instead of nothing. Returning nothing would turn a real chain into
    "no data"; returning the farthest means the DTE column tells the reader
    exactly what they got, which is the property that matters.
    """
    today = today or datetime.now(IST).date()
    parsed: List[Tuple[date, str, int]] = []
    for raw in expiries:
        d = parse_expiry(raw)
        if d is not None:
            parsed.append((d, raw, (d - today).days))
    if not parsed:
        return None, None
    parsed.sort()
    for _drop, raw, dte in parsed:
        if dte >= min_dte:
            return raw, dte
    _drop, raw, dte = parsed[-1]
    log.warning("every expiry for this chain is under %d DTE (farthest is %s, %d DTE)",
                min_dte, raw, dte)
    return raw, dte


def _oi(leg: Optional[Dict[str, Any]]) -> int:
    try:
        return int((leg or {}).get("openInterest") or 0)
    except (TypeError, ValueError):
        return 0


def _iv(leg: Optional[Dict[str, Any]]) -> Optional[float]:
    try:
        v = float((leg or {}).get("impliedVolatility") or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _volume(leg: Optional[Dict[str, Any]]) -> int:
    try:
        return int((leg or {}).get("totalTradedVolume") or 0)
    except (TypeError, ValueError):
        return 0


def _last_price(leg: Optional[Dict[str, Any]]) -> Optional[float]:
    try:
        v = float((leg or {}).get("lastPrice") or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _shifting_ratios(items: Sequence[Tuple[float, int]], spot: float
                     ) -> Tuple[Optional[float], float, float]:
    """Dominant strike plus (WTT above-ratio, WTB below-ratio) for one metric."""
    positive = [(k, v) for k, v in items if v > 0]
    if not positive:
        return None, 0.0, 0.0
    # Highest count; tie-break closest to spot
    primary, p_val = max(positive, key=lambda kv: (kv[1], -abs(kv[0] - spot)))
    above = [v for k, v in positive if k > primary]
    below = [v for k, v in positive if k < primary]
    wtt = (max(above) / p_val) if above else 0.0
    wtb = (max(below) / p_val) if below else 0.0
    return primary, round(wtt, 3), round(wtb, 3)


def _shift_state(wtt: float, wtb: float, threshold: float = 0.75) -> str:
    is_wtt = wtt >= threshold
    is_wtb = wtb >= threshold
    if is_wtt and is_wtb:
        return "WTT+WTB"
    if is_wtt:
        return "WTT"
    if is_wtb:
        return "WTB"
    return "STRONG"


def _coa_regime(sup_state: str, res_state: str) -> str:
    if sup_state == "STRONG" and res_state == "STRONG":
        return "RANGE_BOUND"
    if res_state == "WTT" and sup_state in ("STRONG", "WTT"):
        return "BULLISH"
    if sup_state == "WTB" and res_state in ("STRONG", "WTB"):
        return "BEARISH"
    return "MIXED"


def ltp_features(record: Dict[str, Any], band: float = DEFAULT_BAND,
                 wtt_threshold: float = 0.75) -> Optional[Dict[str, Any]]:
    """Record-only LTP Calculator metrics (S/R, WTT/WTB ratios, COA, EOR/EOS).

    Purely descriptive and observational — NOT wired into trade ranking until
    40+ live sessions prove predictive power via :mod:`tvlab.chain_study`.
    """
    spot_raw = record.get("underlyingValue")
    rows = record.get("data") or []
    try:
        spot = float(spot_raw)
    except (TypeError, ValueError):
        return None
    if spot <= 0 or not rows:
        return None

    lo, hi = spot * (1 - band), spot * (1 + band)
    in_band = [r for r in rows
               if lo <= float(r.get("strikePrice") or 0) <= hi and r.get("strikePrice")]
    if not in_band:
        return None

    ce_oi_items: List[Tuple[float, int]] = []
    ce_vol_items: List[Tuple[float, int]] = []
    pe_oi_items: List[Tuple[float, int]] = []
    pe_vol_items: List[Tuple[float, int]] = []
    by_strike: Dict[float, Dict[str, Any]] = {}
    for r in in_band:
        k = float(r["strikePrice"])
        by_strike[k] = r
        ce, pe = r.get("CE"), r.get("PE")
        ce_oi_items.append((k, _oi(ce)))
        ce_vol_items.append((k, _volume(ce)))
        pe_oi_items.append((k, _oi(pe)))
        pe_vol_items.append((k, _volume(pe)))

    res_oi_k, res_oi_wtt, res_oi_wtb = _shifting_ratios(ce_oi_items, spot)
    ce_vol_k, res_vol_wtt, res_vol_wtb = _shifting_ratios(ce_vol_items, spot)
    sup_oi_k, sup_oi_wtt, sup_oi_wtb = _shifting_ratios(pe_oi_items, spot)
    pe_vol_k, sup_vol_wtt, sup_vol_wtb = _shifting_ratios(pe_vol_items, spot)

    res_strike = res_oi_k if res_oi_k is not None else ce_vol_k
    sup_strike = sup_oi_k if sup_oi_k is not None else pe_vol_k
    res_wtt = round(max(res_oi_wtt, res_vol_wtt), 3)
    res_wtb = round(max(res_oi_wtb, res_vol_wtb), 3)
    sup_wtt = round(max(sup_oi_wtt, sup_vol_wtt), 3)
    sup_wtb = round(max(sup_oi_wtb, sup_vol_wtb), 3)

    res_state = _shift_state(res_wtt, res_wtb, threshold=wtt_threshold)
    sup_state = _shift_state(sup_wtt, sup_wtb, threshold=wtt_threshold)
    coa = _coa_regime(sup_state, res_state)

    eor = None
    if res_strike is not None:
        ce_ltp = _last_price((by_strike.get(res_strike) or {}).get("CE"))
        eor = round(res_strike + (ce_ltp or 0.0), 2)
    eos = None
    if sup_strike is not None:
        pe_ltp = _last_price((by_strike.get(sup_strike) or {}).get("PE"))
        eos = round(sup_strike - (pe_ltp or 0.0), 2)

    def dist(lvl: Optional[float]) -> Optional[float]:
        return round((lvl / spot - 1) * 100, 2) if lvl else None

    return {
        "spot": round(spot, 2),
        "res_strike": res_strike,
        "sup_strike": sup_strike,
        "ce_vol_wall": ce_vol_k,
        "pe_vol_wall": pe_vol_k,
        "res_wtt_ratio": res_wtt,
        "res_wtb_ratio": res_wtb,
        "sup_wtt_ratio": sup_wtt,
        "sup_wtb_ratio": sup_wtb,
        "res_state": res_state,
        "sup_state": sup_state,
        "coa_regime": coa,
        "eor": eor,
        "eos": eos,
        "eor_dist_pct": dist(eor),
        "eos_dist_pct": dist(eos),
    }


def features(record: Dict[str, Any], band: float = DEFAULT_BAND
             ) -> Optional[Dict[str, Any]]:
    """Positioning summary from one chain response, or ``None``.

    ``None`` means the payload could not describe this trade: no underlying
    price, no strikes, or no strikes inside the band. That is different from a
    zero, and the caller records it as such.

    Every field is in the panel because it was measured: the walls and PCR are
    kept even though they did not discriminate, because a filter that was never
    written down cannot be checked later.
    """
    spot = record.get("underlyingValue")
    rows = record.get("data") or []
    try:
        spot = float(spot)
    except (TypeError, ValueError):
        return None
    if spot <= 0 or not rows:
        return None

    lo, hi = spot * (1 - band), spot * (1 + band)
    in_band = [r for r in rows
               if lo <= float(r.get("strikePrice") or 0) <= hi and r.get("strikePrice")]
    if not in_band:
        return None

    ce_oi = sum(_oi(r.get("CE")) for r in in_band)
    pe_oi = sum(_oi(r.get("PE")) for r in in_band)
    traded = sum(_volume(r.get("CE")) + _volume(r.get("PE")) for r in in_band)
    live = sum(1 for r in in_band if _oi(r.get("CE")) or _oi(r.get("PE")))

    calls = [(float(r["strikePrice"]), _oi(r.get("CE"))) for r in in_band
             if float(r["strikePrice"]) > spot]
    puts = [(float(r["strikePrice"]), _oi(r.get("PE"))) for r in in_band
            if float(r["strikePrice"]) < spot]
    call_wall, call_wall_oi = max(calls, key=lambda x: x[1]) if calls else (None, 0)
    put_wall, put_wall_oi = max(puts, key=lambda x: x[1]) if puts else (None, 0)

    atm = min(in_band, key=lambda r: abs(float(r["strikePrice"]) - spot))
    atm_ivs = [v for v in (_iv(atm.get("CE")), _iv(atm.get("PE"))) if v]
    below = [r for r in in_band if float(r["strikePrice"]) <= spot * 0.95]
    above = [r for r in in_band if float(r["strikePrice"]) >= spot * 1.05]
    put_iv = _iv(min(below, key=lambda r: abs(float(r["strikePrice"]) - spot))["PE"]) if below else None
    call_iv = _iv(min(above, key=lambda r: abs(float(r["strikePrice"]) - spot))["CE"]) if above else None

    def dist(strike: Optional[float]) -> Optional[float]:
        return round((strike / spot - 1) * 100, 2) if strike else None

    return {
        "spot": round(spot, 2),
        "strikes_in_band": len(in_band),
        "live_strikes": live,
        "ce_oi": ce_oi,
        "pe_oi": pe_oi,
        "pcr": round(pe_oi / ce_oi, 3) if ce_oi else None,
        "traded_volume": traded,
        "call_wall": call_wall,
        "call_wall_dist_pct": dist(call_wall),
        "call_wall_oi": call_wall_oi,
        "put_wall": put_wall,
        "put_wall_dist_pct": dist(put_wall),
        "atm_iv": round(sum(atm_ivs) / len(atm_ivs), 2) if atm_ivs else None,
        "skew_put_minus_call": (round(put_iv - call_iv, 2)
                                if put_iv is not None and call_iv is not None else None),
    }


def snapshot_symbol(http: ChainHttp, symbol: str, *, today: Optional[date] = None,
                    min_dte: int = DEFAULT_MIN_DTE, band: float = DEFAULT_BAND
                    ) -> Dict[str, Any]:
    """One panel row for ``symbol``. Never raises for that symbol.

    A universe of 218 names will always contain a few that NSE answers badly
    today. Losing the other 217 rows to one exception would make the panel
    unmaintainable, so failures are recorded in ``error`` with ``ok=False`` -
    which also makes chain coverage measurable after the fact.
    """
    as_of = today or datetime.now(IST).date()
    row: Dict[str, Any] = {c: None for c in SNAPSHOT_COLUMNS}
    row.update({"as_of": as_of.isoformat(), "symbol": symbol, "ok": False,
                "fetched_at": datetime.now(IST).isoformat(timespec="seconds")})
    try:
        expiries = contract_expiries(http, symbol)
        if not expiries:
            row["error"] = "no listed options"
            return row
        expiry, dte = pick_expiry(expiries, as_of, min_dte)
        if expiry is None:
            row["error"] = f"unparseable expiry list ({expiries[:3]})"
            return row
        row["expiry"], row["dte"] = expiry, dte
        feats = features(chain_record(http, symbol, expiry), band=band)
        if feats is None:
            row["error"] = "chain carried no usable positioning"
            return row
    except DataError as exc:
        row["error"] = str(exc)
        return row
    except Exception as exc:  # noqa: BLE001 - one bad payload, not a bad panel
        log.warning("chain snapshot failed for %s: %s", symbol, exc)
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row.update(feats)
    row["ok"] = True
    return row


# ------------------------------------------------------------ snapshot panel


#: Explicit panel dtypes. Not cosmetic: pandas infers, and an all-NA column in
#: one frame concatenated with a typed one in the other is exactly the case that
#: emits a FutureWarning today and picks a different dtype tomorrow. A panel that
#: silently changes dtype between days is a panel that breaks the test it exists
#: to enable. ``call_wall`` stays float because a strike is a price, not a count.
_INT_COLUMNS: Tuple[str, ...] = (
    "dte", "strikes_in_band", "live_strikes", "ce_oi", "pe_oi",
    "traded_volume", "call_wall_oi",
)
_FLOAT_COLUMNS: Tuple[str, ...] = (
    "spot", "pcr", "call_wall", "call_wall_dist_pct", "put_wall",
    "put_wall_dist_pct", "atm_iv", "skew_put_minus_call",
)
_TEXT_COLUMNS: Tuple[str, ...] = ("as_of", "symbol", "error", "fetched_at", "expiry")


def _coerce(frame: pd.DataFrame) -> pd.DataFrame:
    """Give a frame the panel's exact columns and dtypes."""
    out = frame.copy()
    for col in SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    for col in _INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in _FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in _TEXT_COLUMNS:
        out[col] = out[col].astype("object").where(out[col].notna(), None)
    out["ok"] = out["ok"].fillna(False).astype(bool)
    return out[list(SNAPSHOT_COLUMNS)]


def load_panel(path: Any) -> pd.DataFrame:
    """The panel as written, or an empty frame with the right columns."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return _coerce(pd.DataFrame(columns=list(SNAPSHOT_COLUMNS)))
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            f"cannot read the chain panel at {path} ({type(exc).__name__}: {exc}). "
            f"This file is forward-collected and NSE publishes no history, so it "
            f"cannot be regenerated - move it aside (or fix it) rather than "
            f"letting a new snapshot overwrite it."
        ) from exc
    missing = [c for c in SNAPSHOT_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            f"{path} is missing column(s) {missing}; it is not a chain panel this "
            f"version wrote. Move it aside rather than appending to it."
        )
    return _coerce(df)


def append_rows(path: Any, rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Merge ``rows`` into the panel and return the whole frame.

    Re-running a snapshot on the same day **replaces** that day's rows for those
    symbols instead of duplicating them: a snapshot taken at 10:00 and repeated
    at 15:00 is the same trading day, and a panel with two rows per name per day
    would silently double-weight that day in any later test. The write is atomic
    (tmp file + ``os.replace``) so an interrupted run cannot leave a truncated
    parquet where the history used to be.
    """
    from pathlib import Path

    path = Path(path)
    new = _coerce(pd.DataFrame(list(rows)))
    if new.empty:
        return load_panel(path)

    existing = load_panel(path)
    if not existing.empty:
        keys = set(zip(new["as_of"], new["symbol"], strict=False))
        keep = ~existing.apply(lambda r: (r["as_of"], r["symbol"]) in keys, axis=1)
        existing = existing[keep]
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(["as_of", "symbol"], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return merged


# ------------------------------------------------------------ LTP logger panel


_LTP_INT_COLUMNS: Tuple[str, ...] = ("dte",)
_LTP_FLOAT_COLUMNS: Tuple[str, ...] = (
    "spot", "res_strike", "sup_strike", "ce_vol_wall", "pe_vol_wall",
    "res_wtt_ratio", "res_wtb_ratio", "sup_wtt_ratio", "sup_wtb_ratio",
    "eor", "eos", "eor_dist_pct", "eos_dist_pct",
)
_LTP_TEXT_COLUMNS: Tuple[str, ...] = (
    "as_of", "symbol", "error", "fetched_at", "expiry",
    "res_state", "sup_state", "coa_regime",
)


def snapshot_ltp(http: ChainHttp, symbol: str, *, today: Optional[date] = None,
                 min_dte: int = 0, band: float = DEFAULT_BAND) -> Dict[str, Any]:
    """One record-only LTP Calculator snapshot row for ``symbol``."""
    as_of = today or datetime.now(IST).date()
    row: Dict[str, Any] = {c: None for c in LTP_SNAPSHOT_COLUMNS}
    row.update({"as_of": as_of.isoformat(), "symbol": symbol, "ok": False,
                "fetched_at": datetime.now(IST).isoformat(timespec="seconds")})
    try:
        expiries = contract_expiries(http, symbol)
        if not expiries:
            row["error"] = "no listed options"
            return row
        expiry, dte = pick_expiry(expiries, as_of, min_dte)
        if expiry is None:
            row["error"] = f"unparseable expiry list ({expiries[:3]})"
            return row
        row["expiry"], row["dte"] = expiry, dte
        feats = ltp_features(chain_record(http, symbol, expiry), band=band)
        if feats is None:
            row["error"] = "chain carried no usable positioning"
            return row
    except DataError as exc:
        row["error"] = str(exc)
        return row
    except Exception as exc:  # noqa: BLE001
        log.warning("ltp snapshot failed for %s: %s", symbol, exc)
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row.update(feats)
    row["ok"] = True
    return row


def _coerce_ltp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in LTP_SNAPSHOT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    for col in _LTP_INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in _LTP_FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in _LTP_TEXT_COLUMNS:
        out[col] = out[col].astype("object").where(out[col].notna(), None)
    out["ok"] = out["ok"].fillna(False).astype(bool)
    return out[list(LTP_SNAPSHOT_COLUMNS)]


def load_ltp_panel(path: Any) -> pd.DataFrame:
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return _coerce_ltp(pd.DataFrame(columns=list(LTP_SNAPSHOT_COLUMNS)))
    df = pd.read_parquet(path)
    missing = [c for c in LTP_SNAPSHOT_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path} is missing column(s) {missing}")
    return _coerce_ltp(df)


def append_ltp_rows(path: Any, rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Append record-only LTP Calculator rows (idempotent per day + symbol)."""
    from pathlib import Path

    path = Path(path)
    new = _coerce_ltp(pd.DataFrame(list(rows)))
    if new.empty:
        return load_ltp_panel(path)
    existing = load_ltp_panel(path)
    if not existing.empty:
        keys = set(zip(new["as_of"], new["symbol"], strict=False))
        keep = ~existing.apply(lambda r: (r["as_of"], r["symbol"]) in keys, axis=1)
        existing = existing[keep]
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.sort_values(["as_of", "symbol"], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return merged
