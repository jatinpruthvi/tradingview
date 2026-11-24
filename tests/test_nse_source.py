"""NSE + Yahoo-chart source: parsers, session filter, politeness and universes.

The payload fixtures here are *reconstructed* from the field names NSE and Yahoo
document in their responses and from ``vendor/market-lab``'s usage of them. This
sandbox cannot reach either host, so none of these shapes was captured live.
They pin the parsing contract; they are not proof the live API still looks like
this. Run ``tvlab data --data-source nse`` on a machine with network access to
find out, and fix the fixtures if NSE has moved.
"""
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tvlab.data.base import DataError
from tvlab.data.nse import (
    ALL_INDICES,
    HOLIDAY_MASTER,
    Http,
    INDIA_INDEX_SYMBOLS,
    IST,
    MARKET_WATCH,
    NIFTY500_CSV,
    NseSource,
    SESSION_END_MIN,
    SESSION_START_MIN,
    YahooChartSource,
    filter_ist_session,
    top_by_liquidity,
    yahoo_symbol,
)

# --------------------------------------------------------------- fixtures
#
# The payload builders live in tests/_nse_payloads.py so the CLI tests exercise
# exactly the same fixtures.
from _nse_payloads import (  # noqa: E402
    NIFTY500_CSV_BODY,
    _all_indices_payload,
    _holiday_payload,
    _market_watch_payload,
    _recording,
    _yahoo_payload,
)


# ------------------------------------------------------- symbol mapping


@pytest.mark.parametrize("symbol,expected", [
    ("RELIANCE", "RELIANCE.NS"),
    ("TCS", "TCS.NS"),
    ("RELIANCE.NS", "RELIANCE.NS"),          # already suffixed: not doubled
    ("NIFTY", "^NSEI"),
    ("NIFTY 50", "^NSEI"),
    ("BANKNIFTY", "^NSEBANK"),
    ("SENSEX", "^BSESN"),
    ("INDIAVIX", "^INDIAVIX"),
    ("^CNXIT", "^CNXIT"),                    # caret passes through
    ("EURUSD=X", "EURUSD=X"),                # FX passes through
    ("BTC/USDT", "BTC/USDT"),                # crypto passes through
    ("INFY.BO", "INFY.BO"),                  # BSE suffix passes through
])
def test_yahoo_symbol_mapping(symbol, expected):
    assert yahoo_symbol(symbol, "indian_equities") == expected


def test_yahoo_symbol_does_not_suffix_non_indian_markets():
    assert yahoo_symbol("AAPL", "us_equities") == "AAPL"
    assert yahoo_symbol("RELIANCE", "us_equities") == "RELIANCE"


def test_index_map_covers_the_names_used_in_configs():
    for name in ("NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "INDIAVIX"):
        assert name in INDIA_INDEX_SYMBOLS, name


# ------------------------------------------------------- Yahoo chart source


def test_fetch_ohlcv_parses_a_chart_response():
    fetcher, calls = _recording({"finance/chart": json.dumps(_yahoo_payload())})
    src = YahooChartSource(market="indian_equities", http=Http(fetcher=fetcher, delay=0))
    df = src.fetch_ohlcv("TCS", "5m")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 40
    assert df["close"].is_monotonic_increasing
    assert ".NS" in calls[0], "an Indian cash symbol must be suffixed for Yahoo"


def test_fetch_ohlcv_survives_null_bars():
    """Yahoo emits nulls for bars with no trade. Those rows must go; a null
    volume is a real zero and must stay."""
    fetcher, _ = _recording({"finance/chart": json.dumps(_yahoo_payload(with_nulls=True))})
    src = YahooChartSource(http=Http(fetcher=fetcher, delay=0))
    df = src.fetch_ohlcv("TCS", "5m")

    assert len(df) == 39, "the null OHLC bar must be dropped"
    assert not df[["open", "high", "low", "close"]].isna().any().any()
    assert (df["volume"] >= 0).all(), "null volume becomes 0, never NaN"


def test_fetch_ohlcv_raises_on_an_unknown_ticker():
    """An unknown ticker must raise, not return an empty frame - empty looks
    like 'no trades today' and produces a flat, plausible equity curve."""
    fetcher, _ = _recording({"finance/chart": json.dumps({"chart": {"result": None,
                                                                   "error": {"code": "Not Found"}}})})
    src = YahooChartSource(http=Http(fetcher=fetcher, delay=0))
    with pytest.raises(DataError, match="no chart data"):
        src.fetch_ohlcv("NOTAREALTICKER", "5m")


def test_fetch_ohlcv_raises_when_zero_bars_returned():
    fetcher, _ = _recording({"finance/chart": json.dumps(
        {"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{}]}}]}})})
    src = YahooChartSource(http=Http(fetcher=fetcher, delay=0))
    with pytest.raises(DataError, match="0 bars"):
        src.fetch_ohlcv("TCS", "5m")


def test_fetch_ohlcv_raises_on_non_json_interstitial():
    """NSE and Yahoo both serve HTML when they want a cookie. That must be a
    loud error naming the cause, not a JSON decode traceback."""
    fetcher, _ = _recording({"finance/chart": "<html><body>Access denied</body></html>"})
    src = YahooChartSource(http=Http(fetcher=fetcher, delay=0))
    with pytest.raises(DataError, match="did not return JSON"):
        src.fetch_ohlcv("TCS", "5m")


def test_unsupported_timeframe_resamples_from_a_supported_one():
    """4h has no Yahoo interval for Indian equities; it must come from daily and
    be resampled through the shared resampler rather than failing."""
    fetcher, calls = _recording({"finance/chart": json.dumps(_yahoo_payload(n=10))})
    src = YahooChartSource(http=Http(fetcher=fetcher, delay=0))
    src.fetch_ohlcv("TCS", "4h")
    assert "interval=1d" in calls[0]


def test_list_symbols_is_refused_with_guidance():
    src = YahooChartSource()
    with pytest.raises(DataError, match="NseSource"):
        src.list_symbols()


# ------------------------------------------------------- session filtering


def test_filter_ist_session_drops_pre_and_post_market():
    """Yahoo returns pre-open rows for Indian symbols. Left in they stretch the
    bar count and corrupt ATR across the overnight gap."""
    idx = pd.to_datetime([
        "2026-11-20 03:30:00+00:00",   # 09:00 IST - pre-open, drop
        "2026-11-20 03:45:00+00:00",   # 09:15 IST - open, keep
        "2026-11-20 09:55:00+00:00",   # 15:25 IST - keep
        "2026-11-20 10:00:00+00:00",   # 15:30 IST - close, drop
        "2026-11-20 10:30:00+00:00",   # 16:00 IST - drop
    ])
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    out = filter_ist_session(df)
    assert len(out) == 2, out.index.tolist()
    hours = out.index.tz_convert(IST).hour * 60 + out.index.tz_convert(IST).minute
    assert list(hours) == [SESSION_START_MIN, SESSION_END_MIN - 5]


def test_filter_ist_session_handles_a_naive_index_and_empties():
    idx = pd.to_datetime(["2026-11-20 03:45:00"])
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}, index=idx)
    assert len(filter_ist_session(df)) == 1
    assert filter_ist_session(df.iloc[0:0]).empty


def test_session_bounds_are_the_real_nse_hours():
    assert (SESSION_START_MIN, SESSION_END_MIN) == (555, 930), "09:15 to 15:30 IST"


# ------------------------------------------------------------------ NSE API


def test_market_watch_returns_only_the_eq_series():
    fetcher, _ = _recording({"marketWatchApi": json.dumps(_market_watch_payload())})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    rows = src.market_watch("NIFTY 50")
    assert [r["symbol"] for r in rows] == ["RELIANCE", "TCS", "YESBANK"]


def test_market_watch_raises_when_the_shape_changes():
    fetcher, _ = _recording({"marketWatchApi": json.dumps({"somethingElse": 1})})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    with pytest.raises(DataError, match="no data.data"):
        src.market_watch("NIFTY 50")


def test_snapshot_is_sorted_by_traded_value_not_volume():
    """Volume alone would rank YESBANK (90M shares, Rs 1.9bn) above TCS (1.2M
    shares, Rs 3.7bn). Traded value is what says you can actually get out."""
    fetcher, _ = _recording({"marketWatchApi": json.dumps(_market_watch_payload())})
    snap = NseSource(http=Http(fetcher=fetcher, delay=0)).snapshot("NIFTY 50")
    assert list(snap["symbol"]) == ["RELIANCE", "TCS", "YESBANK"]
    assert snap["traded_value"].is_monotonic_decreasing
    assert snap.iloc[0]["company"] == "Reliance Industries"   # title-cased


def test_snapshot_handles_an_empty_response():
    fetcher, _ = _recording({"marketWatchApi": json.dumps({"data": {"data": []}})})
    snap = NseSource(http=Http(fetcher=fetcher, delay=0)).snapshot()
    assert snap.empty and "traded_value" in snap.columns


def test_indices_computes_breadth():
    fetcher, _ = _recording({"allIndices": json.dumps(_all_indices_payload())})
    idx = NseSource(http=Http(fetcher=fetcher, delay=0)).indices()
    assert len(idx) == 2
    nifty = idx[idx["index"] == "NIFTY 50"].iloc[0]
    assert nifty["advances"] == 34 and nifty["declines"] == 15
    assert nifty["breadth"] == pytest.approx((34 - 15) / 50)
    assert idx[idx["index"] == "NIFTY BANK"].iloc[0]["breadth"] < 0


def test_indices_raises_when_data_is_missing():
    fetcher, _ = _recording({"allIndices": json.dumps({"error": "nope"})})
    with pytest.raises(DataError, match="allIndices"):
        NseSource(http=Http(fetcher=fetcher, delay=0)).indices()


def test_trading_holidays_parses_and_skips_bad_rows():
    fetcher, _ = _recording({"holiday-master": json.dumps(_holiday_payload())})
    days = NseSource(http=Http(fetcher=fetcher, delay=0)).trading_holidays()
    assert days == {date(2026, 11, 26), date(2026, 12, 25)}, "the unparseable row is skipped"


def test_trading_holidays_raises_on_an_unexpected_shape():
    fetcher, _ = _recording({"holiday-master": json.dumps({"EQ": []})})
    with pytest.raises(DataError, match="no 'CM' key"):
        NseSource(http=Http(fetcher=fetcher, delay=0)).trading_holidays()


def test_list_symbols_unions_the_csv_and_the_fo_list():
    fetcher, calls = _recording({
        "nsearchives": NIFTY500_CSV_BODY,
        "marketWatchApi": json.dumps({"data": {"data": [
            {"symbol": "TCS", "series": "EQ"},
            {"symbol": "IDEA", "series": "EQ"},      # F&O only, not in the Nifty 500 CSV
        ]}}),
    })
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    syms = src.list_symbols()
    assert syms == ["HDFCBANK", "IDEA", "RELIANCE", "TCS"]
    assert any("nsearchives" in c for c in calls) and any("marketWatchApi" in c for c in calls)


def test_list_symbols_degrades_when_one_leg_fails():
    """A partial universe is still usable; no universe is not."""
    def fetcher(url, timeout):
        if "nsearchives" in url:
            raise DataError("HTTP 403 from nsearchives")
        return json.dumps({"data": {"data": [{"symbol": "TCS", "series": "EQ"}]}}).encode()

    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    assert src.list_symbols() == ["TCS"]


def test_list_symbols_raises_only_when_every_leg_fails():
    def fetcher(url, timeout):
        raise DataError("HTTP 403")

    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    with pytest.raises(DataError, match="could not build an NSE universe"):
        src.list_symbols()


def test_nse_source_conforms_to_the_data_source_protocol():
    """--data-source nse must work anywhere a source is accepted, which means
    history has to come from somewhere even though NSE has no OHLCV endpoint."""
    fetcher, _ = _recording({"finance/chart": json.dumps(_yahoo_payload())})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    df = src.fetch_ohlcv("TCS", "5m")
    assert len(df) == 40 and src.name == "nse"


# ------------------------------------------------------------- session clock


def test_is_session_open_uses_the_calendar_and_the_clock():
    fetcher, _ = _recording({"holiday-master": json.dumps(_holiday_payload())})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    holidays = src.trading_holidays()
    at = lambda h, m, d=20: datetime(2026, 11, d, h, m, tzinfo=IST)   # noqa: E731

    assert src.is_session_open(at(10, 0), holidays) is True
    assert src.is_session_open(at(9, 14), holidays) is False, "one minute before the open"
    assert src.is_session_open(at(15, 30), holidays) is False, "at the close"
    assert src.is_session_open(at(12, 0, d=21), holidays) is False, "Saturday"
    assert src.is_session_open(at(12, 0, d=26), holidays) is False, "a listed holiday"


def test_is_session_open_survives_no_network():
    """Without the calendar it must still answer from weekday and clock, and say
    it guessed rather than silently treating a holiday as a trading day."""
    def fetcher(url, timeout):
        raise DataError("unreachable")

    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    assert src.is_session_open(datetime(2026, 11, 20, 10, 0, tzinfo=IST)) is True
    assert src.is_session_open(datetime(2026, 11, 21, 10, 0, tzinfo=IST)) is False


def test_minutes_to_close():
    fetcher, _ = _recording({"holiday-master": json.dumps({"CM": []})})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    assert src.minutes_to_close(datetime(2026, 11, 20, 14, 3, tzinfo=IST)) == 87
    assert src.minutes_to_close(datetime(2026, 11, 20, 9, 15, tzinfo=IST)) == 375
    assert src.minutes_to_close(datetime(2026, 11, 20, 16, 0, tzinfo=IST)) is None


def test_naive_timestamps_are_read_as_ist():
    """Documented convention: a naive datetime is IST, because this is an
    India-specific API and a caller passing datetime.now() in India is passing
    IST. is_session_open and minutes_to_close must agree on it."""
    fetcher, _ = _recording({"holiday-master": json.dumps({"CM": []})})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    naive = datetime(2026, 11, 20, 14, 3)
    aware = datetime(2026, 11, 20, 14, 3, tzinfo=IST)
    assert src.minutes_to_close(naive) == src.minutes_to_close(aware) == 87
    assert src.is_session_open(naive) == src.is_session_open(aware) is True


def test_aware_utc_timestamps_are_converted_to_ist():
    """Bar timestamps from Yahoo are tz-aware UTC; 08:33 UTC is 14:03 IST."""
    fetcher, _ = _recording({"holiday-master": json.dumps({"CM": []})})
    src = NseSource(http=Http(fetcher=fetcher, delay=0))
    utc = datetime(2026, 11, 20, 8, 33, tzinfo=ZoneInfo("UTC"))
    assert src.minutes_to_close(utc) == 87
    assert src.is_session_open(utc) is True


# ------------------------------------------------------------------ liquidity


def test_top_by_liquidity_ranks_by_traded_value():
    """The cheap stock has 75x the volume and half the traded value. Ranking on
    volume would put you in the name you cannot exit."""
    fetcher, _ = _recording({"marketWatchApi": json.dumps(_market_watch_payload())})
    snap = NseSource(http=Http(fetcher=fetcher, delay=0)).snapshot()
    pick = top_by_liquidity(["RELIANCE", "TCS", "YESBANK"], snap, 2)
    assert pick == ["RELIANCE", "TCS"], "YESBANK has more volume but less value"


def test_top_by_liquidity_survives_an_empty_snapshot():
    assert top_by_liquidity(["B", "A", "C"], pd.DataFrame(), 2) == ["A", "B"]


def test_top_by_liquidity_keeps_requested_symbols_missing_from_the_snapshot():
    fetcher, _ = _recording({"marketWatchApi": json.dumps(_market_watch_payload())})
    snap = NseSource(http=Http(fetcher=fetcher, delay=0)).snapshot()
    pick = top_by_liquidity(["RELIANCE", "UNKNOWNNAME"], snap, 5)
    assert "RELIANCE" in pick and "UNKNOWNNAME" in pick


# ---------------------------------------------------------------- politeness


def test_http_is_sequential_and_delays_between_requests():
    """One symbol at a time, with a pause. The difference between a research
    tool and something that gets an IP banned from NSE."""
    seen = []

    def fetcher(url, timeout):
        seen.append(url)
        return b"{}"

    http = Http(fetcher=fetcher, delay=0.01)
    for i in range(3):
        http.get_json(f"https://example.test/{i}")
    assert seen == [f"https://example.test/{i}" for i in range(3)]
    assert http.requests == 3


def test_http_reports_an_unreachable_host_clearly():
    """A network failure must name the host and say it has to be reachable from
    where tvlab runs - not surface as an empty result.

    Targets a closed loopback port rather than a real endpoint. The original
    version pointed at Yahoo's chart API and only passed because this machine
    had no route to it: as soon as the data route worked, the test failed on a
    200. A test of offline behaviour must not require the internet to be down.
    """
    from tvlab.data.nse import _default_fetcher
    with pytest.raises(DataError, match="could not reach 127.0.0.1:9"):
        _default_fetcher("http://127.0.0.1:9/v8/finance/chart/AAPL?interval=1d&range=1d", 5)


def test_note_rate_limit_backs_off():
    http = Http(fetcher=lambda url, t: b"{}", delay=0)
    http.note_rate_limit(0.05)
    assert http.rate_limited_until > 0


def test_endpoint_constants_are_https():
    for url in (NIFTY500_CSV, ALL_INDICES, MARKET_WATCH.format(q="x"), HOLIDAY_MASTER):
        assert url.startswith("https://"), url
