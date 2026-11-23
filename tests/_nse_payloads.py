"""Recorded-shape payloads for the NSE and Yahoo chart endpoints.

Shared by ``test_nse_source.py`` and ``test_cli_nse.py`` so both exercise the
same fixtures. The shapes are *reconstructed* from the field names NSE and Yahoo
return and from ``vendor/market-lab``'s use of them - this sandbox reaches
neither host, so none of this was captured live. They pin the parsing contract;
they are not proof the live API still looks like this.
"""
import pandas as pd

def _recording(responses):
    """A fetcher that answers by URL substring and records what was asked for."""
    calls = []

    def fetcher(url, timeout):
        calls.append(url)
        for needle, body in responses.items():
            if needle in url:
                return body.encode() if isinstance(body, str) else body
        raise AssertionError(f"unexpected URL requested: {url}")

    return fetcher, calls


def _yahoo_payload(n=40, start_min=555, day="2026-11-20", price0=1000.0, with_nulls=False):
    """A Yahoo v8 chart response of 5m bars inside the NSE session."""
    base = pd.Timestamp(f"{day} 00:00", tz="UTC") + pd.Timedelta(minutes=start_min - 330)
    ts = [int((base + pd.Timedelta(minutes=5 * i)).timestamp()) for i in range(n)]
    close = [price0 * (1 + 0.001 * i) for i in range(n)]
    quote = {
        "open": [c * 0.999 for c in close],
        "high": [c * 1.004 for c in close],
        "low": [c * 0.996 for c in close],
        "close": close,
        "volume": [1000 * (i + 1) for i in range(n)],
    }
    if with_nulls:
        quote["close"][3] = None
        quote["open"][3] = None
        quote["high"][3] = None
        quote["low"][3] = None
        quote["volume"][5] = None
    return {"chart": {"result": [{
        "meta": {"symbol": "TCS.NS", "regularMarketPrice": close[-1]},
        "timestamp": ts,
        "indicators": {"quote": [quote]},
    }]}}


def _market_watch_payload():
    return {"data": {"data": [
        {"symbol": "RELIANCE", "series": "EQ", "lastPrice": 2950.5, "pChange": 1.2,
         "previousClose": 2915.0, "open": 2920.0, "dayHigh": 2960.0, "dayLow": 2910.0,
         "totalTradedVolume": 8_000_000, "totalTradedValue": 2.36e10,
         "yearHigh": 3200.0, "yearLow": 2200.0, "companyName": "reliance industries"},
        {"symbol": "TCS", "series": "EQ", "lastPrice": 3100.0, "pChange": -0.4,
         "previousClose": 3112.0, "open": 3110.0, "dayHigh": 3120.0, "dayLow": 3090.0,
         "totalTradedVolume": 1_200_000, "totalTradedValue": 3.7e9,
         "yearHigh": 4200.0, "yearLow": 2800.0, "companyName": "tata consultancy"},
        # a cheap stock with LARGER volume but far smaller traded value
        {"symbol": "YESBANK", "series": "EQ", "lastPrice": 21.0, "pChange": 0.5,
         "previousClose": 20.9, "open": 20.9, "dayHigh": 21.4, "dayLow": 20.8,
         "totalTradedVolume": 90_000_000, "totalTradedValue": 1.9e9,
         "yearHigh": 26.0, "yearLow": 15.0, "companyName": "yes bank"},
        # a non-EQ series that must be filtered out
        {"symbol": "RELIANCE", "series": "BE", "lastPrice": 1.0},
    ]}}


def _all_indices_payload():
    return {"data": [
        {"index": "NIFTY 50", "last": 24500.0, "percentChange": 0.6, "variation": 146.0,
         "previousClose": 24354.0, "advances": 34, "declines": 15, "unchanged": 1,
         "pe": 22.1, "yearHigh": 26300.0, "yearLow": 21000.0, "open": 24400.0},
        {"index": "NIFTY BANK", "last": 52000.0, "percentChange": -0.3, "variation": -156.0,
         "previousClose": 52156.0, "advances": 4, "declines": 8, "unchanged": 0,
         "pe": 18.0, "yearHigh": 58000.0, "yearLow": 44000.0, "open": 52100.0},
    ]}


def _holiday_payload():
    return {"CM": [
        {"tradingDate": "26-Nov-2026", "weekDay": "Thursday", "tradingDate1": "2026-11-26"},
        {"tradingDate": "25-Dec-2026", "weekDay": "Friday"},
        {"tradingDate": "not-a-date", "weekDay": "?"},          # must be skipped, not fatal
    ]}


NIFTY500_CSV_BODY = (
    "Company Name,Industry,Symbol,Series,ISIN Code\n"
    "Reliance Industries Ltd,ENERGY,RELIANCE,Equity,INE002A01018\n"
    "Tata Consultancy Services Ltd,IT,TCS,Equity,INE467B01029\n"
    "HDFC Bank Ltd,FINANCIAL SERVICES,HDFCBANK,Equity,INE040A01034\n"
)


