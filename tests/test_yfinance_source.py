"""yfinance adapter: what it asks the vendor for.

Every test here stubs ``_download`` rather than reaching Yahoo, so the suite stays
offline. What is being pinned is the *request*, because that is where the adapter
went wrong twice and both failures were silent:

1. ``interval`` was passed both positionally and in ``**kwargs``, so
   ``fetch_ohlcv`` raised ``TypeError`` before any HTTP call - every supported
   interval was unusable and the error read like a data outage.
2. ``limit`` was dropped on the resample path (``4h`` resamples from ``1h``), and
   with no dates yfinance falls back to ``period="1mo"``. A 1200-bar 4h scan
   therefore received ~130 bars, which is enough to pass the store's 60-bar floor
   but not enough for daily bias history: the trend filter quietly disabled
   itself instead of the run failing.
"""
import numpy as np
import pandas as pd
import pytest

from tvlab.data.network_sources import YF_DEFAULT_PERIOD, YF_INTERVALS, YFinanceSource


def _frame(n: int, freq: str = "1D", start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    base = 100.0 + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * 1.002,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _source(n: int, freq: str = "1D"):
    """A YFinanceSource with its importable dependency bypassed.

    Constructing it normally needs the optional ``yfinance`` package; the request
    construction under test does not, so the tests must not skip on a machine
    without it (which is the documented "core only" install).
    """
    src = YFinanceSource.__new__(YFinanceSource)
    src._yf = None
    src.session = None
    src.adjust = "auto"
    calls = []

    def fake_download(symbol, interval, start=None, end=None, **kwargs):
        calls.append({"symbol": symbol, "interval": interval, "start": start,
                      "end": end, **kwargs})
        return _frame(n, freq)

    src._download = fake_download
    return src, calls


def test_native_interval_is_passed_once():
    """Regression: interval reached _download twice and raised TypeError."""
    src, calls = _source(400)
    df = src.fetch_ohlcv("SPY", "1d", limit=200)

    assert len(calls) == 1
    assert calls[0]["interval"] == "1d"
    assert calls[0]["auto_adjust"] is True
    assert not df.empty


def test_explicit_limit_becomes_a_period():
    """`limit` with no dates must still produce a period, never the 1-month default."""
    src, calls = _source(400)
    src.fetch_ohlcv("SPY", "1d", limit=200)

    assert calls[0]["period"] == "1y"
    assert calls[0]["period"] != "1mo"


def test_dates_are_forwarded_instead_of_a_period():
    src, calls = _source(400)
    src.fetch_ohlcv("SPY", "1d", start="2025-01-01", end="2025-06-01")

    assert calls[0]["start"] == "2025-01-01"
    assert calls[0]["end"] == "2025-06-01"
    assert "period" not in calls[0]


def test_no_dates_and_no_limit_asks_for_full_history():
    """The bias frame requests daily bars with neither dates nor a limit."""
    src, calls = _source(400)
    src.fetch_ohlcv("SPY", "1d")

    assert calls[0]["period"] == YF_DEFAULT_PERIOD["1d"]
    assert calls[0]["period"] != "1mo"


@pytest.mark.parametrize("timeframe, base_interval", [("4h", "1h"), ("12h", "1h")])
def test_resample_path_scales_the_requested_limit(timeframe, base_interval):
    """A 4h scan of 1200 bars must not ask for 1 month of 1h bars.

    The limit counts bars of the requested timeframe, so the base fetch has to
    over-ask; anything near "1mo" is the regression.
    """
    src, calls = _source(5000, freq="1h")
    src.fetch_ohlcv("SPY", timeframe, limit=1200)

    assert calls[0]["interval"] == base_interval
    # 1200 4h bars is ~200 days, so the request has to reach at least a year of 1h
    assert calls[0]["period"] in ("1y", "2y", "5y", "10y", "max")
    assert calls[0]["period"] != "1mo"


def test_resample_path_without_a_limit_uses_full_history():
    src, calls = _source(5000, freq="1h")
    src.fetch_ohlcv("SPY", "4h")

    assert calls[0]["period"] == YF_DEFAULT_PERIOD[YF_INTERVALS["1h"]]
    assert calls[0]["period"] != "1mo"


def test_every_supported_interval_has_a_default_period():
    for interval in set(YF_INTERVALS.values()):
        assert interval in YF_DEFAULT_PERIOD
