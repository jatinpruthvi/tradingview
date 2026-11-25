"""The NSE delivery join: session alignment, dedup, and what it refuses to do.

Every test here is offline. The risky part of this module is not the HTTP call,
it is the *date*, so the fixtures reproduce the payload shape that jugaad-data
0.37.4 actually returns - including the 18:30-UTC stamp and the duplicate T0
rows - rather than a tidied-up version of it.
"""
from datetime import date
import sys

import numpy as np
import pandas as pd
import pytest

from tvlab.config import load_config
from tvlab.data.base import DataError
from tvlab.data.delivery import (
    DeliveryAugmentedSource,
    attach_delivery,
    fetch_delivery,
    session_dates,
)
from tvlab.data.store import DataStore

IST = "Asia/Kolkata"
DAYS = [date(2025, 1, 20), date(2025, 1, 21), date(2025, 1, 22),
        date(2025, 1, 23), date(2025, 1, 24)]


def _jugaad_stamp(d):
    """IST midnight as jugaad-data 0.37.4 stamps it: 18:30 UTC the day before."""
    return pd.Timestamp(d).tz_localize(IST).tz_convert("UTC").tz_localize(None)


def _payload(days, delivery_pct=40.0, value=1e9, trades=1000, series="EQ"):
    return pd.DataFrame([{
        "DATE": _jugaad_stamp(d), "SERIES": series, "OPEN": 100.0, "HIGH": 101.0,
        "LOW": 99.0, "PREV. CLOSE": 99.5, "LTP": 100.5, "CLOSE": 100.4,
        "VWAP": 100.2, "VOLUME": 9_000_000, "VALUE": value, "NO OF TRADES": trades,
        "DELIVERY QTY": 3_600_000, "DELIVERY %": delivery_pct, "SYMBOL": "TEST",
    } for d in days])


def _prices(days):
    """A daily frame stamped the way the Yahoo source stamps Indian dailies."""
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(IST).tz_convert("UTC") for d in days])
    n = len(days)
    return pd.DataFrame({
        "open": 100.0 + np.arange(n), "high": 102.0 + np.arange(n),
        "low": 99.0 + np.arange(n), "close": 101.0 + np.arange(n),
        "volume": 1e6,
    }, index=idx)


# ------------------------------------------------------------------ session date
def test_utc_stamped_ist_midnight_maps_to_the_session_date():
    """18:30 UTC is IST midnight *of the next day*; taking .dt.date shifts a day
    and turns a causal filter into a lookahead one."""
    out = session_dates(_payload([date(2025, 1, 20)])["DATE"])
    assert list(out) == [pd.Timestamp("2025-01-20")]


def test_naive_midnight_is_left_alone():
    out = session_dates(pd.Series([pd.Timestamp("2025-01-20")]))
    assert list(out) == [pd.Timestamp("2025-01-20")]


def test_dates_are_normalised_to_midnight():
    out = session_dates(pd.Series([_jugaad_stamp(date(2025, 1, 20))]))
    assert out[0] == pd.Timestamp("2025-01-20")
    assert out[0].hour == 0


# ----------------------------------------------------------------------- fetch
def test_t0_series_rows_are_dropped_and_dates_deduped():
    frame = pd.concat([_payload(DAYS), _payload(DAYS, delivery_pct=95.0, series="T0")],
                      ignore_index=True)
    out = fetch_delivery("TEST", DAYS[0], DAYS[-1], fetch=lambda *a: frame)
    assert len(out) == len(DAYS)
    assert out["delivery_pct"].eq(40.0).all()


def test_avg_trade_value_is_value_over_trade_count():
    out = fetch_delivery("TEST", DAYS[0], DAYS[-1],
                         fetch=lambda *a: _payload(DAYS[:1], value=1e9, trades=1000))
    assert out["avg_trade_value"].iloc[0] == pytest.approx(1e6)


def test_zero_trade_count_is_nan_not_infinity():
    out = fetch_delivery("TEST", DAYS[0], DAYS[-1],
                         fetch=lambda *a: _payload(DAYS[:1], value=1e9, trades=0))
    assert np.isnan(out["avg_trade_value"].iloc[0])


def test_delivery_percentage_is_clipped_to_its_range():
    out = fetch_delivery("TEST", DAYS[0], DAYS[-1],
                         fetch=lambda *a: _payload(DAYS[:1], delivery_pct=140.0))
    assert out["delivery_pct"].iloc[0] == 100.0


def test_empty_payload_raises_instead_of_returning_an_empty_frame():
    with pytest.raises(DataError, match="no delivery rows"):
        fetch_delivery("TEST", DAYS[0], DAYS[-1], fetch=lambda *a: pd.DataFrame())


def test_payload_without_a_delivery_column_names_the_missing_field():
    bad = _payload(DAYS).drop(columns=["DELIVERY %"])
    with pytest.raises(DataError, match="DELIVERY %"):
        fetch_delivery("TEST", DAYS[0], DAYS[-1], fetch=lambda *a: bad)


def test_missing_package_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "jugaad_data.nse", None)
    with pytest.raises(DataError, match=r"tvlab\[delivery\]"):
        fetch_delivery("TEST", DAYS[0], DAYS[-1])


# ------------------------------------------------------------------------ join
def test_join_keeps_every_price_column_and_adds_the_two_nse_ones():
    px = _prices(DAYS)
    out = attach_delivery(px, "TEST", fetch=lambda *a: _payload(DAYS))

    pd.testing.assert_frame_equal(out[px.columns], px)
    assert out["_delivery_pct"].eq(40.0).all()
    assert out["_avg_trade_value"].eq(1e6).all()
    assert out.attrs["delivery_coverage"] == 1.0


def test_join_uses_the_session_date_not_the_price_timestamp():
    """The price index is 09:15 IST, the delivery row is IST midnight; both are
    the same session and must land on the same row."""
    out = attach_delivery(_prices(DAYS), "TEST", fetch=lambda *a: _payload(DAYS, delivery_pct=55.0))
    assert out["_delivery_pct"].eq(55.0).all()


def test_missing_sessions_stay_nan_and_are_never_forward_filled():
    px = _prices(DAYS)
    out = attach_delivery(px, "TEST", fetch=lambda *a: _payload(DAYS[:2], delivery_pct=30.0))
    assert out["_delivery_pct"].iloc[:2].eq(30.0).all()
    assert out["_delivery_pct"].iloc[2:].isna().all()
    assert out.attrs["delivery_coverage"] == pytest.approx(0.4)


def test_intraday_frames_are_returned_without_delivery_columns():
    idx = pd.DatetimeIndex([pd.Timestamp("2025-01-20 09:15", tz=IST)
                            + pd.Timedelta(minutes=5 * i) for i in range(80)])
    px = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                       "volume": 1e4}, index=idx)

    def _boom(*a):  # the fetch must not even be attempted
        raise AssertionError("delivery fetch on an intraday frame")

    out = attach_delivery(px, "TEST", fetch=_boom)
    assert "_delivery_pct" not in out.columns
    pd.testing.assert_frame_equal(out, px)


# ---------------------------------------------------------------------- source
class _Inner:
    name = "inner"

    def fetch_ohlcv(self, symbol, timeframe="1d", start=None, end=None, limit=None):
        return _prices(DAYS)

    def list_symbols(self):
        return ["TEST"]

    def market_watch(self):
        return [{"symbol": "TEST"}]


def test_source_serves_inner_prices_with_delivery_attached():
    src = DeliveryAugmentedSource(_Inner(), fetch=lambda *a: _payload(DAYS))
    df = src.fetch_ohlcv("TEST", "1d")
    assert src.name == "nse_delivery"
    assert df["_delivery_pct"].eq(40.0).all()


def test_source_delegates_everything_else_to_the_inner_source():
    src = DeliveryAugmentedSource(_Inner(), fetch=lambda *a: _payload(DAYS))
    assert src.list_symbols() == ["TEST"]
    assert src.market_watch() == [{"symbol": "TEST"}]


def test_store_can_build_it_by_name_without_network():
    cfg = load_config(market="indian_equities", style="swing")
    src = DataStore(cfg).source("nse_delivery")
    assert src.name == "nse_delivery"
    assert type(src).__name__ == "DeliveryAugmentedSource"
