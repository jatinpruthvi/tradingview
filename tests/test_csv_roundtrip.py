"""The ``tvlab data --export`` -> ``--data-source csv`` round trip.

This workflow is what a local setup with live data actually runs: fetch real bars
once, then study them offline. It was documented in the README from the start and
had never worked, for three independent reasons, each of which failed at a
different distance from its cause:

1. ``DataStore`` hardcoded the CSV root to ``<cache_dir>/csv`` - *inside the cache
   directory*, so it could not be pointed anywhere else, did not exist on a fresh
   clone, and was deleted whenever the cache was cleared. There is now a
   ``data.csv_root`` config field.
2. ``CsvSource`` never tried the *underscore*-flattened symbol form. The exporter
   writes ``BTC_USDT_4h.parquet``; the matcher tried ``BTC/USDT``, ``BTC-USDT``,
   ``BTC.USDT`` and ``BTCUSDT``. It also ignored the ``_<timeframe>`` suffix, and
   ``list_symbols()`` reported ``BTC_USDT_4h`` as a symbol, which no config names.
3. ``CsvSource`` raised ``KeyError``/``FileNotFoundError`` where the data layer's
   contract is ``DataError``. That is the one that mattered most: ``bias_frame()``
   catches ``DataError`` and resamples the higher timeframe up from the execution
   bars, which is the correct behaviour when a directory holds only one timeframe.
   A ``KeyError`` escaped instead, killed ``run_symbol``, and surfaced as
   "no symbol produced a result for strategy 'smc_trend'" - blaming the strategy
   for a missing file.
"""
from pathlib import Path

import pandas as pd
import pytest

from tvlab.config import load_config
from tvlab.data.base import DataError
from tvlab.data.csv_source import CsvSource, _symbol_filenames
from tvlab.data.store import DataStore, _csv_root


@pytest.fixture
def bars():
    idx = pd.date_range("2022-01-01", periods=200, freq="4h", tz="UTC")
    import numpy as np

    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 1, len(idx)).cumsum()
    return pd.DataFrame({
        "open": close + rng.normal(0, .2, len(idx)),
        "high": close + abs(rng.normal(0, .5, len(idx))),
        "low": close - abs(rng.normal(0, .5, len(idx))),
        "close": close,
        "volume": rng.uniform(1e5, 1e6, len(idx)),
    }, index=idx)


@pytest.fixture
def exported(tmp_path, bars):
    """A directory laid out exactly the way `tvlab data --export` writes it."""
    root = tmp_path / "live"
    root.mkdir()
    for sym in ("BTC/USDT", "ETH/USDT"):
        safe = "".join(ch if ch.isalnum() else "_" for ch in sym)
        bars.to_parquet(root / f"{safe}_4h.parquet")
    return root


# ------------------------------------------------------- 2. symbol name matching


def test_symbol_filenames_includes_the_underscore_form():
    """The exporter replaces every non-alphanumeric character with `_`. Without
    this candidate the round trip cannot find its own files."""
    cands = _symbol_filenames("BTC/USDT")
    assert "BTC_USDT" in cands
    assert "BTC/USDT" in cands and "BTC-USDT" in cands and "BTCUSDT" in cands


def test_symbol_filenames_handles_exchange_colon_suffixes():
    cands = _symbol_filenames("BTC/USDT:BINANCE")
    assert "BTC_USDT" in cands, "the exchange suffix must not defeat the flat form"


def test_csv_source_reads_the_exported_layout(exported):
    src = CsvSource(exported)
    for sym in ("BTC/USDT", "ETH/USDT"):
        df = src.fetch_ohlcv(sym, "4h")
        assert len(df) == 200
        assert list(df.columns)[:4] == ["open", "high", "low", "close"]


def test_list_symbols_strips_the_timeframe_suffix(exported):
    """Otherwise the universe reports `BTC_USDT_4h`, a symbol no config will ever
    name, and the mismatch looks like a missing file."""
    assert CsvSource(exported).list_symbols() == ["BTC_USDT", "ETH_USDT"]


def test_list_symbols_does_not_strip_a_real_underscore_name(tmp_path, bars):
    """`RELIANCE_NSDQ` is not a timeframe suffix and must survive intact."""
    bars.to_parquet(tmp_path / "RELIANCE_NSDQ_1d.parquet")
    bars.to_parquet(tmp_path / "BTC_USDT.parquet")
    assert CsvSource(tmp_path).list_symbols() == ["BTC_USDT", "RELIANCE_NSDQ"]


def test_timeframe_suffix_is_preferred_over_a_bare_match(tmp_path, bars):
    """With several timeframes on disk, asking for 4h must get the 4h file rather
    than whichever matched first."""
    bars.to_parquet(tmp_path / "BTC_USDT_4h.parquet")
    bars.iloc[::4].to_parquet(tmp_path / "BTC_USDT_1d.parquet")
    src = CsvSource(tmp_path)
    assert len(src.fetch_ohlcv("BTC/USDT", "4h")) == 200
    assert len(src.fetch_ohlcv("BTC/USDT", "1d")) == 50


def test_a_bare_filename_still_matches(exported, bars):
    """The pre-existing layout, without a timeframe suffix, must keep working."""
    bars.to_parquet(exported / "SOL_USDT.parquet")
    assert len(CsvSource(exported).fetch_ohlcv("SOL/USDT", "4h")) == 200


# --------------------------------------------------------- 3. the error contract


def test_a_missing_symbol_raises_data_error_not_key_error(exported):
    """The bug that broke the whole workflow. `bias_frame()` catches DataError and
    resamples; anything else escapes and becomes
    "no symbol produced a result for strategy ..."."""
    with pytest.raises(DataError) as exc:
        CsvSource(exported).fetch_ohlcv("DOES/NOTEXIST", "4h")
    assert not isinstance(exc.value, KeyError)
    msg = str(exc.value)
    assert "DOES/NOTEXIST" in msg
    assert "BTC_USDT" in msg, "must list what IS available, or the fix is guesswork"


def test_a_missing_timeframe_raises_data_error_so_bias_can_resample(exported):
    """The directory holds only 4h. Asking for the 1d bias timeframe must fail in a
    way `bias_frame()` can recover from - that is the designed fallback."""
    with pytest.raises(DataError):
        CsvSource(exported).fetch_ohlcv("BTC/USDT", "1d")


def test_bias_frame_resamples_when_only_one_timeframe_was_exported(exported):
    """End to end: the situation the README tells people to create. Export the
    execution timeframe, then study - the bias frame has to come from a resample."""
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": str(exported)})
    store = DataStore(cfg)
    bias = store.bias_frame("BTC/USDT")
    assert bias is not None, "bias_frame gave up instead of resampling"
    assert bias.index.freq is not None or len(bias) > 0
    # the resampled frame must be COARSER than the execution bars
    exec_df = store.ohlcv("BTC/USDT", "4h")
    assert len(bias) < len(exec_df)


def test_a_missing_root_raises_data_error_naming_the_config_key(tmp_path):
    with pytest.raises(DataError) as exc:
        CsvSource(tmp_path / "nope")
    msg = str(exc.value)
    assert "data.csv_root" in msg, "must name the knob that fixes it"
    assert "tvlab data --export" in msg


def test_a_long_format_symbol_miss_is_also_a_data_error(tmp_path):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=5, freq="4h", tz="UTC"),
        "symbol": ["A", "A", "A", "A", "A"],
        "open": [1.0] * 5, "high": [2.0] * 5, "low": [0.5] * 5,
        "close": [1.5] * 5, "volume": [10.0] * 5,
    })
    f = tmp_path / "long.csv"
    df.to_csv(f, index=False)
    with pytest.raises(DataError):
        CsvSource(f).fetch_ohlcv("NOT_THERE", "4h")


# ------------------------------------------------------------ 1. csv_root config


def test_data_config_has_a_csv_root_field():
    cfg = load_config(market="crypto", style="swing")
    assert cfg.data.csv_root == "data/live", \
        "the default must be where the README tells people to export"


def test_csv_root_is_settable_from_the_cli(tmp_path):
    """`--set data.csv_root=...` has to reach the store, or the flag is decoration."""
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": str(tmp_path)})
    assert cfg.data.csv_root == str(tmp_path)
    assert _csv_root(cfg) == Path(tmp_path)


def test_csv_root_resolves_relative_paths_against_the_repo_root(tmp_path, monkeypatch):
    """Relative to the repo, not the process CWD, so the same config works from a
    script, a cron job and the dashboard."""
    import tvlab.data.store as store_mod

    monkeypatch.setattr(store_mod, "REPO_ROOT", tmp_path)
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": "my/bars"})
    (tmp_path / "my" / "bars").mkdir(parents=True)
    assert _csv_root(cfg) == tmp_path / "my" / "bars"


def test_csv_root_accepts_an_absolute_path(tmp_path):
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": str(tmp_path)})
    assert _csv_root(cfg) == tmp_path


def test_legacy_cache_csv_location_is_still_honoured(tmp_path, monkeypatch):
    """An existing setup pointed at `<cache_dir>/csv` must keep loading after the
    upgrade rather than failing on a path it never chose."""
    import logging

    import tvlab.data.store as store_mod

    monkeypatch.setattr(store_mod, "REPO_ROOT", tmp_path)
    legacy = tmp_path / "data" / "cache" / "csv"
    legacy.mkdir(parents=True)
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": "data/live"})   # does not exist
    assert _csv_root(cfg) == legacy


def test_unresolvable_csv_root_falls_through_to_the_configured_path(tmp_path,
                                                                    monkeypatch):
    """When neither location exists, return the configured one so CsvSource raises
    naming the path the user actually asked for - not the legacy fallback."""
    import tvlab.data.store as store_mod

    monkeypatch.setattr(store_mod, "REPO_ROOT", tmp_path)
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": "wanted/here"})
    assert _csv_root(cfg) == tmp_path / "wanted" / "here"
    with pytest.raises(DataError) as exc:
        CsvSource(_csv_root(cfg))
    assert "wanted/here" in str(exc.value)


def test_store_dispatches_to_csv_with_the_configured_root(exported):
    cfg = load_config(market="crypto", style="swing").merge(
        {"data.source": "csv", "data.csv_root": str(exported)})
    src = DataStore(cfg).source()
    assert isinstance(src, CsvSource)
    assert Path(src.root) == exported
