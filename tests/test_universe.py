"""Universe resolution: who decides which symbols get screened.

The bug these pin is a silent one. `tvlab scan --market indian_equities
--data-source nse` looked at 12 hand-typed large caps, and `scan` then cut that to
`syms[:top * 2]`, so `--top 20` examined 40 names at most. Meanwhile NSE publishes
~500. The output still read "0 live signal(s) of 12 symbols scanned" - a claim
about the market that only covered 2% of it.

Nothing here needs a network: the enumerable-source behaviour is driven through a
stub source, and the one NSE-specific assertion is skipped if the listing is
unreachable.
"""
import pandas as pd
import pytest
from click.testing import CliRunner

from tvlab.cli import _resolve_universe, cli
from tvlab.config import load_config
from tvlab.data.base import DataError
from tvlab.data.store import DataStore
from tvlab.screener import scan_universe


class _ListingSource:
    """Minimal DataSource that enumerates a fake exchange."""

    name = "stub"

    def __init__(self, symbols):
        self._symbols = list(symbols)
        self.listed = 0

    def list_symbols(self):
        self.listed += 1
        return list(self._symbols)

    def fetch_ohlcv(self, symbol, timeframe="1d", start=None, end=None, limit=None):
        raise DataError("not used by these tests")


class _MuteSource(_ListingSource):
    """A source with no listing endpoint at all - like yfinance."""

    def list_symbols(self):
        raise DataError("yfinance has no symbol listing endpoint")


def _cfg(source="stub", **universe):
    cfg = load_config(market="indian_equities", style="swing").merge(
        {"data.source": source, **{f"universe.{k}": v for k, v in universe.items()}}
    )
    return cfg


# ------------------------------------------------------------------- the store
def test_universe_prefers_what_the_source_lists(monkeypatch):
    cfg = _cfg()
    store = DataStore(cfg)
    stub = _ListingSource(["AAA", "BBB", "CCC"])
    monkeypatch.setattr(store, "source", lambda name=None: stub)

    syms, origin = store.universe()
    assert syms == ["AAA", "BBB", "CCC"]
    assert origin == "source"
    assert stub.listed == 1


def test_universe_falls_back_to_config_when_the_source_cannot_list(monkeypatch):
    cfg = _cfg()
    store = DataStore(cfg)
    monkeypatch.setattr(store, "source", lambda name=None: _MuteSource([]))

    syms, origin = store.universe()
    assert origin == "config"
    assert syms == list(cfg.universe.symbols)


def test_universe_prefer_source_false_never_calls_the_source(monkeypatch):
    cfg = _cfg()
    store = DataStore(cfg)
    stub = _ListingSource(["AAA"])
    monkeypatch.setattr(store, "source", lambda name=None: stub)

    syms, origin = store.universe(prefer_source=False)
    assert origin == "config" and stub.listed == 0
    assert syms == list(cfg.universe.symbols)


# --------------------------------------------------------------- the resolver
def test_explicit_symbols_win_over_everything(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: (_ for _ in ()).throw(
                            AssertionError("the source must not be consulted")))
    syms, note = _resolve_universe(cfg, "X.NS,Y.NS", "source")
    assert syms == ["X.NS", "Y.NS"]
    assert "--symbols" in note


def test_config_mode_uses_the_typed_list():
    cfg = _cfg()
    syms, note = _resolve_universe(cfg, None, "config")
    assert syms == list(cfg.universe.symbols)
    assert "config" in note


def test_source_mode_applies_the_cap(monkeypatch):
    cfg = _cfg(max_symbols=3)
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: ([f"S{i}" for i in range(10)], "source"))
    syms, note = _resolve_universe(cfg, None, "source")
    assert syms == ["S0", "S1", "S2"]
    assert "capped from 10" in note


def test_zero_means_no_cap(monkeypatch):
    """`universe.max_symbols: 0` is how the Indian config asks for the whole market."""
    cfg = _cfg(max_symbols=0)
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: ([f"S{i}" for i in range(10)], "source"))
    syms, note = _resolve_universe(cfg, None, "source")
    assert len(syms) == 10 and "capped" not in note


def test_cli_max_symbols_overrides_the_config(monkeypatch):
    cfg = _cfg(max_symbols=0)
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: ([f"S{i}" for i in range(10)], "source"))
    syms, _ = _resolve_universe(cfg, None, "source", max_symbols=2)
    assert syms == ["S0", "S1"]


def test_auto_only_resolves_enumerable_sources(monkeypatch):
    """A source that cannot list must not be asked, and must not be an error."""
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: (_ for _ in ()).throw(
                            AssertionError("yfinance cannot list; auto must not ask")))
    syms, note = _resolve_universe(_cfg(source="yfinance"), None, "auto")
    assert note.endswith("config") or "config" in note
    assert len(syms) == 12


def test_bad_universe_mode_is_a_usage_error():
    import click

    with pytest.raises(click.BadParameter):
        _resolve_universe(_cfg(), None, "everything")


# ------------------------------------------------------------ parallel scanning
@pytest.fixture(scope="module")
def scan_cfg():
    return load_config(market="crypto", style="swing").merge({"data.synthetic_bars": 900})


def test_parallel_scan_matches_sequential_and_keeps_input_order(scan_cfg):
    store = DataStore(scan_cfg)
    syms = list(scan_cfg.universe.symbols)[:5]
    seq_rows, seq_err = scan_universe(store, scan_cfg, syms)
    par_rows, par_err = scan_universe(store, scan_cfg, syms, workers=4)

    assert [r.symbol for r in par_rows] == syms == [r.symbol for r in seq_rows]
    assert par_err == seq_err
    assert [(r.symbol, round(r.last, 6), r.signal) for r in par_rows] == \
           [(r.symbol, round(r.last, 6), r.signal) for r in seq_rows]


def test_parallel_scan_collects_errors_per_symbol(scan_cfg, monkeypatch):
    """One bad ticker must not take the other 499 rows with it."""
    store = DataStore(scan_cfg)
    real = store.ohlcv

    def flaky(symbol, *args, **kwargs):
        if symbol == "BROKEN/USDT":
            raise DataError("no bars for this one")
        return real(symbol, *args, **kwargs)

    monkeypatch.setattr(store, "ohlcv", flaky)
    rows, errors = scan_universe(store, scan_cfg, ["BTC/USDT", "BROKEN/USDT"], workers=3)
    assert [r.symbol for r in rows] == ["BTC/USDT"]
    assert "BROKEN/USDT" in errors and "no bars" in errors["BROKEN/USDT"]


# ------------------------------------------------------------ the CLI contract
def test_scan_screens_the_whole_universe_and_only_truncates_the_table():
    """`--top` is a display limit, not a scan limit."""
    runner = CliRunner()
    res = runner.invoke(cli, ["scan", "--market", "us_equities", "--style", "swing",
                              "--top", "2", "--workers", "2"])
    assert res.exit_code == 0, res.output
    assert "of 12 symbols scanned" in res.output
    assert "showing the 2 most relevant of 12" in res.output


def test_scan_says_which_universe_it_used():
    runner = CliRunner()
    res = runner.invoke(cli, ["scan", "--market", "us_equities", "--style", "swing",
                              "--top", "1"])
    assert res.exit_code == 0, res.output
    assert "universe: 12 symbol(s) from the us_equities config" in res.output


def test_daily_reports_the_universe_it_scanned():
    runner = CliRunner()
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing",
                              "--no-journal"])
    assert res.exit_code == 0, res.output
    assert "universe: 12 symbol(s)" in res.output


# ------------------------------------------------------------------- NSE itself
def test_nse_lists_a_market_universe_not_a_handful():
    """Live check, skipped without network.

    The number is asserted loosely (>100) because the index membership genuinely
    changes; what matters is that it is the exchange's list and not the config's 12.
    """
    from tvlab.data.nse import NseSource

    try:
        syms = NseSource().list_symbols()
    except Exception as exc:  # noqa: BLE001 - offline is not a failure of the code
        pytest.skip(f"NSE listing unreachable: {type(exc).__name__}: {exc}")

    assert len(syms) > 100
    assert "RELIANCE" in syms
    assert len(syms) != 12


def test_indian_config_does_not_cap_the_source_universe():
    cfg = load_config(market="indian_equities", style="swing")
    assert cfg.universe.max_symbols == 0, (
        "a cap here silently truncates the NSE listing back to a handful of names"
    )
