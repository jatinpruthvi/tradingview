"""The dashboard's long reads: universe resolution, progress, and background jobs.

Three separate things are pinned here, and they fail in three different ways:

* **Universe resolution** - the dashboard used ``cfg.universe.symbols`` directly,
  so an Indian scan meant the 12 hand-typed large caps while NSE listed 502. The
  CLI had the same bug (see ``test_universe.py``); keeping the resolver in one
  module is what stops the two front ends drifting again.
* **Progress** - a 500-name scan is 1,500 HTTP requests. If the counter only
  advances when a symbol *succeeds*, a run with a dead ticker stalls the bar and
  the user concludes it is hung.
* **Jobs** - one scan at a time, and a second press joins the run already going
  rather than queueing a second 500-name scan behind it.
"""
import time

import pytest

from tvlab.config import load_config
from tvlab.dashboard.app import _scan_payload
from tvlab.dashboard.jobs import JobRunner
from tvlab.data.base import DataError
from tvlab.data.store import DataStore
from tvlab.screener import scan_universe
from tvlab.screener.universe import ENUMERABLE_SOURCES, MODES, resolve_universe


class _ListingSource:
    name = "stub"

    def __init__(self, symbols):
        self._symbols = list(symbols)

    def list_symbols(self):
        return list(self._symbols)

    def fetch_ohlcv(self, symbol, timeframe="1d", start=None, end=None, limit=None):
        raise DataError("not used")


def _cfg(**universe):
    return load_config(market="indian_equities", style="swing").merge(
        {"data.source": "stub", **{f"universe.{k}": v for k, v in universe.items()}}
    )


# --------------------------------------------------------------------- resolver
def test_resolver_is_the_single_definition_of_a_universe(monkeypatch):
    """The dashboard must be able to ask the source instead of the typed list.

    ``auto`` only trusts sources that enumerate an *exchange* (NSE), so the stub
    is asked through ``source`` - which is the switch the dashboard's Universe
    control exposes.
    """
    cfg = _cfg()
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: ([f"S{i}.NS" for i in range(40)],
                                                          "source"))
    syms, note = resolve_universe(cfg, None, "source", None)
    assert len(syms) == 40
    assert "listed by 'stub'" in note

    # and the note is honest when the source cannot list at all
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: (["A.NS"], "config"))
    syms, note = resolve_universe(cfg, None, "source", None)
    assert "lists no universe" in note and syms != [f"S{i}.NS" for i in range(40)]


def test_auto_does_not_ask_a_source_that_cannot_list(monkeypatch):
    cfg = load_config(market="us_equities", style="swing")
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: (_ for _ in ()).throw(
                            AssertionError("yfinance cannot list; auto must not ask")))
    syms, note = resolve_universe(cfg, None, "auto")
    assert syms == list(cfg.universe.symbols)
    assert "config" in note


def test_an_explicit_list_beats_every_mode(monkeypatch):
    monkeypatch.setattr(DataStore, "universe",
                        lambda self, prefer_source=True: (_ for _ in ()).throw(
                            AssertionError("an explicit list must not consult the source")))
    syms, note = resolve_universe(_cfg(), ["A", "B", "C"], "source")
    assert syms == ["A", "B", "C"]
    assert "3 symbol(s)" in note


def test_modes_are_a_closed_set():
    with pytest.raises(ValueError):
        resolve_universe(_cfg(), None, "everything")
    assert set(MODES) == {"auto", "config", "source"}
    assert ENUMERABLE_SOURCES == frozenset({"nse"})


# --------------------------------------------------------------------- progress
@pytest.fixture(scope="module")
def crypto_cfg():
    return load_config(market="crypto", style="swing").merge({"data.synthetic_bars": 600})


@pytest.mark.parametrize("workers", [1, 4])
def test_progress_reaches_total_and_never_goes_backwards(crypto_cfg, workers):
    seen = []
    scan_universe(DataStore(crypto_cfg), crypto_cfg, ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                  workers=workers, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (3, 3), "the bar must finish full"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen), seen


def test_progress_counts_failures_too(crypto_cfg, monkeypatch):
    """A dead ticker must not freeze the counter - the scan is still advancing."""
    store = DataStore(crypto_cfg)
    real = store.ohlcv

    def flaky(symbol, *args, **kwargs):
        if symbol == "BROKEN/USDT":
            raise DataError("no bars")
        return real(symbol, *args, **kwargs)

    monkeypatch.setattr(store, "ohlcv", flaky)
    seen = []
    rows, errors = scan_universe(store, crypto_cfg, ["BROKEN/USDT", "BTC/USDT"],
                                 workers=2, progress=lambda d, t: seen.append((d, t)))
    assert "BROKEN/USDT" in errors and [r.symbol for r in rows] == ["BTC/USDT"]
    assert seen[-1] == (2, 2)


def test_a_broken_progress_callback_does_not_lose_rows(crypto_cfg):
    def boom(done, total):
        raise RuntimeError("display bug")

    rows, errors = scan_universe(DataStore(crypto_cfg), crypto_cfg, ["BTC/USDT"],
                                 progress=boom)
    assert [r.symbol for r in rows] == ["BTC/USDT"] and not errors


# ------------------------------------------------------------------------- jobs
def _wait(job, timeout=5.0):
    deadline = time.time() + timeout
    while job.state not in ("done", "error") and time.time() < deadline:
        time.sleep(0.01)
    return job


def test_a_job_carries_progress_and_its_result():
    runner = JobRunner()

    def work(progress):
        progress.set_total(3)
        for i in range(3):
            progress.note(f"step {i}")
            progress.advance(i + 1, 3)
        return {"answer": 42}

    job = _wait(runner.submit("scan", work))
    assert job.state == "done"
    payload = job.to_dict()
    assert payload["result"] == {"answer": 42}
    assert payload["done"] == payload["total"] == 3
    assert payload["fraction"] == 1.0
    assert "step 2" == job.message


def test_a_second_request_joins_the_run_in_progress():
    """Pressing the button twice must not queue two 500-name scans."""
    runner = JobRunner()
    started = runner.submit("scan", lambda p: time.sleep(0.2) or {"n": 1})
    again = runner.active("scan")
    assert again is not None and again.id == started.id
    assert runner.active("daily") is None, "a different kind is not the same job"
    _wait(started)


def test_a_failed_job_reports_why_and_stops():
    runner = JobRunner()

    def boom(progress):
        raise DataError("NSE served an HTML interstitial")

    job = _wait(runner.submit("daily", boom))
    assert job.state == "error"
    assert "HTML interstitial" in job.error
    assert job.to_dict()["error"].startswith("DataError:")


def test_history_is_bounded():
    runner = JobRunner(keep=3)
    for _ in range(8):
        _wait(runner.submit("suggest", lambda p: {}))
    assert len(runner.recent(50)) <= 3


def test_a_running_job_is_never_trimmed():
    """Trimming must not delete the job the browser is polling."""
    runner = JobRunner(keep=2)
    hold = runner.submit("scan", lambda p: time.sleep(0.3) or {})
    for _ in range(6):
        runner.submit("suggest", lambda p: {})
    ids = [j.id for j in runner.recent(50)]
    assert hold.id in ids


# ------------------------------------------------------- the dashboard payload
def test_top_is_a_display_limit_not_a_scan_limit():
    """`top` used to slice the symbol list: asking for 2 rows scanned 2 names."""
    cfg = load_config(market="us_equities", style="swing")
    full = _scan_payload(cfg, universe="config", workers=2)
    cut = _scan_payload(cfg, universe="config", top=2, workers=2)
    assert cut["n_rows"] == full["n_rows"] == 12
    assert cut["n_shown"] == 2 and len(cut["rows"]) == 2


def test_the_payload_says_where_the_universe_came_from():
    cfg = load_config(market="us_equities", style="swing")
    payload = _scan_payload(cfg, universe="config", workers=2)
    assert "12 symbol(s) from the us_equities config" in payload["universe"]


def test_a_bad_universe_mode_is_a_400_not_a_traceback():
    from fastapi import HTTPException

    from tvlab.dashboard.app import create_app

    app = create_app(load_config(market="us_equities", style="swing"))
    scan_route = next(r for r in app.routes if getattr(r, "path", "") == "/api/scan")
    endpoint = scan_route.endpoint
    with pytest.raises(HTTPException) as exc:
        endpoint(symbols=None, timeframe=None, top=None, trendlines=False,
                 universe="everything", max_symbols=None, workers=1)
    assert exc.value.status_code == 400
