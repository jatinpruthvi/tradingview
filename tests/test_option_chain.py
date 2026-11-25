"""Option chain: features, expiry choice, and the forward-collected panel.

The payloads here are shaped from the live responses captured on 2026-11-25
(``output/option_chain_study.json``) - field names ``strikePrice``, ``CE``/``PE``,
``openInterest``, ``impliedVolatility``, ``totalTradedVolume``, ``underlyingValue``
and ``expiryDates``. They pin the contract this module parses; they are not a
promise that NSE keeps it.

The panel tests matter more than the parsing tests: this file cannot be
regenerated (NSE publishes no chain history), so appends must be idempotent per
day and a damaged panel must be refused rather than overwritten.
"""
import json
import re
from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from tvlab import cli as cli_mod
from tvlab.cli import cli
from tvlab.data.base import DataError
from tvlab.data.option_chain import (
    SNAPSHOT_COLUMNS,
    ChainHttp,
    append_rows,
    contract_expiries,
    features,
    load_panel,
    pick_expiry,
    snapshot_symbol,
)

TODAY = date(2026, 11, 25)


# ------------------------------------------------------------------ payloads


def _leg(oi: int, iv: float = 25.0, volume: int = 1000) -> dict:
    return {"openInterest": oi, "impliedVolatility": iv, "totalTradedVolume": volume,
            "lastPrice": 1.0}


def _row(strike: float, ce_oi: int = 0, pe_oi: int = 0,
         ce_iv: float = 25.0, pe_iv: float = 26.0) -> dict:
    return {"strikePrice": strike, "CE": _leg(ce_oi, ce_iv), "PE": _leg(pe_oi, pe_iv)}


def _chain(spot: float = 1000.0, rows=None, expiries=("01-Dec-2026", "29-Dec-2026")):
    return {"records": {
        "expiryDates": list(expiries),
        "underlyingValue": spot,
        "data": rows if rows is not None else [
            _row(900, ce_oi=100, pe_oi=500),
            _row(950, ce_oi=150, pe_oi=2500, pe_iv=30.0),
            _row(1000, ce_oi=900, pe_oi=700, ce_iv=24.0, pe_iv=26.0),
            _row(1050, ce_oi=3000, pe_oi=300, ce_iv=28.0),
            _row(1100, ce_oi=400, pe_oi=100),
        ],
    }}


def _fetcher(record=None, expiries=("01-Dec-2026", "29-Dec-2026")):
    """A fetcher routing the two chain URLs to payloads."""

    def fetch(url: str, timeout: float) -> bytes:
        if "contract-info" in url:
            return json.dumps({"expiryDates": list(expiries)}).encode()
        if "option-chain-v3" in url:
            return json.dumps(record if record is not None else _chain()).encode()
        raise AssertionError(f"unexpected URL: {url}")

    return fetch


# ------------------------------------------------------------------ features


def test_features_summarises_the_band():
    f = features(_chain()["records"])
    assert f is not None
    assert f["spot"] == 1000.0
    assert f["call_wall"] == 1050.0 and f["call_wall_dist_pct"] == 5.0
    assert f["put_wall"] == 950.0 and f["put_wall_dist_pct"] == -5.0
    # pe_oi = 500+2500+700+300+100, ce_oi = 100+150+900+3000+400
    assert f["ce_oi"] == 4550 and f["pe_oi"] == 4100
    assert f["pcr"] == round(4100 / 4550, 3)
    assert f["live_strikes"] == 5 and f["strikes_in_band"] == 5
    assert f["atm_iv"] == pytest.approx(25.0)          # mean of the 1000 CE/PE
    assert f["skew_put_minus_call"] == pytest.approx(2.0)


def test_a_huge_strike_outside_the_band_is_not_a_wall():
    """A 1400-strike position is next quarter's story, not this trade's stop.

    Without the band the "wall" would be 40% away and any rule built on it would
    describe a different trade than the one on the table.
    """
    rows = _chain()["records"]["data"] + [_row(1400, ce_oi=99_999, pe_oi=99_999)]
    f = features({**_chain()["records"], "data": rows})
    assert f["call_wall"] == 1050.0
    assert f["ce_oi"] == 4550, "out-of-band open interest must not enter the totals"


def test_features_say_nothing_when_the_payload_says_nothing():
    assert features({"underlyingValue": None, "data": [_row(1000)]}) is None
    assert features({"underlyingValue": 1000.0, "data": []}) is None
    assert features({"data": [_row(1000)]}) is None
    # every listed strike outside the band: no local positioning to report
    assert features({"underlyingValue": 1000.0, "data": [_row(2000, ce_oi=5)]}) is None


def test_features_survive_missing_legs_and_strings():
    f = features({"underlyingValue": "1000",
                  "data": [{"strikePrice": "1000", "CE": {"openInterest": "10"},
                            "PE": {"impliedVolatility": "-1"}}]})
    assert f["spot"] == 1000.0
    assert f["ce_oi"] == 10 and f["pe_oi"] == 0
    assert f["pcr"] == 0.0, "puts with no calls against them is a real ratio"
    assert f["atm_iv"] is None, "a negative IV is not an IV"
    assert f["call_wall"] is None and f["put_wall"] is None, "ATM is neither"


def test_pcr_is_absent_when_there_are_no_calls_at_all():
    """Zero call OI is missing, not infinite: a 0.0 would read as extreme fear."""
    f = features({"underlyingValue": 1000.0,
                  "data": [{"strikePrice": 950, "CE": {}, "PE": {"openInterest": 50}}]})
    assert f["ce_oi"] == 0 and f["pe_oi"] == 50
    assert f["pcr"] is None


# ------------------------------------------------------------------- expiries


def test_pick_expiry_skips_the_near_week():
    raw, dte = pick_expiry(["01-Dec-2026", "29-Dec-2026"], today=TODAY, min_dte=21)
    assert raw == "29-Dec-2026"
    assert dte == 34


def test_pick_expiry_falls_back_to_the_farthest_and_still_reports_its_dte():
    """Better a chain with an unflattering DTE than 'no data' about a real chain.

    The dte column is what makes this honest: the reader is told exactly what
    they got instead of being handed a silently excluded expiry.
    """
    raw, dte = pick_expiry(["01-Dec-2026", "04-Dec-2026"], today=TODAY, min_dte=21)
    assert raw == "04-Dec-2026"
    assert dte == 9


def test_pick_expiry_ignores_unparseable_entries():
    raw, dte = pick_expiry(["not-a-date", "29-Dec-2026"], today=TODAY, min_dte=21)
    assert raw == "29-Dec-2026" and dte == 34
    assert pick_expiry(["nonsense"], today=TODAY) == (None, None)


def test_a_name_without_derivatives_is_empty_not_an_error():
    http = ChainHttp(fetcher=lambda url, t: b"{}", delay=0)
    assert contract_expiries(http, "NH") == []


def test_an_html_interstitial_is_a_data_error_with_the_cookie_hint():
    http = ChainHttp(fetcher=lambda url, t: b"<html>blocked</html>", delay=0)
    with pytest.raises(DataError) as exc:
        contract_expiries(http, "TCS")
    assert "interstitial" in str(exc.value) or "did not return JSON" in str(exc.value)


# -------------------------------------------------------------- snapshot rows


def test_a_snapshot_row_carries_the_strategy_columns_and_the_features():
    http = ChainHttp(fetcher=_fetcher(), delay=0)
    row = snapshot_symbol(http, "TCS", today=TODAY)
    assert row["ok"] is True and row["error"] is None
    assert row["as_of"] == "2026-11-25" and row["symbol"] == "TCS"
    assert row["expiry"] == "29-Dec-2026" and row["dte"] == 34
    assert set(SNAPSHOT_COLUMNS) <= set(row)
    assert row["call_wall_dist_pct"] == 5.0


def test_a_name_without_options_becomes_a_row_not_a_crash():
    http = ChainHttp(fetcher=_fetcher(expiries=()), delay=0)
    row = snapshot_symbol(http, "NH", today=TODAY)
    assert row["ok"] is False
    assert row["error"] == "no listed options"
    assert row["spot"] is None, "a missing chain must not look like a zero price"


def test_a_broken_chain_is_recorded_and_the_run_continues():
    def fetch(url, timeout):
        if "contract-info" in url:
            return json.dumps({"expiryDates": ["29-Dec-2026"]}).encode()
        return b"<html>nope</html>"

    http = ChainHttp(fetcher=fetch, delay=0)
    row = snapshot_symbol(http, "TCS", today=TODAY)
    assert row["ok"] is False
    assert "did not return JSON" in row["error"]
    assert row["expiry"] == "29-Dec-2026", "what was known before failing is kept"


def test_an_unexpected_exception_type_is_also_contained():
    def fetch(url, timeout):
        raise RuntimeError("boom")

    row = snapshot_symbol(ChainHttp(fetcher=fetch, delay=0), "TCS", today=TODAY)
    assert row["ok"] is False and "RuntimeError: boom" in row["error"]


def test_a_chain_with_no_positioning_is_not_ok():
    http = ChainHttp(fetcher=_fetcher(record={"records": {"underlyingValue": 1000,
                                                          "data": []}}), delay=0)
    row = snapshot_symbol(http, "TCS", today=TODAY)
    assert row["ok"] is False
    assert row["error"] == "chain carried no usable positioning"


# ----------------------------------------------------------------- the panel


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive rich's line wrapping."""
    return re.sub(r"\s+", " ", text)


def _row_for(symbol: str, as_of: str = "2026-11-25", **kw) -> dict:
    row = {c: None for c in SNAPSHOT_COLUMNS}
    row.update({"as_of": as_of, "symbol": symbol, "ok": True, "spot": 100.0,
                "pcr": 0.8, "atm_iv": 25.0, "dte": 30,
                "call_wall_dist_pct": 4.0})
    row.update(kw)
    return row


def test_appending_twice_on_the_same_day_replaces_instead_of_duplicating(tmp_path):
    path = tmp_path / "panel.parquet"
    append_rows(path, [_row_for("TCS", atm_iv=25.0), _row_for("INFY")])
    panel = append_rows(path, [_row_for("TCS", atm_iv=31.0)])

    assert len(panel) == 2, "one row per (day, symbol)"
    tcs = panel[panel["symbol"] == "TCS"]
    assert len(tcs) == 1 and float(tcs["atm_iv"].iloc[0]) == 31.0
    assert set(panel["symbol"]) == {"TCS", "INFY"}, "the other symbol survives"


def test_a_new_day_appends_and_the_panel_stays_ordered(tmp_path):
    path = tmp_path / "panel.parquet"
    append_rows(path, [_row_for("TCS", as_of="2026-11-25")])
    panel = append_rows(path, [_row_for("ABB", as_of="2026-11-26"),
                               _row_for("TCS", as_of="2026-11-26")])

    assert len(panel) == 3
    assert list(panel["as_of"]) == ["2026-11-25", "2026-11-26", "2026-11-26"]
    assert list(panel["symbol"]) == ["TCS", "ABB", "TCS"]
    assert panel["as_of"].nunique() == 2


def test_the_panel_survives_a_parquet_round_trip(tmp_path):
    path = tmp_path / "panel.parquet"
    append_rows(path, [_row_for("TCS")])
    again = load_panel(path)
    assert list(again.columns) == list(SNAPSHOT_COLUMNS)
    assert bool(again["ok"].astype(bool).iloc[0]) is True


def test_a_damaged_panel_is_refused_not_overwritten(tmp_path):
    """NSE has no chain history: clobbering this file destroys the only copy."""
    path = tmp_path / "panel.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(DataError) as exc:
        append_rows(path, [_row_for("TCS")])
    assert "move it aside" in str(exc.value)
    assert path.read_bytes() == b"not a parquet file"


def test_a_panel_missing_columns_is_refused(tmp_path):
    path = tmp_path / "panel.parquet"
    pd.DataFrame({"symbol": ["TCS"]}).to_parquet(path, index=False)
    with pytest.raises(DataError) as exc:
        load_panel(path)
    assert "missing column" in str(exc.value)


def test_an_empty_append_leaves_an_absent_panel_absent(tmp_path):
    path = tmp_path / "panel.parquet"
    panel = append_rows(path, [])
    assert panel.empty and not path.exists()


# ---------------------------------------------------------------------- CLI


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def pinned_console_width(monkeypatch):
    """Rich truncates cells to the console width; pin it so assertions are stable."""
    monkeypatch.setattr(cli_mod.console, "_width", 160, raising=False)


@pytest.fixture
def fake_chain(monkeypatch):
    monkeypatch.setattr(cli_mod, "_chain_client",
                        lambda: ChainHttp(fetcher=_fetcher(), delay=0))


def test_chain_group_is_registered(runner):
    res = runner.invoke(cli, ["chain", "--help"])
    assert res.exit_code == 0, res.output
    assert "snapshot" in res.output and "show" in res.output


def test_chain_help_does_not_pretend_to_be_a_signal(runner):
    out = " ".join(runner.invoke(cli, ["chain", "--help"]).output.split())
    assert "cannot be backfilled" in out


def test_snapshot_dry_run_writes_nothing(runner, fake_chain, tmp_path):
    out = tmp_path / "panel.parquet"
    res = runner.invoke(cli, ["chain", "snapshot", "--symbols", "TCS,INFY",
                              "--out", str(out), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "nothing written" in res.output
    assert not out.exists()


def test_snapshot_appends_and_show_reads_it_back(runner, fake_chain, tmp_path):
    out = tmp_path / "panel.parquet"
    for _ in range(2):     # the second run must not duplicate the first
        res = runner.invoke(cli, ["chain", "snapshot", "--symbols", "TCS",
                                  "--out", str(out)])
        assert res.exit_code == 0, res.output
    panel = pd.read_parquet(out)
    assert len(panel) == 1
    assert "1 row(s) across 1 day(s)" in _flat(res.output)

    shown = runner.invoke(cli, ["chain", "show", "--panel", str(out)])
    assert shown.exit_code == 0, shown.output
    assert "TCS" in shown.output


def test_show_says_so_when_the_panel_is_empty(runner, tmp_path):
    res = runner.invoke(cli, ["chain", "show", "--panel", str(tmp_path / "none.parquet")])
    assert res.exit_code == 0, res.output
    assert "is empty" in res.output


def test_a_failed_snapshot_is_reported_not_hidden(runner, monkeypatch, tmp_path):
    """The command exits 0 with fewer usable rows - that is the honest outcome.

    Exiting non-zero would make a cron job look broken on the one day NSE
    throttles, and the failure is already recorded in the panel's error column.
    """
    def fetch(url, timeout):
        if "contract-info" in url:
            return json.dumps({"expiryDates": []}).encode()
        raise AssertionError("no chain should be requested")

    monkeypatch.setattr(cli_mod, "_chain_client",
                        lambda: ChainHttp(fetcher=fetch, delay=0))
    res = runner.invoke(cli, ["chain", "snapshot", "--symbols", "NH,COHANCE",
                              "--out", str(tmp_path / "p.parquet")])
    assert res.exit_code == 0, res.output
    assert "0/2 with a usable chain" in res.output
    panel = pd.read_parquet(tmp_path / "p.parquet")
    assert set(panel["error"]) == {"no listed options"}
