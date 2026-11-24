"""The ``tvlab daily`` command, driven by injected scan rows and fundamentals.

No strategy reliably fires on the final bar of synthetic data and the sandbox has no
route to ``scanner.tradingview.com``, so both inputs are monkeypatched. That is what
makes the assertions possible: with known inputs the framing, the exit codes and the
journal can be checked exactly, rather than hoping the market and the network
cooperate.

What these tests exist to protect:

* The validation verdict is printed **above** the shortlist, not below it. A tool
  that ranks candidates and mentions their unvalidated status afterwards has already
  done the damage.
* A fundamental outage says so on every surviving name. Silence would read as
  "screened and clean".
* ``all symbols failed`` exits 2 as a data failure, distinct from "nothing
  qualifies", which exits 0. Opposite remedies, so they must not look alike.
* The journal lands in the *configured* output dir under a patched ``REPO_ROOT``,
  never in the real one.
"""
import json
import re

import pandas as pd
import pytest
from click.testing import CliRunner

from tvlab.cli import cli
from tvlab.data.fundamentals import FundamentalSnapshot
from tvlab.screener.scanner import ScanRow


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _row(symbol, signal=1, last=100.0, stop=98.0, target=104.0, trend=1, adx=25.0,
         atr_pct=2.0, bars=1200):
    rr = abs(target - last) / abs(last - stop) if target is not None else None
    return ScanRow(symbol=symbol, last=last, signal=signal, stop=stop, target=target,
                   reward_risk=rr, trend=trend, adx=adx, atr_pct=atr_pct, bars=bars)


def _snap(symbol, **kw):
    """Clears the earnings blackout by default, so tests about market cap, cash flow
    or rendering are not silently decided by an unrelated missing date."""
    later = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=60)
    base = dict(symbol=symbol, market_cap=5e11, roe=0.25, pe_ttm=20.0,
                free_cash_flow=1e10, debt_to_equity=0.5, current_ratio=1.8,
                sector="Technology", next_earnings=later)
    base.update(kw)
    return FundamentalSnapshot(**base)


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich truncates cells to the terminal width; at CliRunner's 80 columns a
    symbol renders as "AA..." and substring assertions silently pass or fail."""
    monkeypatch.setenv("COLUMNS", "250")


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    """Keep the journal out of the real repo. A test that writes to
    ``output/daily/`` clobbers whatever the user's last run produced."""
    monkeypatch.setattr("tvlab.cli.REPO_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patch_scan(monkeypatch):
    def _apply(rows, errors=None):
        def fake(store, cfg, symbols, **kwargs):
            return list(rows), dict(errors or {})

        monkeypatch.setattr("tvlab.screener.scan_universe", fake)
        monkeypatch.setattr("tvlab.screener.scanner.scan_universe", fake)
        return rows
    return _apply


@pytest.fixture
def patch_fundamentals(monkeypatch):
    """Stub the network call. Returns a recorder so a test can assert whether the
    command even tried - ``--no-fundamentals`` must not touch the endpoint."""
    calls = []

    def _apply(snaps=None, exc=None):
        import tvlab.data.fundamentals as mod

        def fake(symbols=None, market=None, **kwargs):
            calls.append({"symbols": list(symbols or []), "market": market, **kwargs})
            if exc is not None:
                raise exc
            return dict(snaps or {})

        monkeypatch.setattr(mod, "fetch_fundamentals", fake)
        return calls
    return _apply


# --------------------------------------------------------------------- framing


def test_the_advisory_precedes_the_shortlist(runner, patch_scan, patch_fundamentals,
                                              isolated_output):
    """Order, not just presence: the warning has to come first."""
    patch_scan([_row("AAPL"), _row("MSFT")])
    patch_fundamentals({"AAPL": _snap("AAPL"), "MSFT": _snap("MSFT")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "Daily briefing" in out
    assert "No validation study on record" in out
    assert out.index("No validation study on record") < out.index("AAPL")


def test_an_unvalidated_study_is_named_above_the_list(runner, patch_scan,
                                                      patch_fundamentals,
                                                      isolated_output, monkeypatch):
    d = isolated_output / "output" / "smc_trend_us_equities_4h"
    d.mkdir(parents=True)
    (d / "smc_trend_us_equities_report.json").write_text(json.dumps({
        "portfolio_metrics": {"win_rate": 0.29, "n_trades": 31},
        "verdict": {"verdict": "NOT VALIDATED", "n_passed": 3, "n_checks": 8,
                    "checks": [{"check": "Deflated Sharpe", "passed": False}]},
    }), encoding="utf-8")
    patch_scan([_row("AAPL")])
    patch_fundamentals({"AAPL": _snap("AAPL")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "NOT VALIDATED" in out
    assert "3/8 checks" in out
    assert "screen, not as advice" in out


# ------------------------------------------------------------------ exit codes


def test_all_symbols_failing_exits_2_as_a_data_failure(runner, patch_scan,
                                                       patch_fundamentals,
                                                       isolated_output):
    patch_scan([], {"AAPL": "DataError: no bars", "MSFT": "DataError: no bars"})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 2
    out = _flat(res.output)
    assert "data failure, not a market verdict" in out
    assert "tvlab data --market us_equities" in out, "must give the retry command"


def test_nothing_qualifying_exits_0_and_says_it_is_normal(runner, patch_scan,
                                                          patch_fundamentals,
                                                          isolated_output):
    patch_scan([_row("AAPL", signal=0), _row("MSFT", signal=0)])
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0
    assert "Nothing qualifies today" in _flat(res.output)
    assert "normal state" in _flat(res.output)


def test_partial_coverage_exits_0_and_states_its_coverage(runner, patch_scan,
                                                          patch_fundamentals,
                                                          isolated_output):
    patch_scan([_row("AAPL")], {"MSFT": "DataError: no bars"})
    patch_fundamentals({"AAPL": _snap("AAPL")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0
    assert "failed to load" in _flat(res.output)


# ----------------------------------------------------------------- fundamentals


def test_a_fundamental_outage_flags_every_surviving_name(runner, patch_scan,
                                                         patch_fundamentals,
                                                         isolated_output):
    from tvlab.data.base import DataError
    patch_scan([_row("AAPL"), _row("MSFT")])
    patch_fundamentals(exc=DataError("TradingView fundamentals request failed: "
                                     "TLS/SSL connection has been closed"))
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "fundamentals NOT applied" in out
    assert "NOT fundamentally screened" in out


def test_no_fundamentals_flag_never_calls_the_endpoint(runner, patch_scan,
                                                       patch_fundamentals,
                                                       isolated_output):
    patch_scan([_row("AAPL")])
    calls = patch_fundamentals({"AAPL": _snap("AAPL")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing",
                              "--no-fundamentals"])
    assert res.exit_code == 0, res.output
    assert calls == [], "--no-fundamentals must not touch the network at all"
    assert "AAPL" in _flat(res.output)


def test_crypto_reports_that_there_is_nothing_to_gate_on(runner, patch_scan,
                                                         patch_fundamentals,
                                                         isolated_output):
    """crypto.yaml leaves fundamentals disabled, but if a user enables it the
    command must explain why it still cannot run, not fail quietly."""
    patch_scan([_row("BTC/USDT")])
    calls = patch_fundamentals({})
    res = runner.invoke(cli, ["daily", "--market", "crypto", "--style", "swing",
                              "--set", "fundamentals.enabled=true"])
    assert res.exit_code == 0, res.output
    assert "no earnings" in _flat(res.output)
    assert calls == [], "crypto must not be sent to the equity scanner"


def test_an_earnings_blackout_shows_up_as_screened_out(runner, patch_scan,
                                                       patch_fundamentals,
                                                       isolated_output):
    patch_scan([_row("AAPL"), _row("SOON")])
    soon = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    patch_fundamentals({"AAPL": _snap("AAPL"), "SOON": _snap("SOON", next_earnings=soon)})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "Screened out on fundamentals" in out
    assert "SOON" in out
    assert "does not survive the gap" in out


# --------------------------------------------------------------------- journal


def test_the_journal_is_written_to_the_configured_output_dir(runner, patch_scan,
                                                             patch_fundamentals,
                                                             isolated_output):
    patch_scan([_row("AAPL")])
    patch_fundamentals({"AAPL": _snap("AAPL")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    jp = isolated_output / "output" / "daily" / "journal.jsonl"
    assert jp.exists(), "the daily run must be journaled by default"
    line = json.loads(jp.read_text(encoding="utf-8").strip())
    assert line["market"] == "us_equities"
    assert line["n_shortlist"] == 1
    assert line["shortlist"][0]["symbol"] == "AAPL"
    assert line["is_validated"] is False
    assert "journaled" in _flat(res.output)


def test_no_journal_flag_writes_nothing(runner, patch_scan, patch_fundamentals,
                                        isolated_output):
    patch_scan([_row("AAPL")])
    patch_fundamentals({"AAPL": _snap("AAPL")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing",
                              "--no-journal"])
    assert res.exit_code == 0, res.output
    assert not (isolated_output / "output" / "daily" / "journal.jsonl").exists()


def test_a_quiet_day_is_still_journaled(runner, patch_scan, patch_fundamentals,
                                        isolated_output):
    """The record that matters most is the empty one: it is what lets you say later
    "the tool ran 20 days and found nothing", rather than remembering only the hits."""
    patch_scan([_row("AAPL", signal=0)])
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0
    jp = isolated_output / "output" / "daily" / "journal.jsonl"
    assert jp.exists()
    assert json.loads(jp.read_text(encoding="utf-8").strip())["outcome"] == "no_signals"


# -------------------------------------------------------------------- rendering


def test_the_shortlist_table_carries_the_fundamental_columns(runner, patch_scan,
                                                             patch_fundamentals,
                                                             isolated_output):
    patch_scan([_row("AAPL")])
    patch_fundamentals({"AAPL": _snap("AAPL", sector="Technology", pe_ttm=28.5,
                                      roe=0.42)})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    for col in ("sector", "mkt cap", "P/E", "ROE", "earnings", "fundamentals"):
        assert col in out
    assert "Technology" in out
    assert "28.5" in out
    assert "approved" in out


def test_market_cap_is_abbreviated_not_printed_raw(runner, patch_scan,
                                                   patch_fundamentals, isolated_output):
    patch_scan([_row("AAPL")])
    patch_fundamentals({"AAPL": _snap("AAPL", market_cap=3.2e12)})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    out = _flat(res.output)
    assert "3.2T" in out
    assert "3200000000000" not in out


def test_top_truncates_the_shortlist(runner, patch_scan, patch_fundamentals,
                                     isolated_output):
    syms = ["S1", "S2", "S3", "S4"]
    patch_scan([_row(s) for s in syms])
    patch_fundamentals({s: _snap(s) for s in syms})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing",
                              "--top", "2"])
    assert res.exit_code == 0, res.output
    jp = isolated_output / "output" / "daily" / "journal.jsonl"
    assert json.loads(jp.read_text(encoding="utf-8").strip())["n_shortlist"] == 2


def test_position_sizing_follows_the_risk_mandate(runner, patch_scan,
                                                  patch_fundamentals, isolated_output):
    patch_scan([_row("AAPL")])
    patch_fundamentals({"AAPL": _snap("AAPL")})
    res = runner.invoke(cli, ["daily", "--market", "us_equities", "--style", "swing"])
    out = _flat(res.output)
    # 0.50% risk per trade / 2.00% stop = 25% of equity notional
    assert "risk_per_trade 0.50%" in out
    assert "capped at 25%" in out
