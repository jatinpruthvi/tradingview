"""The ``tvlab suggest`` command, driven by injected scan rows.

No strategy reliably fires a signal on the final bar of synthetic data, so the
rendering path is tested by monkeypatching ``scan_universe`` with controlled rows
rather than by hoping the market cooperates. That also makes the assertions
possible: with known inputs, the ranking and the arithmetic can be checked exactly.
"""
import re
from datetime import datetime

import pytest
from click.testing import CliRunner

from tvlab.cli import cli
from tvlab.data.store import DataStore
from tvlab.screener.scanner import ScanRow

CRYPTO_FEE, CRYPTO_SLIP = 10.0, 5.0      # 30 bp round trip


def _flat(text):
    return re.sub(r"\s+", " ", text)


def _row(symbol, signal=1, last=100.0, stop=98.0, target=104.0, trend=1, adx=25.0,
         atr_pct=2.0, zones="-", sweep_age=None, bars=1200, support_pct=None):
    rr = abs(target - last) / abs(last - stop) if target is not None else None
    return ScanRow(symbol=symbol, last=last, signal=signal, stop=stop, target=target,
                   reward_risk=rr, trend=trend, adx=adx, atr_pct=atr_pct, zones=zones,
                   sweep_age=sweep_age, bars=bars, support_pct=support_pct)


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich truncates table cells to the terminal width, so at CliRunner's default
    80 columns a symbol renders as "WI..." and substring assertions silently fail.
    Rich reads COLUMNS at render time, so widening it makes the tables assertable."""
    monkeypatch.setenv("COLUMNS", "250")


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patch_scan(monkeypatch):
    """Replace the universe scan with a fixed set of rows."""
    def _apply(rows, errors=None):
        import tvlab.cli as cli_mod

        def fake(store, cfg, symbols, **kwargs):
            return list(rows), dict(errors or {})

        # scan_universe is imported inside the command body, so patch the module
        monkeypatch.setattr("tvlab.screener.scan_universe", fake)
        monkeypatch.setattr("tvlab.screener.scanner.scan_universe", fake)
        return rows
    return _apply


# ------------------------------------------------------------------ happy path


def test_suggest_ranks_the_wider_stop_above_the_tighter_one(runner, patch_scan):
    """The whole point of the command: TIGHT has the better ratio on paper (8:1 vs
    2:1) and the naive sort would list it first."""
    patch_scan([
        _row("TIGHT", last=100.0, stop=99.80, target=101.60),   # R:R 8, 0.20% stop
        _row("WIDE", last=100.0, stop=98.00, target=104.00),     # R:R 2, 2.00% stop
    ])
    res = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = res.output
    flat = _flat(out)
    # row 1 of the ranked table, and the prose summary, must both name WIDE
    assert "Top candidate: WIDE" in flat, "the cost-aware ranking must invert the naive one"
    assert flat.index("1 │ WIDE") < flat.index("2 │ TIGHT"), flat


def test_suggest_shows_the_breakeven_win_rate(runner, patch_scan):
    """The breakeven threshold is the number that makes the R:R interpretable, so
    it must be on screen next to the ratio, not buried in JSON."""
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    out = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output
    assert "breakeven" in out.lower()
    # R:R 2, 2% stop, 30 bp -> cost_R 0.15, breakeven = 1.15/3 = 38.3%
    assert "38.3%" in out
    # and the cost-free comparison value, 1/(1+2) = 33.3%
    assert "33.3%" in out


def test_suggest_prints_the_top_candidate_with_entry_stop_and_target(runner, patch_scan):
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "Top candidate: WIDE long" in out
    assert "entry ~100" in out and "stop 98" in out and "target 104" in out


def test_suggest_lists_the_reasons_and_the_warnings(runner, patch_scan):
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0, zones="FVG", sweep_age=2)])
    out = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output
    assert "R:R" in out and "costs eat" in out
    assert "price inside FVG" in out


def test_suggest_warns_about_a_stop_inside_the_noise(runner, patch_scan):
    """A 0.20% stop clears the mandate's 0.15% floor but is only 0.10 ATR on a 2%
    ATR instrument - inside the noise, so it stays a candidate and must warn."""
    patch_scan([_row("NOISY", last=100.0, stop=99.80, target=101.00, atr_pct=2.0)])
    out = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output
    assert "Top candidate: NOISY" in _flat(out)
    assert "ATR away" in out


def test_a_stop_below_the_mandate_floor_is_gated_not_warned(runner, patch_scan):
    """The mandate's min_stop_distance_pct is a hard floor - the sizer would reject
    the trade - so such a row never reaches the warnings list. It is reported as
    rejected, which is a different and stronger statement."""
    patch_scan([_row("HAIR", last=100.0, stop=99.95, target=100.20, atr_pct=2.0)])
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "Nothing qualifies" in out
    assert "HAIR" in out
    assert "risk mandate" in out


def test_the_mandate_floor_cannot_be_overridden_by_max_breakeven(runner, patch_scan):
    """Loosening the breakeven gate must not resurrect a stop the risk config
    forbids. Two independent gates, and the stricter one wins."""
    patch_scan([_row("HAIR", last=100.0, stop=99.95, target=100.20)])
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing",
                                    "--max-breakeven", "1.0"]).output)
    assert "Nothing qualifies" in out and "HAIR" in out


def test_suggest_reports_position_size_from_the_mandate(runner, patch_scan):
    """The user still has to know how big to make it. risk_per_trade / stop
    distance, capped at max_position_pct."""
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "risk_per_trade" in out
    # 0.5% risk / 2.0% stop = 25.0% of equity notional
    assert "25.0%" in out


# ------------------------------------------------------------------ empty path


def test_suggest_says_plainly_when_nothing_qualifies(runner, patch_scan):
    """The normal state. It must read as an answer, not as a crash or an empty table."""
    patch_scan([_row("QUIET", signal=0)])
    res = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "Nothing qualifies" in out
    assert "This is the normal state" in out


def test_suggest_rejects_untradeable_stops_and_says_why(runner, patch_scan):
    patch_scan([
        _row("HAIR", last=100.0, stop=99.95, target=100.20),   # 0.05% stop -> 6R of cost
        _row("WIDE", last=100.0, stop=98.00, target=104.00),
    ])
    out = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output
    assert "Rejected" in out
    assert "HAIR" in out
    assert "WIDE" in out


def test_suggest_max_breakeven_is_configurable(runner, patch_scan):
    """At a loose gate the hair-thin stop is a candidate; at a strict one it is not."""
    # 0.20% stop clears the mandate floor; R:R 5 at 1.5R of cost needs 41.7%
    rows = [_row("PRICEY", last=100.0, stop=99.80, target=101.00)]
    patch_scan(rows)
    loose = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing",
                                "--max-breakeven", "0.60"]).output
    strict = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing",
                                 "--max-breakeven", "0.30"]).output
    assert "Top candidate: PRICEY" in _flat(loose)
    assert "41.7%" in loose
    assert "Nothing qualifies" in _flat(strict)


def test_suggest_reports_load_failures(runner, patch_scan):
    patch_scan([_row("WIDE")], errors={"DEAD/COIN": "DataError: unknown symbol"})
    out = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output
    assert "failed to load" in out.lower()


# ------------------------------------------------- comparison against the study


def test_suggest_compares_breakeven_against_the_measured_win_rate(runner, patch_scan, monkeypatch):
    """This is what turns a threshold into a decision: the strategy's own recorded
    win rate, next to the win rate the idea needs."""
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate",
                        lambda cfg, name: {"status": "ok", "win_rate": 0.55, "n_trades": 400,
                                           "profit_factor": 1.4, "path": "x.json",
                                           "created_at": None})
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "measured on the last study" in out.lower()
    assert "55.0%" in out
    assert "clears it" in out
    assert "adequate sample" in out


def test_suggest_flags_when_the_measured_rate_does_not_clear_it(runner, patch_scan, monkeypatch):
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate",
                        lambda cfg, name: {"status": "ok", "win_rate": 0.20, "n_trades": 500,
                                           "profit_factor": 0.8, "path": "x.json",
                                           "created_at": None})
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "does NOT clear it" in out


def test_suggest_distrusts_a_thin_sample(runner, patch_scan, monkeypatch):
    """38% on 13 trades is not 38%. The command must say so rather than letting a
    green PASS marker imply the comparison is settled."""
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate",
                        lambda cfg, name: {"status": "ok", "win_rate": 0.90, "n_trades": 9,
                                           "profit_factor": 3.0, "path": "x.json",
                                           "created_at": None})
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "far too few trades to trust" in out
    assert "Below 30" in out


def test_suggest_copes_with_no_previous_study(runner, patch_scan, monkeypatch):
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate", lambda cfg, name: None)
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"]).output)
    assert "no previous study" in out
    assert "tvlab study" in out, "must tell the user how to get one"


def _write_report(tmp_path, monkeypatch, portfolio_metrics, market="crypto",
                  strat="smc_trend", tf="4h"):
    """Point REPO_ROOT at a temp dir holding one study report.

    Reading whatever report happens to be on disk makes this test depend on what
    anyone ran last: it passed while a 13-trade report sat there and started
    failing the moment a 2-symbol run overwrote it with a 0-trade one. The
    ambient-disk version is kept separately, below, as an opportunistic check.
    """
    import json
    from tvlab.cli import _measured_win_rate
    from tvlab.config import load_config

    d = tmp_path / "output" / f"{strat}_{market}_{tf}"
    d.mkdir(parents=True)
    path = d / f"{strat}_{market}_report.json"
    path.write_text(json.dumps({"portfolio_metrics": portfolio_metrics,
                                "created_at": "2026-11-23T09:00:00"}), encoding="utf-8")
    monkeypatch.setattr("tvlab.cli.REPO_ROOT", tmp_path)
    cfg = load_config(market=market, style="swing")
    return _measured_win_rate(cfg, strat), path


def test_measured_win_rate_status_ok(tmp_path, monkeypatch):
    m, path = _write_report(tmp_path, monkeypatch,
                            {"win_rate": 0.385, "n_trades": 13, "profit_factor": 1.09,
                             "payoff_ratio": 1.7, "expectancy": 0.05})
    assert m["status"] == "ok"
    assert m["win_rate"] == pytest.approx(0.385)
    assert m["n_trades"] == 13
    assert m["path"] == str(path)


def test_measured_win_rate_status_no_trades(tmp_path, monkeypatch):
    """A study that ran and took no trades is an answer, not an absent one. This is
    the state a short CSV sample or a small universe actually produces, and it used
    to be reported as "no previous study" - telling the user to re-run what they ran."""
    m, _ = _write_report(tmp_path, monkeypatch, {"win_rate": None, "n_trades": 0})
    assert m["status"] == "no_trades"
    assert m["win_rate"] is None
    assert m["n_trades"] == 0


def test_measured_win_rate_status_corrupt(tmp_path, monkeypatch):
    """A half-written report must not kill the scan, but must not read as "no
    study" either."""
    import json
    from tvlab.cli import _measured_win_rate
    from tvlab.config import load_config

    d = tmp_path / "output" / "smc_trend_crypto_4h"
    d.mkdir(parents=True)
    (d / "smc_trend_crypto_report.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("tvlab.cli.REPO_ROOT", tmp_path)
    m = _measured_win_rate(load_config(market="crypto", style="swing"), "smc_trend")
    assert m["status"] == "corrupt"
    assert "JSONDecodeError" in m["error"]


def test_measured_win_rate_reads_the_report_on_disk_if_there_is_one():
    """Opportunistic: exercises the real path resolution against whatever the last
    study wrote. Skips when there is nothing, and accepts any status."""
    from tvlab.cli import _measured_win_rate
    from tvlab.config import load_config

    cfg = load_config(market="crypto", style="swing")
    m = _measured_win_rate(cfg, cfg.strategy.name)
    if m is None:
        pytest.skip("no study report on disk yet")
    assert m["status"] in {"ok", "no_trades", "corrupt"}
    assert m["path"].endswith("_report.json")
    if m["status"] == "ok":
        assert 0.0 <= m["win_rate"] <= 1.0
        assert m["n_trades"] >= 0


def test_suggest_does_not_crash_when_the_study_took_no_trades(runner, patch_scan,
                                                             monkeypatch):
    """The regression this whole distinction exposed: with candidates present AND a
    no_trades study, `measured["win_rate"] > be` raised
    TypeError: '>' not supported between NoneType and float. Three sites guarded on
    `if measured:` - truthy for a dict carrying a path but no win rate."""
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate",
                        lambda cfg, name: {"status": "no_trades", "win_rate": None,
                                           "n_trades": 0, "path": "x.json"})
    res = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    assert res.exception is None
    out = _flat(res.output)
    assert "Top candidate: WIDE" in out, "the table and detail block must still render"
    assert "took no trades" in out
    assert "PASS" not in out and "FAIL" not in out, "no win rate means no verdict"


def test_suggest_renders_a_corrupt_study_report(runner, patch_scan, monkeypatch):
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate",
                        lambda cfg, name: {"status": "corrupt", "win_rate": None,
                                           "n_trades": 0, "path": "x.json",
                                           "error": "JSONDecodeError:Expecting value"})
    res = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "unreadable" in out
    assert "Top candidate: WIDE" in out


def test_suggest_infers_status_when_the_dict_omits_it(runner, patch_scan, monkeypatch):
    """The dashboard round-trips this dict through JSON and other callers may build
    it by hand, so a missing status must not silently demote a real measurement to
    "no previous study"."""
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    monkeypatch.setattr("tvlab.cli._measured_win_rate",
                        lambda cfg, name: {"win_rate": 0.60, "n_trades": 250,
                                           "profit_factor": 1.5, "path": "x.json"})
    out = _flat(runner.invoke(cli, ["suggest", "--market", "crypto",
                                    "--style", "swing"]).output)
    assert "measured on the last study" in out.lower()
    assert "60.0%" in out


def test_measured_win_rate_returns_none_for_an_unknown_strategy():
    from tvlab.cli import _measured_win_rate
    from tvlab.config import load_config

    cfg = load_config(market="crypto", style="swing")
    assert _measured_win_rate(cfg, "no_such_strategy_xyz") is None


def test_measured_win_rate_survives_a_corrupt_report(tmp_path, monkeypatch):
    """A half-written report must not kill the scan."""
    from tvlab.cli import _measured_win_rate
    from tvlab.config import load_config

    cfg = load_config(market="crypto", style="swing")
    d = tmp_path / "smc_trend_crypto_4h"
    d.mkdir(parents=True)
    (d / "smc_trend_crypto_report.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("tvlab.cli.REPO_ROOT", tmp_path)
    assert _measured_win_rate(cfg, "smc_trend") is None


# -------------------------------------------------------------------- options


def test_suggest_help_disclaims_being_a_recommendation(runner):
    out = _flat(runner.invoke(cli, ["suggest", "--help"]).output)
    assert "NOT VALIDATED" in out
    assert "not a recommendation" in out.lower()
    assert "breakeven win rate" in out.lower()


def test_suggest_exposes_the_gate_and_the_trendline_options(runner):
    out = runner.invoke(cli, ["suggest", "--help"]).output
    assert "--max-breakeven" in out
    assert "--trendlines" in out
    assert "--top" in out


def test_suggest_accepts_the_nse_data_source(runner):
    """The command must accept every source the store does, or the new NSE path is
    unreachable from the one command that matters most for Indian equities."""
    out = runner.invoke(cli, ["suggest", "--help"]).output
    assert "nse" in out and "yahoo_chart" in out


def test_suggest_passes_the_trendline_flag_through(runner, patch_scan, monkeypatch):
    patch_scan([_row("WIDE", last=100.0, stop=98.0, target=104.0)])
    seen = {}

    def fake(store, cfg, symbols, **kwargs):
        seen.update(kwargs)
        return [_row("WIDE", last=100.0, stop=98.0, target=104.0)], {}

    monkeypatch.setattr("tvlab.screener.scan_universe", fake)
    monkeypatch.setattr("tvlab.screener.scanner.scan_universe", fake)
    res = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing", "--trendlines"])
    assert res.exit_code == 0, res.output
    assert seen.get("with_trendlines") is True


def test_structural_stop_is_surfaced_as_a_reason(runner, patch_scan):
    """A stop beyond the nearest fitted trendline is a real invalidation level, and
    that is worth more than an arbitrary ATR multiple."""
    patch_scan([_row("STRUCT", last=100.0, stop=98.0, target=104.0, support_pct=-6.0)])
    out = runner.invoke(cli, ["suggest", "--market", "crypto", "--style", "swing",
                              "--trendlines"]).output
    assert "structural" in out.lower()


# ------------------------------------------- data failure vs. a quiet market


def test_suggest_reports_a_total_data_failure_as_a_failure_not_a_verdict(runner, patch_scan):
    """The central distinction. "Nothing qualifies" is a claim about the market and
    is only licensed if the market was observed. With every symbol failing to load
    there is no observation, and the reassuring phrasing would turn a network
    outage into a quiet-market verdict - the same shape as an empty frame reading
    as "no trades today"."""
    patch_scan([], errors={"RELIANCE.NS": "DataError: could not reach Yahoo",
                           "TCS.NS": "DataError: could not reach Yahoo"})
    res = runner.invoke(cli, ["suggest", "--market", "indian_equities", "--style", "swing"])
    assert res.exit_code == 2, res.output
    out = _flat(res.output)
    assert "data failure, not a market verdict" in out
    assert "Nothing qualifies" not in out, "must not claim a market verdict it did not earn"
    assert "normal state" not in out


def test_suggest_data_failure_lists_the_errors_and_a_retry(runner, patch_scan):
    patch_scan([], errors={f"S{i}.NS": "DataError: unreachable" for i in range(9)})
    out = _flat(runner.invoke(cli, ["suggest", "--market", "indian_equities",
                                    "--style", "swing"]).output)
    assert "All 9 symbol(s) failed to load" in out
    assert "... and 4 more" in out, "should truncate rather than dump every error"
    assert "tvlab data --market indian_equities" in out, "must give a concrete next step"
    assert "None" not in out.split("tvlab data")[1][:80], "unresolved flag leaked into the retry"


def test_suggest_partial_failure_still_gives_a_market_verdict(runner, patch_scan):
    """Over-warning is its own failure mode: if two of ten symbols loaded and none
    fired, that IS a legitimate (if partial) market observation and must not exit
    non-zero."""
    patch_scan([_row("QUIET", signal=0)], errors={"DEAD.NS": "DataError: unreachable"})
    res = runner.invoke(cli, ["suggest", "--market", "indian_equities", "--style", "swing"])
    assert res.exit_code == 0, res.output
    assert "Nothing qualifies" in _flat(res.output)


def test_scan_reports_a_total_data_failure_as_a_failure(runner, monkeypatch):
    """`scan` had the identical bug: every symbol failed to load and it still
    printed "That is the normal state - this setup should not fire often." """
    def fake(store, cfg, symbols, **kwargs):
        return [], {s: "DataError: could not reach query1.finance.yahoo.com" for s in symbols}

    monkeypatch.setattr("tvlab.cli.scan_universe", fake, raising=False)
    monkeypatch.setattr("tvlab.screener.scan_universe", fake, raising=False)
    monkeypatch.setattr("tvlab.screener.scanner.scan_universe", fake, raising=False)
    res = runner.invoke(cli, ["scan", "--market", "crypto", "--style", "swing"])
    assert res.exit_code == 2, res.output
    out = _flat(res.output)
    assert "data failure, not a market verdict" in out
    assert "normal state" not in out


def test_scan_partial_failure_qualifies_its_verdict_with_coverage(runner, monkeypatch):
    """A partial view must not present itself as a complete one. The verdict is
    still allowed, but it has to say how much of the universe it actually saw."""
    def fake(store, cfg, symbols, **kwargs):
        rows = [ScanRow(symbol=symbols[0], signal=0, bars=500)] if symbols else []
        return rows, {s: "DataError: unreachable" for s in symbols[1:]}

    monkeypatch.setattr("tvlab.cli.scan_universe", fake, raising=False)
    monkeypatch.setattr("tvlab.screener.scan_universe", fake, raising=False)
    monkeypatch.setattr("tvlab.screener.scanner.scan_universe", fake, raising=False)
    res = runner.invoke(cli, ["scan", "--market", "crypto", "--style", "swing",
                              "--symbols", "BTC/USDT,DEAD/COIN"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "normal state" in out
    assert "based on 1 of 2 symbols" in out, "must state the coverage it actually had"
