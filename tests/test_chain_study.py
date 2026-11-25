"""The panel study: forward returns, cross-sectional IC, and the verdict gate.

The statistics are the easy part. What these tests really pin down is the two
ways a study like this lies to its author: measuring the forward return from the
snapshot day itself (a one-bar lookahead), and printing a mean IC from a sample
far too small for one. Both are asserted against directly.
"""
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from tvlab import cli as cli_mod
from tvlab.chain_study import (
    MIN_NAMES_PER_DAY,
    FeatureResult,
    _spearman,
    add_derived,
    forward_returns,
    forward_window_open,
    study,
)
from tvlab.cli import cli
from tvlab.data.option_chain import SNAPSHOT_COLUMNS, append_rows


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _bars(closes, start="2026-09-01", tz="Asia/Kolkata"):
    """Business-day bars, stamped 09:15 in the market's own timezone."""
    idx = pd.bdate_range(start=start, periods=len(closes), tz=tz) + pd.Timedelta(hours=9, minutes=15)
    return pd.DataFrame({"close": [float(c) for c in closes]}, index=idx)


def _panel(rows) -> pd.DataFrame:
    out = {c: None for c in SNAPSHOT_COLUMNS}
    df = pd.DataFrame(rows)
    for c in SNAPSHOT_COLUMNS:
        if c not in df.columns:
            df[c] = out[c]
    return df[list(SNAPSHOT_COLUMNS)]


def _row(as_of: str, symbol: str, **kw) -> dict:
    base = {"as_of": as_of, "symbol": symbol, "ok": True, "atm_iv": 25.0,
            "pcr": 0.8, "ce_oi": 1000, "pe_oi": 800, "live_strikes": 10,
            "call_wall_dist_pct": 4.0, "put_wall_dist_pct": -4.0,
            "skew_put_minus_call": 1.0}
    base.update(kw)
    return base


# ------------------------------------------------------------ forward returns


def test_the_window_starts_at_the_first_bar_on_or_after_the_snapshot():
    """Using the snapshot day's own close would be a one-bar lookahead."""
    closes = [100, 110, 121, 133.1]
    bars = {"AAA": _bars(closes)}
    panel = _panel([_row("2026-09-01", "AAA")])
    ret, stats = forward_returns(panel, bars, horizon=2)

    assert stats["rows"] == 1
    got = float(ret["fwd_ret"].iloc[0])
    assert got == pytest.approx(121 / 100 - 1), "entry 100 (day 0) -> exit 121 (day 2)"
    assert ret["entry_date"].iloc[0] == "2026-09-01"


def test_a_snapshot_before_the_open_enters_on_the_same_bar_not_the_day_before():
    closes = [100, 110, 121]
    bars = {"AAA": _bars(closes)}
    panel = _panel([_row("2026-09-01", "AAA")])
    ret, _ = forward_returns(panel, bars, horizon=1)
    assert float(ret["fwd_ret"].iloc[0]) == pytest.approx(110 / 100 - 1)


def test_a_snapshot_on_a_non_trading_day_rolls_forward():
    # 2026-09-05 is a Saturday; the next bar is Monday 2026-09-07.
    bars = {"AAA": _bars([100, 110, 121, 133.1, 146.41, 161.05])}
    panel = _panel([_row("2026-09-05", "AAA")])
    ret, _ = forward_returns(panel, bars, horizon=1)
    assert ret["entry_date"].iloc[0] == "2026-09-07"
    # bars are 09-01..09-04 then 09-07 (146.41) and 09-08 (161.05): entry Monday,
    # exit the next bar, and the Thu/Fri closes are never used.
    assert float(ret["fwd_ret"].iloc[0]) == pytest.approx(161.05 / 146.41 - 1)


def test_an_unclosed_window_is_dropped_rather_than_filled_with_zero():
    """A zero would read as 'no move' and pull every mean toward it."""
    bars = {"AAA": _bars([100, 110, 121])}
    panel = _panel([_row("2026-09-03", "AAA")])     # last bar, nothing after it
    ret, stats = forward_returns(panel, bars, horizon=2)
    assert ret.empty
    assert stats["censored"] == 1


def test_a_symbol_with_no_bars_is_counted_not_silently_skipped():
    panel = _panel([_row("2026-09-01", "AAA"), _row("2026-09-01", "ZZZ")])
    ret, stats = forward_returns(panel, {"AAA": _bars([100, 110, 121])}, horizon=1)
    assert stats["no_bars"] == 1
    assert set(ret["symbol"]) == {"AAA"}


def test_the_timestamp_is_read_in_market_time_not_utc():
    """03:45 UTC is 09:15 IST; a UTC normalise would already be the right date,
    but 20:00 UTC is 01:30 IST the *next* day, and reading that in UTC would put
    every forward return one bar early."""
    idx = pd.DatetimeIndex([pd.Timestamp("2026-09-01 20:00", tz="UTC"),
                            pd.Timestamp("2026-09-02 20:00", tz="UTC"),
                            pd.Timestamp("2026-09-03 20:00", tz="UTC")])
    bars = {"AAA": pd.DataFrame({"close": [100.0, 110.0, 121.0]}, index=idx)}
    panel = _panel([_row("2026-09-02", "AAA")])
    ret, _ = forward_returns(panel, bars, horizon=1)
    assert ret["entry_date"].iloc[0] == "2026-09-02", "20:00 UTC is the 2nd in IST"
    assert float(ret["fwd_ret"].iloc[0]) == pytest.approx(121 / 110 - 1)


# ---------------------------------------------------------------- derived data


def test_iv_rv_ratio_needs_the_bars_and_is_absent_without_them():
    panel = _panel([_row("2026-09-01", "AAA", atm_iv=25.0)])
    returns = pd.DataFrame([{"as_of": "2026-09-01", "symbol": "AAA",
                             "entry_date": "2026-09-01", "fwd_ret": 0.01, "rv20": 0.20}])
    frame = add_derived(panel, returns)
    assert frame["iv_rv_ratio"].iloc[0] == pytest.approx(25.0 / 20.0)

    frame = add_derived(panel, returns.assign(rv20=np.nan))
    assert not np.isfinite(frame["iv_rv_ratio"].iloc[0])


def test_oi_total_is_logged_so_size_does_not_dominate_the_rank():
    panel = _panel([_row("2026-09-01", "AAA", ce_oi=10, pe_oi=10),
                    _row("2026-09-01", "BBB", ce_oi=100_000, pe_oi=100_000)])
    returns = pd.DataFrame({"as_of": ["2026-09-01"] * 2, "symbol": ["AAA", "BBB"],
                            "entry_date": ["2026-09-01"] * 2, "fwd_ret": [0.0, 0.0],
                            "rv20": [0.2, 0.2]})
    frame = add_derived(panel, returns)
    assert set(np.round(frame["oi_total_log"], 3)) == {round(np.log1p(20), 3),
                                                       round(np.log1p(200_000), 3)}


# ------------------------------------------------------------------ statistics


def test_spearman_is_exact_for_monotone_relationships():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert _spearman([1, 2, 3], [1, 5, 2]) < 0.9


def test_a_constant_feature_has_no_rank_correlation_rather_than_zero():
    assert np.isnan(_spearman([5, 5, 5, 5], [1, 2, 3, 4]))


# ---------------------------------------------------------------------- study


def _synthetic(days=60, names=25, seed=0, predictor=None):
    """A panel + returns frame whose forward return is a chosen function of the feature."""
    rng = np.random.default_rng(seed)
    panel_rows, ret_rows = [], []
    for d in range(days):
        as_of = (date(2026, 1, 5) + timedelta(days=d)).isoformat()
        pcr = rng.normal(0.8, 0.25, names)
        noise = rng.normal(0, 0.02, names)
        fwd = noise if predictor is None else predictor(pcr) + noise * 0.1
        for i in range(names):
            panel_rows.append(_row(as_of, f"S{i:02d}", pcr=float(pcr[i])))
            ret_rows.append({"as_of": as_of, "symbol": f"S{i:02d}",
                             "entry_date": as_of, "fwd_ret": float(fwd[i]), "rv20": 0.25})
    return _panel(panel_rows), pd.DataFrame(ret_rows)


def test_a_predictive_feature_is_reported_as_a_candidate():
    panel, returns = _synthetic(predictor=lambda f: f)
    res = study(panel, returns, horizon=20, min_days=40, features=["pcr"],
                min_names=MIN_NAMES_PER_DAY)
    item = res.results[0]
    assert item.days == 60 and item.rows == 60 * 25
    assert item.mean_ic == pytest.approx(1.0, abs=0.02)
    assert item.consistency == 1.0
    assert item.quintile_spread > 0
    assert item.notable
    assert "CANDIDATE" in res.verdict


def test_an_inverse_feature_is_also_notable():
    """A feature that predicts *down* moves is information, not a dud: the sign
    of the relationship is the finding, so consistency is sign-agnostic."""
    panel, returns = _synthetic(predictor=lambda f: -f)
    res = study(panel, returns, min_days=40, features=["pcr"], min_names=MIN_NAMES_PER_DAY)
    item = res.results[0]
    assert item.mean_ic == pytest.approx(-1.0, abs=0.02)
    assert item.consistency == 1.0
    assert item.notable
    assert item.quintile_spread < 0


def test_noise_generates_no_candidate():
    panel, returns = _synthetic(seed=7, predictor=None)
    res = study(panel, returns, min_days=40, features=["pcr"], min_names=MIN_NAMES_PER_DAY)
    assert res.results[0].days == 60
    assert not res.results[0].notable
    assert "NO EVIDENCE" in res.verdict


def test_the_verdict_refuses_before_min_days_even_when_the_numbers_look_great():
    """The failure this protects against is a beautiful IC from a fortnight."""
    panel, returns = _synthetic(days=10, predictor=lambda f: f)
    res = study(panel, returns, min_days=40, features=["pcr"])
    assert res.evaluated_days == 10
    assert res.insufficient
    assert "INSUFFICIENT SAMPLE" in res.verdict
    assert res.results[0].mean_ic > 0.9, "the numbers are computed, the verdict still refuses"


def test_thin_days_do_not_count_as_a_cross_section():
    panel, returns = _synthetic(days=60, names=10, predictor=lambda f: f)
    res = study(panel, returns, min_days=5, min_names=MIN_NAMES_PER_DAY, features=["pcr"])
    assert res.results[0].days == 0
    assert any("no day had" in n for n in res.notes)
    assert res.insufficient


def test_a_missing_column_is_reported_not_crashed():
    panel, returns = _synthetic(days=5)
    res = study(panel, returns, min_days=1, features=["not_a_column"])
    assert any("not a panel column" in n for n in res.notes)


def test_an_empty_panel_says_what_to_run_first():
    res = study(_panel([]), pd.DataFrame(), min_days=40)
    assert res.results == [] or all(not r.has_sample for r in res.results)
    assert any("tvlab chain snapshot" in n for n in res.notes)
    assert res.insufficient


def test_a_feature_result_needs_more_than_one_day_to_count_as_a_sample():
    assert FeatureResult("pcr", "x", days=1, mean_ic=1.0, ic_t=9).has_sample is False
    assert FeatureResult("pcr", "x", days=2, mean_ic=1.0, ic_t=9).has_sample is True


# --------------------------------------------------------------- window gate


def test_the_window_is_closed_on_the_day_the_panel_is_born():
    panel = _panel([_row(date.today().isoformat(), "AAA")])
    assert forward_window_open(panel, horizon=20, today=date.today()) is False


def test_the_window_opens_once_enough_calendar_days_have_passed():
    panel = _panel([_row((date.today() - timedelta(days=30)).isoformat(), "AAA")])
    assert forward_window_open(panel, horizon=20, today=date.today()) is True


# ---------------------------------------------------------------------- CLI


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def pinned_console_width(monkeypatch):
    monkeypatch.setattr(cli_mod.console, "_width", 200, raising=False)


def _write_panel(path, rows):
    append_rows(path, rows)


def test_study_on_a_fresh_panel_refuses_and_fetches_nothing(runner, tmp_path, monkeypatch):
    """Day one: no forward window exists, so 218 bar requests would be wasted and
    the only honest output is 'not yet'."""
    called = {"returns": False}

    def fake_returns(*a, **kw):
        called["returns"] = True
        return pd.DataFrame(), {}

    monkeypatch.setattr(cli_mod, "_chain_returns", fake_returns)
    path = tmp_path / "panel.parquet"
    _write_panel(path, [_row(date.today().isoformat(), "AAA"),
                        _row(date.today().isoformat(), "BBB")])

    res = runner.invoke(cli, ["chain", "study", "--panel", str(path)])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "no panel day is old enough" in out
    assert "INSUFFICIENT SAMPLE" in out
    assert called["returns"] is False


def test_study_with_a_real_sample_prints_the_candidate(runner, tmp_path, monkeypatch):
    panel, returns = _synthetic(days=6, predictor=lambda f: f)
    monkeypatch.setattr(cli_mod, "_chain_returns", lambda *a, **kw: (returns, {}))
    path = tmp_path / "panel.parquet"
    _write_panel(path, panel.to_dict("records"))

    res = runner.invoke(cli, ["chain", "study", "--panel", str(path),
                              "--min-days", "5", "--features", "pcr"])
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "pcr" in out
    assert "CANDIDATE, NOT A FILTER" in out
    # The marker column is the visual "this one cleared the bar"; rich draws the
    # box-drawing bar (U+2502), which cannot be confused with the ASCII pipes in
    # the prose below the table.
    assert "\u2502 * \u2502" in out, "the candidate row carries the marker"


def test_study_reports_missing_bars_without_killing_the_run(runner, tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("should not be reached")

    panel, returns = _synthetic(days=6, predictor=lambda f: f)
    monkeypatch.setattr(cli_mod, "_chain_returns", lambda *a, **kw: (returns, {"no_bars": 4}))
    path = tmp_path / "panel.parquet"
    _write_panel(path, panel.to_dict("records"))

    res = runner.invoke(cli, ["chain", "study", "--panel", str(path), "--min-days", "99"])
    assert res.exit_code == 0, res.output
    assert "INSUFFICIENT SAMPLE" in _flat(res.output)
    _ = boom
