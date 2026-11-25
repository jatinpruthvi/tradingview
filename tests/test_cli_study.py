"""``tvlab study --strategy`` must actually change the strategy.

The flag was accepted and silently ignored: ``study --strategy orb`` ran the
configured default with its own 243-combo grid and wrote a report labelled
``smc_trend``, which reads like a successful run of the wrong thing. That is the
worst failure mode a research tool has, so it is pinned here.
"""
import json

import pytest
from click.testing import CliRunner

from tvlab.cli import cli
from tvlab.config import load_config


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich truncates to the terminal width; CliRunner's default 80 columns
    mangles table cells and makes output assertions unreliable."""
    monkeypatch.setenv("COLUMNS", "250")


def _study(tmp_path, *args):
    out = tmp_path / "study"
    res = CliRunner().invoke(cli, [
        "study", "--market", "crypto", "--symbols", "BTC/USDT",
        "--data-source", "synthetic", "--no-walk-forward", "--no-monte-carlo",
        "--no-nulls", "--no-costs", "--out", str(out), *args,
    ])
    return res, out


def _report(out):
    reports = list(out.glob("*_report.json"))
    assert reports, f"no report written in {out}"
    return reports[0], json.loads(reports[0].read_text())


def test_strategy_flag_selects_the_strategy(tmp_path):
    res, out = _study(tmp_path, "--strategy", "ema_pullback")
    assert res.exit_code == 0, res.output
    path, data = _report(out)
    assert data["strategy"] == "ema_pullback"
    assert "ema_pullback" in path.name


def test_without_the_flag_the_configured_default_runs(tmp_path):
    expected = load_config(market="crypto", style="swing").strategy.name
    res, out = _study(tmp_path)
    assert res.exit_code == 0, res.output
    _, data = _report(out)
    assert data["strategy"] == expected
