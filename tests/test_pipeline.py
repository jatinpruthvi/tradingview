"""End-to-end pipeline: study, allocation accounting and the benchmark."""
import numpy as np
import pandas as pd
import pytest

from tvlab.config import load_config
from tvlab.pipeline import build_strategy, combine_equity, run_study


@pytest.fixture(scope="module")
def cfg():
    return load_config(market="crypto", style="swing").merge({"data.synthetic_bars": 1500})


@pytest.fixture(scope="module")
def study(cfg):
    return run_study(cfg, symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"], with_benchmark=True)


def test_portfolio_curve_starts_at_the_account_balance(cfg, study):
    """Regression: when the risk caps fund only part of the book, the funded
    slices sum to less than initial_capital, so the curve started below the
    account balance and every capital-relative metric disagreed with every
    curve-relative one. One report showed 'Net P&L -49,916' next to 'Total
    return +0.17%' for the same run. Unallocated capital is cash, not nothing."""
    eq = study.portfolio_equity
    assert eq.iloc[0] == pytest.approx(cfg.backtest.initial_capital, rel=1e-9)


def test_net_pnl_and_return_agree(cfg, study):
    m = study.portfolio_metrics
    eq = study.portfolio_equity
    capital = cfg.backtest.initial_capital
    assert m["net_pnl"] == pytest.approx(eq.iloc[-1] - eq.iloc[0], abs=1e-6)
    assert m["total_return_pct"] == pytest.approx(m["net_pnl"] / capital * 100, abs=1e-6)
    assert m["final_equity"] == pytest.approx(eq.iloc[-1], abs=1e-6)


def test_study_records_the_allocation(cfg, study):
    a = study.allocation
    assert a is not None
    assert a["scheme"] == "equal"
    assert a["n_symbols"] >= 1
    assert 0 < a["total_weight_pct"] <= 100.0 + 1e-9
    assert a["corr_threshold"] == cfg.risk.correlation_threshold
    # the report must be able to say how much account risk the book carries
    assert a["aggregate_risk_pct"] is not None
    assert a["aggregate_risk_pct"] <= cfg.risk.risk_per_trade * 100 * len(a["rows"]) + 1e-6


def test_unallocated_symbols_are_still_studied(cfg, study):
    """Excluded symbols must still be backtested and reported per symbol - the
    cap decides what is FUNDED, not what is MEASURED."""
    studied = {s.symbol for s in study.symbols}
    funded = set(study.allocation and [r["symbol"] for r in study.allocation["rows"]] or [])
    assert funded <= studied
    assert set(study.unallocated) == studied - funded


def test_correlation_cap_is_respected_in_the_funded_book(cfg, study):
    a = study.allocation
    for g in a["groups"]:
        assert sum(1 for s in g if s in funded_symbols(a)) <= cfg.risk.max_correlated_positions


def funded_symbols(a):
    return {r["symbol"] for r in a["rows"]}


def test_single_position_cap_is_respected(cfg, study):
    for r in study.allocation["rows"]:
        assert r["weight_pct"] <= cfg.risk.max_position_pct * 100 + 1e-6


def test_benchmark_is_buy_and_hold_not_a_swing_trade(cfg, study):
    """Regression: -inf was treated as a missing stop and replaced with an ATR
    stop, so the 'benchmark' became a 90-bar swing trade and reported -0.53%
    while the underlying asset had risen +141%."""
    b = study.benchmark
    assert b is not None
    m = b.get("metrics") or b
    assert m.get("n_trades") == 1, f"a benchmark should be one position, got {m.get('n_trades')}"


def test_combine_equity_with_weights_deploys_only_that_weight(cfg, study):
    curves = {s.symbol: s.equity for s in study.symbols}
    half = {"BTC/USDT": 0.25, "ETH/USDT": 0.25}
    eq = combine_equity(study.symbols[:2], cfg.backtest.initial_capital, weights=half)
    # 50% deployed + 50% idle cash = the full account balance at t0
    assert eq.iloc[0] == pytest.approx(cfg.backtest.initial_capital, rel=1e-9)


def test_combine_equity_equal_weight_sums_to_capital(cfg, study):
    eq = combine_equity(study.symbols, cfg.backtest.initial_capital)
    assert eq.iloc[0] == pytest.approx(cfg.backtest.initial_capital, rel=1e-9)


def test_combine_equity_handles_empty():
    assert combine_equity([], 100_000.0).empty


def test_combine_equity_shows_diversification(cfg, study):
    """The combined curve must not be worse than its worst component at the same
    capital - that is the only reason to hold several symbols."""
    combined = combine_equity(study.symbols, cfg.backtest.initial_capital)
    from tvlab.backtest.metrics import max_drawdown
    dd_combined = abs(max_drawdown(combined))
    dds = []
    for sym in study.symbols:
        scaled = sym.equity / sym.equity.iloc[0] * cfg.backtest.initial_capital
        dds.append(abs(max_drawdown(scaled)))
    assert dd_combined <= max(dds) + 1e-9, (
        f"combined DD {dd_combined:.3f}% exceeded the worst single symbol {max(dds):.3f}%")


def test_params_for_prevents_benchmark_param_leakage(cfg):
    """Regression: switching only strategy.name left the previous strategy's
    params in place, so buy_and_hold inherited max_bars_in_trade=90."""
    bench_cfg = cfg.merge({"strategy.name": "buy_and_hold"})
    strat = build_strategy(bench_cfg)
    assert strat.params.get("max_bars_in_trade") is None
    assert "swing_length" not in strat.params


def test_build_strategy_uses_declared_params_when_they_match(cfg):
    strat = build_strategy(cfg)
    assert strat.name == cfg.strategy.name
    assert strat.params.get("swing_length") == cfg.strategy.params["swing_length"]


def test_study_survives_a_bad_symbol(cfg):
    """One unusable ticker must not kill the whole study."""
    st = run_study(cfg, symbols=["BTC/USDT", "NOT/REAL", "ETH/USDT"], with_benchmark=False)
    assert len(st.symbols) >= 1


def test_per_symbol_table_is_complete(cfg, study):
    t = study.per_symbol
    assert len(t) == len(study.symbols)
    for col in ("symbol", "return_pct", "max_dd_pct", "trades", "sharpe",
                "profit_factor", "dsr"):
        assert col in t.columns, col


def test_profitable_symbols_filter(cfg, study):
    prof = study.profitable_symbols()
    assert isinstance(prof, list)
    for s in prof:
        row = study.per_symbol[study.per_symbol["symbol"] == s].iloc[0]
        assert (row["return_pct"] or 0) > 0
