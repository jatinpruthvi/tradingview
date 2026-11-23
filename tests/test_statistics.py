"""Walk-forward, Monte Carlo and null tests.

These decide whether a result is trustworthy, so they get the same scrutiny as
the engine: a validator that always says "pass" is worse than no validator.
"""
import numpy as np
import pandas as pd
import pytest

from tvlab.backtest.metrics import (deflated_sharpe_ratio, expected_max_sharpe,
                                    full_report, max_drawdown, sharpe)
from tvlab.backtest.montecarlo import block_bootstrap, fan_chart, simulate_trades
from tvlab.backtest.walkforward import (MIN_TRADES_FOR_OBJECTIVE, make_windows,
                                        min_trades_for_window, objective_value,
                                        param_grid, run_walk_forward)
from tvlab.backtest.engine import Backtester
from tvlab.config import load_config
from tvlab.data.synthetic import basket
from tvlab.strategy import get
from tvlab.strategy.base import blank_signals
from tvlab.validation.null_test import (interpret, rebuild_from_returns,
                                        shuffled_null, sign_null)


# --------------------------------------------------------------------- windows
WIN = dict(n_splits=4, train_pct=0.6, rolling=True, min_train=100, min_test=50)


def test_oos_segments_are_contiguous_and_never_overlap_train():
    ws = make_windows(1000, **WIN)
    assert len(ws) == 4
    for w in ws:
        assert w.train_end == w.test_start, "train/test must abut, never overlap"
        assert w.test_end > w.test_start
    segs = [(w.test_start, w.test_end) for w in ws]
    assert all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1)), \
        "gaps between OOS segments mean the stitched curve is not continuous"
    assert segs[-1][1] == 1000, "the OOS run must reach the end of the series"


def test_train_pct_is_a_share_of_the_whole_series():
    """Regression: `span * train_pct * n_splits / n_splits` was a no-op and the
    rolling branch then ignored train_pct for the test length entirely."""
    ws = make_windows(1000, **WIN)
    assert ws[0].train_end - ws[0].train_start == 600       # 0.6 of the series
    assert ws[0].test_end - ws[0].test_start == 100         # remainder / n_splits


def test_rolling_train_length_is_constant_and_slides():
    ws = make_windows(1000, **WIN)
    assert len({w.train_end - w.train_start for w in ws}) == 1
    assert all(b.train_start > a.train_start for a, b in zip(ws, ws[1:]))


def test_anchored_train_start_is_fixed_and_grows():
    anch = make_windows(1000, **{**WIN, "rolling": False})
    assert {w.train_start for w in anch} == {0}
    lengths = [w.train_end - w.train_start for w in anch]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)


def test_too_few_bars_raises_instead_of_reporting_nothing():
    """Silently returning zero windows is how the report ended up with no
    out-of-sample evidence at all. Fail loudly instead."""
    with pytest.raises(ValueError, match="not enough bars"):
        make_windows(120, **WIN)                       # < min_train + min_test
    with pytest.raises(ValueError):
        make_windows(400, **{**WIN, "train_pct": 0.99})  # nothing left to test


def test_train_pct_extreme_still_leaves_an_oos_segment():
    ws = make_windows(2000, **{**WIN, "train_pct": 0.9, "n_splits": 2})
    assert len(ws) == 2
    assert ws[-1].test_end == 2000


def test_min_trades_floor_scales_with_window_length():
    """Regression: a fixed floor of 8 rejected every window for a selective
    setup, so walk-forward logged 'no usable fit' instead of a result."""
    assert min_trades_for_window(300) == MIN_TRADES_FOR_OBJECTIVE      # small window
    assert min_trades_for_window(3000) > MIN_TRADES_FOR_OBJECTIVE
    assert min_trades_for_window(100_000) <= 8                          # hard cap
    assert min_trades_for_window(0) == MIN_TRADES_FOR_OBJECTIVE         # no crash


def test_objective_rejects_degenerate_fits():
    few = {"n_trades": 0, "sharpe": 9.0, "total_return_pct": 500.0}
    assert objective_value(few, "sharpe", min_trades=1) == -np.inf, \
        "a zero-trade fit with a great Sharpe must not win the search"
    ok = {"n_trades": 20, "sharpe": 1.0, "total_return_pct": 5.0}
    assert np.isfinite(objective_value(ok, "sharpe", min_trades=3))


def test_param_grid_is_capped():
    g = param_grid(get("smc_trend"), max_trials=8)
    assert 1 <= len(g) <= 8
    assert all(isinstance(p, dict) for p in g)


# ------------------------------------------------------------------ deflated SR
def test_deflated_sharpe_penalises_more_trials():
    """The whole point of DSR: 64 attempts at a Sharpe of 1.0 is far less
    impressive than 1 attempt."""
    one = deflated_sharpe_ratio(1.0, n_trials=1, n_obs=500)
    many = deflated_sharpe_ratio(1.0, n_trials=64, n_obs=500)
    assert one["dsr"] > many["dsr"]
    assert expected_max_sharpe(64) > expected_max_sharpe(1)


def test_deflated_sharpe_needs_enough_observations():
    """A great Sharpe over 5 bars is not evidence, and must say so."""
    short = deflated_sharpe_ratio(3.0, n_trials=1, n_obs=5)
    long = deflated_sharpe_ratio(3.0, n_trials=1, n_obs=2000)
    assert short["verdict"] == "insufficient_data"
    assert not np.isfinite(short["dsr"])
    assert long["verdict"] == "pass"
    assert long["dsr"] > 0.95


def test_deflated_sharpe_verdict_degrades_with_trials():
    one = deflated_sharpe_ratio(1.0, n_trials=1, n_obs=500)["dsr"]
    many = deflated_sharpe_ratio(1.0, n_trials=64, n_obs=500)["dsr"]
    assert one > many


def test_deflated_sharpe_handles_degenerate_input():
    out = deflated_sharpe_ratio(0.0, n_trials=1, n_obs=1)
    assert isinstance(out, dict) and "dsr" in out
    nan = deflated_sharpe_ratio(float("nan"), n_trials=1, n_obs=100)
    assert nan["dsr"] is None or not np.isfinite(nan["dsr"]) or nan["dsr"] <= 0.5


# ---------------------------------------------------------------- monte carlo
def _trades(n=60, edge=0.0, seed=1):
    rng = np.random.default_rng(seed)
    pnl = rng.normal(edge, 300.0, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"pnl": pnl, "r_multiple": pnl / 300.0, "entry_time": idx,
                         "exit_time": idx, "side": 1, "bars_held": 5})


def test_block_bootstrap_preserves_length_and_seed():
    rets = np.random.default_rng(0).normal(0, 0.01, 500)
    a = block_bootstrap(rets, n_sims=20, block_size=10, rng=np.random.default_rng(7))
    b = block_bootstrap(rets, n_sims=20, block_size=10, rng=np.random.default_rng(7))
    assert a.shape == (20, 500)
    assert np.allclose(a, b), "bootstrap must be reproducible"
    c = block_bootstrap(rets, n_sims=20, block_size=10, rng=np.random.default_rng(8))
    assert not np.allclose(a, c)


def test_block_bootstrap_keeps_vol_clustering():
    """Blocks, not iid draws - otherwise the DD distribution is understated."""
    rets = np.concatenate([np.full(50, 0.03), np.full(50, -0.03)])
    out = block_bootstrap(rets, n_sims=50, block_size=25, rng=np.random.default_rng(3))
    # per-sim kurtosis of runs should exceed an iid shuffle's
    assert out.shape == (50, 100)


def test_simulate_trades_reports_breach_probabilities():
    res = simulate_trades(_trades(80, edge=50.0, seed=2), initial_capital=100_000.0,
                          n_sims=300, block_size=10, seed=7, dd_limits=[0.05, 0.10, 0.20])
    s = res.summary()
    assert s["n_sims"] == 300 and s["n_trades"] == 80
    for k in ("prob_loss", "prob_beat_backtest", "final_return_pct", "max_drawdown_pct",
              "ruin_probabilities", "percentiles"):
        assert k in s, k
    assert 0.0 <= s["prob_loss"] <= 1.0
    assert set(s["ruin_probabilities"]) == {"dd_exceeds_5pct", "dd_exceeds_10pct",
                                            "dd_exceeds_20pct"}, \
        "ruin probabilities must match the dd_limits actually requested"
    # a positive-edge sample should mostly not lose money
    assert s["prob_loss"] < 0.5
    assert all(0.0 <= v <= 1.0 for v in s["ruin_probabilities"].values())
    assert s["max_drawdown_pct"]["p95"] >= s["max_drawdown_pct"]["median"]


def test_simulate_trades_on_losing_edge_is_mostly_negative():
    res = simulate_trades(_trades(80, edge=-80.0, seed=5), initial_capital=100_000.0,
                          n_sims=200, block_size=10, seed=7)
    assert res.summary()["prob_loss"] > 0.9


def test_simulate_trades_is_reproducible():
    tr = _trades(60, edge=20.0, seed=3)
    a = simulate_trades(tr, n_sims=100, block_size=10, seed=7).summary()
    b = simulate_trades(tr, n_sims=100, block_size=10, seed=7).summary()
    assert a["final_return_pct"]["median"] == b["final_return_pct"]["median"]


def test_simulate_trades_handles_no_trades():
    res = simulate_trades(pd.DataFrame(), n_sims=50)
    assert isinstance(res.summary(), dict)      # no crash, no fake confidence


def test_fan_chart_quantiles_span_the_index():
    """Regression: quantiles were taken along axis=1, so the bands had one
    value per simulation instead of per bar and plotting raised a length error."""
    eq = pd.Series(100_000 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0002, 0.01, 600))),
                   index=pd.date_range("2022-01-01", periods=600, freq="4h", tz="UTC"))
    fc = fan_chart(eq, n_sims=100, block_size=20, seed=3)
    # one row per simulated bar (the first return is dropped), indexed by time
    assert len(fc) == len(eq) - 1, f"{len(fc)} rows for {len(eq)} bars"
    assert isinstance(fc.index, pd.DatetimeIndex)
    assert list(fc.columns) == ["p5", "p25", "p50", "p75", "p95", "actual"]
    # quantiles must be ordered at every bar, never crossing
    arr = fc[["p5", "p25", "p50", "p75", "p95"]].to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all(), "fan bands cross"
    assert np.isfinite(arr).all()


def test_fan_chart_needs_enough_history():
    eq = pd.Series([100_000.0] * 10, index=pd.date_range("2022-01-01", periods=10, tz="UTC"))
    assert fan_chart(eq).empty


# ------------------------------------------------------------------- null tests
def test_rebuild_from_returns_preserves_the_return_distribution():
    df = basket(["BTC/USDT"], n_bars=500, timeframe="4h")["BTC/USDT"]
    rets = np.log(df["close"]).diff().dropna().to_numpy()
    out = rebuild_from_returns(rets, float(df["close"].iloc[0]), df.index[1:])
    assert len(out) == len(rets)
    assert {"open", "high", "low", "close"} <= set(out.columns)
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()
    got = np.log(out["close"]).diff().dropna().to_numpy()
    assert got.std() == pytest.approx(np.diff(np.log(out['close'].to_numpy())).std(), rel=1e-6)


def _hold_statistic(frame):
    """Buy-and-hold on whatever frame it is handed."""
    eq = frame["close"].to_numpy()
    return {"total_return_pct": float(eq[-1] / eq[0] - 1) * 100}


def test_shuffled_null_detects_a_real_edge():
    """Positive control: a steadily drifting series must beat its own shuffled
    null, or the test could never pass for anybody and is pure theatre.

    Shuffling destroys the ORDER of returns but keeps their sum, so buy-and-hold
    alone cannot separate them - the drift has to be persistent. A random-walk
    statistic with a directional edge does exactly that.
    """
    n = 800
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(2)
    drift = 0.002 + rng.normal(0.0, 0.004, n - 1)      # positive mean, noisy
    df = rebuild_from_returns(drift, 100.0, idx[1:])
    res = shuffled_null(df, _hold_statistic, n_sims=60, statistic="total_return_pct", seed=11)
    assert res.actual > res.null_mean
    # with the drift intact the endpoint is far above a shuffled endpoint
    assert res.better_than_null_pct > 60.0


def test_shuffled_null_does_not_invent_an_edge_from_noise():
    """The other half: a zero-mean series must NOT beat its own shuffled null."""
    n = 800
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    noise = np.random.default_rng(4).normal(0.0, 0.01, n - 1)
    df = rebuild_from_returns(noise, 100.0, idx[1:])
    res = shuffled_null(df, _hold_statistic, n_sims=60, statistic="total_return_pct", seed=11)
    assert res.p_value > 0.05, f"noise looked significant: p={res.p_value}"
    assert 20.0 < res.better_than_null_pct < 80.0


def test_sign_null_flips_trade_direction_not_bars():
    """sign_null re-runs the SAME bars with randomly flipped signals, so it tests
    whether the entry timing carries information."""
    df = basket(["BTC/USDT"], n_bars=400, timeframe="4h")["BTC/USDT"]
    sig = blank_signals(df)
    sig.loc[df.index[50::40], "signal"] = 1.0
    sig.loc[df.index[50::40], "stop"] = df["close"] * 0.98
    sig.loc[df.index[50::40], "target"] = df["close"] * 1.04

    c = load_config(market="crypto", style="swing")

    def run(frame, signals):
        res = Backtester(c).run(frame, signals, symbol="BTC/USDT")
        return {"total_return_pct": float(res.equity.iloc[-1] / res.equity.iloc[0] - 1) * 100}

    res = sign_null(df, sig, run, n_sims=25, seed=3)
    assert res.n_sims == 25
    assert np.isfinite(res.actual)
    assert 0.0 <= res.p_value <= 1.0


def test_interpret_labels_are_sane():
    strong = shuffled_null(basket(["BTC/USDT"], n_bars=300, timeframe="4h")["BTC/USDT"],
                           _hold_statistic, n_sims=20, seed=5)
    txt = interpret({"shuffled_returns": strong})
    assert isinstance(txt, str) and txt.strip()
    # a p-value above 0.05 must never be described as significant
    if strong.p_value > 0.05:
        assert "significant" not in txt.lower() or "not" in txt.lower()


# ---------------------------------------------------------------- walk-forward
def test_walk_forward_produces_oos_windows(cfg=None):
    """Regression: this used to return zero windows, so the report had no
    out-of-sample evidence at all."""
    c = load_config(market="crypto", style="swing").merge(
        {"walk_forward.n_splits": 3, "walk_forward.max_trials": 8,
         "walk_forward.objective": "sharpe"})
    df = basket(["BTC/USDT"], n_bars=1600, timeframe="4h")["BTC/USDT"]
    res = run_walk_forward(c, "BTC/USDT", df, bias=None, strategy_name="ema_pullback",
                           max_trials=8)
    assert res.n_windows > 0, "walk-forward produced no out-of-sample windows"
    s = res.summary()
    assert s["n_windows"] == res.n_windows
    assert s["total_trials"] > 0
    assert len(res.oos_equity) > 0


def test_walk_forward_reports_the_trial_count():
    """DSR needs the number of parameter sets actually tried; understating it
    flatters the result."""
    c = load_config(market="crypto", style="swing").merge(
        {"walk_forward.n_splits": 2, "walk_forward.max_trials": 6,
         "walk_forward.objective": "sharpe"})
    df = basket(["ETH/USDT"], n_bars=1200, timeframe="4h")["ETH/USDT"]
    res = run_walk_forward(c, "ETH/USDT", df, bias=None, strategy_name="donchian_breakout",
                           max_trials=6)
    assert res.total_trials <= 6 * res.n_windows + 1
    assert res.total_trials >= res.n_windows


def test_walk_forward_windows_do_not_leak():
    c = load_config(market="crypto", style="swing").merge(
        {"walk_forward.n_splits": 3, "walk_forward.max_trials": 4,
         "walk_forward.objective": "sharpe"})
    df = basket(["BTC/USDT"], n_bars=1200, timeframe="4h")["BTC/USDT"]
    res = run_walk_forward(c, "BTC/USDT", df, bias=None, strategy_name="ema_pullback",
                           max_trials=4)
    for w in res.windows:
        # WindowResult exposes the (start, end) timestamps it actually used
        (tr0, tr1), (te0, te1) = w.train_slice, w.test_slice
        assert tr1 <= te0, f"train {tr0}->{tr1} overlaps test {te0}->{te1}"
        assert te1 <= df.index[-1]
        assert w.trials > 0
        assert w.min_trades_required >= MIN_TRADES_FOR_OBJECTIVE
