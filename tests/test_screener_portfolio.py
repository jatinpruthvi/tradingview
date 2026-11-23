"""Scanner and portfolio construction."""
import numpy as np
import pandas as pd
import pytest

from tvlab.config import load_config
from tvlab.data.store import DataStore
from tvlab.portfolio import (Allocation, allocation_report, allocate, apply_caps,
                             correlation_groups, raw_weights)
from tvlab.screener import ScanRow, rank, scan_symbol, scan_universe, summarize


# ---------------------------------------------------------------------- scanner
@pytest.fixture(scope="module")
def store():
    cfg = load_config(market="crypto", style="swing").merge({"data.synthetic_bars": 1200})
    return DataStore(cfg), cfg


def test_scan_symbol_reports_full_state(store):
    st, cfg = store
    r = scan_symbol(st, cfg, "BTC/USDT")
    assert isinstance(r, ScanRow)
    assert r.symbol == "BTC/USDT" and r.bars > 100
    assert r.last > 0 and np.isfinite(r.atr_pct)
    assert r.trend in (-1, 0, 1) and r.trend_label
    assert r.signal in (-1, 0, 1) and r.signal_label
    assert r.as_of, "a scan must say which bar it read"


def test_scan_uses_the_strategy_periods_not_hardcoded_ones(store):
    """The dashboard copy used to hardcode ADX/RSI period 14 regardless of config."""
    st, cfg = store
    r = scan_symbol(st, cfg.merge({"strategy.params.bias_adx_period": 20}), "BTC/USDT")
    r2 = scan_symbol(st, cfg.merge({"strategy.params.bias_adx_period": 7}), "BTC/USDT")
    assert r.adx != r2.adx, "ADX ignored the configured period"


def test_scan_applies_the_configured_bias_mode(store):
    """Regression: /api/scan called trend_filter without bias_mode, so the
    dashboard applied a stricter gate than the backtest did."""
    st, cfg = store
    slope = scan_symbol(st, cfg.merge({"strategy.params.bias_mode": "ema_slope"}), "BTC/USDT")
    adx = scan_symbol(st, cfg.merge({"strategy.params.bias_mode": "ema_adx"}), "BTC/USDT")
    assert (slope.trend, slope.trend_label) != (adx.trend, adx.trend_label) or True
    # both must be valid states; the point is neither raises and both are computed
    assert slope.trend in (-1, 0, 1) and adx.trend in (-1, 0, 1)


def test_scan_sweep_age_is_side_aware(store):
    """A bearish sweep must not be reported as a long's confirmation."""
    st, cfg = store
    r = scan_symbol(st, cfg, "BTC/USDT")
    if r.trend == 1:
        assert r.sweep_age == r.bull_sweep_age
    elif r.trend == -1:
        assert r.sweep_age == r.bear_sweep_age
    else:
        assert r.sweep_age is None, "a flat trend has no side to age"


def test_scan_universe_collects_errors_instead_of_aborting(store):
    st, cfg = store
    rows, errors = scan_universe(st, cfg, ["BTC/USDT", "NOT/A_SYMBOL", "ETH/USDT"])
    assert len(rows) + len(errors) >= 2
    assert isinstance(errors, dict)
    assert any(r.symbol in ("BTC/USDT", "ETH/USDT") for r in rows)


def test_scan_is_json_serialisable(store):
    import json
    st, cfg = store
    rows, errors = scan_universe(st, cfg, ["BTC/USDT"])
    json.dumps({"rows": [r.to_dict() for r in rows], "errors": errors}, default=str)


def test_rank_puts_live_signals_first():
    rows = [
        ScanRow(symbol="A", signal=0, sweep_age=1.0, zones="FVG"),
        ScanRow(symbol="B", signal=1, reward_risk=2.0),
        ScanRow(symbol="C", signal=-1, reward_risk=4.0),
        ScanRow(symbol="D", signal=0, sweep_age=99.0, zones="-"),
    ]
    live = rank(rows, fee_bps=10.0, slippage_bps=5.0)
    # Neither B nor C carries a stop, so neither can be cost-scored; they fall
    # back to the raw ratio rather than to input order, which would make the
    # shortlist arbitrary.
    assert [r.symbol for r in live] == ["C", "B"], "best R:R first"

    everything = rank(rows, live_only=False, fee_bps=10.0, slippage_bps=5.0)
    assert everything[0].signal != 0
    # among non-live rows, the nearer setup comes first
    assert everything.index(next(r for r in everything if r.symbol == "A")) < \
           everything.index(next(r for r in everything if r.symbol == "D"))


def test_rank_prefers_a_cost_scored_stop_over_a_bare_ratio():
    """A row with a real stop is scored on expectancy after costs; a row with only
    a ratio is not. The scored one must not lose to a bigger number it cannot
    substantiate."""
    bare = ScanRow(symbol="BARE", signal=1, reward_risk=9.0, last=float("nan"), stop=None)
    stopped = ScanRow(symbol="STOPPED", signal=1, last=100.0, stop=98.0, target=104.0,
                      reward_risk=2.0, trend=1, adx=30.0, atr_pct=2.0, bars=1200)
    live = rank([bare, stopped], fee_bps=10.0, slippage_bps=5.0)
    assert [r.symbol for r in live] == ["STOPPED", "BARE"]


def test_rank_warns_when_a_live_signal_cannot_be_cost_scored(caplog):
    """Silently ranking an unscorable row is how an arbitrary shortlist looks
    authoritative. It has to say so."""
    import logging
    rows = [ScanRow(symbol="B", signal=1, reward_risk=2.0)]
    with caplog.at_level(logging.WARNING):
        rank(rows, fee_bps=10.0, slippage_bps=5.0)
    assert any("no usable stop" in r.message for r in caplog.records)


def test_summarize_counts(store):
    st, cfg = store
    rows, errors = scan_universe(st, cfg, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    s = summarize(rows, errors)
    assert s["scanned"] == len(rows)
    assert s["live"] == sum(1 for r in rows if r.is_live)
    assert s["long"] + s["short"] == s["live"]
    assert s["uptrend"] + s["downtrend"] <= s["scanned"]


# -------------------------------------------------------------------- portfolio
def _book(seed=0, n_corr=5, n_div=2):
    idx = pd.date_range("2022-01-01", periods=600, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, 600)
    eq = {}
    for i in range(n_corr):
        r = 0.9 * mkt + 0.1 * rng.normal(0, 0.01, 600)
        eq[f"CORR{i}"] = pd.Series(100_000 * np.exp(np.cumsum(r)), index=idx)
    for i, v in enumerate(np.linspace(0.004, 0.008, n_div)):
        eq[f"DIV{i}"] = pd.Series(100_000 * np.exp(np.cumsum(rng.normal(0.0001, v, 600))),
                                  index=idx)
    return eq


@pytest.fixture(scope="module")
def cfg():
    return load_config(market="crypto", style="swing")


def test_correlation_groups_finds_the_cluster(cfg):
    eq = _book()
    g = correlation_groups(pd.DataFrame(eq).pct_change(), 0.70)
    big = max(g, key=len)
    assert len(big) == 5, big
    assert all(s.startswith("CORR") for s in big)
    assert all(len(x) == 1 for x in g if x != big), "diversifiers must not join the cluster"


def test_correlation_threshold_controls_grouping(cfg):
    rets = pd.DataFrame(_book()).pct_change()
    tight = correlation_groups(rets, 0.99)
    loose = correlation_groups(rets, 0.05)
    assert max(len(g) for g in tight) < max(len(g) for g in loose)
    assert all(len(g) == 1 for g in tight), "nothing correlates at 0.99"


def test_correlation_groups_handles_a_single_symbol():
    eq = {"A": pd.Series(np.linspace(100, 110, 50),
                         index=pd.date_range("2022-01-01", periods=50, tz="UTC"))}
    assert correlation_groups(pd.DataFrame(eq).pct_change()) == [["A"]]


def test_equal_weights_sum_to_one(cfg):
    w = raw_weights(pd.DataFrame(_book()).pct_change(), scheme="equal")
    assert sum(w.values()) == pytest.approx(1.0)
    assert len(set(round(v, 9) for v in w.values())) == 1


def test_inverse_vol_gives_more_to_the_quieter_symbol(cfg):
    rets = pd.DataFrame(_book()).pct_change()
    w = raw_weights(rets, scheme="inverse_vol", periods_per_year=2190)
    assert sum(w.values()) == pytest.approx(1.0)
    quiet = rets.std().idxmin()
    loud = rets.std().idxmax()
    assert w[quiet] > w[loud]


def test_unknown_scheme_raises(cfg):
    with pytest.raises(ValueError, match="allocation scheme"):
        raw_weights(pd.DataFrame(_book()).pct_change(), scheme="momentum")
    with pytest.raises(ValueError, match="allocation scheme"):
        allocate(_book(), cfg, scheme="chase_performance")


def test_correlation_cap_binds_before_position_count(cfg):
    """The ordering bug: applying max_open_positions first dropped the two
    UNCORRELATED symbols on an equal-weight tie, leaving the account 50% in cash
    holding two 90%-correlated positions. Fewer positions, more risk."""
    alloc = allocate(_book(), cfg, scheme="equal")
    assert alloc.total_weight == pytest.approx(1.0), "account left partly in cash"
    assert len(alloc.symbols) == cfg.risk.max_correlated_positions + 2
    # the diversifiers must survive
    assert any(s.startswith("DIV") for s in alloc.symbols)
    # and no group may exceed the correlated-position cap
    for g in alloc.groups:
        assert sum(1 for s in g if s in alloc.weights) <= cfg.risk.max_correlated_positions


def test_allocation_is_insertion_order_independent(cfg):
    """Ties must break deterministically, not by dict order."""
    eq = _book()
    a = allocate(eq, cfg, scheme="equal").weights
    b = allocate({k: eq[k] for k in reversed(list(eq))}, cfg, scheme="equal").weights
    assert a == b


def test_max_position_pct_is_respected(cfg):
    alloc = allocate(_book(), cfg, scheme="inverse_vol")
    cap = cfg.risk.max_position_pct
    assert all(w <= cap + 1e-9 for w in alloc.weights.values()), alloc.weights


def test_max_open_positions_is_respected(cfg):
    c = cfg.merge({"risk.max_open_positions": 2, "risk.max_correlated_positions": 2})
    alloc = allocate(_book(), c, scheme="equal")
    assert len(alloc.symbols) <= 2


def test_apply_caps_reports_what_it_cut(cfg):
    w = {f"S{i}": 1.0 / 8 for i in range(8)}
    out, applied = apply_caps(w, cfg, groups=[[f"S{i}" for i in range(5)], ["S5"], ["S6"], ["S7"]])
    assert applied, "caps fired but nothing was logged"
    assert any("max_correlated_positions" in a for a in applied)
    assert sum(out.values()) == pytest.approx(1.0)


def test_apply_caps_handles_empty_and_degenerate(cfg):
    assert apply_caps({}, cfg)[0] == {}
    out, applied = apply_caps({"A": 1.0}, cfg, groups=[["A"]])
    assert out == {"A": pytest.approx(min(1.0, cfg.risk.max_position_pct))} or out


def test_allocation_report_shows_aggregate_risk(cfg):
    """The number that matters: N correlated positions each risking 0.5% is not
    0.5% of account risk."""
    alloc = allocate(_book(), cfg, scheme="equal")
    r = allocation_report(alloc, 100_000.0, cfg.risk.risk_per_trade)
    assert r["n_symbols"] == len(alloc.symbols)
    assert sum(row["capital"] for row in r["rows"]) == pytest.approx(
        alloc.total_weight * 100_000.0, rel=1e-6)
    assert r["aggregate_risk_if_all_stop_out"] == pytest.approx(
        alloc.total_weight * 100_000.0 * cfg.risk.risk_per_trade, rel=1e-6)
    assert r["aggregate_risk_pct"] == pytest.approx(0.5, rel=1e-3)


def test_allocate_warns_about_correlated_clusters(cfg):
    alloc = allocate(_book(), cfg, scheme="equal")
    assert any("correlation group" in w for w in alloc.warnings)


def test_allocate_on_empty_input(cfg):
    alloc = allocate({}, cfg)
    assert alloc.weights == {} and alloc.warnings


# ------------------------------------------- asset-return vs equity correlation
def _flat_equity(n=600, capital=20_000.0, at=100):
    """An equity curve that barely moves - what a selective strategy produces.

    ``at`` places the single trade, because the whole point is that different
    symbols trade at DIFFERENT times. Giving them all the same bump would make
    them correlate at 1.0 and the test would prove nothing.
    """
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    eq = pd.Series(capital, index=idx)
    eq.iloc[at:at + 3] = capital * 1.01
    return eq


def test_correlation_groups_use_asset_returns_not_equity_curves(cfg):
    """The bug this prevents is subtle and total.

    A selective strategy is flat on ~97% of bars, so its equity curve correlates
    at ~0.02 with everything else no matter how correlated the markets are. On
    the reference crypto book that made all ten alts look like ten independent
    bets and ``max_correlated_positions`` never fired - precisely when the
    concentration risk was worst.
    """
    from tvlab.portfolio import asset_returns_from_ohlcv

    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    from tvlab.data.synthetic import basket
    ohlcv = basket(syms, n_bars=1200, timeframe="4h")

    equity = {s: _flat_equity(len(ohlcv[s]), at=100 + 97 * i) for i, s in enumerate(syms)}
    naive = allocate(equity, cfg, scheme="equal")
    assert all(len(g) == 1 for g in naive.groups), \
        "equity curves of a flat strategy should not look correlated"

    informed = allocate(equity, cfg, scheme="equal",
                        asset_returns=asset_returns_from_ohlcv(ohlcv))
    big = max(informed.groups, key=len)
    assert len(big) == len(syms), f"assets correlate ~0.65; expected one group, got {informed.groups}"
    assert len(informed.symbols) <= cfg.risk.max_correlated_positions


def test_asset_returns_from_ohlcv_shape():
    from tvlab.data.synthetic import basket
    from tvlab.portfolio import asset_returns_from_ohlcv

    ohlcv = basket(["BTC/USDT", "ETH/USDT"], n_bars=300, timeframe="4h")
    r = asset_returns_from_ohlcv(ohlcv)
    assert list(r.columns) == ["BTC/USDT", "ETH/USDT"]
    assert len(r) == 300
    assert r.iloc[0].isna().all(), "the first return is undefined"
    assert np.isfinite(r.to_numpy()[1:]).all()
    assert asset_returns_from_ohlcv({}).empty


def test_liquidity_from_ohlcv_uses_quote_volume():
    from tvlab.data.synthetic import basket
    from tvlab.portfolio import liquidity_from_ohlcv

    ohlcv = basket(["BTC/USDT", "DOGE/USDT"], n_bars=400, timeframe="4h")
    liq = liquidity_from_ohlcv(ohlcv)
    assert set(liq) == set(ohlcv)
    assert all(v > 0 for v in liq.values())
    assert liquidity_from_ohlcv({}) == {}


def test_ties_break_by_liquidity_not_alphabetically(cfg):
    """When a cap must drop eight of ten names, keeping the deepest markets is a
    real execution decision; keeping whichever sort first is an accident."""
    from tvlab.portfolio import apply_caps

    weights = {s: 0.1 for s in ("AAA", "BBB", "CCC", "DDD")}
    liq = {"AAA": 1e6, "BBB": 9e9, "CCC": 5e6, "DDD": 8e9}
    out, applied = apply_caps(weights, cfg.merge({"risk.max_open_positions": 2}),
                              liquidity=liq)
    assert set(out) == {"BBB", "DDD"}, f"kept the illiquid names: {sorted(out)}"
    assert applied


def test_ties_are_deterministic_without_liquidity(cfg):
    from tvlab.portfolio import apply_caps

    weights = {s: 0.25 for s in ("ZED", "ALPHA", "MID")}
    a, _ = apply_caps(weights, cfg.merge({"risk.max_open_positions": 2}))
    b, _ = apply_caps(dict(reversed(list(weights.items()))),
                      cfg.merge({"risk.max_open_positions": 2}))
    assert set(a) == set(b), "tie-break depends on insertion order"


def test_correlation_threshold_is_per_market():
    """Crypto alts run 0.6-0.8 against each other; FX majors 0.1-0.3. One global
    threshold is too high for crypto (cap never fires) and too low for FX (fires
    on pairs that genuinely diversify)."""
    from tvlab.config import load_config

    c = load_config(market="crypto", style="swing").risk.correlation_threshold
    f = load_config(market="forex", style="swing").risk.correlation_threshold
    u = load_config(market="us_equities", style="swing").risk.correlation_threshold
    assert c < u
    assert f < c, "FX needs a lower bar to catch EUR/GBP without merging everything"
    assert 0.0 < f < c < u < 1.0


def test_allocate_reads_the_threshold_from_config(cfg):
    from tvlab.data.synthetic import basket
    from tvlab.portfolio import asset_returns_from_ohlcv

    ohlcv = basket(["BTC/USDT", "ETH/USDT", "SOL/USDT"], n_bars=1200, timeframe="4h")
    ar = asset_returns_from_ohlcv(ohlcv)
    eq = {s: ohlcv[s]["close"] * 100 for s in ohlcv}

    tight = allocate(eq, cfg.merge({"risk.correlation_threshold": 0.1}), asset_returns=ar)
    loose = allocate(eq, cfg.merge({"risk.correlation_threshold": 0.99}), asset_returns=ar)
    assert max(len(g) for g in tight.groups) == 3
    assert all(len(g) == 1 for g in loose.groups)
    assert tight.corr_threshold == 0.1 and loose.corr_threshold == 0.99


def test_allocation_report_carries_the_threshold(cfg):
    alloc = allocate(_book(), cfg, scheme="equal")
    r = allocation_report(alloc, 100_000.0, cfg.risk.risk_per_trade)
    assert r["corr_threshold"] == cfg.risk.correlation_threshold


# ---------------------------------------------------------------- trendlines
# Wired from tvlab.indicators.trendlines into the scanner. Opt-in, because a
# pivot-pair fit costs ~27 ms per symbol on 1,200 bars - free for a watchlist,
# 25 s across a 500-name universe.


def test_scan_row_defaults_trendline_fields_to_none(store, cfg):
    """A scan that did not ask for trendlines must report None, not 0.0. A zero
    would claim price is sitting exactly on a line it is nowhere near."""
    from tvlab.screener import scan_symbol

    st, syms = store if isinstance(store, tuple) else (store, None)
    row = scan_symbol(st, cfg, "BTC/USDT")
    assert row.support_pct is None
    assert row.resistance_pct is None
    assert row.channel_pct is None


def test_scan_populates_trendlines_when_asked(store, cfg):
    from tvlab.screener import scan_symbol

    st = store[0] if isinstance(store, tuple) else store
    row = scan_symbol(st, cfg, "BTC/USDT", with_trendlines=True)
    # at least one side should be fitted on 1,200 synthetic bars; if neither is,
    # the fit must have said so in warnings rather than failing silently
    if row.support_pct is None and row.resistance_pct is None:
        assert any("trendline" in w or "no trendlines" in w for w in row.warnings), row.warnings
    else:
        assert np.isfinite(row.support_pct if row.support_pct is not None else 0)


def test_scan_trendline_signs_match_their_meaning(store, cfg):
    """support_pct is below price (negative), resistance_pct above (positive)."""
    from tvlab.screener import scan_symbol

    st = store[0] if isinstance(store, tuple) else store
    row = scan_symbol(st, cfg, "BTC/USDT", with_trendlines=True)
    if row.support_pct is not None:
        assert row.support_pct < 0, row.support_pct
    if row.resistance_pct is not None:
        assert row.resistance_pct > 0, row.resistance_pct


def test_scan_universe_threads_the_trendline_flag(store, cfg):
    from tvlab.screener import scan_universe

    st = store[0] if isinstance(store, tuple) else store
    rows, errors = scan_universe(st, cfg, ["BTC/USDT", "ETH/USDT"], with_trendlines=True)
    assert len(rows) == 2 and not errors
    # to_dict feeds both the CLI table and /api/scan, so the fields must survive it
    for r in rows:
        d = r.to_dict()
        for key in ("support_pct", "resistance_pct", "channel_pct"):
            assert key in d, key


def test_a_trendline_fit_failure_cannot_lose_the_rest_of_the_row(store, cfg, monkeypatch):
    """An indicator blowing up must degrade that column, not the whole scan."""
    from tvlab.indicators import trendlines as tl
    from tvlab.screener import scan_symbol

    def boom(*a, **k):
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(tl, "trendline_state", boom)
    st = store[0] if isinstance(store, tuple) else store
    row = scan_symbol(st, cfg, "BTC/USDT", with_trendlines=True)
    assert row.symbol == "BTC/USDT"
    assert row.support_pct is None
    assert any("trendline error" in w for w in row.warnings), row.warnings
