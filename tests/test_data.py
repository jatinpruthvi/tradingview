import numpy as np
import pandas as pd
import pytest

from tvlab.data.base import (DataError, align_to_execution, bars_per_year,
                             detect_timeframe, resample, validate_ohlcv)
from tvlab.data.store import DataStore
from tvlab.data.synthetic import SynthSpec, basket, generate


def test_ohlcv_envelope_holds():
    for sym, df in basket(["BTC/USDT", "SPY", "EURUSD"], n_bars=400, timeframe="1d").items():
        assert (df.high >= df[["open", "close"]].max(axis=1) - 1e-9).all(), sym
        assert (df.low <= df[["open", "close"]].min(axis=1) + 1e-9).all(), sym
        assert (df.high >= df.low).all(), sym
        assert (df[["open", "high", "low", "close"]] > 0).all().all(), sym


def test_validate_rejects_empty_and_bad():
    with pytest.raises(DataError):
        validate_ohlcv(pd.DataFrame(), "x")
    bad = pd.DataFrame({"open": [1, 2], "high": [0.5, 1], "low": [1, 2], "close": [1, 2],
                        "volume": [1, 1]},
                       index=pd.date_range("2022-01-01", periods=2, tz="UTC"))
    with pytest.raises(DataError):
        validate_ohlcv(bad, "x", fix=False)
    # fix=True repairs the envelope instead of raising
    fixed = validate_ohlcv(bad, "x", fix=True)
    assert (fixed.high >= fixed[["open", "close"]].max(axis=1)).all()


def test_validate_drops_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2022-01-01", "2022-01-01", "2022-01-02"], tz="UTC")
    df = pd.DataFrame({"open": [1, 1, 1], "high": [2, 2, 2], "low": [.5, .5, .5],
                       "close": [1.5, 1.5, 1.5], "volume": [1, 1, 1]}, index=idx)
    out = validate_ohlcv(df, "x")
    assert len(out) == 2
    assert not out.index.duplicated().any()


def test_metadata_columns_survive_validation():
    df = basket(["BTC/USDT"], n_bars=300, timeframe="1d")["BTC/USDT"]
    assert "_regime" in df.columns
    assert set(df["_regime"].unique()) <= {"bull", "bear", "chop"}


def test_resample_is_ohlc_correct():
    df = basket(["BTC/USDT"], n_bars=400, timeframe="1h")["BTC/USDT"]
    d = resample(df, "4h")
    assert len(d) < len(df)
    # the daily high must be the max of the constituent highs
    assert d["high"].max() == pytest.approx(df["high"].max(), rel=1e-9)
    assert d["volume"].sum() == pytest.approx(df["volume"].sum(), rel=1e-6)


def test_bias_alignment_does_not_leak_the_in_progress_bar():
    df = basket(["BTC/USDT"], n_bars=600, timeframe="4h")["BTC/USDT"]
    bias = resample(df, "1d")
    aligned = align_to_execution(bias, df)
    assert len(aligned) == len(df)
    # every execution bar sees a bias close that existed at or before its timestamp
    for ts in aligned.index[::37]:
        row = aligned.loc[ts]
        if pd.isna(row["close"]):
            continue
        prior = bias.loc[:ts, "close"].dropna()
        assert len(prior) and float(row["close"]) == pytest.approx(float(prior.iloc[-1]))


def test_detect_timeframe():
    for tf in ("5m", "1h", "4h", "1d"):
        df = basket(["BTC/USDT"], n_bars=200, timeframe=tf)["BTC/USDT"]
        assert detect_timeframe(df) == tf


def test_bars_per_year_distinguishes_crypto_from_equities():
    assert bars_per_year("1d", "crypto") == 365.0
    assert bars_per_year("1d", "us") == 252.0
    assert bars_per_year("4h", "crypto") > bars_per_year("4h", "us")


def test_generation_is_deterministic():
    a = generate(SynthSpec(seed=5), n_bars=200, timeframe="1d")
    b = generate(SynthSpec(seed=5), n_bars=200, timeframe="1d")
    pd.testing.assert_frame_equal(a[["open", "high", "low", "close"]],
                                  b[["open", "high", "low", "close"]])
    c = generate(SynthSpec(seed=6), n_bars=200, timeframe="1d")
    assert not np.allclose(a["close"], c["close"])


def test_realistic_volatility_by_asset_class():
    """FX must not look like crypto. Getting this wrong invalidates every test."""
    cases = {"BTC/USDT": (0.30, 1.50), "SPY": (0.10, 0.60), "EURUSD": (0.03, 0.30)}
    for sym, (lo, hi) in cases.items():
        df = basket([sym], n_bars=2000, timeframe="1d")[sym]
        r = np.log(df["close"]).diff().dropna()
        ann = r.std() * np.sqrt(252 if "/" not in sym else 365)
        assert lo <= ann <= hi, f"{sym} annualised vol {ann:.3f} outside [{lo},{hi}]"


def test_injected_sweeps_are_recorded():
    df = basket(["BTC/USDT"], n_bars=1500, timeframe="4h", inject_sweeps=10)["BTC/USDT"]
    ev = df.attrs["injected_sweeps"]
    assert len(ev) == 10
    assert all(e["side"] in ("bullish", "bearish") for e in ev)
    assert all(pd.Timestamp(e["timestamp"]) in df.index for e in ev)


# ------------------------------------------------- multi-timeframe consistency
def test_bias_frame_is_the_same_path_as_the_execution_frame(cfg, store):
    """The most damaging bug found here.

    The synthetic generator draws an INDEPENDENT path per (symbol, timeframe), so
    a natively fetched daily series is not a resample of the 4h series - the two
    were uncorrelated (r = -0.64). Gating entries on an unrelated series makes
    every multi-timeframe result meaningless while looking completely normal.
    """
    ex = store.ohlcv("BTC/USDT", cfg.timeframe.execution)
    bi = store.bias_frame("BTC/USDT")
    assert bi is not None
    ref = resample(ex, cfg.timeframe.bias)["close"]
    joined = pd.DataFrame({"bias": bi["close"], "ref": ref}).dropna()
    assert len(joined) > 50
    drift = ((joined["bias"] - joined["ref"]).abs() / joined["ref"]).max()
    assert drift < 1e-9, f"bias frame is a different series (drift {drift:.2%})"


def test_bias_frame_never_extends_past_the_execution_window(cfg, store):
    """A bias bar dated after the last execution bar is future information."""
    ex = store.ohlcv("ETH/USDT", cfg.timeframe.execution)
    bi = store.bias_frame("ETH/USDT")
    assert bi.index.max() <= ex.index.max()


def test_series_drift_detects_an_unrelated_series(store):
    from tvlab.data.store import _MAX_BIAS_DRIFT, _series_drift

    ex = store.ohlcv("BTC/USDT", "4h")
    same = resample(ex, "1d")
    assert _series_drift(same, ex, "1d") == pytest.approx(0.0, abs=1e-12)

    # an independent daily draw must be rejected
    other = basket(["BTC/USDT"], n_bars=700, timeframe="1d")["BTC/USDT"]
    d = _series_drift(other, ex, "1d")
    assert d is not None and d > _MAX_BIAS_DRIFT


def test_series_drift_returns_none_without_an_overlap(store):
    from tvlab.data.store import _series_drift

    ex = store.ohlcv("BTC/USDT", "4h")
    far = ex.copy()
    far.index = far.index + pd.Timedelta(days=3650)
    assert _series_drift(resample(far, "1d"), ex, "1d") is None


def test_walk_forward_bias_slicing_is_by_time_not_by_index(cfg, store):
    """Regression: ``bias.reindex(exec_idx)`` matched daily stamps against 4h
    stamps and returned an almost all-NaN frame, so walk-forward windows ran
    with no trend filter at all - a different strategy from the full run."""
    from tvlab.backtest.walkforward import _slice_bias

    ex = store.ohlcv("BTC/USDT", cfg.timeframe.execution)
    bi = store.bias_frame("BTC/USDT")
    win = ex.index[:600]

    naive = bi.reindex(win)
    assert naive["close"].notna().sum() < len(win) * 0.5, \
        "expected the naive reindex to be mostly NaN - if this passes, recheck the test"

    sliced = _slice_bias(bi, win)
    assert len(sliced) > 0
    assert sliced["close"].notna().all()
    assert sliced.index.max() <= win[-1], "window bias must not reach past the window"
    # warmup history before the window start must be preserved for the EMA/ADX
    assert sliced.index.min() <= win[0]


# --------------------------------------------------------- market factor / bpy
def test_bars_per_year_is_market_specific():
    """Regression: every non-crypto market was annualised with a 6.5h US equity
    session, so 4h forex got 409 bars/year instead of 1560 - understating
    annualised vol and Sharpe by ~1.95x."""
    from tvlab.data.base import bars_per_year

    assert bars_per_year("4h", "crypto") == 2190.0
    assert bars_per_year("4h", "forex") == 1560.0
    assert bars_per_year("4h", "us_equities") == 409.5
    assert bars_per_year("4h", "indian_equities") == pytest.approx(393.75)
    # 24h markets must annualise far above session-bound ones at the same bar size
    assert bars_per_year("4h", "forex") > 3 * bars_per_year("4h", "us_equities")


def test_market_name_aliases_resolve():
    from tvlab.data.base import bars_per_year, normalise_market

    assert normalise_market("us_equities") == "us"
    assert normalise_market("indian_equities") == "india"
    assert normalise_market("FX") == "forex"
    assert normalise_market(None) == "crypto"
    # long and short spellings must agree, or the same run annualises differently
    # depending on which name the caller happened to pass
    assert bars_per_year("1h", "us") == bars_per_year("1h", "us_equities")
    assert bars_per_year("1h", "forex") == bars_per_year("1h", "fx")


def test_basket_has_realistic_cross_asset_correlation():
    """Regression: each symbol was an independent draw, so pairwise correlation
    was ~0 and the portfolio's max_correlated_positions cap could never bind.
    A book of ten alt pairs is one bet sized ten times, and a backtest that
    cannot see that reports a survivable drawdown for an unsurvivable book."""
    b = basket(["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"], n_bars=1500, timeframe="4h")
    r = pd.DataFrame({k: np.log(v["close"]).diff() for k, v in b.items()}).dropna()
    corr = r.corr().to_numpy()
    off = corr[np.triu_indices(len(b), 1)]
    assert off.mean() > 0.4, f"crypto alts should be strongly correlated, got {off.mean():.3f}"


def test_forex_is_less_correlated_than_crypto():
    c = basket(["BTC/USDT", "ETH/USDT", "SOL/USDT"], n_bars=1500, timeframe="4h")
    f = basket(["EURUSD", "GBPUSD", "USDJPY"], n_bars=1500, timeframe="4h")

    def mean_corr(b):
        r = pd.DataFrame({k: np.log(v["close"]).diff() for k, v in b.items()}).dropna()
        m = r.corr().to_numpy()
        return m[np.triu_indices(len(b), 1)].mean()

    assert mean_corr(f) < mean_corr(c)
    assert mean_corr(f) < 0.35, "FX majors should be only weakly correlated"


def test_market_factor_preserves_marginal_volatility():
    """The single-factor blend must change correlation, not volatility.

    z' = beta*F + sqrt(1-beta^2)*z has unit variance when F and z are both
    unit-variance and independent, so the symbol's own vol profile survives.
    """
    from tvlab.data.synthetic import SynthSpec, generate, market_factor

    f = market_factor(1501, 20261123, "4h")
    vols = {}
    for beta in (0.0, 0.15, 0.85):
        # session hours must be the FX ones (24h), or the session filter cuts
        # the intraday bars this test is trying to measure
        spec = SynthSpec(symbol="EURUSD", seed=999, market="forex",
                         vol_scale=0.12, drift_scale=0.15, market_beta=beta,
                         session_start_hour=0, session_end_hour=23)
        df = generate(spec, n_bars=1500, timeframe="4h", market_factor_arr=f)
        vols[beta] = np.log(df["close"]).diff().dropna().std()
    assert max(vols.values()) / min(vols.values()) < 1.05, vols


def test_market_factor_is_reproducible_across_calls():
    """It must not be seeded from the builtin hash(), which is salted per process."""
    from tvlab.data.synthetic import market_factor

    a = market_factor(500, 20261123, "4h")
    b = market_factor(500, 20261123, "4h")
    assert np.allclose(a, b)
    assert not np.allclose(a, market_factor(500, 20261124, "4h"))
    assert a.std() == pytest.approx(1.0, rel=1e-9)


def test_basket_is_identical_across_two_calls():
    a = basket(["BTC/USDT", "EURUSD"], n_bars=400, timeframe="4h")
    b = basket(["BTC/USDT", "EURUSD"], n_bars=400, timeframe="4h")
    for k in a:
        pd.testing.assert_frame_equal(a[k][["open", "high", "low", "close"]],
                                      b[k][["open", "high", "low", "close"]])


def test_cache_key_includes_the_schema_version(cfg):
    """Regression: the cache key covered the request but not the code that
    produced it. After the generator gained a shared market factor, previously
    cached frames were still independent per-symbol paths, so a 1500-bar run
    measured ~0.00 cross-asset correlation while an 1800-bar run measured ~0.67
    - same config, same code, two different answers, nothing to say which was
    stale."""
    from tvlab.data.store import CACHE_SCHEMA_VERSION

    st = DataStore(cfg)
    k = st._cache_key("BTC/USDT", "4h", None, None, 1500, "synthetic")
    # the version is hashed into the digest, so it is not literally in the key;
    # what matters is that changing it invalidates the entry
    assert st._cache_key("BTC/USDT", "4h", None, None, 1500, "synthetic") == k, \
        "the cache key must be deterministic"

    import tvlab.data.store as mod
    old = mod.CACHE_SCHEMA_VERSION
    try:
        mod.CACHE_SCHEMA_VERSION = "vTEST"
        bumped = st._cache_key("BTC/USDT", "4h", None, None, 1500, "synthetic")
        assert bumped != k, "bumping the schema version must invalidate cached frames"
    finally:
        mod.CACHE_SCHEMA_VERSION = old
    assert CACHE_SCHEMA_VERSION == old


def test_store_matches_basket_correlation(store):
    """The store must serve the same correlated paths as basket() directly."""
    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    direct = basket(syms, n_bars=1500, timeframe="4h")
    via_store = {s: store.ohlcv(s, "4h") for s in syms}

    def mean_corr(b):
        r = pd.DataFrame({k: np.log(v["close"]).diff() for k, v in b.items()}).dropna()
        m = r.corr().to_numpy()
        return m[np.triu_indices(len(b), 1)].mean()

    assert mean_corr(via_store) == pytest.approx(mean_corr(direct), abs=0.02)
    assert mean_corr(via_store) > 0.4
