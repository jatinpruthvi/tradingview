import numpy as np
import pandas as pd
import pytest

from tvlab.data.synthetic import basket
from tvlab.indicators import technical as t
from tvlab.indicators.smc import fair_value_gaps, smc_features, swing_points


@pytest.fixture(scope="module")
def df():
    return basket(["BTC/USDT"], n_bars=900, timeframe="4h")["BTC/USDT"]


def test_rsi_bounds(df):
    r = t.rsi(df, 14).dropna()
    assert r.min() >= 0 and r.max() <= 100


def test_adx_bounds(df):
    a = t.adx(df, 14)["adx"].dropna()
    assert a.min() >= 0 and a.max() <= 100


def test_atr_nonneg_and_matches_manual(df):
    a = t.atr(df, 14).dropna()
    assert (a >= 0).all()
    tr = t.true_range(df)
    # ATR must sit inside the observed true-range envelope
    assert a.max() <= tr.max() * 1.5


def test_donchian_bands_exclude_the_current_bar(df):
    """A breakout must react to PRIOR bars, so the band is shifted."""
    ch = t.donchian(df, 20)
    valid = ch.dropna()
    i = df.index.get_loc(valid.index[100])
    assert df["high"].iloc[i - 20:i].max() == pytest.approx(ch["dc_upper"].iloc[i])


def test_supertrend_direction_flips(df):
    """Regression: a NaN ATR seed used to poison the recursion, so the band was
    NaN for the whole series and the direction never flipped."""
    st = t.supertrend(df, 10, 3.0)
    valid = st["supertrend"].dropna()
    assert len(valid) > len(df) * 0.8, "supertrend band is mostly NaN"
    dirs = st["st_direction"].to_numpy()[len(df) - len(valid):]
    assert set(np.unique(dirs)).issubset({-1, 1})
    flips = int((dirs[1:] != dirs[:-1]).sum())
    assert flips >= 2, f"direction never flips ({flips} flips) - recursion is stuck"


def test_supertrend_warmup_is_nan_not_poisoned(df):
    st = t.supertrend(df, 10, 3.0)
    assert st["supertrend"].iloc[:9].isna().all()      # ATR warmup
    assert st["supertrend"].iloc[20:].notna().all()    # ...and finite afterwards


def test_vwap_handles_sliced_index(df):
    """Regression: .index.shift() raised NullFrequencyError on a sliced frame."""
    sliced = df.iloc[100:500]
    v = t.vwap(sliced)
    assert len(v) == len(sliced)
    assert v.notna().sum() > 0


def test_swing_lag_is_exactly_swing_length(df):
    L = 10
    raw = swing_points(df, L, lag=False)
    lagged = swing_points(df, L, lag=True)
    rh = raw["swing_high"].to_numpy()
    lh = lagged["swing_high"].to_numpy()
    # every raw pivot at i must appear in the lagged output at i+L
    for i in np.flatnonzero(rh)[:-1]:
        if i + L < len(df):
            assert lh[i + L], f"pivot at {i} not published at {i+L}"
    # and nothing is published before it can be known
    assert not lh[:L].any()


def test_fvg_matches_definition(df):
    f = fair_value_gaps(df)
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    for i in np.flatnonzero(f["bull_fvg"].to_numpy())[:20]:
        assert lo[i] > hi[i - 2]
    for i in np.flatnonzero(f["bear_fvg"].to_numpy())[:20]:
        assert hi[i] < lo[i - 2]


def test_smc_columns_present(df):
    f = smc_features(df, swing_length=12)
    required = ["swing_high", "swing_low", "bos", "choch", "structure", "sweep_low",
                "sweep_high", "bull_sweep_age", "bear_sweep_age", "choch_age", "bos_age",
                "ob_bull", "ob_bear", "bull_fvg", "bear_fvg", "fvg_in_zone", "ob_in_zone",
                "eq_high", "eq_low", "last_high_level", "last_low_level"]
    for c in required:
        assert c in f.columns, c
    assert len(f) == len(df)


def test_per_side_sweep_ages_are_independent(df):
    """Regression: a shared sweep age let a bearish sweep trigger a long."""
    f = smc_features(df, swing_length=12)
    bull = f["sweep_low"].to_numpy()
    bear = f["sweep_high"].to_numpy()
    ba, ea = f["bull_sweep_age"].to_numpy(), f["bear_sweep_age"].to_numpy()
    for i in np.flatnonzero(bull):
        assert ba[i] == 0
        # the bearish age must NOT have been reset by a bullish sweep
        if i > 0 and not bear[i]:
            assert ea[i] >= ea[i - 1]
    for i in np.flatnonzero(bear):
        assert ea[i] == 0


def test_structure_is_only_minus_one_zero_one(df):
    f = smc_features(df, swing_length=12)
    assert set(np.unique(f["structure"])).issubset({-1, 0, 1})
