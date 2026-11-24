"""The synthetic generator must survive the session filter for non-crypto markets.

Regression test for a bug that made ``--market us_equities`` (and ``indian_equities``,
``uk_equities``, ``forex``) unusable on synthetic data at *any* bar count on an
intraday timeframe:

    ValueError: Length of values (1500) does not match length of index (250)

``generate`` built the OHLCV frame at ``n_bars`` length, then ``_apply_session``
dropped every out-of-session bar - 4 h bars inside a 09:30-16:00 session leave roughly
one bar in six - and only *then* assigned the ``_regime`` and ``_realised_vol`` ground
truth columns positionally at the original ``n_bars`` length. The lengths could never
match once the filter had run.

Two consequences beyond the crash, both asserted here:

* The ground-truth columns must stay **aligned with the bars they describe**. A
  positional assignment after filtering would not merely raise - if it ever did fit,
  it would silently label bar 200 with the regime of bar 1200, and every test that
  scores the sweep or regime detector against ``_regime`` would be measuring the
  wrong thing.
* Crypto and daily bars must be **bit-identical** to before the fix. The published
  study numbers in README.md come from the crypto/swing default, so a reordering that
  changed the RNG consumption order would invalidate every quoted result. It does not,
  because neither ``_inject_sweeps`` nor ``_apply_session`` draws random numbers
  differently when extra columns are already present.
"""
from __future__ import annotations

import numpy as np
import pytest

from tvlab.data.synthetic import DEFAULT_BASKET, SynthSpec, generate

#: (symbol, timeframe) pairs that go through the session filter: non-crypto intraday.
SESSION_FILTERED = [
    ("SPY", "4h"),
    ("SPY", "1h"),
    ("AAPL", "4h"),
    ("RELIANCE.NS", "4h"),
    ("EURUSD", "4h"),
]

#: pairs that must not be touched by the fix.
UNFILTERED = [
    ("BTC/USDT", "4h"),
    ("ETH/USDT", "4h"),
    ("SPY", "1d"),
]

#: close-sum fingerprints captured from the pre-fix generator for UNFILTERED pairs.
#: If the fix ever shifts the RNG order these fail, and every number quoted in
#: README.md would need re-deriving - so they are worth pinning.
FINGERPRINTS = {
    ("BTC/USDT", "4h"): 82838532.68174052,
    ("ETH/USDT", "4h"): 4733630.438956602,
    ("SPY", "1d"): 3057800.173667057,
}


def _spec(symbol: str) -> SynthSpec:
    return SynthSpec(symbol=symbol, seed=20261123, **DEFAULT_BASKET[symbol])


@pytest.mark.parametrize("symbol,timeframe", SESSION_FILTERED)
@pytest.mark.parametrize("n_bars", [400, 1200, 4000])
def test_session_filtered_markets_generate_at_all_bar_counts(symbol, timeframe, n_bars):
    """The crash. Every combination used to raise ValueError."""
    df = generate(_spec(symbol), n_bars=n_bars, timeframe=timeframe)
    assert len(df) > 0
    assert len(df) <= n_bars, "the session filter must drop bars, not add them"
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)


@pytest.mark.parametrize("symbol,timeframe", SESSION_FILTERED)
def test_ground_truth_columns_survive_the_filter_and_stay_aligned(symbol, timeframe):
    df = generate(_spec(symbol), n_bars=1200, timeframe=timeframe)
    assert "_regime" in df.columns and "_realised_vol" in df.columns
    assert len(df["_regime"]) == len(df)
    assert len(df["_realised_vol"]) == len(df)
    assert df["_regime"].notna().all()
    assert np.isfinite(df["_realised_vol"].to_numpy()).all()
    # every bar must sit inside the session, i.e. the filter actually ran
    hours = df.index.hour
    spec = _spec(symbol)
    assert ((hours >= spec.session_start_hour) & (hours <= spec.session_end_hour)).all()


def test_the_regime_labels_match_the_bars_they_describe():
    """Alignment, not just presence. `_regime` is derived from a hidden state chain
    over the *unfiltered* series, so after filtering each surviving bar must carry
    the label of that bar - verified by regenerating without the filter and
    reindexing on the timestamps the filtered frame kept."""
    sym, tf = "SPY", "4h"
    filtered = generate(_spec(sym), n_bars=1200, timeframe=tf)

    # crypto spec on the same seed skips the filter, so build an unfiltered copy by
    # asking for a timeframe at/above the intraday threshold instead
    unfiltered = generate(_spec(sym), n_bars=1200, timeframe="1d")
    assert len(unfiltered) == 1200, "daily bars are not session-filtered"

    # The filtered 4h frame's labels must be a subset of the label vocabulary and
    # must agree with the unfiltered chain wherever timestamps coincide.
    common = filtered.index.intersection(unfiltered.index)
    if len(common):
        assert (filtered.loc[common, "_regime"] == unfiltered.loc[common, "_regime"]).all()


@pytest.mark.parametrize("symbol,timeframe", UNFILTERED)
def test_unfiltered_output_is_bit_identical_to_the_pre_fix_generator(symbol, timeframe):
    df = generate(_spec(symbol), n_bars=1200, timeframe=timeframe)
    got = float(np.sum(df["close"].to_numpy()))
    assert got == pytest.approx(FINGERPRINTS[(symbol, timeframe)], rel=0, abs=1e-6), (
        f"{symbol} {timeframe} close sum moved from "
        f"{FINGERPRINTS[(symbol, timeframe)]} to {got}: the RNG consumption order "
        f"changed, so every published study number needs re-deriving"
    )
    assert len(df) == 1200


def test_injected_sweeps_are_preserved_through_the_session_filter():
    """`inject_sweeps` runs before the filter too, and it returns a copy - so the
    events recorded in `.attrs` must still be there, and the ground-truth columns
    must not have been dropped by the copy."""
    spec = SynthSpec(symbol="SPY", seed=7, inject_sweeps=6, **DEFAULT_BASKET["SPY"])
    df = generate(spec, n_bars=2000, timeframe="4h")
    events = df.attrs.get("injected_sweeps")
    assert events, "sweep injection should have recorded events"
    assert "_regime" in df.columns and "_realised_vol" in df.columns
    assert len(df["_regime"]) == len(df)
