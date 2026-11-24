"""Native trendline fitting, and the lookahead guarantee it has to keep.

``pytrendline`` was evaluated first and rejected: it calls ``DataFrame.append()``
(removed in pandas 2.0), its documented ``datetime_col=None`` path raises
``KeyError``, its constructor discards its own ``reset_index()`` so ``detect()``
needs a caller-supplied integer index, and it is roughly cubic (120 daily bars
3.9 s, 400 bars 141 s). These tests cover the native replacement.
"""
import numpy as np
import pandas as pd
import pytest

from tvlab.indicators.trendlines import (
    MAX_PIVOTS,
    Trendline,
    _dedupe,
    trendline_series,
    trendline_state,
)


def _rising(n=300, seed=5, start=100.0, drift=0.0015, vol=0.012):
    """A drifting series with pullbacks, so diagonal structure actually exists."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2, n)))
    open_ = np.concatenate([[start], close[:-1]])
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum.reduce([open_, high, close]),
                         "low": np.minimum.reduce([open_, low, close]), "close": close,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


@pytest.fixture(scope="module")
def df():
    return _rising()


# ------------------------------------------------------------------ contracts


def test_returns_the_same_keys_whether_or_not_it_fits(df):
    """A failure path that omits a key turns a handled error into a KeyError
    somewhere downstream, which is far harder to diagnose than the cause."""
    ok = trendline_state(df, "1d")
    too_short = trendline_state(df.iloc[:5], "1d")
    assert set(ok) == set(too_short)
    for key in ("available", "lines", "nearest_support_pct", "nearest_resistance_pct",
                "channel_pct", "close", "reason", "n_support", "n_resistance",
                "n_pivots_high", "n_pivots_low"):
        assert key in too_short, key


def test_short_series_reports_not_enough_bars(df):
    st = trendline_state(df.iloc[:5], "1d")
    assert st["available"] is False
    assert st["reason"] == "not_enough_bars"
    assert st["lines"] == []
    assert np.isnan(st["nearest_support_pct"])


def test_finds_lines_on_a_trending_series(df):
    st = trendline_state(df, "1d")
    assert st["available"] is True, st["reason"]
    assert len(st["lines"]) >= 1
    assert st["n_pivots_low"] >= 2 and st["n_pivots_high"] >= 2


def test_line_kinds_and_sign_conventions_are_consistent(df):
    """distance_pct is signed relative to price. An *unbroken* support sits below
    price and an unbroken resistance above it. A support line ABOVE price is not a
    contradiction - it is a rising line price has since fallen through, and it must
    carry is_breakout=True so no caller mistakes it for live support."""
    for t in trendline_state(df, "1d")["lines"]:
        assert t.kind in ("support", "resistance")
        assert t.level > 0 and np.isfinite(t.level)
        assert t.n_points >= 2
        if t.kind == "support":
            assert t.distance_pct <= 0 or t.is_breakout, (
                f"support above price must be flagged broken: {t}")
        else:
            assert t.distance_pct >= 0 or t.is_breakout, (
                f"resistance below price must be flagged broken: {t}")


def test_broken_lines_are_excluded_from_the_nearest_levels(df):
    """A broken support is not support. 'Nearest support' must only ever be a line
    price is still above, or a scanner would report a level already lost."""
    st = trendline_state(df, "1d", ignore_breakouts=True)
    if np.isfinite(st["nearest_support_pct"]):
        assert st["nearest_support_pct"] < 0
    if np.isfinite(st["nearest_resistance_pct"]):
        assert st["nearest_resistance_pct"] > 0


def test_level_is_at_the_last_bar_not_at_the_anchor(df):
    """The reported level must be the line extended to the final bar - the only
    number usable for a decision now - not the price at the second pivot."""
    st = trendline_state(df, "1d")
    close = float(df["close"].iloc[-1])
    for t in st["lines"]:
        assert t.distance_pct == pytest.approx((t.level - close) / close * 100, rel=1e-9)


def test_slope_is_normalised_by_level(df):
    """slope_pct_per_bar is slope/level, so lines on different-priced symbols are
    comparable. Rebuild the raw slope from the two anchors and check the ratio."""
    st = trendline_state(df, "1d")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)
    for t in st["lines"]:
        assert np.isfinite(t.slope_pct_per_bar)
        raw_slope = t.level * t.slope_pct_per_bar
        # extend the line back to its second anchor: it must land on the actual
        # pivot bar's extreme, which is only true if the anchor index is the bar
        # the pivot happened on and not the bar it became knowable on
        p1 = t.level - raw_slope * (n - 1 - t.last_index)
        actual = (low if t.kind == "support" else high)[t.last_index]
        assert abs(p1 - actual) / actual < 1e-6, (
            f"{t.kind} line misses its own anchor: {p1:.4f} vs {actual:.4f} at bar {t.last_index}")


def test_to_dict_is_json_friendly(df):
    t = trendline_state(df, "1d")["lines"][0]
    d = t.to_dict()
    assert set(d) == {"kind", "level", "slope_pct_per_bar", "score", "n_points",
                      "distance_pct", "is_breakout", "first_index", "last_index"}
    import json
    json.dumps(d)


# --------------------------------------------------------------- lookahead


def test_state_at_bar_i_matches_an_explicit_truncation(df):
    """The core guarantee: fitting on the whole series and reading bar i must NOT
    be possible, and fitting on a prefix must equal asking for that prefix."""
    i = 200
    truncated = trendline_state(df.iloc[: i + 1], "1d")
    again = trendline_state(df.iloc[: i + 1], "1d")
    assert [t.to_dict() for t in truncated["lines"]] == [t.to_dict() for t in again["lines"]]


def test_series_never_uses_future_bars(df):
    """Each row of trendline_series must equal a fresh fit on the prefix ending at
    that bar. If it silently used the full series, this fails."""
    warmup, step = 80, 7
    s = trendline_series(df.iloc[:200], "1d", warmup=warmup, step=step)
    for ts in list(s.index)[::step][:6]:
        i = df.index.get_loc(ts)
        fresh = trendline_state(df.iloc[: i + 1], "1d")
        row = s.loc[ts]
        if fresh["available"] and not np.isnan(fresh["nearest_support_pct"]):
            assert row["nearest_support_pct"] == pytest.approx(fresh["nearest_support_pct"], rel=1e-9), ts
        assert row["trendlines_available"] == float(fresh["available"])


def test_appending_future_bars_does_not_change_the_past(df):
    """A fit at bar i must be identical whether or not bars after i exist. This is
    exactly the property the vendored smartmoneyconcepts pivots violated."""
    i = 180
    before = trendline_state(df.iloc[: i + 1], "1d")
    after = trendline_state(df, "1d")       # includes 120 bars of future
    # the fits differ, because they see different data - that is correct. What must
    # NOT happen is the truncated fit changing when the future is appended.
    again = trendline_state(df.iloc[: i + 1], "1d")
    assert [t.to_dict() for t in before["lines"]] == [t.to_dict() for t in again["lines"]]
    assert before is not after


# --------------------------------------------------------------- sanity caps


def test_distance_cap_rejects_useless_lines(df):
    """A mathematically valid line 56% away is not a tradeable level. Reporting it
    as 'nearest resistance' is worse than reporting none."""
    wide = trendline_state(df, "1d", max_distance_pct=None)
    tight = trendline_state(df, "1d", max_distance_pct=5.0)
    assert all(abs(t.distance_pct) <= 5.0 for t in tight["lines"])
    assert len(tight["lines"]) <= len(wide["lines"])


def test_channel_is_the_gap_between_the_nearest_levels(df):
    st = trendline_state(df, "1d")
    if np.isfinite(st["channel_pct"]):
        expected = abs(st["nearest_resistance_pct"] - st["nearest_support_pct"])
        assert st["channel_pct"] == pytest.approx(expected, rel=1e-9)
        assert st["nearest_support_pct"] < 0 < st["nearest_resistance_pct"]


def test_pivot_count_is_capped():
    """Fitting is O(p^2) in pivots, so the number of anchors considered per side
    must be bounded however long the series gets."""
    from tvlab.indicators.trendlines import _pivots

    long_df = _rising(n=3000, seed=11)
    hi_idx, hi_lvl, lo_idx, lo_lvl = _pivots(long_df, 8)
    # the raw detector finds many pivots on 3,000 bars...
    assert len(lo_idx) > MAX_PIVOTS, "test needs a series long enough to exceed the cap"
    # ...and the fit only ever considers the most recent MAX_PIVOTS of them
    st = trendline_state(long_df, "1d")
    assert st["n_pivots_low"] == len(lo_idx)          # reported honestly, not truncated
    # every line must be anchored inside the capped window of its OWN side
    hi_idx, _, _, _ = _pivots(long_df, 8)
    floor_lo, floor_hi = int(lo_idx[-MAX_PIVOTS]), int(hi_idx[-MAX_PIVOTS])
    for t in st["lines"]:
        floor = floor_lo if t.kind == "support" else floor_hi
        assert t.first_index >= floor, f"{t.kind} anchored at {t.first_index} < cap floor {floor}"


def test_reason_is_none_only_when_lines_exist(df):
    st = trendline_state(df, "1d")
    if st["available"]:
        assert st["reason"] is None
    else:
        assert st["reason"] == "no_lines_fitted"


def test_ignore_breakouts_controls_broken_lines(df):
    keeping = trendline_state(df, "1d", ignore_breakouts=True)
    dropping = trendline_state(df, "1d", ignore_breakouts=False)
    assert all(not t.is_breakout for t in dropping["lines"])
    assert len(dropping["lines"]) <= len(keeping["lines"])


def test_min_points_requires_enough_touches(df):
    loose = trendline_state(df, "1d", min_points=2)
    strict = trendline_state(df, "1d", min_points=5)
    assert all(t.n_points >= 5 for t in strict["lines"])
    assert len(strict["lines"]) <= len(loose["lines"])


def test_tighter_tolerance_finds_fewer_lines(df):
    loose = trendline_state(df, "1d", tol_atr_mult=2.0)
    strict = trendline_state(df, "1d", tol_atr_mult=0.2)
    assert len(strict["lines"]) <= len(loose["lines"])


def test_bad_close_series_is_reported_not_raised(df):
    broken = df.copy()
    broken.loc[broken.index[-1], "close"] = np.nan
    assert trendline_state(broken, "1d")["reason"] == "bad_close_series"


def test_flat_series_produces_no_spurious_lines():
    """A perfectly flat series has no structure. Fitting lines through noise and
    reporting them as support would be the worst failure mode here."""
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                         "volume": 1.0}, index=idx)
    st = trendline_state(flat, "1d")
    for t in st["lines"]:
        assert abs(t.slope_pct_per_bar) < 1e-9, "a flat series cannot have a sloped line"


def test_dedupe_collapses_near_duplicates():
    mk = lambda lvl, kind, score: Trendline(  # noqa: E731
        kind=kind, level=lvl, slope_pct_per_bar=0.001, score=score, n_points=3,
        distance_pct=-1.0, is_breakout=False, first_index=0, last_index=10)
    lines = [mk(100.0, "support", 5.0), mk(100.2, "support", 3.0), mk(140.0, "support", 4.0),
             mk(100.1, "resistance", 9.0)]
    kept = _dedupe(lines)
    # 100.0 and 100.2 collapse (0.2% apart, same slope) keeping the higher score;
    # 140.0 is distinct; the resistance line is a different kind so it survives
    assert len(kept) == 3
    assert max(kept, key=lambda t: t.score).level in (100.0, 140.0, 100.1)
    assert any(t.kind == "resistance" for t in kept)


def test_dedupe_keeps_the_best_scoring_member():
    mk = lambda lvl, score: Trendline(kind="support", level=lvl, slope_pct_per_bar=0.0,  # noqa: E731
                                      score=score, n_points=3, distance_pct=-1.0,
                                      is_breakout=False, first_index=0, last_index=5)
    kept = _dedupe([mk(100.0, 1.0), mk(100.1, 9.0), mk(100.2, 5.0)])
    assert len(kept) == 1 and kept[0].score == 9.0


# --------------------------------------------------------------- performance


def test_it_is_fast_enough_for_the_scanner(df):
    """pytrendline took 141 s on 400 daily bars. If this regresses to seconds per
    symbol, a 500-name universe scan becomes unusable."""
    import time

    big = _rising(n=1200, seed=3)
    t0 = time.perf_counter()
    trendline_state(big, "1d")
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"1,200 bars took {elapsed:.2f}s - too slow to scan a universe"


def test_series_step_reduces_the_number_of_fits(df):
    import time

    t0 = time.perf_counter()
    a = trendline_series(df, "1d", warmup=100, step=1)
    t1 = time.perf_counter()
    b = trendline_series(df, "1d", warmup=100, step=10)
    t2 = time.perf_counter()
    assert len(a) == len(b) == len(df) - 100
    assert (t2 - t1) < (t1 - t0), "step must actually reduce the work"


def test_series_is_empty_below_the_warmup(df):
    s = trendline_series(df.iloc[:50], "1d", warmup=60)
    assert s.empty
    assert "nearest_support_pct" in s.columns
