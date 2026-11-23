"""Cross-validate ``indicators/technical.py`` against Pine Script semantics.

The reference is PyneCore's ``ta`` module, which implements TradingView's built-in
Pine functions, driven by ``scripts/vendor_xval_indicator.py`` over bars this repo
generates itself. That closes the last high-value item in ``VENDOR.md`` and puts
``pynecore-strategies`` / ``pynecore`` to actual use rather than as shelfware.

The findings are the point of the file, so they are asserted rather than merely
recorded:

1. **Non-recursive indicators match from bar zero.** ``sma``, the Bollinger bands
   and ``stdev`` agree to float64 noise. Bollinger matching confirms the ``ddof=0``
   choice: Pine's ``ta.stdev`` defaults to ``biased=True``, i.e. the population
   standard deviation, and ``df.rolling(n).std()`` defaults to ``ddof=1``. Had that
   been left at the pandas default the bands would have been quietly too wide by a
   factor of ``sqrt(n/(n-1))``.

2. **Recursive indicators converge to machine epsilon.** ``atr``, ``rsi``, ``ema``,
   ``rma``, ``macd``, ``supertrend`` and all three ``dmi`` columns differ only
   during warmup, decaying geometrically to ~1e-15 relative error. This confirms
   the ``atr()`` docstring claim that the Wilder method matches ``ta.atr``, and the
   ``adx()`` claim that it matches ``ta.dmi``.

   The ``adx()`` claim needed a separate route: Pine has no ``ta.adx``, so there is
   no function of that name to compare against, and ADX gates the higher-timeframe
   bias in ``SmcTrendStrategy`` via ``bias_adx_min``. It was therefore the one
   indicator sitting in a live decision path with no external oracle. ``ta.dmi``
   returns ``(+DI, -DI, ADX)`` as a triple, which is what TradingView's own
   ``ta.dmi()`` exposes, and tvlab uses a single period for both the DI length and
   the ADX smoothing - so the mapping is ``dmi(p, p)``, matching TradingView's
   default ``(14, 14)``.

   Its warmup is the largest of anything validated here (+DI 28%, -DI 46% relative
   over the first hundred bars): tvlab's ``true_range`` fills bar zero with
   ``high - low`` where ``ta.dmi``'s internal ``tr()`` is na there, and that seed is
   then smoothed by two chained RMAs. Because ``bias_adx_min: 22.0`` is a
   *threshold*, reading ADX inside that window compares a materially different
   number against the same cutoff.

3. **The warmup difference is real and bounded.** Pine seeds a recursive indicator
   from its first bar; pandas ``ewm(adjust=False, min_periods=n)`` also recurses
   from bar zero but publishes NaN until ``n``. RSI disagrees by up to ~27%
   relative in the first hundred bars and is indistinguishable afterwards. Any
   comparison of these indicators must therefore skip warmup, and a backtest that
   trades the first bars of a short sample is comparing different numbers.

4. **``st_direction`` has the opposite sign to Pine's ``direction``.** Pine's own
   built-in plots ``direction < 0`` as "Up Direction", so Pine is ``+1 = downtrend``
   while tvlab is ``+1 = uptrend``. The *band* matches Pine exactly. Nothing in
   ``src/`` reads ``st_direction`` today, so this is a porting trap rather than a
   live bug - but 449 Pine strategies sit in ``vendor/`` as reference material, and
   copying one across would invert every signal.

Skips cleanly when PyneCore is not installed: it is an optional dependency, like
the rest of the vendored surface.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tvlab.indicators import technical as t

REPO_ROOT = Path(__file__).resolve().parents[1]
XVAL_SCRIPT = REPO_ROOT / "scripts" / "vendor_xval_indicator.py"

N_BARS = 600
SEED = 7

def _pynecore_importable() -> bool:
    try:
        import pynecore.standalone  # noqa: F401
        return True
    except Exception:
        return False


def _pyne_cli() -> str | None:
    """Locate the `pyne` CLI, needed for the CSV -> OHLCV conversion step.

    ``shutil.which`` is not enough: under pytest the venv's bin directory is often
    not on PATH even though the interpreter running the tests came from it, so the
    console script sits right next to ``sys.executable`` and which() returns None.
    Treating that as "PyneCore unavailable" silently skipped all 21 tests.
    """
    found = shutil.which("pyne")
    if found:
        return found
    # Do NOT .resolve() here: a venv's python is a symlink to the system
    # interpreter, so resolve() lands in /usr/bin, which has no `pyne`, and the
    # console script sitting beside the venv interpreter is missed.
    beside = Path(sys.executable).parent / ("pyne.exe" if os.name == "nt" else "pyne")
    return str(beside) if beside.exists() else None


pytestmark = pytest.mark.skipif(
    not (_pynecore_importable() and _pyne_cli()),
    reason="PyneCore is not installed (optional dependency; pip install pynesys-pynecore[cli])",
)


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> pd.DataFrame:
    """Run the Pine reference script over bars this repo generated.

    Everything happens in a temp directory: PyneCore's standalone runner writes its
    plot CSV next to the script and calls ``sys.exit`` on failure, so it is driven
    in a subprocess with a copied script rather than imported.
    """
    if not XVAL_SCRIPT.exists():
        pytest.skip(f"reference script missing: {XVAL_SCRIPT}")

    work = tmp_path_factory.mktemp("pynecore_xval")
    py = Path(sys.executable)

    from tvlab.data.synthetic import SynthSpec, generate

    bars = generate(SynthSpec(seed=SEED), n_bars=N_BARS, timeframe="4h", start="2022-01-01")
    bars = bars[["open", "high", "low", "close", "volume"]].astype(float)
    csv = work / "BTC_USDT_240.csv"
    out = bars.reset_index()
    out.columns = ["timestamp"] + list(bars.columns)
    out.to_csv(csv, index=False)

    # CSV -> PyneCore OHLCV (+ the syminfo .toml the runner requires alongside it).
    #
    # `pyne` wants a working directory and, if it does not find one, *prompts* to
    # create it. Under pytest stdin is closed, so the prompt aborts and every test
    # skips with "Aborted." - which reads like a missing dependency rather than an
    # interactive prompt. Passing -w and PYNE_WORK_DIR makes it non-interactive.
    wd = work / "wd"
    wd.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYNE_WORK_DIR"] = str(wd)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    r = subprocess.run([_pyne_cli(), "-w", str(wd), "data", "convert-from", str(csv),
                        "-p", "tvlab", "-s", "BTC/USDT"],
                       capture_output=True, text=True, cwd=str(work), timeout=600,
                       env=env, stdin=subprocess.DEVNULL)
    ohlcv = work / "BTC_USDT_240.ohlcv"
    if not ohlcv.exists():
        pytest.skip(f"pyne data convert-from failed (rc={r.returncode}): "
                    f"{(r.stderr or r.stdout)[-400:]}")

    script = work / XVAL_SCRIPT.name
    shutil.copy2(XVAL_SCRIPT, script)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([str(py), str(script), str(ohlcv)], capture_output=True,
                       text=True, cwd=str(work), timeout=600, env=env,
                       stdin=subprocess.DEVNULL)
    plot = work / f"{XVAL_SCRIPT.stem}.csv"
    if not plot.exists():
        pytest.skip(f"Pine reference script failed: {(r.stderr or r.stdout)[-400:]}")

    ref = pd.read_csv(plot, parse_dates=["time"]).set_index("time")
    assert len(ref) == N_BARS, f"expected {N_BARS} plotted bars, got {len(ref)}"

    # Both sides must consume identical bars: use the OHLC PyneCore plotted back,
    # not the frame we wrote, so a conversion rounding difference cannot masquerade
    # as an indicator difference.
    ref.attrs["bars"] = ref[["open", "high", "low", "close", "volume"]].astype(float)
    return ref


@pytest.fixture(scope="module")
def bars(reference) -> pd.DataFrame:
    return reference.attrs["bars"]


# --------------------------------------------------------------- helpers


def _rel_err(a: pd.Series, b: pd.Series, skip: int = 0) -> pd.Series:
    a, b = a.astype(float), b.astype(float)
    both = a.notna() & b.notna()
    if skip:
        both &= pd.Series(False, index=a.index)
        both.iloc[skip:] = True
    rel = (a - b).abs() / a.abs().replace(0.0, np.nan)
    return rel.where(both)


def _assert_converges(ref, mine, name, tail=200, tol=1e-9):
    rel = _rel_err(ref, mine)
    assert rel.notna().any(), f"{name}: no overlapping bars - warmup mismatch"
    worst = float(rel.iloc[-tail:].max())
    assert worst <= tol, f"{name} did not converge: max relative error {worst:.3g} in last {tail} bars"


# ------------------------------------------- 1. non-recursive: exact from bar 0


@pytest.mark.parametrize("col,fn", [
    ("sma_20", lambda df: t.sma(df["close"], 20)),
    ("bb_mid_20", lambda df: t.bollinger(df, 20, 2.0)["bb_mid"]),
    ("bb_upper_20", lambda df: t.bollinger(df, 20, 2.0)["bb_upper"]),
    ("bb_lower_20", lambda df: t.bollinger(df, 20, 2.0)["bb_lower"]),
    ("stdev_20_biased", lambda df: df["close"].rolling(20, min_periods=20).std(ddof=0)),
])
def test_non_recursive_indicators_match_from_the_first_bar(reference, bars, col, fn):
    """No recursion, no seeding question: these must agree everywhere they are both
    defined, including the very first bar of the warmup."""
    rel = _rel_err(reference[col], fn(bars))
    assert rel.notna().sum() > 100
    assert float(rel.max()) <= 1e-9, f"{col} disagrees: max rel {float(rel.max()):.3g}"


def test_bollinger_uses_population_std_like_pine(bars):
    """The ddof=0 choice, pinned directly: Pine's ta.stdev defaults to biased=True.
    ddof=1 would widen the bands by sqrt(n/(n-1)) - 1.3% at n=20, enough to move a
    band-cross signal without looking like a bug."""
    n = 20
    pop = bars["close"].rolling(n, min_periods=n).std(ddof=0)
    samp = bars["close"].rolling(n, min_periods=n).std(ddof=1)
    assert not np.allclose(pop.dropna(), samp.dropna()), "the two ddof choices must differ"
    assert float((samp / pop).iloc[-1]) == pytest.approx(np.sqrt(n / (n - 1)), rel=1e-9)
    bb = t.bollinger(bars, n, 2.0)
    width_from_pop = (4.0 * pop / bb["bb_mid"]).dropna()
    assert np.allclose(width_from_pop, bb["bb_width"].dropna(), rtol=1e-9)


# ------------------------------------------------ 2. recursive: converge exactly


@pytest.mark.parametrize("col,fn", [
    ("atr_14", lambda df: t.atr(df, 14)),
    ("rsi_14", lambda df: t.rsi(df, 14)),
    ("ema_20", lambda df: t.ema(df["close"], 20)),
    ("rma_14", lambda df: t.rma(df["close"], 14)),
    ("macd_12_26_9", lambda df: t.macd(df, 12, 26, 9)["macd"]),
    ("macd_signal_9", lambda df: t.macd(df, 12, 26, 9)["macd_signal"]),
    ("macd_hist", lambda df: t.macd(df, 12, 26, 9)["macd_hist"]),
    ("supertrend_3_10", lambda df: t.supertrend(df, 10, 3.0)["supertrend"]),
    # ta.dmi returns the triple (+DI, -DI, ADX); tvlab's adx() uses one period for
    # both the DI length and the ADX smoothing, so the mapping is dmi(p, p).
    ("dmi_plus_di_14", lambda df: t.adx(df, 14)["plus_di"]),
    ("dmi_minus_di_14", lambda df: t.adx(df, 14)["minus_di"]),
    ("dmi_adx_14", lambda df: t.adx(df, 14)["adx"]),
])
def test_recursive_indicators_converge_to_pine(reference, bars, col, fn):
    """The steady state must be identical to float64 precision. This is what
    validates the Wilder/RMA claims in the docstrings."""
    _assert_converges(reference[col], fn(bars), col)


def test_adx_matches_ta_dmi_in_steady_state(reference, bars):
    """adx()'s docstring claims it "Matches TradingView's ta.dmi". Pine has no
    ``ta.adx``, so that claim was unverifiable by name and the ADX gate in
    ``SmcTrendStrategy`` (``bias_adx_min``) was the one live decision input with no
    external oracle. ``ta.dmi`` returns the whole triple, which is the function the
    claim is about, and all three columns converge to float64 precision.

    The warmup is not cosmetic here: ``bias_adx_min: 22.0`` is a threshold, and +DI
    disagrees by up to 46% relative in the first hundred bars. A study short enough
    to read ADX inside that window is comparing a different number to the same
    threshold. See ``bars_to_forget_seed`` for how far the window actually reaches.
    """
    mine = t.adx(bars, 14)
    _assert_converges(reference["dmi_adx_14"], mine["adx"], "adx", tol=1e-9)
    _assert_converges(reference["dmi_plus_di_14"], mine["plus_di"], "plus_di", tol=1e-9)
    _assert_converges(reference["dmi_minus_di_14"], mine["minus_di"], "minus_di", tol=1e-9)
    # The DX identity ADX is the RMA of. Checking it holds on tvlab's own columns
    # pins the relationship between the three, not just each against Pine: a
    # +DI/-DI pair that matched Pine but fed the wrong DX would still pass above.
    di_sum = (mine["plus_di"] + mine["minus_di"]).replace(0.0, np.nan)
    dx = 100.0 * (mine["plus_di"] - mine["minus_di"]).abs() / di_sum
    both = dx.notna() & mine["adx"].notna()
    # ADX is rma(dx), so it can only equal dx where dx has been constant; assert
    # the bounded range and that ADX tracks dx's own smoothing instead.
    adx = mine["adx"][both]
    assert float(adx.min()) >= 0.0 and float(adx.max()) <= 100.0, "ADX out of range"
    assert np.allclose(adx, t.rma(dx, 14)[both], rtol=1e-12), \
        "adx is not the RMA of its own DX"


def test_atr_wilder_matches_ta_atr_exactly_in_steady_state(reference, bars):
    """Named separately because atr()'s docstring makes this claim explicitly."""
    _assert_converges(reference["atr_14"], t.atr(bars, 14), "atr(wilder)", tol=1e-12)
    # and the other methods must NOT be what Pine does
    wilder = t.atr(bars, 14, method="wilder")
    as_ema = t.atr(bars, 14, method="ema")
    assert not np.allclose(wilder.dropna(), as_ema.dropna()), \
        "wilder and ema ATR must differ, or the method parameter is inert"


# ------------------------------------------- 3. warmup differs, and that is known


@pytest.mark.parametrize("col,fn", [
    ("atr_14", lambda df: t.atr(df, 14)),
    ("rsi_14", lambda df: t.rsi(df, 14)),
    ("macd_12_26_9", lambda df: t.macd(df, 12, 26, 9)["macd"]),
    # The DI columns carry the largest warmup error of anything cross-validated
    # here (+DI 28%, -DI 46% relative in the first 100 bars), because tvlab's
    # true_range fills bar zero with high-low while ta.dmi's internal tr() is na
    # there, and that seed is smoothed by two chained RMAs.
    ("dmi_plus_di_14", lambda df: t.adx(df, 14)["plus_di"]),
    ("dmi_adx_14", lambda df: t.adx(df, 14)["adx"]),
])
def test_warmup_disagreement_is_real_but_decays(reference, bars, col, fn):
    """Pine seeds from bar zero; pandas publishes NaN until min_periods. The early
    bars genuinely differ, so this documents the magnitude and asserts it decays.
    A test that compared these indicators without skipping warmup would fail, and
    the failure would look like a formula bug."""
    rel = _rel_err(reference[col], fn(bars))
    head = float(rel.iloc[:100].max())
    tail = float(rel.iloc[-200:].max())
    assert head > 1e-6, f"{col}: expected a visible warmup difference, got {head:.3g}"
    assert tail < head / 1e4, f"{col}: warmup difference did not decay ({head:.3g} -> {tail:.3g})"


def bars_to_forget_seed(period: int, initial_rel_err: float, tol: float) -> int:
    """How many bars an RMA needs to forget its seed.

    Wilder's smoothing recurses with alpha = 1/period, so the seed's influence
    after k bars is (1 - 1/period)**k. Solving for the bar at which an initial
    relative error falls below a tolerance gives the warmup a comparison must skip.
    This is why "skip 100 bars" is not a safe default: at period 14 it takes ~260
    bars to turn a 0.27 disagreement into 1e-9.
    """
    import math
    decay = math.log(1.0 - 1.0 / period)
    return int(math.ceil(math.log(tol / initial_rel_err) / decay))


def test_comparing_recursive_indicators_without_warmup_would_fail(reference, bars):
    """The methodological point, made executable: the naive whole-series comparison
    of RSI is visibly wrong, and skipping a *correctly sized* warmup makes it exact.

    Skipping a round 100 bars is not enough and this asserts that too - a test that
    quietly compared the unconverged region would fail intermittently depending on
    the series, which is worse than failing always.
    """
    pine = reference["rsi_14"]
    mine = t.rsi(bars, 14)

    naive = float(_rel_err(pine, mine).max())
    assert naive > 1e-3, "expected the naive comparison to be visibly wrong"

    short = float(_rel_err(pine, mine, skip=100).max())
    assert short < naive / 100, "the difference should be decaying by bar 100"
    assert short > 1e-9, (
        "100 bars is NOT enough warmup for RSI(14); if this passes, the decay is "
        "faster than modelled and bars_to_forget_seed is too conservative"
    )

    need = bars_to_forget_seed(14, naive, 1e-9)
    assert 200 <= need <= 400, f"modelled warmup {need} bars is outside the expected range"
    converged = float(_rel_err(pine, mine, skip=need + 40).max())
    assert converged < 1e-9, f"RSI still {converged:.3g} after {need + 40} bars"


def test_the_warmup_error_decays_monotonically_to_machine_precision(reference, bars):
    """The robust form of the decay claim.

    It is tempting to assert the per-block ratio equals (1 - 1/period)**block_len,
    and that is wrong: these are *relative* errors, so each block's maximum is
    normalised by whatever the indicator happened to be at its worst bar. A block
    where the RMA dips toward zero inflates the ratio, so the observed decay is
    lumpier than the algebra (measured ~0.06 against a predicted ~0.025 for the
    first step). What is true and worth pinning is monotone decay across many
    orders of magnitude, ending at float64 noise.
    """
    pine, mine = reference["rma_14"], t.rma(bars["close"], 14)
    rel = _rel_err(pine, mine)
    blocks = [float(rel.iloc[i:i + 50].max()) for i in range(0, 600, 50)]
    blocks = [b for b in blocks if np.isfinite(b) and b > 0]
    assert len(blocks) >= 6, f"too few measurable blocks: {blocks}"

    # Monotonicity is only a meaningful claim while the error is above float64
    # noise. Once it reaches ~1e-15 the blocks just jitter (measured 1.15e-15 ->
    # 1.18e-15), and asserting decay there fails on rounding, not on the seeding.
    NOISE = 1e-12
    decaying = [b for b in blocks if b > NOISE]
    assert len(decaying) >= 3, f"expected several blocks above the noise floor: {blocks}"
    for prev, cur in zip(decaying, decaying[1:]):
        assert cur < prev, f"warmup error grew ({prev:.3g} -> {cur:.3g}); seeding is not decaying"

    settled = blocks[len(decaying):]
    assert settled and max(settled) <= NOISE * 10, \
        f"tail never settled at machine precision: {settled}"
    total = blocks[0] / max(blocks[-1], 1e-300)
    assert total > 1e9, f"expected decay over many orders of magnitude, got {total:.3g}x"


def test_absolute_error_also_decays_not_just_relative(reference, bars):
    """Relative error can look large where the indicator is near zero, so confirm
    the absolute disagreement collapses too - that is the version that matters for
    a signal threshold like RSI crossing 70."""
    pine, mine = reference["rsi_14"], t.rsi(bars, 14)
    both = pine.notna() & mine.notna()
    absd = (pine - mine).abs().where(both)
    head = float(absd.iloc[:50].max())
    tail = float(absd.iloc[-100:].max())
    assert head > 1.0, f"expected a large early absolute RSI gap, got {head:.3g}"
    assert tail < 1e-9, f"tail absolute RSI gap did not collapse: {tail:.3g}"


# --------------------------------------------- 4. supertrend direction sign trap


def test_supertrend_band_matches_pine_but_direction_sign_is_inverted(reference, bars):
    """The single most portable hazard this file found.

    Pine's built-in plots ``direction < 0`` as "Up Direction", so Pine's convention
    is +1 = downtrend. tvlab's is +1 = uptrend. The band agrees to machine epsilon;
    only the sign of the direction flag is flipped. Porting a Pine strategy that
    branches on ``direction`` would invert every entry.
    """
    st = t.supertrend(bars, 10, 3.0)
    _assert_converges(reference["supertrend_3_10"], st["supertrend"], "supertrend band")

    pine = reference["supertrend_dir_3_10"].astype(float)
    mine = st["st_direction"].astype(float)
    settled = pine.notna() & mine.notna() & (mine != 0)     # skip tvlab's warmup zeros
    assert settled.sum() > 100
    assert int((mine[settled] == -pine[settled]).sum()) == int(settled.sum()), \
        "tvlab's st_direction is expected to be the exact negation of Pine's direction"
    assert int((mine[settled] == pine[settled]).sum()) == 0


def test_tvlab_direction_sign_is_internally_consistent(bars):
    """Given the inversion, tvlab's own convention must at least be coherent:
    +1 means the band is the LOWER band (support under price, i.e. an uptrend)."""
    st = t.supertrend(bars, 10, 3.0)
    up = st["st_direction"] == 1
    valid = up & st["supertrend"].notna()
    assert valid.sum() > 50
    # in an uptrend the plotted line is the lower band, and price sits above it
    assert np.allclose(st.loc[valid, "supertrend"], st.loc[valid, "st_lower"], rtol=1e-9)
    assert (bars.loc[valid, "close"] >= st.loc[valid, "supertrend"]).mean() > 0.9


# ------------------------------------------------------- 5. donchian is deliberate


def test_donchian_shifts_where_pine_highest_does_not(reference, bars):
    """tvlab's Donchian bands are shifted one bar so bar i's band excludes bar i -
    a breakout entry must react to the previous bars' extreme, not its own. Pine's
    ta.highest includes the current bar. Both are correct for their purpose; this
    pins the relationship so the shift cannot silently disappear."""
    hh = reference["hh_20"].astype(float)
    dc = t.donchian(bars, 20)

    unshifted = bars["high"].rolling(20, min_periods=20).max()
    rel = _rel_err(hh, unshifted)
    assert float(rel.max()) <= 1e-9, "ta.highest must equal an unshifted rolling max"

    rel_shift = _rel_err(hh, dc["dc_upper"])
    assert float(rel_shift.max()) > 1e-6, \
        "dc_upper should differ from ta.highest because of the deliberate shift"
    # and the shift is exactly one bar
    assert float(_rel_err(hh.shift(1), dc["dc_upper"]).max()) <= 1e-9


# ------------------------------------------------------------- 6. the bars agree


def test_pynecore_received_the_bars_we_generated(bars, reference):
    """Guard against the round trip doing something *gross* - resampling,
    reordering, a unit change - which would invalidate every comparison above.

    It is not bit-exact and is not asserted to be. PyneCore reports
    ``lossless_prices=True`` for its own binary format, yet prices still move by
    ~2e-9 relative through CSV -> OHLCV -> plot CSV (and unevenly: open and high
    survive while low and close shift). That is far below anything an indicator
    comparison can see, and more importantly it cannot affect the results at all:
    the ``bars`` fixture is the OHLC PyneCore itself plotted back, so both sides of
    every comparison in this file compute on byte-identical numbers by construction.
    """
    from tvlab.data.synthetic import SynthSpec, generate

    original = generate(SynthSpec(seed=SEED), n_bars=N_BARS, timeframe="4h",
                        start="2022-01-01")[["open", "high", "low", "close"]].astype(float)
    assert len(bars) == len(original), "bar count changed - resampled or truncated"
    for col in original.columns:
        rel = float(((bars[col] - original[col]).abs() / original[col].abs()).max())
        assert rel < 1e-6, f"{col} changed grossly in the round trip (max rel {rel:.3g})"
    # timestamps must be the same series, in the same order
    assert (bars.index == original.index).all(), "timestamps reordered or shifted"
    # and the plotted hl column is just high-low: a sanity check on the plot wiring
    hl = reference["hl"].astype(float)
    assert float(_rel_err(hl, bars["high"] - bars["low"]).max()) < 1e-9


def test_both_sides_of_every_comparison_use_the_same_bars(bars, reference):
    """The design guarantee the tolerances above rest on, stated explicitly: the
    bars handed to tvlab's indicators are the bars PyneCore plotted, not the ones
    we generated. If that ever changes, the round-trip precision loss stops being
    irrelevant and the convergence tests start measuring storage error."""
    for col in ("open", "high", "low", "close", "volume"):
        assert np.array_equal(bars[col].to_numpy(), reference[col].astype(float).to_numpy())
