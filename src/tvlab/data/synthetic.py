"""Regime-aware synthetic OHLCV generator.

WHY THIS EXISTS
---------------
Real exchange data needs network access and API keys, so the test suite, the
validation harness and the dashboard demo all need *some* bars. This generator
produces bars with the properties that actually matter for strategy testing:

* Markov regime switching (bull trend / bear trend / chop)
* GARCH(1,1)-style volatility clustering
* Student-t innovations, i.e. fat tails
* volume correlated with absolute returns
* session filtering + overnight gaps for equity/FX markets, 24/7 for crypto
* optional injection of *known* liquidity-sweep-then-reversal events so the
  signal logic can be scored for precision/recall independently of any market

IMPORTANT: profits measured on synthetic data prove the plumbing works, NOT that
a strategy has an edge. A trend-follower "wins" on data generated with trending
regimes by construction. See ``tvlab.validation`` for the checks that do carry
scientific weight (shuffled-null and pattern-recovery).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import RESAMPLE_RULE, bars_per_year, timeframe_seconds, validate_ohlcv

#: per-regime (annualised drift, annualised vol, mean bars spent in regime)
REGIMES: Dict[str, Tuple[float, float, float]] = {
    "bull": (0.55, 0.55, 420),
    "bear": (-0.45, 0.85, 260),
    "chop": (0.02, 0.38, 620),
}
REGIME_NAMES = list(REGIMES)

INTRADAY_SECONDS = 86_400


@dataclass(frozen=True)
class SynthSpec:
    """Per-symbol personality, so a basket does not look like one cloned asset."""

    symbol: str = "SYNTH/USD"
    start_price: float = 100.0
    regime_mix: Tuple[float, float, float] = (0.34, 0.22, 0.44)  # bull, bear, chop
    vol_of_vol: float = 0.06        # GARCH alpha
    vol_persistence: float = 0.90   # GARCH beta
    tail_df: float = 5.0            # Student-t dof (lower = fatter tails)
    market: str = "crypto"          # crypto = 24/7, anything else = session-bound
    vol_scale: float = 1.0          # multiply regime vols: FX needs ~0.12, index ETFs ~0.45
    drift_scale: float = 1.0        # multiply regime drifts
    session_start_hour: int = 9
    session_end_hour: int = 15
    gap_pct: float = 0.004          # overnight gap stdev
    wick_mult: float = 0.6          # intrabar wick size, in units of bar vol
    base_volume: float = 25_000.0   # typical volume per bar, in BASE units
    volume_noise: float = 0.35
    inject_sweeps: int = 0          # n synthetic sweep->reversal events (recovery test)
    sweep_size_pct: float = 0.012
    sweep_reversal_pct: float = 0.028
    #: loading on a shared market factor, in [0, 1]. 0 = an independent path.
    #: Passed with ``generate(..., market_factor=...)``; see ASSET_CLASS_BETA.
    market_beta: float = 0.0
    seed: int = 0


#: Loading on the shared market factor per asset class.
#:
#: Without this, every symbol in a basket is an independent draw, so pairwise
#: correlation is ~0 and the portfolio's ``max_correlated_positions`` cap never
#: binds. That flatters every multi-symbol result: a book of ten alt pairs is not
#: ten bets, it is one bet sized ten times, and a backtest that ignores it
#: reports a drawdown the account could not have survived. Real numbers are
#: roughly 0.8-0.9 within crypto, 0.6-0.7 for large-cap equities against their
#: index, and 0.1-0.2 for FX majors against each other.
ASSET_CLASS_BETA: Dict[str, float] = {
    "crypto": 0.85,
    "us":     0.65,
    "india":  0.60,
    "forex":  0.15,
}


def _student_t(rng: np.random.Generator, df: float, n: int) -> np.ndarray:
    """Unit-variance Student-t innovations (fat tails, no vol scaling applied)."""
    if df <= 2 or not np.isfinite(df):
        return rng.standard_normal(n)
    t = rng.standard_t(df, size=n)
    return t / np.sqrt(df / (df - 2.0))


def market_factor(n: int, seed: int, timeframe: str = "4h") -> np.ndarray:
    """A unit-variance shared shock series for a basket.

    Deterministic in ``(n, seed, timeframe)`` so every symbol in one basket sees
    the same factor, and so a re-run reproduces it. The seed is derived with
    sha256, never with the builtin ``hash()``: ``hash()`` on anything containing
    a string is salted per process, which would make the whole basket - and
    therefore every backtest built on it - irreproducible.

    Fat-tailed like the idiosyncratic innovations, because a Gaussian factor
    would understate how hard correlated books get hit together.
    """
    digest = hashlib.sha256(f"mkt|{int(seed)}|{int(n)}|{timeframe}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    f = _student_t(rng, 5.0, n)
    sd = f.std()
    return f / sd if sd > 0 else f


def generate(
    spec: SynthSpec,
    n_bars: int = 4_000,
    timeframe: str = "4h",
    start: str = "2022-01-01",
    end: Optional[str] = None,
    market_factor_arr: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build a validated OHLCV frame from ``spec``.

    Extra columns ``_regime`` and ``_realised_vol`` carry ground truth used by
    the validation suite; they are dropped by consumers that index OHLCV.

    ``market_factor_arr`` is a shared unit-variance shock series (see
    :func:`market_factor`). When supplied together with ``spec.market_beta``, a
    fraction ``beta`` of each bar's innovation comes from the factor and the
    rest from the symbol's own draw, which produces cross-asset correlation
    without changing the symbol's marginal volatility.
    """
    rng = np.random.default_rng(int(spec.seed))
    freq = RESAMPLE_RULE.get(timeframe, timeframe)

    if end is not None:
        idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    else:
        idx = pd.date_range(start=start, periods=n_bars, freq=freq, tz="UTC")
    n = len(idx)
    if n < 50:
        raise ValueError(f"need at least 50 bars, got {n}")

    bpy = bars_per_year(timeframe, spec.market)

    # ---- regime path: Markov chain with regime-specific persistence ----------
    mix = np.asarray(spec.regime_mix, dtype=float)
    if mix.sum() <= 0:
        mix = np.ones(3)
    mix = mix / mix.sum()
    states = np.empty(n, dtype=int)
    cur = int(rng.choice(len(REGIME_NAMES), p=mix))
    stay = np.array([1.0 - 1.0 / REGIMES[name][2] for name in REGIME_NAMES])
    for i in range(n):
        states[i] = cur
        if rng.random() > stay[cur]:
            nxt = int(rng.choice(len(REGIME_NAMES)))
            cur = nxt if nxt != cur else (nxt + 1) % len(REGIME_NAMES)

    drift = np.array([REGIMES[REGIME_NAMES[s]][0] for s in states]) / bpy * spec.drift_scale
    base_vol = np.array([REGIMES[REGIME_NAMES[s]][1] for s in states]) / np.sqrt(bpy) * spec.vol_scale

    # ---- unit-variance innovations, then GARCH(1,1) conditional vol ----------
    z = _student_t(rng, spec.tail_df, n)

    # Single-factor blend: z' = beta * F + sqrt(1 - beta^2) * z keeps Var(z') = 1
    # when F and z are both unit-variance and independent, so the symbol's own
    # vol profile is untouched while its correlation with other symbols sharing
    # F becomes approximately beta_i * beta_j.
    beta = float(getattr(spec, "market_beta", 0.0) or 0.0)
    if market_factor_arr is not None and beta > 0:
        f = np.asarray(market_factor_arr, dtype="float64")
        if len(f) < n:
            raise ValueError(f"market factor has {len(f)} bars, need {n}")
        f = f[:n]
        fsd = f.std()
        if fsd > 0:
            f = f / fsd
        beta = min(max(beta, 0.0), 0.999)
        z = beta * f + np.sqrt(1.0 - beta * beta) * z

    shock = z * base_vol                      # per-bar return shock, in returns
    var = np.empty(n)
    var[0] = base_vol[0] ** 2
    alpha, beta = spec.vol_of_vol, spec.vol_persistence
    for i in range(1, n):
        omega = (1.0 - alpha - beta) * base_vol[i] ** 2
        var[i] = max(omega + alpha * shock[i - 1] ** 2 + beta * var[i - 1],
                     (0.2 * base_vol[i]) ** 2)
    vol = np.sqrt(var)

    rets = drift + vol * z                    # <-- single application of vol

    # ---- close path, then intrabar OHLC --------------------------------------
    close = spec.start_price * np.exp(np.cumsum(rets))
    prev_close = np.concatenate([[spec.start_price], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0, spec.gap_pct / 4.0, n))

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = spec.wick_mult * vol * np.abs(rng.standard_normal(n))
    high = body_hi * (1.0 + wick)
    low = body_lo * (1.0 - wick)

    # ---- volume in BASE units, correlated with |return| / vol ----------------
    activity = 1.0 + 3.0 * np.abs(rets) / np.maximum(vol, 1e-12)
    volume = spec.base_volume * activity * np.exp(rng.normal(0, spec.volume_noise, n))

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    if spec.inject_sweeps:
        df = _inject_sweeps(df, spec, rng)

    if spec.market != "crypto" and timeframe_seconds(timeframe) < INTRADAY_SECONDS:
        df = _apply_session(df, spec, rng)

    df["_regime"] = [REGIME_NAMES[s] for s in states]
    df["_realised_vol"] = vol
    # validate_ohlcv preserves "_"-prefixed ground-truth columns and .attrs
    return validate_ohlcv(df, spec.symbol)


def _inject_sweeps(df: pd.DataFrame, spec: SynthSpec, rng: np.random.Generator) -> pd.DataFrame:
    """Carve known stop-hunt patterns into the series.

    Each event runs price just beyond a recent swing extreme (taking the resting
    liquidity there) and then reverses hard. Event indices are recorded in
    ``df.attrs['injected_sweeps']`` so a detector can be scored for
    precision/recall against known ground truth.
    """
    n = len(df)
    warmup = 120
    events: List[Dict[str, object]] = []
    o = df["open"].to_numpy().copy()
    h = df["high"].to_numpy().copy()
    lo = df["low"].to_numpy().copy()
    c = df["close"].to_numpy().copy()

    attempts, placed = 0, 0
    while placed < spec.inject_sweeps and attempts < spec.inject_sweeps * 50:
        attempts += 1
        if n <= warmup + 16:
            break
        i = int(rng.integers(warmup, n - 14))
        if any(abs(i - int(e["index"])) < 25 for e in events):
            continue

        side = int(rng.choice([1, -1]))   # +1 bullish sweep (low grabbed), -1 bearish
        steps = min(7, n - i - 1)
        if side == 1:
            swing_low = float(np.min(lo[i - 40:i - 5]))
            lo[i] = swing_low * (1.0 - spec.sweep_size_pct)      # the stop hunt
            target = c[i] * (1.0 + spec.sweep_reversal_pct)      # then reverse up
            h[i] = max(h[i], c[i])
            path = np.linspace(c[i], target, steps + 1)[1:]
            for k, px in enumerate(path, start=1):
                c[i + k] = px
                o[i + k] = path[k - 2] if k > 1 else c[i]
                h[i + k] = max(h[i + k], px * 1.002)
                lo[i + k] = min(lo[i + k], px * 0.998)
            c[i] = max(c[i], lo[i] * 1.001)
        else:
            swing_high = float(np.max(h[i - 40:i - 5]))
            h[i] = swing_high * (1.0 + spec.sweep_size_pct)
            target = c[i] * (1.0 - spec.sweep_reversal_pct)
            lo[i] = min(lo[i], c[i])
            path = np.linspace(c[i], target, steps + 1)[1:]
            for k, px in enumerate(path, start=1):
                c[i + k] = px
                o[i + k] = path[k - 2] if k > 1 else c[i]
                lo[i + k] = min(lo[i + k], px * 0.998)
                h[i + k] = max(h[i + k], px * 1.002)
            c[i] = min(c[i], h[i] * 0.999)

        events.append({"index": i, "timestamp": df.index[i],
                       "side": "bullish" if side == 1 else "bearish"})
        placed += 1

    out = df.copy()
    out["open"], out["high"], out["low"], out["close"] = o, h, lo, c
    out.attrs["injected_sweeps"] = events
    return out


def _apply_session(df: pd.DataFrame, spec: SynthSpec, rng: np.random.Generator) -> pd.DataFrame:
    """Keep only in-session bars and add overnight gaps for equity/FX markets."""
    hours = df.index.hour
    mask = (hours >= spec.session_start_hour) & (hours <= spec.session_end_hour)
    out = df[mask].copy()
    if len(out) < 50:
        return df

    days = out.index.normalize()
    first_of_day = np.concatenate([[True], days[1:] != days[:-1]])
    o = out["open"].to_numpy().copy()
    c = out["close"].to_numpy().copy()
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    gaps = rng.normal(0, spec.gap_pct, len(out))
    o[first_of_day] = prev_close[first_of_day] * (1.0 + gaps[first_of_day])

    out["open"] = o
    out["high"] = np.maximum.reduce([out["high"].to_numpy(), o, c])
    out["low"] = np.minimum.reduce([out["low"].to_numpy(), o, c])
    return out


# ---------------------------------------------------------------------- basket
#: (start price, regime mix, tail dof, market, base volume per bar)
DEFAULT_BASKET: Dict[str, Dict[str, object]] = {
    # --- crypto majors: strong trends, fat tails, 24/7
    "BTC/USDT": dict(start_price=42_000, regime_mix=(0.40, 0.20, 0.40), tail_df=4.0,
                     market="crypto", base_volume=22_000),
    "ETH/USDT": dict(start_price=2_400, regime_mix=(0.38, 0.24, 0.38), tail_df=4.0,
                     market="crypto", base_volume=320_000),
    "SOL/USDT": dict(start_price=110, regime_mix=(0.44, 0.26, 0.30), tail_df=3.5,
                     market="crypto", base_volume=2_400_000),
    "BNB/USDT": dict(start_price=310, regime_mix=(0.36, 0.24, 0.40), tail_df=4.5,
                     market="crypto", base_volume=180_000),
    "XRP/USDT": dict(start_price=0.62, regime_mix=(0.32, 0.28, 0.40), tail_df=3.8,
                     market="crypto", base_volume=180_000_000),
    "ADA/USDT": dict(start_price=0.45, regime_mix=(0.30, 0.30, 0.40), tail_df=3.6,
                     market="crypto", base_volume=95_000_000),
    "AVAX/USDT": dict(start_price=35, regime_mix=(0.38, 0.28, 0.34), tail_df=3.8,
                      market="crypto", base_volume=3_200_000),
    "LINK/USDT": dict(start_price=15, regime_mix=(0.36, 0.28, 0.36), tail_df=4.0,
                      market="crypto", base_volume=14_000_000),
    "DOGE/USDT": dict(start_price=0.12, regime_mix=(0.26, 0.30, 0.44), tail_df=3.0,
                      market="crypto", base_volume=700_000_000),
    "DOT/USDT": dict(start_price=7.2, regime_mix=(0.32, 0.30, 0.38), tail_df=4.0,
                     market="crypto", base_volume=26_000_000),
    # --- US equities/ETFs: session-bound, lower vol, persistent drift
    "SPY": dict(start_price=470, regime_mix=(0.52, 0.12, 0.36), tail_df=6.0,
                market="us", base_volume=55_000_000),
    "QQQ": dict(start_price=400, regime_mix=(0.54, 0.14, 0.32), tail_df=5.5,
                market="us", base_volume=38_000_000),
    "NVDA": dict(start_price=480, regime_mix=(0.58, 0.16, 0.26), tail_df=4.5,
                 market="us", base_volume=42_000_000),
    "AAPL": dict(start_price=185, regime_mix=(0.46, 0.18, 0.36), tail_df=5.5,
                 market="us", base_volume=52_000_000),
    "MSFT": dict(start_price=375, regime_mix=(0.48, 0.16, 0.36), tail_df=5.5,
                 market="us", base_volume=21_000_000),
    "AMZN": dict(start_price=145, regime_mix=(0.44, 0.20, 0.36), tail_df=5.0,
                 market="us", base_volume=44_000_000),
    "META": dict(start_price=350, regime_mix=(0.46, 0.22, 0.32), tail_df=4.8,
                 market="us", base_volume=17_000_000),
    "GOOGL": dict(start_price=140, regime_mix=(0.45, 0.19, 0.36), tail_df=5.2,
                  market="us", base_volume=24_000_000),
    "JPM": dict(start_price=170, regime_mix=(0.40, 0.22, 0.38), tail_df=5.5,
                market="us", base_volume=9_500_000),
    "XOM": dict(start_price=105, regime_mix=(0.36, 0.26, 0.38), tail_df=5.0,
                market="us", base_volume=16_000_000),
    "AMD": dict(start_price=125, regime_mix=(0.44, 0.26, 0.30), tail_df=4.5,
                market="us", base_volume=48_000_000),
    "TSLA": dict(start_price=245, regime_mix=(0.40, 0.32, 0.28), tail_df=4.0,
                 market="us", base_volume=95_000_000),
    # --- Indian large caps: session-bound, gap-prone
    "RELIANCE.NS": dict(start_price=2_450, regime_mix=(0.40, 0.20, 0.40), tail_df=5.0,
                        market="india", base_volume=7_500_000),
    "TCS.NS": dict(start_price=3_600, regime_mix=(0.36, 0.22, 0.42), tail_df=5.5,
                   market="india", base_volume=2_200_000),
    "HDFCBANK.NS": dict(start_price=1_620, regime_mix=(0.30, 0.26, 0.44), tail_df=5.0,
                        market="india", base_volume=11_000_000),
    "INFY.NS": dict(start_price=1_450, regime_mix=(0.32, 0.28, 0.40), tail_df=5.0,
                    market="india", base_volume=8_500_000),
    "ICICIBANK.NS": dict(start_price=980, regime_mix=(0.38, 0.22, 0.40), tail_df=5.0,
                         market="india", base_volume=9_000_000),
    "SBIN.NS": dict(start_price=620, regime_mix=(0.34, 0.28, 0.38), tail_df=4.8,
                    market="india", base_volume=18_000_000),
    "ITC.NS": dict(start_price=440, regime_mix=(0.30, 0.24, 0.46), tail_df=5.5,
                   market="india", base_volume=14_000_000),
    "BHARTIARTL.NS": dict(start_price=1_050, regime_mix=(0.42, 0.20, 0.38), tail_df=5.0,
                          market="india", base_volume=6_500_000),
    "KOTAKBANK.NS": dict(start_price=1_750, regime_mix=(0.28, 0.26, 0.46), tail_df=5.2,
                         market="india", base_volume=3_400_000),
    "LT.NS": dict(start_price=3_200, regime_mix=(0.36, 0.24, 0.40), tail_df=5.0,
                  market="india", base_volume=2_800_000),
    "TATAMOTORS.NS": dict(start_price=650, regime_mix=(0.38, 0.30, 0.32), tail_df=4.5,
                          market="india", base_volume=13_000_000),
    "BAJFINANCE.NS": dict(start_price=6_900, regime_mix=(0.32, 0.30, 0.38), tail_df=4.5,
                          market="india", base_volume=1_600_000),
    # --- FX / metals: mean-reverting, tight ranges
    "EURUSD": dict(start_price=1.085, regime_mix=(0.20, 0.20, 0.60), tail_df=8.0,
                   market="forex", base_volume=1.2e8, wick_mult=0.4),
    "GBPUSD": dict(start_price=1.265, regime_mix=(0.20, 0.22, 0.58), tail_df=7.5,
                   market="forex", base_volume=1.2e8, wick_mult=0.4),
    "USDJPY": dict(start_price=148.5, regime_mix=(0.26, 0.18, 0.56), tail_df=7.0,
                   market="forex", base_volume=1.2e8, wick_mult=0.4),
    "AUDUSD": dict(start_price=0.655, regime_mix=(0.18, 0.24, 0.58), tail_df=7.5,
                   market="forex", base_volume=1.2e8, wick_mult=0.4),
    "USDCAD": dict(start_price=1.355, regime_mix=(0.22, 0.20, 0.58), tail_df=7.5,
                   market="forex", base_volume=1.2e8, wick_mult=0.4),
    "USDCHF": dict(start_price=0.885, regime_mix=(0.20, 0.22, 0.58), tail_df=8.0,
                   market="forex", base_volume=1.2e8, wick_mult=0.4),
    "XAUUSD": dict(start_price=2_030, regime_mix=(0.46, 0.18, 0.36), tail_df=5.0,
                   market="forex", base_volume=9.0e7, wick_mult=0.5),
}

#: (vol_scale, drift_scale) per asset class - regime vols are calibrated to crypto
ASSET_CLASS_SCALE: Dict[str, Tuple[float, float]] = {
    "crypto": (1.00, 1.00),
    "us":     (0.42, 0.35),   # large-cap equities/ETFs
    "india":  (0.58, 0.45),   # higher-beta EM large caps
    "forex":  (0.12, 0.15),   # majors: ~8-10% annualised vol
}

#: session hours in UTC-ish local terms used for the synthetic session filter
SESSION_HOURS: Dict[str, Tuple[int, int]] = {
    "us": (9, 15),       # 09:30-16:00 ET
    "india": (9, 15),    # 09:15-15:30 IST
    "forex": (0, 23),    # 24h, but weekends still cut below
    "crypto": (0, 23),
}


def _personality(symbol: str, seed: int) -> Dict[str, object]:
    """Known personality, or a deterministic hash-derived one for any symbol."""
    if symbol in DEFAULT_BASKET:
        return dict(DEFAULT_BASKET[symbol])
    # stable across processes: hash() is salted per-run, so use a real digest
    digest = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(digest % (2**31))
    bull = float(rng.uniform(0.20, 0.55))
    bear = float(rng.uniform(0.12, 0.30))
    market = rng.choice(["crypto", "us", "india", "forex"], p=[0.4, 0.3, 0.2, 0.1])
    return dict(
        start_price=float(10 ** rng.uniform(0.3, 3.5)),
        regime_mix=(bull, bear, max(0.08, 1 - bull - bear)),
        tail_df=float(rng.uniform(3.0, 8.0)),
        market=str(market),
        base_volume=float(10 ** rng.uniform(3.0, 7.0)),
    )


def basket(
    symbols: Optional[List[str]] = None,
    n_bars: int = 4_000,
    timeframe: str = "4h",
    start: str = "2022-01-01",
    end: Optional[str] = None,
    seed: int = 20261123,
    inject_sweeps: int = 0,
) -> Dict[str, pd.DataFrame]:
    """symbol -> OHLCV for a reproducible synthetic basket.

    Symbols within one asset class share a common market factor, so the basket
    has realistic cross-asset correlation instead of being a set of independent
    paths. That matters: the portfolio risk caps only mean something if the
    data can actually trip them.
    """
    syms = list(symbols or DEFAULT_BASKET)

    # the index length depends on start/end/timeframe, so build it once here and
    # size the factors to match rather than guessing n_bars
    freq = RESAMPLE_RULE.get(timeframe, timeframe)
    if end is not None:
        n = len(pd.date_range(start=start, end=end, freq=freq, tz="UTC"))
    else:
        n = n_bars

    factors: Dict[str, np.ndarray] = {}
    out: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        personality = _personality(sym, seed)
        sym_salt = int(hashlib.sha256(sym.encode()).hexdigest()[8:16], 16) % 100_000
        market_kind = str(personality.get("market", "crypto"))
        hours = SESSION_HOURS.get(market_kind, (9, 15))
        vol_scale, drift_scale = ASSET_CLASS_SCALE.get(market_kind, (1.0, 1.0))
        beta = ASSET_CLASS_BETA.get(market_kind, 0.0)
        if market_kind not in factors:
            factors[market_kind] = market_factor(n + 1, seed, timeframe)
        spec = SynthSpec(
            symbol=sym,
            seed=int(seed) + sym_salt,
            inject_sweeps=inject_sweeps,
            session_start_hour=hours[0],
            session_end_hour=hours[1],
            vol_scale=vol_scale,
            drift_scale=drift_scale,
            market_beta=beta,
            **personality,
        )
        out[sym] = generate(spec, n_bars=n_bars, timeframe=timeframe, start=start, end=end,
                            market_factor_arr=factors[market_kind])
    return out
