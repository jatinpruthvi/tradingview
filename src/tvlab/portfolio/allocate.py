"""Portfolio construction: weights, correlation caps and mandate checks.

The backtester sizes a *single* trade from a risk budget. This module decides
how much of the account each symbol may use at all, and enforces the three caps
that a single-symbol backtest cannot see:

``max_position_pct``
    no symbol may hold more than this fraction of equity.
``max_open_positions``
    how many symbols may be allocated concurrently.
``max_correlated_positions``
    how many *mutually correlated* symbols may be allocated concurrently. Ten
    alt pairs are not ten independent bets; in a risk-off move they are one bet,
    and a portfolio that ignores this reports a drawdown it cannot actually
    survive.

On weighting
------------
``equal`` is the default, and that is deliberate. The tempting alternative -
weight symbols by their backtested return - selects on the in-sample outcome and
is indistinguishable from overfitting: it will always look better on the data
that chose it. ``inverse_vol`` is the one data-driven scheme offered here, and
it is defensible only because volatility is far more persistent and far slower
to change than edge. It allocates by *risk*, not by past profit.

There is no ``performance`` scheme in this module on purpose. If you want one,
build it inside a walk-forward loop where the weights are chosen on training
data only, and then report the out-of-sample result - not the in-sample one.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import AppConfig

log = logging.getLogger(__name__)

SCHEMES = ("equal", "inverse_vol")

#: Fallback only. The live value comes from ``risk.correlation_threshold``, which
#: is calibrated per market - see config/markets/*.yaml. Crypto alts correlate
#: around 0.6-0.8 and FX majors around 0.1-0.3, so no single global number is
#: right for both.
DEFAULT_CORR_THRESHOLD = 0.60


@dataclass
class Allocation:
    """Per-symbol capital weights after every cap has been applied."""

    scheme: str
    weights: Dict[str, float] = field(default_factory=dict)
    #: what the weights were before capping, for transparency
    raw_weights: Dict[str, float] = field(default_factory=dict)
    groups: List[List[str]] = field(default_factory=list)
    caps_applied: List[str] = field(default_factory=list)
    #: the threshold actually used, from ``risk.correlation_threshold``
    corr_threshold: float = DEFAULT_CORR_THRESHOLD
    warnings: List[str] = field(default_factory=list)

    @property
    def symbols(self) -> List[str]:
        return [s for s, w in self.weights.items() if w > 0]

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights.values()))

    def capital(self, symbol: str, equity: float) -> float:
        """Dollars this symbol may commit at the given equity level."""
        return float(self.weights.get(symbol, 0.0) * equity)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_weight"] = self.total_weight
        d["n_symbols"] = len(self.symbols)
        return d


# ------------------------------------------------------------------- correlation
def correlation_groups(returns: pd.DataFrame,
                       threshold: float = DEFAULT_CORR_THRESHOLD) -> List[List[str]]:
    """Cluster symbols whose returns move together, single-linkage.

    Single-linkage (a symbol joins a group if it correlates with *any* member
    above the threshold) is deliberately conservative: it produces fewer, larger
    groups than average-linkage, which means the cap binds sooner. For risk
    limits, over-grouping is the safe error.

    Returns groups sorted largest-first; symbols with no correlate come back as
    groups of one so the caller can treat every symbol uniformly.
    """
    cols = [c for c in returns.columns if returns[c].notna().sum() > 10]
    if len(cols) < 2:
        return [[c] for c in returns.columns]

    corr = returns[cols].corr().abs()
    # NaN correlations (a symbol that never traded) are treated as uncorrelated
    corr = corr.fillna(0.0)

    parent = {c: c for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                union(a, b)

    buckets: Dict[str, List[str]] = {}
    for c in cols:
        buckets.setdefault(find(c), []).append(c)
    # keep any symbol dropped for insufficient history as its own group
    for c in returns.columns:
        if c not in cols:
            buckets[c] = [c]

    groups = sorted(buckets.values(), key=lambda g: (-len(g), g[0]))
    return groups


def _returns_from_equity(equity: Dict[str, pd.Series]) -> pd.DataFrame:
    """Align equity curves on their time index and convert to simple returns."""
    if not equity:
        return pd.DataFrame()
    frame = pd.DataFrame(equity).sort_index()
    return frame.pct_change().replace([np.inf, -np.inf], np.nan)


def liquidity_from_ohlcv(ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """symbol -> median quote volume (close x volume), for tie-breaking only.

    A median rather than a mean, because one wash-trade bar can dominate the
    average of a thin market. Used solely to break exact weight ties: deeper
    markets are cheaper and easier to exit, which is an execution fact rather
    than a performance claim.
    """
    out: Dict[str, float] = {}
    for sym, df in (ohlcv or {}).items():
        try:
            if df is None or len(df) == 0 or "volume" not in df.columns:
                continue
            qv = (df["close"] * df["volume"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(qv):
                out[sym] = float(qv.median())
        except Exception:  # noqa: BLE001 - a bad symbol must not break the allocation
            continue
    return out


def asset_returns_from_ohlcv(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """symbol -> underlying close returns, aligned on the union of timestamps.

    Feed the result to :func:`allocate` as ``asset_returns``. This is the correct
    basis for correlation grouping: it is dense (every bar moves) and it reflects
    the market rather than the strategy's decision to be in it.
    """
    if not ohlcv:
        return pd.DataFrame()
    closes = {s: df["close"] for s, df in ohlcv.items() if df is not None and len(df)}
    if not closes:
        return pd.DataFrame()
    return (pd.DataFrame(closes).sort_index().pct_change()
            .replace([np.inf, -np.inf], np.nan))


def causal_rs_mask(
    ohlcv: Dict[str, pd.DataFrame],
    lookback_sessions: int = 20,
    top_pct: float = 0.4,
) -> Dict[str, pd.DataFrame]:
    """Causal cross-sectional relative-strength mask per symbol.

    For each session date ``D``, symbols are ranked strictly on their trailing
    ``lookback_sessions`` return ending at the close of session ``D - 1``
    (shifted by one full session so nothing from session ``D`` is ever read).

    Returns ``symbol -> DataFrame({"allow_long": bool, "allow_short": bool})``
    aligned to each symbol's index:
    * ``allow_long`` is True for the top ``top_pct`` leaders.
    * ``allow_short`` is True for the bottom ``top_pct`` laggards.
    * Warmup sessions before ``lookback_sessions`` completed days stay True
      (fail-open on insufficient history rather than killing short samples).
    """
    if not ohlcv or not (0.0 < float(top_pct) < 1.0):
        return {}

    daily_closes: Dict[str, pd.Series] = {}
    for sym, df in ohlcv.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            continue
        day = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        # last close per session date
        s_close = df["close"].groupby(day).last()
        daily_closes[sym] = s_close

    if len(daily_closes) < 2:
        return {}

    panel = pd.DataFrame(daily_closes).sort_index().ffill()
    lb = max(int(lookback_sessions), 1)
    # Strictly shift by 1 completed session so day D never sees day D's close.
    trailing_ret = panel.pct_change(lb).shift(1)
    long_rank = trailing_ret.rank(axis=1, ascending=False, pct=True, method="average")
    short_rank = trailing_ret.rank(axis=1, ascending=True, pct=True, method="average")
    n_valid = trailing_ret.notna().sum(axis=1).replace(0, np.nan)
    eff_pct = np.maximum(float(top_pct), (1.0 / n_valid).to_numpy())

    out: Dict[str, pd.DataFrame] = {}
    for sym, df in ohlcv.items():
        if sym not in long_rank.columns or df is None or df.empty:
            continue
        idx = df.index
        day = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        lr = long_rank[sym].reindex(day).to_numpy()
        sr = short_rank[sym].reindex(day).to_numpy()
        cut = pd.Series(eff_pct, index=panel.index).reindex(day).to_numpy()
        no_hist = np.isnan(lr)
        allow_long = no_hist | (lr <= cut)
        allow_short = no_hist | (sr <= cut)
        out[sym] = pd.DataFrame(
            {"allow_long": allow_long, "allow_short": allow_short},
            index=idx,
        )
    return out


def _volatility(returns: pd.DataFrame, periods_per_year: float) -> pd.Series:
    vol = returns.std() * np.sqrt(periods_per_year)
    return vol.replace(0.0, np.nan)


# --------------------------------------------------------------------- weighting
def raw_weights(returns: pd.DataFrame, scheme: str = "equal",
                periods_per_year: float = 252.0) -> Dict[str, float]:
    """Unconstrained weights. Sums to 1.0 before caps are applied."""
    cols = list(returns.columns)
    if not cols:
        return {}
    if scheme not in SCHEMES:
        raise ValueError(f"unknown allocation scheme {scheme!r}; use one of {SCHEMES}")

    if scheme == "equal":
        w = {c: 1.0 / len(cols) for c in cols}
        return w

    vol = _volatility(returns, periods_per_year)
    # a symbol with no measurable volatility (never traded) gets zero weight;
    # inventing one would put capital behind a strategy that produced nothing
    usable = {c: float(v) for c, v in vol.items() if np.isfinite(v) and v > 0}
    if not usable:
        log.warning("inverse_vol found no usable volatility; falling back to equal weight")
        return {c: 1.0 / len(cols) for c in cols}
    inv = {c: 1.0 / v for c, v in usable.items()}
    total = sum(inv.values())
    return {c: inv[c] / total for c in cols if c in inv}


def _renormalise(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(w.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in w.items()}


def apply_caps(weights: Dict[str, float], cfg: AppConfig,
               groups: Optional[Sequence[Sequence[str]]] = None,
               liquidity: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, float], List[str]]:
    """Enforce the mandate. Returns capped weights and a log of what was cut.

    Order matters, and it is not the obvious one:

    1. ``max_correlated_positions`` - inside each correlation group, keep at
       most this many.
    2. ``max_open_positions`` - if too many remain, drop the smallest.
    3. ``max_position_pct`` - clip any single symbol, then redistribute what is
       left into the remaining headroom.

    The correlation cap must come FIRST. Applying the position-count cap first
    drops symbols by weight alone, and under equal weighting every symbol has
    the same weight - so the tie is broken by dictionary order, which in
    practice discards the uncorrelated diversifiers and keeps the redundant
    ones. On a 7-symbol crypto/FX book that left the account 50% in cash while
    holding two symbols that were 90% correlated with each other: fewer
    positions, more risk, and a result that depended on insertion order.

    Ties are broken by ``liquidity`` (median quote volume) when supplied, then
    alphabetically. Both are deterministic and neither looks at performance, so
    the allocation is reproducible across runs and Python versions and cannot
    tilt toward whatever happened to make money in the sample. Breaking ties by
    liquidity matters in practice: when a cap has to drop eight of ten crypto
    names, keeping the two deepest markets is a real execution decision, while
    keeping whichever two sort first alphabetically is an accident.
    """
    applied: List[str] = []
    w = {k: float(v) for k, v in weights.items() if v > 0}
    if not w:
        return {}, applied

    risk = cfg.risk
    liq = liquidity or {}

    def _rank(sym: str, weight: float) -> Tuple[float, float, str]:
        """Sort key for KEEPING: heaviest weight, then deepest market, then A-Z."""
        return (-weight, -float(liq.get(sym, 0.0) or 0.0), sym)

    # 1. correlated positions - the cap that actually controls risk
    max_corr = int(getattr(risk, "max_correlated_positions", 0) or 0)
    if max_corr and groups:
        for group in groups:
            members = sorted(g for g in group if g in w)
            if len(members) > max_corr:
                keep = sorted(members, key=lambda m: _rank(m, w[m]))[:max_corr]
                dropped = [m for m in members if m not in keep]
                for m in dropped:
                    w.pop(m, None)
                if not w:
                    return {}, applied + ["correlation cap removed every position"]
                w = _renormalise(w)
                applied.append(
                    f"max_correlated_positions={max_corr} in group "
                    f"[{', '.join(members)}]: kept {', '.join(keep)}, "
                    f"dropped {', '.join(dropped)}"
                )

    # 2. concurrent position count
    max_pos = int(getattr(risk, "max_open_positions", 0) or 0)
    if max_pos and len(w) > max_pos:
        keep = sorted(w.items(), key=lambda kv: _rank(kv[0], kv[1]))[:max_pos]
        dropped = sorted(set(w) - {k for k, _ in keep})
        w = _renormalise(dict(keep))
        applied.append(f"max_open_positions={max_pos}: dropped {', '.join(dropped)}")

    # 3. single-position cap
    cap = float(getattr(risk, "max_position_pct", 1.0) or 1.0)
    if cap and any(v > cap + 1e-12 for v in w.values()):
        clipped = [k for k, v in w.items() if v > cap]
        w = {k: min(v, cap) for k, v in w.items()}
        applied.append(f"max_position_pct={cap:.0%}: clipped {', '.join(clipped)}")
        # renormalise the remainder up to the cap, one pass - iterating to a
        # fixed point can oscillate when every symbol is already at the cap
        total = sum(w.values())
        if total > 0 and total < 1.0 - 1e-9 and len(w) > 1:
            room = {k: cap - v for k, v in w.items()}
            slack = sum(r for r in room.values() if r > 0)
            if slack > 1e-12:
                add = min(1.0 - total, slack)
                w = {k: v + (max(room[k], 0.0) / slack) * add for k, v in w.items()}

    return w, applied


def allocate(equity: Dict[str, pd.Series], cfg: AppConfig, scheme: str = "equal",
             corr_threshold: Optional[float] = None,
             periods_per_year: Optional[float] = None,
             asset_returns: Optional[pd.DataFrame] = None,
             liquidity: Optional[Dict[str, float]] = None) -> Allocation:
    """Build a capped allocation for a set of symbols.

    ``equity`` maps symbol -> strategy equity curve. ``asset_returns`` maps
    symbol -> *underlying price* returns, and when supplied it - not the equity
    curve - drives both the correlation groups and the ``inverse_vol`` weights.

    That distinction is the whole function. A selective strategy is flat most of
    the time: on the reference crypto book the equity curves moved on only
    0.3-5% of bars, which made their pairwise correlation 0.02 - indistinguishable
    from zero - while the underlying assets correlated at 0.65, with 39 of 45
    pairs above the threshold. Grouping on equity curves therefore reported ten
    independent bets and never fired ``max_correlated_positions``, exactly when
    the concentration risk was worst. Correlation is a property of the *market*,
    and it is there whether or not the strategy happened to be holding.

    Neither input's level or cumulative return is ever used, so this cannot tilt
    toward whatever performed well in the sample.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown allocation scheme {scheme!r}; use one of {SCHEMES}")

    bpy = float(periods_per_year or cfg.periods_per_year())
    threshold = float(corr_threshold if corr_threshold is not None
                      else getattr(cfg.risk, "correlation_threshold", DEFAULT_CORR_THRESHOLD))

    returns = _returns_from_equity(equity)
    if returns.empty and asset_returns is None:
        return Allocation(scheme=scheme, warnings=["no equity curves supplied"])

    basis = returns
    if asset_returns is not None:
        basis = (pd.DataFrame(asset_returns) if isinstance(asset_returns, dict)
                 else asset_returns).replace([np.inf, -np.inf], np.nan)
        if basis.empty:
            log.warning("asset_returns supplied but empty; falling back to equity curves")
            basis = returns
    if basis.empty:
        return Allocation(scheme=scheme, warnings=["no usable return series supplied"])

    groups = correlation_groups(basis, threshold=threshold)
    raw = raw_weights(basis, scheme=scheme, periods_per_year=bpy)
    weights, applied = apply_caps(raw, cfg, groups=groups, liquidity=liquidity)

    out = Allocation(scheme=scheme, weights=weights, raw_weights=raw,
                     groups=[list(g) for g in groups], caps_applied=applied)
    out.corr_threshold = threshold

    if not weights:
        out.warnings.append("every symbol was removed by the caps; nothing is allocated")
    multi = [g for g in groups if len(g) > 1]
    if multi:
        out.warnings.append(
            f"{len(multi)} correlation group(s) above {threshold:.0%}: "
            + "; ".join("[" + ", ".join(g) + "]" for g in multi[:4])
            + (" ..." if len(multi) > 4 else "")
        )
    if scheme == "inverse_vol" and any(not np.isfinite(v) or v <= 0
                                       for v in _volatility(basis, bpy).values):
        out.warnings.append("some symbols had no measurable volatility and were given zero weight")
    return out


def allocation_report(alloc: Allocation, equity: float,
                      risk_per_trade: Optional[float] = None) -> Dict[str, Any]:
    """Human-readable summary, including the account-level risk this implies.

    The last figure is the one that matters: summing each symbol's own
    ``risk_per_trade`` across N positions gives N x the intended account risk if
    the positions are correlated. This reports the number honestly so the
    mandate can be checked against it.
    """
    rpt = risk_per_trade if risk_per_trade is not None else 0.005
    rows = []
    for sym, w in sorted(alloc.weights.items(), key=lambda kv: -kv[1]):
        rows.append({
            "symbol": sym,
            "weight_pct": round(w * 100, 3),
            "capital": round(w * equity, 2),
            "risk_at_mandate": round(w * equity * rpt, 2),
        })
    total_risk = sum(r["risk_at_mandate"] for r in rows)
    return {
        "scheme": alloc.scheme,
        "corr_threshold": getattr(alloc, "corr_threshold", None),
        "n_symbols": len(rows),
        "total_weight_pct": round(alloc.total_weight * 100, 3),
        "rows": rows,
        "groups": alloc.groups,
        "caps_applied": alloc.caps_applied,
        "warnings": alloc.warnings,
        "aggregate_risk_if_all_stop_out": round(total_risk, 2),
        "aggregate_risk_pct": round(total_risk / equity * 100, 3) if equity else None,
    }
