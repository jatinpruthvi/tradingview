"""Does the option chain predict anything? The test the panel exists for.

Why forward returns, and not the win rate
-----------------------------------------
The tempting test is "does a chain filter lift the win rate of the shortlist".
It cannot be run: this strategy produces about 19 trades a year on the
configured names, and detecting a shift from 19% to 30% at 80% power needs
roughly 290 trades per arm - around 15 years. A result that far away is not a
result, it is a promise.

Forward returns are a different measurement. Every symbol with a chain gives an
observation every trading day, so 218 names over 60 days is ~13,000 name-days,
and a cross-sectional rank correlation can detect a real relationship long
before the trade count would. So this module asks the only question the panel can
answer quickly: **when a chain feature is high across today's names, do those
names outperform over the next N days?**

What the numbers mean, and what they do not
-------------------------------------------
* ``mean_ic`` is the average of the daily cross-sectional Spearman correlation
  between the feature and the forward return. It is scale-free, so a feature
  measured in lots and one measured in percent are comparable.
* ``ic_t`` treats each *day* as one observation. Days are not independent of
  each other (a market-wide move pushes every name the same way), so this
  overstates significance somewhat; it is a screening statistic, not a p-value
  you would publish.
* ``consistency`` is the share of days whose IC had the *same sign* as the mean.
  A feature with ``mean_ic`` driven by two violent days shows up here as ~50%.
* Eight features are tested at once, so the bar is ``|t| >= 3`` rather than 2 -
  roughly Bonferroni for eight tests. Even that only earns the word
  "candidate": in-sample discovery is the easy half.

Nothing here is allowed to influence ``suggest``. The honest sequence is
discover, confirm out-of-sample, and only then consider a filter - and a chain
feature that merely reproduces realised volatility (measured at IV/RV = 0.91)
should lose to realised volatility, which has history and can be backtested.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data.base import bars_per_year

log = logging.getLogger(__name__)

#: Days of panel needed before any verdict is allowed. One day of 218 names is
#: 218 observations of a *single* market state; the cross-section has to be
#: measured across many states before a mean IC means anything. 40 trading days
#: is about two months - enough for the t statistic to have a denominator.
MIN_DAYS = 40

#: A day with fewer names than this is a narrow cross-section: the rank
#: correlation of 8 names is mostly noise, and letting those days in would let a
#: handful of thin sessions dominate the mean.
MIN_NAMES_PER_DAY = 20

DEFAULT_HORIZON = 20

#: |t| that counts as "worth a second look" once eight features are tested at
#: once. See the module docstring: this is a screening bar, not a p-value.
T_BAR = 3.0

#: Share of days that must agree on the sign for a feature to be called stable.
CONSISTENCY_BAR = 0.60

#: (column, human label). Raw open-interest levels and spot are excluded on
#: purpose: they are size proxies, and a cross-section that ranks by market cap
#: would report "size predicts returns" under the name of an option feature.
FEATURES: Tuple[Tuple[str, str], ...] = (
    ("pcr", "put/call OI in band"),
    ("atm_iv", "ATM implied vol %"),
    ("iv_rv_ratio", "ATM IV / 20d realised vol"),
    ("skew_put_minus_call", "put IV minus call IV"),
    ("call_wall_dist_pct", "nearest big call OI, % from spot"),
    ("put_wall_dist_pct", "nearest big put OI, % from spot"),
    ("oi_total_log", "log total OI in band"),
    ("live_strikes", "strikes with open interest"),
)


# --------------------------------------------------------------------- returns


def _dates_in(index: pd.Index) -> List[date]:
    """Bar timestamps as calendar dates in the market's own timezone.

    A daily bar for an Indian name is stamped 03:45 UTC (09:15 IST). Normalising
    in UTC would move the date back a day for anything stamped after 18:30 UTC
    and silently shift every forward return by one bar.
    """
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata")
    return [ts.date() for ts in idx]


def forward_returns(
    panel: pd.DataFrame,
    bars: Mapping[str, pd.DataFrame],
    horizon: int = DEFAULT_HORIZON,
    market: str = "indian_equities",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Close-to-close return over ``horizon`` bars from each panel row.

    Entry is the **first bar on or after** ``as_of``: a snapshot taken in the
    evening describes a position you could only take at the next close, and using
    the same day's close would build a one-bar lookahead into every observation.

    Rows whose forward window has not closed yet are dropped - not filled with
    zero, which would look like "no move" and bias every mean toward it. The
    counts come back so the caller can say how much was censored instead of
    silently reporting a smaller sample.
    """
    rows: List[Dict[str, Any]] = []
    stats = {"rows": 0, "no_bars": 0, "censored": 0}
    if panel.empty:
        return pd.DataFrame(columns=["as_of", "symbol", "entry_date", "fwd_ret", "rv20"]), stats

    bpy = bars_per_year("1d", market)
    for symbol, grp in panel.groupby("symbol", sort=True):
        df = bars.get(str(symbol))
        if df is None or df.empty or "close" not in df.columns:
            stats["no_bars"] += len(grp)
            continue
        close = df["close"].astype(float).reset_index(drop=True)
        dates = _dates_in(df.index)
        if len(close) < 3:
            stats["no_bars"] += len(grp)
            continue

        for as_of in grp["as_of"]:
            try:
                wanted = date.fromisoformat(str(as_of))
            except ValueError:
                stats["no_bars"] += 1
                continue
            entry = next((i for i, d in enumerate(dates) if d >= wanted), None)
            if entry is None or entry + horizon >= len(close):
                stats["censored"] += 1
                continue
            ret = np.log(close.iloc[entry + horizon] / close.iloc[entry])
            if not np.isfinite(ret):
                stats["censored"] += 1
                continue
            window = close.iloc[max(0, entry - 19):entry + 1]
            logret = np.log(window / window.shift(1)).dropna()
            rv = float(logret.std(ddof=1) * math.sqrt(bpy)) if len(logret) >= 5 else float("nan")
            rows.append({"as_of": str(as_of), "symbol": str(symbol),
                         "entry_date": dates[entry].isoformat(),
                         "fwd_ret": math.expm1(ret), "rv20": rv})
            stats["rows"] += 1
    out = pd.DataFrame(rows, columns=["as_of", "symbol", "entry_date", "fwd_ret", "rv20"])
    return out, stats


def add_derived(panel: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Panel + forward returns + the two features built from both.

    ``iv_rv_ratio`` is the feature the earlier study flagged as the only one with
    a defensible mechanism - options pricing a move the realised series has not
    delivered - and it needs the bars to exist at all.
    """
    if panel.empty or returns.empty:
        return pd.DataFrame()
    frame = panel.merge(returns, on=["as_of", "symbol"], how="inner")
    frame["iv_rv_ratio"] = np.where(
        (frame["rv20"].notna()) & (frame["rv20"] > 0) & (frame["atm_iv"].notna()),
        frame["atm_iv"] / (frame["rv20"] * 100.0), np.nan)
    frame["oi_total_log"] = np.log1p(
        frame[["ce_oi", "pe_oi"]].fillna(0).sum(axis=1).astype(float))
    return frame


# ------------------------------------------------------------------- statistics


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation on average ranks, without a scipy dependency call.

    Returns NaN - not 0 - for a constant input. A feature that is identical
    across the cross-section has no rank order, and calling that "no
    correlation" would make it indistinguishable from a measured zero.
    """
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    xr = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    yr = pd.Series(np.asarray(y, dtype=float)).rank().to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _quintile_spread(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean forward return of the top fifth minus the bottom fifth."""
    if len(x) < 5:
        return float("nan")
    s = pd.Series(np.asarray(x, dtype=float))
    r = pd.Series(np.asarray(y, dtype=float))
    lo, hi = s.quantile([0.2, 0.8])
    bottom = r[s <= lo]
    top = r[s >= hi]
    if bottom.empty or top.empty:
        return float("nan")
    return float(top.mean() - bottom.mean())


@dataclass
class FeatureResult:
    feature: str
    label: str
    days: int = 0
    rows: int = 0
    mean_ic: float = float("nan")
    ic_t: float = float("nan")
    consistency: float = float("nan")
    quintile_spread: float = float("nan")

    @property
    def has_sample(self) -> bool:
        # bool(), not the numpy scalar: callers do `is True` on these flags.
        return bool(self.days > 1 and np.isfinite(self.mean_ic))

    @property
    def notable(self) -> bool:
        if not self.has_sample:
            return False
        return bool(np.isfinite(self.ic_t) and abs(self.ic_t) >= T_BAR
                    and np.isfinite(self.consistency)
                    and self.consistency >= CONSISTENCY_BAR)

    def to_dict(self) -> Dict[str, Any]:
        return {"feature": self.feature, "label": self.label, "days": self.days,
                "rows": self.rows, "mean_ic": self.mean_ic, "ic_t": self.ic_t,
                "consistency": self.consistency,
                "quintile_spread": self.quintile_spread, "notable": self.notable}


@dataclass
class ChainStudyResult:
    horizon: int
    panel_days: int
    panel_rows: int
    min_days: int
    min_names: int
    results: List[FeatureResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def evaluated_days(self) -> int:
        """Days that produced at least one usable cross-section."""
        return max((r.days for r in self.results), default=0)

    @property
    def insufficient(self) -> bool:
        return self.evaluated_days < self.min_days

    @property
    def candidates(self) -> List[FeatureResult]:
        return [r for r in self.results if r.notable]

    @property
    def verdict(self) -> str:
        tested = len([r for r in self.results if r.has_sample])
        if self.insufficient:
            return (f"INSUFFICIENT SAMPLE: {self.evaluated_days} usable day(s) against a bar "
                    f"of {self.min_days}. No verdict is possible yet - keep running "
                    f"`tvlab chain snapshot`.")
        if not self.candidates:
            return (f"NO EVIDENCE: none of the {tested} feature(s) shows a stable "
                    f"cross-sectional relationship with {self.horizon}-day forward returns "
                    f"(bar: |t| >= {T_BAR:g} and sign consistency >= "
                    f"{CONSISTENCY_BAR:.0%}).")
        names = ", ".join(r.feature for r in self.candidates)
        return (f"CANDIDATE, NOT A FILTER: {names} cleared the bar in-sample. Confirm on "
                f"days the panel has not seen before anything is wired into a pick.")

    def to_dict(self) -> Dict[str, Any]:
        return {"horizon": self.horizon, "panel_days": self.panel_days,
                "panel_rows": self.panel_rows, "min_days": self.min_days,
                "min_names": self.min_names, "verdict": self.verdict,
                "insufficient": self.insufficient,
                "results": [r.to_dict() for r in self.results],
                "notes": list(self.notes), "dropped": dict(self.dropped)}


def study(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    min_days: int = MIN_DAYS,
    min_names: int = MIN_NAMES_PER_DAY,
    features: Optional[Iterable[str]] = None,
    dropped: Optional[Dict[str, int]] = None,
) -> ChainStudyResult:
    """Cross-sectional IC of each chain feature against forward returns."""
    wanted = list(features) if features else [f for f, _ in FEATURES]
    labels = dict(FEATURES)
    res = ChainStudyResult(horizon=horizon,
                           panel_days=int(panel["as_of"].nunique()) if not panel.empty else 0,
                           panel_rows=len(panel), min_days=min_days, min_names=min_names,
                           dropped=dict(dropped or {}))
    if panel.empty:
        res.notes.append("the panel is empty - run `tvlab chain snapshot` first")
        return res

    frame = add_derived(panel, returns)
    if frame.empty:
        res.notes.append("no panel row has a closed forward window yet")
        for name in wanted:
            res.results.append(FeatureResult(feature=name, label=labels.get(name, name)))
        return res

    unknown = [f for f in wanted if f not in frame.columns]
    if unknown:
        res.notes.append(f"not a panel column: {', '.join(unknown)}")
        wanted = [f for f in wanted if f in frame.columns]

    for name in wanted:
        item = FeatureResult(feature=name, label=labels.get(name, name))
        ics: List[float] = []
        spreads: List[float] = []
        rows = 0
        for _as_of, day in frame.groupby("as_of", sort=True):
            g = day[[name, "fwd_ret"]].dropna()
            if len(g) < min_names:
                continue
            ic = _spearman(g[name].to_numpy(), g["fwd_ret"].to_numpy())
            if not np.isfinite(ic):
                continue
            ics.append(ic)
            spread = _quintile_spread(g[name].to_numpy(), g["fwd_ret"].to_numpy())
            if np.isfinite(spread):
                spreads.append(spread)
            rows += len(g)
        item.days = len(ics)
        item.rows = rows
        if ics:
            arr = np.asarray(ics, dtype=float)
            item.mean_ic = float(arr.mean())
            if len(arr) > 1 and arr.std(ddof=1) > 0:
                item.ic_t = float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))
            item.consistency = (float(max((arr > 0).mean(), (arr < 0).mean()))
                                if item.mean_ic != 0 else float("nan"))
            if spreads:
                item.quintile_spread = float(np.mean(spreads))
        else:
            res.notes.append(f"{name}: no day had {min_names} usable names")
        res.results.append(item)

    if res.dropped.get("censored"):
        res.notes.append(f"{res.dropped['censored']} name-day(s) dropped: forward window "
                         f"not closed yet")
    if res.dropped.get("no_bars"):
        res.notes.append(f"{res.dropped['no_bars']} name-day(s) dropped: no bars available")
    return res


def forward_window_open(panel: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                        today: Optional[date] = None) -> bool:
    """Whether *any* panel day is old enough to have a forward return yet.

    Used to skip a 218-symbol bar fetch on day one. Calendar days, not trading
    days, so it errs toward fetching rather than toward a false "too early".
    """
    if panel.empty:
        return False
    today = today or date.today()
    try:
        oldest = min(date.fromisoformat(str(d)) for d in panel["as_of"])
    except ValueError:
        return True
    return oldest + timedelta(days=int(math.ceil(horizon * 7 / 5))) <= today
