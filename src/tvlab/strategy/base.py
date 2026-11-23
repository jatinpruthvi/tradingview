"""Strategy protocol and the registry.

A strategy turns an OHLCV frame (plus an optional higher-timeframe bias frame)
into a *signal frame* with the same index:

===========  ==========================================================
``signal``   +1 want long, -1 want short, 0 nothing
``stop``     stop-loss price for an entry taken on this bar
``target``   take-profit price
``tag``      short label, used to break down P&L by setup in the report
===========  ==========================================================

The engine consumes ``signal[i]`` and acts on bar ``i+1``'s open, so strategies
must not pre-shift their own signals - doing so double-counts the lag.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REGISTRY: Dict[str, Type["Strategy"]] = {}


def register(name: str) -> Callable[[Type["Strategy"]], Type["Strategy"]]:
    def deco(cls: Type["Strategy"]) -> Type["Strategy"]:
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def get(name: str) -> Type["Strategy"]:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available() -> List[str]:
    return sorted(REGISTRY)


class Strategy(ABC):
    """Base class. Subclasses implement :meth:`signals` and declare a param grid."""

    name: str = "base"

    #: parameter defaults; overridden by config/styles/*.yaml
    defaults: Dict[str, Any] = {}

    #: risk-config overrides applied only when running this strategy
    #: (e.g. buy-and-hold needs the notional cap lifted to be a fair benchmark)
    risk_overrides: Dict[str, Any] = {}

    #: search space for walk-forward optimisation: name -> tuple of candidates.
    #: Keep it SMALL. Every extra parameter multiplies the overfitting surface;
    #: the walk-forward module reports how many trials were run so the deflated
    #: Sharpe can penalise the search.
    grid: Dict[str, tuple] = {}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged = dict(self.defaults)
        merged.update(params or {})
        unknown = set(merged) - set(self.defaults)
        if unknown:
            log.warning("%s: ignoring unknown params %s", self.name, sorted(unknown))
            merged = {k: v for k, v in merged.items() if k in self.defaults}
        self.params = merged
        for k, v in merged.items():
            # Never let a parameter shadow a method or dunder: a param named
            # "max_bars_in_trade" silently replaced the method of the same name
            # and produced "'int' object is not callable" deep in the engine.
            existing = getattr(type(self), k, None)
            if callable(existing) or k.startswith("_"):
                log.debug("%s: param %r kept in .params only (shadows a method)", self.name, k)
                continue
            setattr(self, k, v)

    @abstractmethod
    def signals(self, df: pd.DataFrame, bias: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return the signal frame. Must be causal - see tvlab.validation.lookahead."""

    # ------------------------------------------------------------------- utils
    def warmup(self) -> int:
        """Bars needed before the first valid signal. Used to trim the sample."""
        vals = [v for v in self.params.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1]
        return int(max(vals) * 2) if vals else 50

    def resolve_max_bars(self, fallback: Optional[int]) -> Optional[int]:
        """Per-strategy time exit, falling back to the backtest config.

        A strategy that declares ``max_bars_in_trade: None`` means "never exit on
        time" - buy-and-hold needs exactly that, and the engine distinguishes it
        from "unspecified" with its own ``UNSET`` sentinel.
        """
        if "max_bars_in_trade" in self.params:
            v = self.params["max_bars_in_trade"]
            return int(v) if v is not None else None
        return fallback

    def describe(self) -> Dict[str, Any]:
        return {"strategy": self.name, "params": dict(self.params)}

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


def blank_signals(df: pd.DataFrame) -> pd.DataFrame:
    """An all-flat signal frame with the required columns."""
    return pd.DataFrame(
        {"signal": np.zeros(len(df)), "stop": np.full(len(df), np.nan),
         "target": np.full(len(df), np.nan), "tag": [""] * len(df)},
        index=df.index,
    )


def trend_filter(df: pd.DataFrame, bias: Optional[pd.DataFrame],
                 ema_period: int, adx_period: int, adx_min: float,
                 mode: str = "ema_adx", slope_lookback: int = 5) -> pd.Series:
    """Higher-timeframe trend gate: +1 up, -1 down, 0 no trend.

    Indicators are computed on the **native** bias bars and only the resulting
    state is aligned onto the execution index (shifted one HTF bar, so an
    execution bar never sees an incomplete HTF bar). Computing the EMA on
    forward-filled HTF prices instead would silently shorten the lookback from
    ``ema_period`` days to ``ema_period`` execution bars.

    Modes
    -----
    ``ema_adx``
        close vs EMA **and** ADX >= ``adx_min`` **and** +DI vs -DI. Three
        conditions, so it is flat a large fraction of the time. Correct when you
        want to trade only established trends; too strict for a selective LTF
        trigger, where it can suppress almost every setup.
    ``ema_slope``
        close vs EMA **and** the EMA itself rising/falling over ``slope_lookback``
        bars. The standard "trend" definition, and much less restrictive - this
        is the default for the SMC setup.
    ``ema``
        close vs EMA only.

    Falls back to the execution frame when no bias frame is supplied, so a
    strategy still works with ``timeframe.bias: null``.
    """
    from ..data.base import align_indicator
    from ..indicators.technical import adx, ema

    if mode not in ("ema_adx", "ema_slope", "ema"):
        raise ValueError(f"unknown bias mode {mode!r}; use ema_adx|ema_slope|ema")

    def _state(src: pd.DataFrame) -> pd.Series:
        trend_ema = ema(src["close"], ema_period)
        above = src["close"] > trend_ema
        below = src["close"] < trend_ema

        if mode == "ema_adx":
            d = adx(src, adx_period)
            trending = d["adx"] >= adx_min
            up = above & trending & (d["plus_di"] > d["minus_di"])
            down = below & trending & (d["minus_di"] > d["plus_di"])
        elif mode == "ema_slope":
            slope = trend_ema.diff(int(slope_lookback))
            up = above & (slope > 0)
            down = below & (slope < 0)
        else:
            up, down = above, below

        st = pd.Series(0, index=src.index, dtype="int8")
        st[up.fillna(False).to_numpy()] = 1
        st[down.fillna(False).to_numpy()] = -1
        return st

    usable_bias = (bias is not None and len(bias) >= max(ema_period, adx_period) + 2
                   and bias["close"].notna().any())
    if usable_bias:
        aligned = align_indicator(_state(bias), df.index)
        out = aligned.fillna(0).astype("int8")
        # a legacy pre-aligned bias frame (same index as df) would double-shift;
        # detect and handle it by using it directly
        if out.notna().all() and len(bias) == len(df) and bias.index.equals(df.index):
            out = _state(bias)
        return out.rename("trend")

    return _state(df).rename("trend")
