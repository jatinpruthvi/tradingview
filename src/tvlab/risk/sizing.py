"""Position sizing and drawdown governance.

Sizing is the part of a system that actually determines whether you survive.
Two strategies with identical signals can differ by 10x in terminal wealth purely
from sizing, and only one of them blows up.

Rules implemented here
----------------------
1. **Risk-based sizing** - the position is sized so that hitting the stop loses
   exactly ``risk_per_trade`` of *current* equity. Notional is a consequence, not
   an input.
2. **Equity throttle** - once drawdown passes ``dd_throttle_start * max_drawdown``
   size scales down linearly to zero at the hard limit. This makes the drawdown
   limit a smooth governor rather than a cliff you fall off.
3. **Hard caps** - notional cap, leverage cap, and a minimum sensible stop
   distance so a 1-tick stop cannot produce an infinite position.
4. **Daily loss limit** - after ``daily_loss_limit_pct`` of the day's starting
   equity is lost, no new positions are opened until the next session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import RiskConfig


@dataclass(frozen=True)
class SizingResult:
    """Outcome of a sizing decision, with the reason when it is zero."""

    units: float
    notional: float
    risk_amount: float
    throttle: float
    reason: str = "ok"
    clipped: bool = False
    unclipped_notional: float = 0.0

    @property
    def accepted(self) -> bool:
        # units are SIGNED (negative for shorts), so test the magnitude
        return abs(self.units) > 0


class Sizer:
    """Fixed-fractional, ATR-aware position sizer with a drawdown throttle."""

    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def throttle_at(self, drawdown: float) -> float:
        """Linear exposure throttle in [0, 1] as a function of current drawdown.

        ``drawdown`` is a positive fraction (0.05 = 5% below the equity peak).
        Returns 1.0 until the soft threshold, then ramps to 0.0 at the hard limit.
        """
        dd = abs(float(drawdown))
        hard = max(self.risk.max_drawdown_pct, 1e-9)
        soft = hard * self.risk.dd_throttle_start
        if dd <= soft:
            return 1.0
        if dd >= hard:
            return 0.0
        return float((hard - dd) / (hard - soft))

    def size(
        self,
        equity: float,
        price: float,
        stop: float,
        side: int,
        peak_equity: Optional[float] = None,
        drawdown: Optional[float] = None,
        atr: Optional[float] = None,
        min_units: float = 0.0,
    ) -> SizingResult:
        """Compute units to buy/sell so that a stop-out loses ``risk_per_trade``.

        Parameters
        ----------
        equity:
            current equity (cash + open-position mark-to-market).
        price:
            intended entry price.
        stop:
            stop-loss price.
        side:
            +1 long, -1 short.
        peak_equity / drawdown:
            either may be given; ``drawdown`` wins if both are.
        atr:
            used only to reject stops that are unreasonably tight relative to noise.
        min_units:
            exchange lot size / minimum order. Below this the trade is skipped
            rather than rounded into a meaningless position.
        """
        if equity <= 0 or price <= 0 or not np.isfinite(price):
            return SizingResult(0.0, 0.0, 0.0, 0.0, "invalid_inputs")

        if side not in (1, -1):
            return SizingResult(0.0, 0.0, 0.0, 0.0, "no_signal")
        if side == -1 and not self.risk.allow_short:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "shorts_disabled")

        # ---- stop distance, with sanity bounds -------------------------------
        # An infinite stop is the explicit "no stop" sentinel used by
        # buy-and-hold style strategies: size on the notional cap instead, since
        # there is no risk distance to divide by.
        dist = (price - stop) * side
        if np.isinf(dist):
            max_notional = equity * self.risk.max_position_pct * self.risk.max_leverage
            units = max_notional / price
            if min_units > 0 and units < min_units:
                return SizingResult(0.0, 0.0, 0.0, 1.0, "below_min_lot")
            return SizingResult(units=float(units * side), notional=float(units * price),
                                risk_amount=float("nan"), throttle=1.0, reason="ok")
        if not np.isfinite(dist) or dist <= 0:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "stop_on_wrong_side")

        min_dist = price * self.risk.min_stop_distance_pct
        max_dist = price * self.risk.max_stop_distance_pct
        if dist < min_dist:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "stop_too_tight")
        if dist > max_dist:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "stop_too_wide")

        if atr is not None and np.isfinite(atr) and atr > 0 and dist < 0.05 * atr:
            return SizingResult(0.0, 0.0, 0.0, 0.0, "stop_inside_noise")

        # ---- drawdown throttle ----------------------------------------------
        if drawdown is None:
            peak = peak_equity if peak_equity is not None else equity
            drawdown = max(0.0, 1.0 - equity / peak) if peak > 0 else 0.0
        throttle = self.throttle_at(drawdown)
        if throttle <= 0.0:
            return SizingResult(0.0, 0.0, 0.0, throttle, "dd_limit_reached")

        # ---- risk-based units -------------------------------------------------
        risk_amount = equity * self.risk.risk_per_trade * throttle
        units = risk_amount / dist

        # ---- caps: notional and leverage -------------------------------------
        max_notional = equity * self.risk.max_position_pct * self.risk.max_leverage * throttle
        unclipped_notional = units * price
        notional = unclipped_notional
        clipped = False
        if notional > max_notional and notional > 0:
            units = max_notional / price
            notional = units * price
            clipped = True

        if units <= 0 or not np.isfinite(units):
            return SizingResult(0.0, 0.0, 0.0, throttle, "zero_size")
        if min_units > 0 and units < min_units:
            return SizingResult(0.0, 0.0, 0.0, throttle, "below_min_lot")

        return SizingResult(
            units=float(units * side),
            notional=float(abs(units) * price),
            risk_amount=float(abs(units) * dist),
            throttle=float(throttle),
            clipped=clipped,
            unclipped_notional=float(unclipped_notional),
        )


class RiskGovernor:
    """Tracks the account-level circuit breakers across a backtest or live run.

    Deliberately stateful and explicit: the engine asks ``can_open()`` before
    every entry and calls ``on_fill()`` / ``on_bar()`` so the governor sees the
    same information a live risk desk would.
    """

    def __init__(self, risk: RiskConfig, initial_capital: float):
        self.risk = risk
        self.initial_capital = float(initial_capital)
        self.peak_equity = float(initial_capital)
        self.equity = float(initial_capital)
        self.day: Optional[pd.Timestamp] = None
        self.day_start_equity = float(initial_capital)
        self.day_loss_locked = False
        self.halted = False
        self.halt_reason: str = ""
        self.open_positions = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ updates
    def on_bar(self, ts: pd.Timestamp, equity: float, open_positions: int = 0) -> None:
        self.equity = float(equity)
        self.open_positions = int(open_positions)
        self.peak_equity = max(self.peak_equity, self.equity)

        d = ts.normalize()
        if self.day is None or d != self.day:
            self.day = d
            self.day_start_equity = self.equity
            self.day_loss_locked = False

        if not self.halted:
            dd = self.drawdown
            if dd >= self.risk.max_drawdown_pct:
                self.halted = True
                self.halt_reason = (
                    f"max drawdown breached: {dd:.2%} >= {self.risk.max_drawdown_pct:.2%}"
                )
                self.history.append({"ts": ts, "event": "halt", "reason": self.halt_reason})
            else:
                day_loss = self.day_start_equity - self.equity
                limit = self.day_start_equity * self.risk.daily_loss_limit_pct
                if day_loss >= limit and not self.day_loss_locked:
                    self.day_loss_locked = True
                    self.history.append({
                        "ts": ts, "event": "day_lock",
                        "reason": f"daily loss {day_loss:.2f} >= {limit:.2f}",
                    })

    # ------------------------------------------------------------------ queries
    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    def can_open(self) -> tuple[bool, str]:
        if self.halted:
            return False, self.halt_reason or "halted"
        if self.day_loss_locked:
            return False, "daily_loss_limit"
        if self.open_positions >= self.risk.max_open_positions:
            return False, "max_open_positions"
        return True, "ok"

    def throttle(self) -> float:
        return Sizer(self.risk).throttle_at(self.drawdown)

    def snapshot(self) -> dict:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "day_loss_locked": self.day_loss_locked,
            "open_positions": self.open_positions,
            "throttle": self.throttle(),
            "events": len(self.history),
        }
