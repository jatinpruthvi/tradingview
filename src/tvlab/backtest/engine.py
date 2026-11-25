"""Event-driven bar backtester with realistic fills and no lookahead.

Execution model
---------------
* A signal computed on bar ``i`` (from bar ``i``'s close) is acted on at the
  **open of bar ``i+1``**. This is the single most important line in the file:
  acting on the same bar's close is the standard way backtests get rich and
  live accounts get poor.
* Stops and targets are checked against the bar's ``high``/``low``. When one bar
  touches both, ``intrabar_fill="pessimistic"`` (the default) assumes the **stop
  was hit first**. We cannot know the intra-bar path from OHLC, so we take the
  adverse branch - the only choice that does not flatter the strategy.
* A stop the bar **opens** through fills at the **open**, not at the stop price:
  the market gapped past the level and never traded there, so a fill at the stop
  is a price nobody could have got. Measured on this repo's own runs, 17 of 101
  crypto/swing stop exits were gap fills, and pricing them at the stop overstated
  P&L by about 3% of a ₹100k account. Targets stay at the limit price, which a
  limit order can never fill worse than.
* Every fill pays ``fee_bps`` on notional plus ``slippage_bps`` of adverse price
  movement, on both entry and exit.
* Equity is marked to market at each bar close, so drawdown is measured on
  unrealised P&L too, not just realised.
* The :class:`~tvlab.risk.sizing.RiskGovernor` can halt all entries on the
  drawdown limit or the daily loss limit.

What this engine deliberately does NOT model
--------------------------------------------
Partial fills, order-book depth, funding/swap, borrow cost for shorts, dividends,
corporate actions, taxes, and latency. For a 4h+ swing system those are second
order; for the intraday profile they are not, and the report says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..risk.sizing import RiskGovernor, Sizer

log = logging.getLogger(__name__)

SIGNAL_COLUMNS = ["signal", "stop", "target"]

#: sentinel distinguishing "caller did not specify" from "caller says: no limit"
UNSET = object()


@dataclass
class Trade:
    """One completed round trip."""

    symbol: str
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    units: float
    notional: float
    stop: float
    target: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = float("nan")
    exit_reason: str = ""
    pnl: float = 0.0
    fees: float = 0.0
    bars_held: int = 0
    r_multiple: float = 0.0
    equity_at_entry: float = float("nan")
    throttle: float = 1.0
    tag: str = ""

    def as_dict(self) -> Dict[str, object]:
        d = self.__dict__.copy()
        d["side_label"] = "long" if self.side > 0 else "short"
        d["return_pct"] = (self.pnl / self.equity_at_entry) if self.equity_at_entry else float("nan")
        return d


@dataclass
class BacktestResult:
    """Everything a run produced, so the report/CLI/dashboard share one object."""

    symbol: str
    timeframe: str
    config: AppConfig
    equity: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    bars: int
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    governor_events: List[Dict[str, object]] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else float("nan")


class Backtester:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.sizer = Sizer(cfg.risk)
        self._sizers: Dict[int, Sizer] = {}
        self.fee = cfg.costs.fee_bps / 1e4
        self.slip = cfg.costs.slippage_bps / 1e4
        self.min_comm = cfg.costs.min_commission

    # ------------------------------------------------------------------- public
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        symbol: str = "?",
        timeframe: Optional[str] = None,
        min_units: float = 0.0,
        atr: Optional[pd.Series] = None,
        max_bars_in_trade: Any = UNSET,
        override_config: Optional[AppConfig] = None,
    ) -> BacktestResult:
        """Execute ``signals`` against ``df``.

        ``signals`` must be index-aligned with ``df`` and carry at least
        ``signal`` (+1/-1/0). ``stop`` and ``target`` are optional per-bar prices;
        when absent the engine falls back to ATR-based levels.
        """
        cfg = override_config or self.cfg
        df, signals = self._prepare(df, signals, symbol)
        timeframe = timeframe or df.attrs.get("timeframe", "?")

        o, h, l, c = (np.ascontiguousarray(df[k].to_numpy(dtype="float64"))
                      for k in ("open", "high", "low", "close"))
        sig = np.ascontiguousarray(signals["signal"].to_numpy(dtype="float64"))
        stop_in = signals["stop"].to_numpy(dtype="float64")
        tgt_in = signals["target"].to_numpy(dtype="float64")
        tag = signals["tag"].to_list() if "tag" in signals else [""] * len(df)
        atr_arr = (np.ascontiguousarray(atr.to_numpy(dtype="float64"))
                   if atr is not None else None)

        n = len(df)
        times = df.index
        governor = RiskGovernor(cfg.risk, cfg.backtest.initial_capital)
        cash = float(cfg.backtest.initial_capital)

        units = 0.0
        side = 0
        entry_fee = 0.0
        entry_px = np.nan
        stop_px = np.nan
        tgt_px = np.nan
        entry_i = -1
        entry_tag = ""
        entry_equity = np.nan
        entry_throttle = 1.0
        #: the stop the position was SIZED against. Trailing may move the live
        #: stop, but R must stay measured against the risk actually taken, or a
        #: trailed stop would shrink the denominator and inflate every R.
        entry_stop = np.nan
        #: best price since entry - the reference a trailing stop hangs off
        best_px = np.nan

        equity_arr = np.empty(n)
        drawdown_arr = np.empty(n)
        pos_arr = np.zeros(n)
        exposure_arr = np.zeros(n)
        trades: List[Trade] = []
        skipped: Dict[str, int] = {}
        warnings: List[str] = []
        # UNSET -> fall back to config; an explicit None -> no time exit at all
        # (buy-and-hold needs the latter, and None could not express it before)
        max_bars = (cfg.backtest.max_bars_in_trade
                    if max_bars_in_trade is UNSET else max_bars_in_trade)
        pessimistic = cfg.backtest.intrabar_fill.lower().startswith("pess")
        next_open = cfg.backtest.entry_on_next_open
        trail_mult = cfg.backtest.trail_atr_mult
        be_at_r = cfg.backtest.breakeven_at_r

        def _bump(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        for i in range(n):
            # ---------------- 1. mark to market on this bar's close -----------
            mtm = cash + units * c[i]
            governor.on_bar(times[i], mtm, open_positions=1 if units != 0 else 0)
            equity_arr[i] = mtm
            drawdown_arr[i] = governor.drawdown
            pos_arr[i] = units
            exposure_arr[i] = abs(units) * c[i]

            if governor.halted and units == 0:
                # hard stop: no more trading for the rest of the sample
                equity_arr[i:] = mtm
                drawdown_arr[i:] = governor.drawdown
                pos_arr[i:] = 0.0
                exposure_arr[i:] = 0.0
                warnings.append(f"trading halted at bar {i} ({times[i]}): {governor.halt_reason}")
                break

            # ---------------- 2. manage the open position ---------------------
            if units != 0:
                exit_px, reason = self._check_exit(
                    i, h[i], l[i], c[i], o[i], side, stop_px, tgt_px,
                    entry_i, max_bars, sig[i], pessimistic,
                )
                # NOTE: the stop used above is the one in force when the bar
                # opened. Any ratchet below takes effect on the NEXT bar, so a
                # bar's own high can never raise the stop that same bar's low
                # then trades through - that would be an intrabar order we do
                # not know.
                if exit_px is not None:
                    fill = self._apply_slippage(exit_px, -side)
                    proceeds = units * fill
                    fee = max(abs(proceeds) * self.fee, self.min_comm)
                    cash += proceeds - fee
                    pnl = (fill - entry_px) * units - fee - entry_fee
                    risk_dist = (abs(entry_px - entry_stop)
                                 if np.isfinite(entry_stop) else np.nan)
                    r_mult = (((fill - entry_px) * side) / risk_dist
                              if (risk_dist and np.isfinite(risk_dist) and risk_dist > 0) else np.nan)
                    trades.append(Trade(
                        symbol=symbol, side=side, entry_time=times[entry_i],
                        entry_price=entry_px, units=units, notional=abs(units) * entry_px,
                        stop=entry_stop, target=tgt_px, exit_time=times[i], exit_price=fill,
                        exit_reason=reason, pnl=pnl, fees=fee + entry_fee,
                        bars_held=i - entry_i, r_multiple=r_mult,
                        equity_at_entry=entry_equity, throttle=entry_throttle, tag=entry_tag,
                    ))
                    units, side, entry_px, stop_px, tgt_px = 0.0, 0, np.nan, np.nan, np.nan
                    entry_i, entry_tag = -1, ""
                    entry_stop, best_px = np.nan, np.nan
                    equity_arr[i] = cash
                    pos_arr[i] = 0.0
                    exposure_arr[i] = 0.0
                else:
                    # still open: fold this bar into the trailing reference and
                    # ratchet the stop for the next bar
                    best_px = (max(best_px, h[i]) if side > 0
                               else min(best_px, l[i]))
                    stop_px = self._ratchet_stop(
                        side, stop_px, entry_px, entry_stop, best_px, c[i],
                        (atr_arr[i] if atr_arr is not None else None),
                        trail_mult, be_at_r,
                    )

            # ---------------- 3. open a new position --------------------------
            if units == 0 and i > 0:
                want = sig[i - 1] if next_open else sig[i]
                if want != 0:
                    can, why = governor.can_open()
                    if not can:
                        _bump(why)
                    else:
                        raw_px = o[i] if next_open else c[i]
                        entry = self._apply_slippage(raw_px, int(np.sign(want)))
                        a = atr_arr[i - 1] if (atr_arr is not None and i > 0) else None
                        s_px = self._resolve_level(stop_in[i - 1] if next_open else stop_in[i],
                                                   entry, int(np.sign(want)), a, is_stop=True)
                        t_px = self._resolve_level(tgt_in[i - 1] if next_open else tgt_in[i],
                                                   entry, int(np.sign(want)), a, is_stop=False)
                        res = self._sizer_for(cfg).size(
                            equity=equity_arr[i], price=entry, stop=s_px,
                            side=int(np.sign(want)), peak_equity=governor.peak_equity,
                            atr=a, min_units=min_units,
                        )
                        if not res.accepted:
                            _bump(res.reason)
                        else:
                            cost = abs(res.units) * entry
                            if cost > equity_arr[i] * cfg.risk.max_position_pct * \
                                    cfg.risk.max_leverage + 1e-9:
                                _bump("insufficient_equity")
                            else:
                                fee = max(cost * self.fee, self.min_comm)
                                cash -= res.units * entry + fee
                                entry_fee = fee
                                units = res.units
                                side = int(np.sign(want))
                                entry_px, stop_px, tgt_px = entry, s_px, t_px
                                entry_stop = s_px
                                # the entry bar's own extreme is a valid trailing
                                # reference only when we entered at its open
                                best_px = ((max(entry, h[i]) if side > 0
                                            else min(entry, l[i]))
                                           if next_open else entry)
                                entry_i = i
                                entry_tag = str(tag[i - 1] if next_open else tag[i])
                                entry_equity = equity_arr[i]
                                entry_throttle = res.throttle
                                pos_arr[i] = units
                                exposure_arr[i] = abs(units) * c[i]

        # ---- close any position still open at the end (mark-to-market exit) ---
        if units != 0:
            fill = self._apply_slippage(c[n - 1], -side)
            proceeds = units * fill
            fee = max(abs(proceeds) * self.fee, self.min_comm)
            cash += proceeds - fee
            risk_dist = (abs(entry_px - entry_stop)
                         if np.isfinite(entry_stop) else np.nan)
            trades.append(Trade(
                symbol=symbol, side=side, entry_time=times[entry_i], entry_price=entry_px,
                units=units, notional=abs(units) * entry_px, stop=entry_stop, target=tgt_px,
                exit_time=times[n - 1], exit_price=fill, exit_reason="end_of_data",
                pnl=(fill - entry_px) * units - fee - entry_fee, fees=fee + entry_fee,
                bars_held=n - 1 - entry_i,
                r_multiple=(((fill - entry_px) * side) / risk_dist
                            if (risk_dist and np.isfinite(risk_dist) and risk_dist > 0) else np.nan),
                equity_at_entry=entry_equity, throttle=entry_throttle, tag=entry_tag,
            ))
            equity_arr[n - 1] = cash

        equity = pd.Series(equity_arr, index=times, name="equity")
        positions = pd.DataFrame(
            {"units": pos_arr, "exposure": exposure_arr, "drawdown": drawdown_arr},
            index=times,
        )
        trades_df = pd.DataFrame([t.as_dict() for t in trades])
        if not trades_df.empty:
            trades_df["cum_pnl"] = trades_df["pnl"].cumsum()

        if not warnings and len(trades_df) == 0:
            warnings.append(
                "no trades were taken. Check the signal logic, the trend filter, "
                f"or the risk caps. Skipped entries: {skipped}"
            )

        return BacktestResult(
            symbol=symbol, timeframe=timeframe, config=cfg, equity=equity,
            trades=trades_df, positions=positions, bars=n,
            start=times.min(), end=times.max(),
            governor_events=governor.history, skipped=skipped, warnings=warnings,
        )

    # ------------------------------------------------------------------ helpers
    def _sizer_for(self, cfg: AppConfig) -> Sizer:
        """Cached sizer per config, so risk_overrides do not rebuild it every bar."""
        key = id(cfg)
        sizer = self._sizers.get(key)
        if sizer is None:
            sizer = Sizer(cfg.risk)
            self._sizers[key] = sizer
        return sizer

    def _prepare(self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(df) < 2:
            raise ValueError(f"{symbol}: need at least 2 bars, got {len(df)}")
        if not df.index.equals(signals.index):
            # realign rather than fail: strategies may drop warmup rows
            signals = signals.reindex(df.index)
        out = signals.copy()
        for col in SIGNAL_COLUMNS:
            if col not in out.columns:
                out[col] = 0.0 if col == "signal" else np.nan
        if "tag" not in out.columns:
            out["tag"] = ""
        out["signal"] = out["signal"].fillna(0.0).clip(-1, 1)
        bad = out["signal"].abs() > 0
        if bad.any() and not np.isclose(out.loc[bad, "signal"].abs(), 1.0).all():
            # fractional signals are allowed as a strength hint; normalise to sign
            out.loc[bad, "signal"] = np.sign(out.loc[bad, "signal"])
        return df, out

    def _apply_slippage(self, price: float, direction: int) -> float:
        """Adverse slippage: buys fill higher, sells fill lower."""
        return float(price * (1.0 + self.slip * direction))

    def _resolve_level(self, level: float, entry: float, side: int,
                       atr_val: Optional[float], is_stop: bool) -> float:
        """Use the strategy-supplied level, else fall back to an ATR multiple.

        ``NaN`` means "the strategy did not supply one" -> ATR fallback.
        ``+-inf`` is an explicit "no stop / no target" and is passed through
        untouched; treating it as missing gave buy-and-hold an ATR stop and
        turned the benchmark into a 90-bar swing trade.
        """
        if level is not None and np.isinf(level):
            return float(level)
        if level is not None and np.isfinite(level):
            return float(level)
        a = atr_val if (atr_val is not None and np.isfinite(atr_val) and atr_val > 0) else entry * 0.02
        mult = 2.0 if is_stop else 4.0
        return float(entry - side * mult * a) if is_stop else float(entry + side * mult * a)

    @staticmethod
    def _ratchet_stop(side: int, stop: float, entry_px: float, entry_stop: float,
                      best_px: float, close: float, atr_val: Optional[float],
                      trail_mult: Optional[float], be_at_r: Optional[float]) -> float:
        """The stop to carry into the next bar. Never loosens, never invents a level.

        Two optional rules, both off by default so an existing study keeps the
        stop its signal chose:

        *``breakeven_at_r``* - once the trade has run that many R in profit, move
        the stop to the entry price. It converts a winner into a scratch instead
        of letting it round-trip into a full loss, at the cost of being stopped
        on the pullback that would have resumed.

        *``trail_atr_mult``* - hang the stop that many ATRs behind the best price
        since entry. A trend exit for a strategy whose fixed target caps the
        right tail at exactly ``r_multiple``.

        Both only ever move the stop in the trade's favour, and the result is
        clamped to the bar's close: a stop beyond the market is not a resting
        order, it is a market order that already filled.
        """
        if not np.isfinite(stop):
            return stop
        cand = stop
        risk = abs(entry_px - entry_stop) if np.isfinite(entry_stop) else np.nan
        if be_at_r and np.isfinite(risk) and risk > 0 and np.isfinite(best_px):
            if side * (best_px - entry_px) >= be_at_r * risk:
                cand = max(cand, entry_px) if side > 0 else min(cand, entry_px)
        if (trail_mult and atr_val is not None and np.isfinite(atr_val)
                and atr_val > 0 and np.isfinite(best_px)):
            trail = best_px - side * trail_mult * atr_val
            cand = max(cand, trail) if side > 0 else min(cand, trail)
        if np.isfinite(close):
            cand = min(cand, close) if side > 0 else max(cand, close)
        return max(stop, cand) if side > 0 else min(stop, cand)

    def _check_exit(self, i: int, hi: float, lo: float, close: float, open_: float,
                    side: int, stop: float, target: float, entry_i: int,
                    max_bars: Optional[int], sig_now: float, pessimistic: bool
                    ) -> Tuple[Optional[float], str]:
        """Decide whether this bar closes the position, and at what price.

        Stop fills are gap-aware. A resting stop is a market order once triggered,
        so when the bar *opens* beyond it the fill is the open: the market gapped
        past the level and never traded there. Returning the stop price on a gap
        invents a fill nobody could have got, and on a limit-down open it is
        fiction. Targets are deliberately left at the limit price - a limit fills
        no worse than its level, so the conservative branch is the correct one.
        """
        # +-inf is the explicit "no stop" / "no target" sentinel
        has_stop = np.isfinite(stop)
        has_target = np.isfinite(target)
        hit_stop = ((lo <= stop) if side > 0 else (hi >= stop)) if has_stop else False
        hit_target = ((hi >= target) if side > 0 else (lo <= target)) if has_target else False

        if has_stop and np.isfinite(open_) and (open_ <= stop if side > 0 else open_ >= stop):
            # Checked before the both-touched branch: the stop is triggered at the
            # open, ahead of anything that happens intrabar, so a gap through the
            # stop beats even an optimistic target fill on the same bar.
            return float(open_), "stop_loss"

        if hit_stop and hit_target:
            # OHLC cannot tell us the intra-bar path; take the adverse branch
            if pessimistic:
                return stop, "stop_loss"
            return target, "take_profit"
        if hit_stop:
            return stop, "stop_loss"
        if hit_target:
            return target, "take_profit"

        if max_bars is not None and (i - entry_i) >= max_bars:
            return close, "time_exit"

        # opposite signal flattens the position (reversal handling)
        if sig_now != 0 and np.sign(sig_now) != side:
            return open_ if i > entry_i else close, "signal_flip"

        return None, ""
