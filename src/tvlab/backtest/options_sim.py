"""Causal Black-Scholes option-buying simulator for underlying bar signals.

Why this exists
---------------
``Backtester`` simulates linear Delta=1 spot/futures P&L. An option buyer
faces non-linear Gamma, continuous Theta decay, implied-volatility markups
above realised volatility (IV > RV), and much wider percentage spreads on
option premium. Without modelling those four forces, backtesting an ORB signal
on spot bars tells you nothing about whether buying Calls/Puts makes money.

This module replays any causal strategy's signal frame (such as ``orb`` with
``min_range_pct`` and ``max_entry_tod``) into synthetic ATM / ITM / OTM Calls
and Puts priced off strictly causal trailing realised volatility (plus an
explicit ``iv_rv_markup``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..data.base import bars_per_year
from ..data.store import DataStore
from ..pipeline import build_strategy
from .metrics import full_report


# ------------------------------------------------------------ Black-Scholes
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


def norm_inv_cdf(p: float) -> float:
    """Acklam's rational approximation to the standard normal quantile (~1e-9 accuracy)."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    a = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def bs_price(spot: float, strike: float, t_years: float, vol: float,
             side: int, rate: float = 0.065) -> float:
    """Black-Scholes European option price (side=+1 Call, side=-1 Put)."""
    if spot <= 0 or strike <= 0:
        return 0.0
    intrinsic = max(0.0, side * (spot - strike))
    if t_years <= 1e-7 or vol <= 1e-6:
        return intrinsic
    sqrt_t = math.sqrt(t_years)
    v_sqrt_t = vol * sqrt_t
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / v_sqrt_t
    d2 = d1 - v_sqrt_t
    disc = math.exp(-rate * t_years)
    if side > 0:
        px = spot * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    else:
        px = strike * disc * norm_cdf(-d2) - spot * norm_cdf(-d1)
    return max(px, intrinsic * 0.999)


def bs_delta(spot: float, strike: float, t_years: float, vol: float,
             side: int, rate: float = 0.065) -> float:
    """Option delta (+0..+1 for Call, -1..0 for Put)."""
    if t_years <= 1e-7 or vol <= 1e-6 or spot <= 0 or strike <= 0:
        if side > 0:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    nd1 = norm_cdf(d1)
    return nd1 if side > 0 else (nd1 - 1.0)


def strike_for_delta(spot: float, t_years: float, vol: float,
                     target_abs_delta: float, side: int,
                     rate: float = 0.065) -> float:
    """Solve Black-Scholes strike corresponding to ``target_abs_delta``."""
    delta = min(max(float(target_abs_delta), 0.05), 0.95)
    if t_years <= 1e-7 or vol <= 1e-6:
        return float(spot)
    d1 = norm_inv_cdf(delta) if side > 0 else norm_inv_cdf(1.0 - delta)
    sqrt_t = math.sqrt(t_years)
    exponent = (rate + 0.5 * vol * vol) * t_years - d1 * vol * sqrt_t
    return float(spot * math.exp(exponent))


# ----------------------------------------------------------- Volatility model
def causal_session_rv(df: pd.DataFrame, window_sessions: int = 20,
                      floor: float = 0.10, cap: float = 1.00) -> np.ndarray:
    """Strictly causal annualised daily realised vol aligned to each bar.

    Uses only completed prior sessions (shifted 1 full session so today's
    bars never peek at today's close). Falls back to 0.18 if fewer than 3 prior
    sessions exist.
    """
    n = len(df)
    out = np.full(n, 0.18, dtype="float64")
    if n == 0 or not isinstance(df.index, pd.DatetimeIndex):
        return out
    idx = df.index
    day = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    daily_close = df["close"].groupby(day).last()
    if len(daily_close) < 4:
        return out
    log_ret = np.log(daily_close / daily_close.shift(1))
    # Shift 1 session so session D uses volatility through session D-1 only.
    rv = (log_ret.rolling(max(int(window_sessions), 3), min_periods=3).std(ddof=1) * math.sqrt(252.0)).shift(1)
    aligned = rv.reindex(day).to_numpy(dtype="float64")
    valid = np.isfinite(aligned) & (aligned > 0)
    out[valid] = np.clip(aligned[valid], floor, cap)
    return out


# ------------------------------------------------------------- Simulator
@dataclass(frozen=True)
class OptionSimConfig:
    """Parameters governing synthetic long-option execution and decay."""

    name: str = "atm_7dte"
    target_delta: float = 0.55       # 0.50 ATM, 0.65 1-ITM, 0.30 OTM
    dte_days: float = 7.0            # days to expiry at entry
    iv_rv_markup: float = 1.05       # implied vol = iv_rv_markup * causal_rv20
    rate: float = 0.065              # risk-free rate
    premium_stop_pct: float = 0.40   # stop out if option loses 40% of premium
    premium_target_mult: Optional[float] = 2.0  # take profit at 2.0x (+100%) or None
    max_hold_bars: Optional[int] = 12  # time stop (12 x 5m = 60 min) or None
    use_underlying_stop: bool = True   # also stop out if underlying breaches signal stop
    option_fee_bps: float = 15.0     # per side on option premium (brokerage + STT/exchange)
    option_spread_bps: float = 35.0  # per side half-spread slippage on option premium
    max_premium_pct: float = 0.05    # cap premium spent per trade at 5% of equity
    min_premium: float = 0.5         # ignore near-zero deep-OTM dust
    session_hours_per_day: float = 6.25  # used only for diagnostics


@dataclass
class OptionSimResult:
    config: OptionSimConfig
    symbol: str
    equity: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)


def simulate_options(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: AppConfig,
    opt_cfg: OptionSimConfig,
    symbol: str = "?",
    timeframe: Optional[str] = None,
) -> OptionSimResult:
    """Simulate long Call (+1) / long Put (-1) trades from ``signals`` on ``df``."""
    timeframe = timeframe or cfg.timeframe.execution
    if not df.index.equals(signals.index):
        signals = signals.reindex(df.index)

    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    sig = signals["signal"].fillna(0.0).to_numpy(dtype="float64")
    stop_in = signals["stop"].to_numpy(dtype="float64") if "stop" in signals else np.full(len(df), np.nan)
    exit_sig = signals["exit"].fillna(0.0).to_numpy(dtype="float64") if "exit" in signals else np.zeros(len(df))
    times = df.index
    n = len(df)

    rv_arr = causal_session_rv(df, window_sessions=20)
    fee_rate = opt_cfg.option_fee_bps / 1e4
    slip_rate = opt_cfg.option_spread_bps / 1e4
    next_open = cfg.backtest.entry_on_next_open
    risk_pct = cfg.risk.risk_per_trade

    cash = float(cfg.backtest.initial_capital)
    eq_arr = np.empty(n, dtype="float64")
    trades: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    in_pos = False
    side = 0
    strike = 0.0
    iv_entry = 0.0
    t_entry_yrs = opt_cfg.dte_days / 365.0
    entry_i = -1
    entry_opt_px = 0.0
    entry_spot = 0.0
    und_stop = np.nan
    contracts = 0.0
    entry_fee = 0.0
    entry_eq = cash
    delta_entry = 0.0

    def _bump(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    def _remaining_t(bar_i: int) -> float:
        elapsed_days = max(0.0, (times[bar_i] - times[entry_i]).total_seconds() / 86400.0)
        return max(1e-7, t_entry_yrs - elapsed_days / 365.0)

    for i in range(n):
        if in_pos:
            t_rem = _remaining_t(i)
            mark_px = bs_price(c[i], strike, t_rem, iv_entry, side, opt_cfg.rate)
            eq_arr[i] = cash + contracts * mark_px

            # ---- check exit conditions (pessimistic intrabar ordering) -----
            exit_opt: Optional[float] = None
            reason = ""

            prem_stop_px = entry_opt_px * (1.0 - opt_cfg.premium_stop_pct)
            prem_tgt_px = (entry_opt_px * opt_cfg.premium_target_mult
                           if opt_cfg.premium_target_mult is not None else np.inf)

            # 1. Gap at open through underlying stop or premium stop
            open_opt = bs_price(o[i], strike, t_rem, iv_entry, side, opt_cfg.rate)
            if opt_cfg.use_underlying_stop and np.isfinite(und_stop):
                if (side > 0 and o[i] <= und_stop) or (side < 0 and o[i] >= und_stop):
                    exit_opt, reason = open_opt, "underlying_stop_gap"
            if exit_opt is None and open_opt <= prem_stop_px:
                exit_opt, reason = open_opt, "premium_stop_gap"

            # 2. Intrabar worst-case touch (adverse spot extreme)
            if exit_opt is None:
                worst_spot = l[i] if side > 0 else h[i]
                best_spot = h[i] if side > 0 else l[i]
                worst_opt = bs_price(worst_spot, strike, t_rem, iv_entry, side, opt_cfg.rate)
                best_opt = bs_price(best_spot, strike, t_rem, iv_entry, side, opt_cfg.rate)

                hit_und = (
                    opt_cfg.use_underlying_stop and np.isfinite(und_stop)
                    and ((side > 0 and l[i] <= und_stop) or (side < 0 and h[i] >= und_stop))
                )
                hit_prem_stop = worst_opt <= prem_stop_px
                hit_prem_tgt = np.isfinite(prem_tgt_px) and best_opt >= prem_tgt_px

                if hit_und:
                    exit_opt = bs_price(und_stop, strike, t_rem, iv_entry, side, opt_cfg.rate)
                    reason = "underlying_stop"
                elif hit_prem_stop:
                    exit_opt, reason = prem_stop_px, "premium_stop"
                elif hit_prem_tgt:
                    exit_opt, reason = prem_tgt_px, "premium_target"

            # 3. Session exit or short time stop
            if exit_opt is None and exit_sig[i] > 0:
                exit_opt, reason = mark_px, "session_exit"
            if exit_opt is None and opt_cfg.max_hold_bars is not None and (i - entry_i) >= opt_cfg.max_hold_bars:
                exit_opt, reason = mark_px, "time_exit"

            if exit_opt is not None:
                # Apply adverse option spread + fees on exit (except limit target gets 0 slip)
                if reason == "premium_target":
                    fill_px = float(exit_opt)
                else:
                    fill_px = max(0.01, float(exit_opt) * (1.0 - slip_rate))
                proceeds = contracts * fill_px
                exit_fee = proceeds * fee_rate
                cash += proceeds - exit_fee
                pnl = proceeds - exit_fee - (contracts * entry_opt_px) - entry_fee
                risk_budget = max(contracts * entry_opt_px * opt_cfg.premium_stop_pct, 1e-9)
                trades.append({
                    "symbol": symbol,
                    "side": side,
                    "option_type": "CE" if side > 0 else "PE",
                    "entry_time": times[entry_i],
                    "exit_time": times[i],
                    "entry_spot": entry_spot,
                    "exit_spot": c[i],
                    "strike": strike,
                    "delta_entry": delta_entry,
                    "iv": iv_entry,
                    "entry_price": entry_opt_px,
                    "exit_price": fill_px,
                    "units": contracts,
                    "notional": contracts * entry_opt_px,
                    "pnl": pnl,
                    "fees": entry_fee + exit_fee,
                    "r_multiple": pnl / risk_budget,
                    "bars_held": i - entry_i,
                    "exit_reason": reason,
                    "equity_at_entry": entry_eq,
                })
                in_pos = False
                eq_arr[i] = cash
        else:
            eq_arr[i] = cash

        # ---- open new option trade ----------------------------------------
        if not in_pos and i > 0 and not exit_sig[i]:
            sig_idx = i - 1 if next_open else i
            want = 0.0 if (next_open and exit_sig[i - 1]) else sig[sig_idx]
            if want != 0:
                want_side = int(np.sign(want))
                spot_in = o[i] if next_open else c[i]
                iv = float(rv_arr[sig_idx] * opt_cfg.iv_rv_markup)
                k = strike_for_delta(spot_in, t_entry_yrs, iv, opt_cfg.target_delta, want_side, opt_cfg.rate)
                fair_opt = bs_price(spot_in, k, t_entry_yrs, iv, want_side, opt_cfg.rate)
                if fair_opt < opt_cfg.min_premium:
                    _bump("premium_too_small")
                    continue
                fill_in = fair_opt * (1.0 + slip_rate)
                # Size by premium-at-risk, capped by max_premium_pct of equity
                risk_dollars = eq_arr[i] * risk_pct
                prem_budget = min(
                    risk_dollars / max(opt_cfg.premium_stop_pct, 0.05),
                    eq_arr[i] * opt_cfg.max_premium_pct,
                )
                qty = prem_budget / fill_in if fill_in > 0 else 0.0
                if qty <= 0:
                    _bump("zero_size")
                    continue
                cost = qty * fill_in
                fee_in = cost * fee_rate
                if cost + fee_in > cash:
                    _bump("insufficient_cash")
                    continue

                cash -= cost + fee_in
                in_pos = True
                side = want_side
                strike = k
                iv_entry = iv
                entry_i = i
                entry_opt_px = fill_in
                entry_spot = spot_in
                und_stop = float(stop_in[sig_idx])
                contracts = qty
                entry_fee = fee_in
                entry_eq = eq_arr[i]
                delta_entry = bs_delta(spot_in, k, t_entry_yrs, iv, side, opt_cfg.rate)

    if in_pos:
        t_rem = _remaining_t(n - 1)
        mark_px = bs_price(c[n - 1], strike, t_rem, iv_entry, side, opt_cfg.rate)
        fill_px = max(0.01, mark_px * (1.0 - slip_rate))
        proceeds = contracts * fill_px
        exit_fee = proceeds * fee_rate
        cash += proceeds - exit_fee
        pnl = proceeds - exit_fee - (contracts * entry_opt_px) - entry_fee
        risk_budget = max(contracts * entry_opt_px * opt_cfg.premium_stop_pct, 1e-9)
        trades.append({
            "symbol": symbol,
            "side": side,
            "option_type": "CE" if side > 0 else "PE",
            "entry_time": times[entry_i],
            "exit_time": times[n - 1],
            "entry_spot": entry_spot,
            "exit_spot": c[n - 1],
            "strike": strike,
            "delta_entry": delta_entry,
            "iv": iv_entry,
            "entry_price": entry_opt_px,
            "exit_price": fill_px,
            "units": contracts,
            "notional": contracts * entry_opt_px,
            "pnl": pnl,
            "fees": entry_fee + exit_fee,
            "r_multiple": pnl / risk_budget,
            "bars_held": (n - 1) - entry_i,
            "exit_reason": "end_of_data",
            "equity_at_entry": entry_eq,
        })
        eq_arr[n - 1] = cash

    eq_ser = pd.Series(eq_arr, index=times, name="equity")
    trades_df = pd.DataFrame(trades)
    bpy = bars_per_year(timeframe, cfg.universe.market)
    metrics = full_report(eq_ser, trades_df, periods_per_year=bpy,
                          initial_capital=cfg.backtest.initial_capital)
    return OptionSimResult(config=opt_cfg, symbol=symbol, equity=eq_ser,
                           trades=trades_df, metrics=metrics, skipped=skipped)


def default_option_sweep() -> List[OptionSimConfig]:
    """Structured option-buying hypotheses across strike Delta, DTE, hold, and IV markup."""
    return [
        OptionSimConfig("atm_7dte_60m", target_delta=0.50, dte_days=7.0, max_hold_bars=12, iv_rv_markup=1.05),
        OptionSimConfig("itm65_7dte_60m", target_delta=0.65, dte_days=7.0, max_hold_bars=12, iv_rv_markup=1.05),
        OptionSimConfig("otm30_7dte_60m", target_delta=0.30, dte_days=7.0, max_hold_bars=12, iv_rv_markup=1.05),
        OptionSimConfig("itm65_14dte_60m", target_delta=0.65, dte_days=14.0, max_hold_bars=12, iv_rv_markup=1.05),
        OptionSimConfig("itm65_7dte_30m", target_delta=0.65, dte_days=7.0, max_hold_bars=6, iv_rv_markup=1.05),
        OptionSimConfig("itm65_7dte_eod", target_delta=0.65, dte_days=7.0, max_hold_bars=None, iv_rv_markup=1.05),
        OptionSimConfig("itm65_3dte_60m", target_delta=0.65, dte_days=3.0, max_hold_bars=12, iv_rv_markup=1.05),
        OptionSimConfig("itm65_7dte_rich_iv", target_delta=0.65, dte_days=7.0, max_hold_bars=12, iv_rv_markup=1.20),
    ]


def run_option_study(
    cfg: AppConfig,
    symbols: Optional[List[str]] = None,
    source: Optional[str] = None,
    configs: Optional[Sequence[OptionSimConfig]] = None,
    store: Optional[DataStore] = None,
) -> pd.DataFrame:
    """Evaluate option-buying configs across ``symbols`` with combined equal-weight portfolio."""
    store = store or DataStore(cfg)
    strat = build_strategy(cfg)
    syms = list(symbols or cfg.universe.symbols) or ["^NSEI"]
    tf = cfg.timeframe.execution
    configs = list(configs or default_option_sweep())

    data: Dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sym in syms:
        try:
            df = store.ohlcv(sym, tf, source=source)
            bias = store.bias_frame(sym, source=source)
            sig = strat.signals(df, bias)
            data[sym] = (df, sig)
        except Exception:  # noqa: BLE001
            continue

    if not data:
        return pd.DataFrame()

    bpy = bars_per_year(tf, cfg.universe.market)
    n_trials = max(len(configs), 1)
    init_cap = float(cfg.backtest.initial_capital)
    rows: List[Dict[str, Any]] = []

    for ocfg in configs:
        curves = []
        all_trades = []
        for sym, (df, sig) in data.items():
            res = simulate_options(df, sig, cfg, ocfg, symbol=sym, timeframe=tf)
            curves.append((res.equity / init_cap) * (init_cap / len(data)))
            if not res.trades.empty:
                all_trades.append(res.trades)

        idx = curves[0].index
        for c_ser in curves[1:]:
            idx = idx.union(c_ser.index)
        aligned = [c_ser.reindex(idx).ffill().bfill() for c_ser in curves]
        port_eq = aligned[0]
        for c_ser in aligned[1:]:
            port_eq = port_eq.add(c_ser)
        port_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

        m = full_report(port_eq, port_trades, periods_per_year=bpy,
                        n_trials=n_trials, initial_capital=init_cap)
        rows.append({
            "config": ocfg.name,
            "delta": ocfg.target_delta,
            "dte": ocfg.dte_days,
            "hold_bars": ocfg.max_hold_bars if ocfg.max_hold_bars is not None else "EOD",
            "iv_markup": ocfg.iv_rv_markup,
            "return_pct": m.get("total_return_pct"),
            "sharpe": m.get("sharpe"),
            "dsr": (m.get("deflated_sharpe") or {}).get("dsr"),
            "profit_factor": m.get("profit_factor"),
            "max_dd_pct": m.get("max_drawdown_pct"),
            "trades": m.get("n_trades"),
            "win_rate": m.get("win_rate"),
            "avg_r": m.get("avg_r"),
            "fees": m.get("total_fees"),
        })
    return pd.DataFrame(rows)
