"""Performance metrics.

Conventions
-----------
* Returns are **log** returns for Sharpe/Sortino (additive, no compounding bias)
  and simple returns for the equity-curve statistics.
* Annualisation uses the actual bar spacing and the market's session calendar,
  not a hardcoded 252 - a 4h crypto series has 2190 bars/year, a 4h equity series
  has ~410. Getting this wrong inflates Sharpe by ~2x and is very common.
* Sharpe is reported both raw and **deflated** (Bailey & Lopez de Prado) using
  the number of optimisation trials, because a search over N parameter sets
  guarantees a high maximum Sharpe even on pure noise.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    end = max(float(equity.iloc[-1]), 1e-12)
    return float((end / equity.iloc[0]) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a positive fraction."""
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(-dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown_duration(equity: pd.Series) -> Optional[pd.Timedelta]:
    """Longest time spent below a previous peak (underwater duration)."""
    if len(equity) < 2:
        return None
    peak = equity.cummax()
    under = equity < peak
    if not under.any():
        return pd.Timedelta(0)
    longest = start = None
    best = pd.Timedelta(0)
    for ts, flag in under.items():
        if flag and start is None:
            start = ts
        elif not flag and start is not None:
            span = ts - start
            if longest is None or span > longest:
                longest, best = span, span
            start = None
    if start is not None:
        span = equity.index[-1] - start
        if span > best:
            best = span
    return best


def sharpe(equity: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    """Annualised Sharpe on log returns of the equity curve."""
    if len(equity) < 3:
        return float("nan")
    r = np.log(equity / equity.shift(1)).dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    per_bar_rf = rf / periods_per_year
    return float((r.mean() - per_bar_rf) / sd * math.sqrt(periods_per_year))


def sortino(equity: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    """Annualised Sortino - downside deviation only."""
    if len(equity) < 3:
        return float("nan")
    r = np.log(equity / equity.shift(1)).dropna()
    per_bar_rf = rf / periods_per_year
    downside = r[r < per_bar_rf] - per_bar_rf
    if len(downside) < 2:
        return float("inf") if r.mean() > 0 else float("nan")
    dd = math.sqrt(float((downside**2).mean()))
    if dd <= 0:
        return float("nan")
    return float((r.mean() - per_bar_rf) / dd * math.sqrt(periods_per_year))


def calmar(equity: pd.Series, periods_per_year: float) -> float:
    mdd = max_drawdown(equity)
    if mdd <= 0:
        return float("inf") if cagr(equity, periods_per_year) > 0 else float("nan")
    return float(cagr(equity, periods_per_year) / mdd)


def volatility(equity: pd.Series, periods_per_year: float) -> float:
    r = np.log(equity / equity.shift(1)).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def var_historical(equity: pd.Series, confidence: float = 0.95) -> float:
    r = np.log(equity / equity.shift(1)).dropna()
    if len(r) < 10:
        return float("nan")
    return float(-np.quantile(r, 1 - confidence))


def cvar_historical(equity: pd.Series, confidence: float = 0.95) -> float:
    r = np.log(equity / equity.shift(1)).dropna()
    if len(r) < 10:
        return float("nan")
    tail = r[r <= np.quantile(r, 1 - confidence)]
    return float(-tail.mean()) if len(tail) else float("nan")


# --------------------------------------------------------------------- trades
def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """Per-trade statistics. Empty frames yield NaN, never fake zeros."""
    nan = float("nan")
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": nan, "profit_factor": nan, "expectancy": nan,
            "avg_win": nan, "avg_loss": nan, "payoff_ratio": nan, "avg_r": nan,
            "median_r": nan, "best_r": nan, "worst_r": nan, "avg_bars_held": nan,
            "max_consecutive_losses": 0, "max_consecutive_wins": 0,
            "total_fees": nan, "avg_fee_per_trade": nan, "long_trades": 0, "short_trades": 0,
            "long_win_rate": nan, "short_win_rate": nan, "bars_in_market": 0,
        }

    pnl = trades["pnl"].to_numpy(dtype="float64")
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0

    r = trades["r_multiple"].to_numpy(dtype="float64") if "r_multiple" in trades else np.array([])
    r = r[np.isfinite(r)] if len(r) else r

    def _streaks(flags: np.ndarray) -> int:
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return best

    longs = trades[trades["side"] > 0] if "side" in trades else trades.iloc[0:0]
    shorts = trades[trades["side"] < 0] if "side" in trades else trades.iloc[0:0]

    bars = trades["bars_held"].to_numpy(dtype="float64") if "bars_held" in trades else np.array([])

    return {
        "n_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else nan,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else
                         (float("inf") if gross_win > 0 else nan),
        "expectancy": float(pnl.mean()) if len(pnl) else nan,
        "avg_win": float(wins.mean()) if len(wins) else nan,
        "avg_loss": float(losses.mean()) if len(losses) else nan,
        "payoff_ratio": (float(wins.mean() / abs(losses.mean()))
                         if len(wins) and len(losses) and losses.mean() != 0 else nan),
        "avg_r": float(r.mean()) if len(r) else nan,
        "median_r": float(np.median(r)) if len(r) else nan,
        "best_r": float(r.max()) if len(r) else nan,
        "worst_r": float(r.min()) if len(r) else nan,
        "avg_bars_held": float(bars.mean()) if len(bars) else nan,
        "max_consecutive_losses": _streaks(pnl <= 0),
        "max_consecutive_wins": _streaks(pnl > 0),
        "total_fees": float(trades["fees"].sum()) if "fees" in trades else nan,
        "avg_fee_per_trade": float(trades["fees"].mean()) if "fees" in trades else nan,
        "long_trades": int(len(longs)),
        "short_trades": int(len(shorts)),
        "long_win_rate": float((longs["pnl"] > 0).mean()) if len(longs) else nan,
        "short_win_rate": float((shorts["pnl"] > 0).mean()) if len(shorts) else nan,
        "bars_in_market": float(bars.sum()) if len(bars) else nan,
    }


def exit_breakdown(trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """P&L grouped by exit reason - the fastest way to see which rule is broken."""
    if trades is None or trades.empty or "exit_reason" not in trades:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for reason, grp in trades.groupby("exit_reason"):
        pnl = grp["pnl"].to_numpy(dtype="float64")
        out[str(reason)] = {
            "n": int(len(grp)),
            "pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
            "avg_r": _nanmean(grp["r_multiple"]) if "r_multiple" in grp else float("nan"),
        }
    return out


def _nanmean(s: pd.Series) -> float:
    """Mean that returns NaN instead of warning when every value is NaN."""
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype="float64")
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def tag_breakdown(trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """P&L grouped by setup tag - tells you which entry variant actually works."""
    if trades is None or trades.empty or "tag" not in trades:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for tag, grp in trades.groupby("tag"):
        if not str(tag):
            continue
        pnl = grp["pnl"].to_numpy(dtype="float64")
        out[str(tag)] = {
            "n": int(len(grp)),
            "pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
            "avg_r": _nanmean(grp["r_multiple"]) if "r_multiple" in grp else float("nan"),
            "expectancy": float(pnl.mean()),
        }
    return out


def monthly_returns(equity: pd.Series) -> pd.Series:
    if len(equity) < 2:
        return pd.Series(dtype="float64")
    m = equity.resample("1ME").last().dropna()
    return m.pct_change().dropna()


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Calendar-year x month return grid, as on a fund fact sheet."""
    mr = monthly_returns(equity)
    if mr.empty:
        return pd.DataFrame()
    tbl = pd.DataFrame({"year": mr.index.year, "month": mr.index.month, "ret": mr.to_numpy()})
    wide = tbl.pivot_table(index="year", columns="month", values="ret", aggfunc="last")
    wide.columns = [pd.Timestamp(2000, int(c), 1).strftime("%b") for c in wide.columns]
    return wide


def rolling_sharpe(equity: pd.Series, periods_per_year: float, window: int = 250) -> pd.Series:
    r = np.log(equity / equity.shift(1))
    return (r.rolling(window).mean() / r.rolling(window).std(ddof=1)
            * math.sqrt(periods_per_year)).rename(f"rolling_sharpe_{window}")


# ------------------------------------------------------------- deflated sharpe
def expected_max_sharpe(n_trials: int, variance: float = 1.0) -> float:
    """Expected maximum Sharpe from ``n_trials`` independent draws (E[Max]).

    Bailey & Lopez de Prado's approximation, using the Euler-Mascheroni constant.
    This is the hurdle a backtest must clear: with enough trials, luck alone
    produces a high Sharpe.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(variance) * ((1 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    sharpe: float, n_trials: int, n_obs: int,
    skew: float = 0.0, kurtosis: float = 3.0,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """Probability that the observed Sharpe is real, after multiple testing.

    Returns the DSR probability and its inputs. A DSR below ~0.95 means the
    result is indistinguishable from the best of ``n_trials`` random strategies.
    """
    if not np.isfinite(sharpe) or n_obs < 10:
        return {"dsr": float("nan"), "sr0": float("nan"), "n_trials": n_trials,
                "n_obs": n_obs, "verdict": "insufficient_data"}

    sr_per_bar = sharpe / math.sqrt(periods_per_year)
    e_max = expected_max_sharpe(max(n_trials, 1)) / math.sqrt(periods_per_year)

    denom = math.sqrt(max(1e-12, (1.0 - skew * sr_per_bar
                                 + (kurtosis - 1.0) / 4.0 * sr_per_bar**2) / max(n_obs - 1, 1)))
    psr = (sr_per_bar - e_max) / denom
    dsr = _norm_cdf(psr)

    if dsr >= 0.95:
        verdict = "pass"
    elif dsr >= 0.80:
        verdict = "weak"
    else:
        verdict = "fail"
    return {"dsr": float(dsr), "sr0": float(e_max * math.sqrt(periods_per_year)),
            "n_trials": int(n_trials), "n_obs": int(n_obs), "verdict": verdict}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's algorithm) - avoids a scipy dependency here."""
    if p <= 0:
        return -float("inf")
    if p >= 1:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ------------------------------------------------------------------- aggregate
def full_report(
    equity: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: float,
    n_trials: int = 1,
    risk_free: float = 0.0,
    initial_capital: Optional[float] = None,
) -> Dict[str, Any]:
    """One dict with every number the report and dashboard need."""
    eq = equity.dropna()
    if len(eq) < 2:
        return {"error": "equity curve too short to evaluate", "n_bars": len(eq)}

    ic = initial_capital if initial_capital is not None else float(eq.iloc[0])
    log_r = np.log(eq / eq.shift(1)).dropna()

    stats = trade_stats(trades)
    mdd = max_drawdown(eq)
    sr = sharpe(eq, periods_per_year, risk_free)
    dsr = deflated_sharpe_ratio(
        sr if np.isfinite(sr) else 0.0, n_trials=n_trials, n_obs=len(log_r),
        skew=float(log_r.skew()) if len(log_r) > 3 else 0.0,
        kurtosis=float(log_r.kurt() + 3.0) if len(log_r) > 3 else 3.0,
        periods_per_year=periods_per_year,
    )

    out: Dict[str, Any] = {
        "initial_capital": float(ic),
        "final_equity": float(eq.iloc[-1]),
        "net_pnl": float(eq.iloc[-1] - ic),
        "total_return_pct": total_return(eq) * 100.0,
        "cagr_pct": cagr(eq, periods_per_year) * 100.0,
        "max_drawdown_pct": mdd * 100.0,
        "max_drawdown_duration": str(max_drawdown_duration(eq) or ""),
        "volatility_pct": volatility(eq, periods_per_year) * 100.0,
        "sharpe": sr,
        "sortino": sortino(eq, periods_per_year, risk_free),
        "calmar": calmar(eq, periods_per_year),
        "var_95_pct": var_historical(eq, 0.95) * 100.0,
        "cvar_95_pct": cvar_historical(eq, 0.95) * 100.0,
        "n_bars": int(len(eq)),
        "years": float(len(eq) / periods_per_year),
        "periods_per_year": float(periods_per_year),
        "deflated_sharpe": dsr,
        **stats,
    }
    out["return_over_max_dd"] = (out["total_return_pct"] / out["max_drawdown_pct"]
                                 if out["max_drawdown_pct"] > 0 else float("nan"))
    out["fees_as_pct_of_gross"] = (
        stats["total_fees"] / max(abs(out["net_pnl"]) + stats["total_fees"], 1e-9) * 100.0
        if np.isfinite(stats["total_fees"]) else float("nan")
    )
    return out


def format_metrics(m: Dict[str, Any]) -> str:
    """Compact human-readable block for the CLI and the markdown report."""

    def num(key: str, dp: int = 2, suffix: str = "") -> str:
        v = m.get(key, float("nan"))
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return f"{'n/a':>10}"
        return f"{v:>10,.{dp}f}{suffix}"

    def pct(key: str, dp: int = 2) -> str:
        return num(key, dp, " %")

    def ratio(key: str, dp: int = 1) -> str:
        v = m.get(key, float("nan"))
        if v is None or not np.isfinite(v):
            return f"{'n/a':>10}"
        return f"{v * 100:>9.{dp}f} %"

    dsr = m.get("deflated_sharpe", {}) or {}
    dsr_val = dsr.get("dsr", float("nan"))
    dsr_str = f"{dsr_val:>10.3f}" if np.isfinite(dsr_val) else f"{'n/a':>10}"

    return "\n".join([
        f"  Net P&L              {num('net_pnl', 0)}",
        f"  Total return         {pct('total_return_pct')}",
        f"  CAGR                 {pct('cagr_pct')}",
        f"  Max drawdown         {pct('max_drawdown_pct')}   "
        f"(underwater {m.get('max_drawdown_duration') or '-'})",
        f"  Sharpe               {num('sharpe')}",
        f"  Sortino              {num('sortino')}",
        f"  Calmar               {num('calmar')}",
        f"  Volatility           {pct('volatility_pct')}",
        f"  VaR95 / CVaR95       {num('var_95_pct')} / {num('cvar_95_pct')} %",
        f"  Trades               {m.get('n_trades', 0):>10}",
        f"  Win rate             {ratio('win_rate')}",
        f"  Profit factor        {num('profit_factor')}",
        f"  Expectancy / trade   {num('expectancy', 2)}",
        f"  Payoff (win/loss)    {num('payoff_ratio', 3)}",
        f"  Avg R                {num('avg_r', 3)}",
        f"  Max consec. losses   {m.get('max_consecutive_losses', 0):>10}",
        f"  Total fees           {num('total_fees', 0)}",
        f"  Deflated Sharpe      {dsr_str}  "
        f"({dsr.get('verdict', '-')}, {dsr.get('n_trials', 1)} trials)",
    ])
