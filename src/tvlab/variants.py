"""Controlled variant runner with honest multiple-testing DSR penalties.

Implements Backlog Tier 3 item #15: run N named config overrides over the exact
same sample, report each variant's metrics and delta vs. baseline, and feed the
true trial count into the deflated Sharpe ratio so searching N variants never
masquerades as N=1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest.walkforward import run_walk_forward
from .config import AppConfig
from .data.store import DataStore
from .pipeline import run_study

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSpec:
    """One named hypothesis to evaluate against the baseline."""

    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


def orb_improvement_variants(session_tz: Optional[str] = None) -> List[VariantSpec]:
    """The 10 isolated ORB hypotheses plus the combined setup.

    Always starts with ``baseline`` so every row reports its marginal lift over
    the unmodified OpeningRangeBreakout rule on the exact same sample.
    """
    base_tz: Dict[str, Any] = {"strategy.params.session_tz": session_tz} if session_tz else {}

    def _p(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"strategy.name": "orb", **base_tz}
        if extra:
            out.update(extra)
        return out

    return [
        VariantSpec("baseline", _p(), "Plain 15m ORB, opposite stop, EOD exit"),
        VariantSpec("confirm_close", _p({"strategy.params.confirm_close": True}),
                    "Require bar close beyond ORB instead of wick pierce"),
        VariantSpec("morning_cutoff_1130", _p({"strategy.params.max_entry_tod": "11:30"}),
                    "Skip late-day breakouts after 11:30"),
        VariantSpec("min_range_0.5pct", _p({"strategy.params.min_range_pct": 0.005}),
                    "Skip tight opening ranges < 0.5%"),
        VariantSpec("max_range_1.5atr", _p({"strategy.params.max_range_atr": 1.5}),
                    "Skip exhausted openings > 1.5x prior-session range"),
        VariantSpec("max_gap_1.5atr", _p({"strategy.params.max_gap_atr": 1.5}),
                    "Skip gap-exhausted opens > 1.5x prior-session range"),
        VariantSpec("limit_entry_no_chase", _p({
            "strategy.params.limit_entry": True,
            "strategy.params.max_chase_ratio": 0.75,
            "backtest.use_limit_entries": True,
        }), "Passive limit fill at ORB boundary + max chase cap"),
        VariantSpec("htf_bias_aligned", _p({"strategy.params.use_bias": True}),
                    "Align ORB direction with causal HTF bias"),
        VariantSpec("midpoint_on_wide", _p({"strategy.params.stop_mode": "midpoint_on_wide"}),
                    "Use range midpoint stop when ORB width >= 1.5%"),
        VariantSpec("target_2.5r", _p({"strategy.params.r_multiple": 2.5}),
                    "Take profit at 2.5R instead of holding all winners to close"),
        VariantSpec("rs_top_40pct", _p({"universe.rs_top_pct": 0.4}),
                    "Trade only causal top/bottom 40% relative-strength names"),
        VariantSpec("mis_costs_shorts_sizing", _p({
            "risk.allow_short": True,
            "risk.max_position_pct": 0.40,
            "risk.max_leverage": 2.0,
            "costs.fee_bps": 2.0,
            "costs.slippage_bps": 2.5,
        }), "NSE MIS intraday shorting, lower cost, unclipped sizing"),
        VariantSpec("combined_enhanced", _p({
            "strategy.params.confirm_close": True,
            "strategy.params.max_entry_tod": "11:30",
            "strategy.params.min_range_pct": 0.004,
            "strategy.params.max_range_atr": 1.5,
            "strategy.params.max_gap_atr": 1.5,
            "strategy.params.stop_mode": "midpoint_on_wide",
            "strategy.params.r_multiple": 2.5,
            "strategy.params.use_bias": True,
            "strategy.params.limit_entry": True,
            "backtest.use_limit_entries": True,
            "universe.rs_top_pct": 0.4,
            "risk.allow_short": True,
            "risk.max_position_pct": 0.40,
            "risk.max_leverage": 2.0,
            "costs.fee_bps": 2.0,
            "costs.slippage_bps": 2.5,
        }), "Combined enhanced ORB MIS setup"),
    ]


def run_variants(
    cfg: AppConfig,
    variants: Sequence[VariantSpec],
    symbols: Optional[List[str]] = None,
    source: Optional[str] = None,
    use_walk_forward: bool = False,
    store: Optional[DataStore] = None,
) -> pd.DataFrame:
    """Evaluate ``variants`` over identical data with DSR trial correction."""
    if not variants:
        return pd.DataFrame()

    store = store or DataStore(cfg)
    syms = list(symbols or cfg.universe.symbols)
    if not syms:
        syms = ["BTC/USDT"]
    n_trials = max(len(variants), 1)

    rows: List[Dict[str, Any]] = []
    base_ret: Optional[float] = None
    base_sharpe: Optional[float] = None
    base_pf: Optional[float] = None

    for spec in variants:
        vc = cfg.merge(spec.overrides) if spec.overrides else cfg
        try:
            st = run_study(
                vc, symbols=syms, store=store, n_trials=n_trials,
                with_benchmark=False, source=source,
            )
            m = st.portfolio_metrics or {}
            ret = m.get("total_return_pct")
            sh = m.get("sharpe")
            pf = m.get("profit_factor")
            dsr = (m.get("deflated_sharpe") or {}).get("dsr")

            if base_ret is None and ret is not None and np.isfinite(ret):
                base_ret = float(ret)
            if base_sharpe is None and sh is not None and np.isfinite(sh):
                base_sharpe = float(sh)
            if base_pf is None and pf is not None and np.isfinite(pf):
                base_pf = float(pf)

            oos_ret = np.nan
            oos_wf_wins = ""
            if use_walk_forward and st.symbols:
                primary = st.symbols[0]
                wf = run_walk_forward(
                    vc, primary.symbol, primary.ohlcv,
                    bias=store.bias_frame(primary.symbol, source=source),
                )
                ws = wf.summary()
                oos_ret = ws.get("oos_return_pct", np.nan)
                oos_wf_wins = f"{ws.get('windows_profitable', 0)}/{ws.get('n_windows', 0)}"

            rows.append({
                "variant": spec.name,
                "return_pct": ret,
                "delta_ret_pct": (float(ret) - base_ret) if (ret is not None and base_ret is not None) else np.nan,
                "sharpe": sh,
                "delta_sharpe": (float(sh) - base_sharpe) if (sh is not None and base_sharpe is not None) else np.nan,
                "dsr": dsr,
                "profit_factor": pf,
                "delta_pf": (float(pf) - base_pf) if (pf is not None and base_pf is not None) else np.nan,
                "max_dd_pct": m.get("max_drawdown_pct"),
                "trades": m.get("n_trades"),
                "win_rate": m.get("win_rate"),
                "avg_r": m.get("avg_r"),
                "fees": m.get("total_fees"),
                "trials": n_trials,
                "oos_return_pct": oos_ret if use_walk_forward else np.nan,
                "wf_windows": oos_wf_wins if use_walk_forward else "",
                "description": spec.description,
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("variant %s failed: %s", spec.name, exc)
            rows.append({
                "variant": spec.name,
                "error": f"{type(exc).__name__}: {exc}",
                "trials": n_trials,
                "description": spec.description,
            })

    df = pd.DataFrame(rows)
    if not use_walk_forward and not df.empty:
        df = df.drop(columns=["oos_return_pct", "wf_windows"], errors="ignore")
    return df
