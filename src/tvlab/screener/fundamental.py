"""Fundamental gates, applied on top of the cost-aware technical ranking.

The technical side of ``suggest`` answers *"is there a setup, and can it be taken
at these costs?"*. This answers the other half of the question an equity trade
actually poses: *"is this a company worth owning while the setup plays out, and is
there an event about to make the stop irrelevant?"*

Three design decisions, each of which is a refusal to do the obvious thing
--------------------------------------------------------------------

**1. Gates, not weights.** Nothing here produces a number that is multiplied into
the ranking. ``screener/quality.py`` ranks on cost-adjusted expectancy because that
is arithmetic - a breakeven win rate follows from the fee schedule and the stop
distance, and needs no forecast. A weighted blend of P/E, ROE and growth has no
such derivation; any weights would be invented, and fitting them on the 31 trades
this repo can measure is noise mining. So fundamentals *veto*, and the ranking of
whatever survives is left to the part of the system that can justify itself.

``quality_marks`` counts how many of the *configured* quality gates a name passes.
It is reported for transparency, and it is **not** a ranking: every approved name
has passed every configured gate, so the count is the same for all of them and
cannot order them. Approved names are therefore listed alphabetically and the
ranking of survivors is left to the cost-adjusted expectancy in ``quality.py``,
which is derived from arithmetic rather than invented. Weighting ROE against P/E
against growth to produce a single "fundamental score" is exactly the kind of
number that looks rigorous and is not, so this module does not produce one.

**2. Missing data is a third outcome.** A symbol whose P/E did not arrive is
neither approved nor rejected - it is ``unrated``, with the missing field named.
Passing it would let an unfiltered name into a shortlist; rejecting it would turn
an endpoint gap into a market verdict, which is the mistake the data-failure
handling in ``cli.py`` and ``screener/scanner.py`` exists to prevent.

**3. The earnings blackout is the gate that matters most.** A swing position held
through an earnings release is an unhedgeable gap: the stop does not protect
against it, because the position opens at the new price. It needs nothing but a
date, it is not a statistical claim, and it is the one fundamental filter whose
value does not depend on believing any valuation model. When the date is unknown
the symbol is unrated, not waved through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..config import FundamentalsConfig
from ..data.fundamentals import FundamentalSnapshot, coverage

log = logging.getLogger(__name__)


@dataclass
class GateOutcome:
    """One gate's verdict on one symbol."""

    gate: str
    #: True = passed, False = failed, None = could not be evaluated
    passed: Optional[bool]
    detail: str = ""


@dataclass
class FundamentalVerdict:
    """What the fundamental gates decided about one symbol."""

    symbol: str
    status: str                       # "approved" | "rejected" | "unrated"
    gates: List[GateOutcome] = field(default_factory=list)
    #: canonical field names that were needed and absent
    missing: Sequence[str] = ()
    days_to_earnings: Optional[int] = None
    days_since_earnings: Optional[int] = None
    #: how many configured quality gates this name passed. Identical for every
    #: approved name, so it is transparency rather than a ranking - see the module
    #: docstring for why this module refuses to produce one.
    quality_marks: int = 0

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @property
    def failed(self) -> List[str]:
        return [g.gate for g in self.gates if g.passed is False]

    @property
    def could_not_evaluate(self) -> List[str]:
        return [g.gate for g in self.gates if g.passed is None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "failed": self.failed,
            "could_not_evaluate": self.could_not_evaluate,
            "missing": list(self.missing),
            "days_to_earnings": self.days_to_earnings,
            "days_since_earnings": self.days_since_earnings,
            "quality_marks": self.quality_marks,
            "detail": self.detail(),
        }

    def detail(self) -> str:
        if self.status == "rejected":
            return "; ".join(g.detail for g in self.gates if g.passed is False)
        if self.status == "unrated":
            parts = [g.detail for g in self.gates if g.passed is None]
            return "; ".join(parts) or "insufficient fundamental data"
        passed = [g for g in self.gates if g.passed is True]
        if not passed:
            return "no fundamental gates configured"
        return "; ".join(g.detail for g in passed)


# ------------------------------------------------------------------- gate table
#: canonical field -> (config attribute, comparison, human label)
#: ``comparison`` is "max" for an upper bound and "min" for a lower bound.
_NUMERIC_GATES = (
    ("market_cap",       "min_market_cap",              "min", "market cap"),
    ("pe_ttm",           "max_pe_ttm",                  "max", "P/E (TTM)"),
    ("price_to_book",    "max_price_to_book",           "max", "P/B"),
    ("price_to_sales",   "max_price_to_sales",          "max", "P/S"),
    ("ev_to_ebitda",     "max_ev_to_ebitda",            "max", "EV/EBITDA"),
    ("roe",              "min_roe",                     "min", "ROE"),
    ("operating_margin", "min_operating_margin",        "min", "operating margin"),
    ("net_margin",       "min_net_margin",              "min", "net margin"),
    ("revenue_growth_yoy", "min_revenue_growth_yoy",    "min", "revenue growth YoY"),
    ("eps_growth_yoy",   "min_eps_growth_yoy",          "min", "EPS growth YoY"),
    ("debt_to_equity",   "max_debt_to_equity",          "max", "debt/equity"),
    ("current_ratio",    "min_current_ratio",           "min", "current ratio"),
    ("dividend_yield",   "min_dividend_yield",          "min", "dividend yield"),
    ("dividend_payout",  "max_dividend_payout",         "max", "dividend payout"),
    ("analyst_rating",   "min_analyst_rating",          "min", "analyst rating"),
)

#: Gates that indicate quality rather than size or price. Used only for the
#: unweighted ``quality_marks`` tie-break.
_QUALITY_GATES = {"roe", "operating_margin", "net_margin", "revenue_growth_yoy",
                  "eps_growth_yoy", "current_ratio"}


def evaluate(snap: FundamentalSnapshot, cfg: FundamentalsConfig,
             as_of: Optional[pd.Timestamp] = None) -> FundamentalVerdict:
    """Apply every configured gate to one snapshot.

    A gate whose threshold is ``None`` is not evaluated at all and produces no
    ``GateOutcome`` - "not configured" and "passed" are different statements, and
    conflating them is how a report ends up claiming eight filters ran when two
    were set.
    """
    gates: List[GateOutcome] = []
    quality_marks = 0
    d2e = snap.days_to_earnings(as_of)
    dse = snap.days_since_earnings(as_of)

    # --- the earnings blackout, first: it is the only gate that is not a
    #     statistical claim, and it can invalidate every other one.
    blackout = int(cfg.earnings_blackout_days or 0)
    aftermath = int(cfg.earnings_aftermath_days or 0)
    if blackout > 0 or aftermath > 0:
        if d2e is None:
            gates.append(GateOutcome(
                "earnings_blackout", None,
                "next earnings date unknown - cannot rule out holding through a report"))
        elif 0 <= d2e <= blackout:
            gates.append(GateOutcome(
                "earnings_blackout", False,
                f"reports earnings in {d2e} day(s) (blackout is {blackout}) - a stop "
                f"does not survive the gap"))
        elif d2e < 0:
            # A scheduled date in the past means the feed has not rolled over.
            gates.append(GateOutcome(
                "earnings_blackout", None,
                f"next earnings date is {abs(d2e)} day(s) in the past - stale feed, "
                f"cannot rule out an imminent report"))
        elif aftermath > 0 and dse is not None and 0 <= dse <= aftermath:
            gates.append(GateOutcome(
                "earnings_blackout", False,
                f"reported earnings {dse} day(s) ago (aftermath window is {aftermath})"))
        else:
            gates.append(GateOutcome(
                "earnings_blackout", True,
                f"no earnings within {blackout} day(s)"
                + (f"; next report in {d2e} days" if d2e is not None else "")))

    # --- positive free cash flow, which is a sign test rather than a threshold
    if cfg.require_positive_free_cash_flow:
        fcf = snap.free_cash_flow
        if fcf is None:
            gates.append(GateOutcome("positive_fcf", None, "free cash flow not reported"))
        else:
            gates.append(GateOutcome(
                "positive_fcf", fcf > 0,
                f"free cash flow {fcf:,.0f}" + ("" if fcf > 0 else " (negative)")))
            if fcf > 0:
                quality_marks += 1

    # --- every numeric threshold that was actually configured
    for canonical, attr, direction, label in _NUMERIC_GATES:
        threshold = getattr(cfg, attr, None)
        if threshold is None:
            continue
        value = snap.get(canonical)
        if value is None:
            gates.append(GateOutcome(canonical, None, f"{label} not reported"))
            continue
        ok = value <= threshold if direction == "max" else value >= threshold
        # The limit is stated as a bound, not as a comparison operator. Printing
        # "debt/equity 6.4 <= 2" for a gate that FAILED asserts something false in
        # the same breath as it reports the failure, and the rejection detail is
        # built by joining exactly these strings.
        limit = f"at most {threshold:g}" if direction == "max" else f"at least {threshold:g}"
        gates.append(GateOutcome(canonical, bool(ok), f"{label} {value:g} ({limit})"))
        if ok and canonical in _QUALITY_GATES:
            quality_marks += 1

    failed = [g for g in gates if g.passed is False]
    unknown = [g for g in gates if g.passed is None]
    if failed:
        status = "rejected"
    elif unknown:
        status = "unrated"
    else:
        status = "approved"

    return FundamentalVerdict(
        symbol=snap.symbol, status=status, gates=gates,
        missing=tuple(snap.missing), days_to_earnings=d2e, days_since_earnings=dse,
        quality_marks=quality_marks,
    )


def screen(symbols: Iterable[str], snaps: Dict[str, FundamentalSnapshot],
           cfg: FundamentalsConfig,
           as_of: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    """Screen a symbol list, reporting the ones we have no data for.

    A symbol in ``symbols`` with no snapshot is *not* the same as one that failed a
    gate - it means the endpoint did not return it, which is a coverage fact and is
    reported as such.
    """
    approved: List[FundamentalVerdict] = []
    rejected: List[FundamentalVerdict] = []
    unrated: List[FundamentalVerdict] = []
    no_data: List[str] = []

    for sym in symbols:
        snap = snaps.get(sym)
        if snap is None:
            no_data.append(sym)
            continue
        v = evaluate(snap, cfg, as_of)
        {"approved": approved, "rejected": rejected, "unrated": unrated}[v.status].append(v)

    # Alphabetical, and deliberately not by quality_marks: every approved name has
    # passed every configured gate, so the marks are identical across them and
    # sorting on them would only make the output order depend on the input order.
    approved.sort(key=lambda v: v.symbol)
    rejected.sort(key=lambda v: v.symbol)
    unrated.sort(key=lambda v: v.symbol)

    evaluated = approved + rejected + unrated
    return {
        "approved": approved,
        "rejected": rejected,
        "unrated": unrated,
        "no_data": no_data,
        "n_approved": len(approved),
        "n_rejected": len(rejected),
        "n_unrated": len(unrated),
        "n_no_data": len(no_data),
        "coverage": coverage(evaluated),
        "enabled": bool(cfg.enabled),
    }


def active_gate_count(cfg: FundamentalsConfig) -> int:
    """How many gates are actually configured. Zero means the screen is a no-op,
    and saying so out loud is better than printing an empty filter table."""
    n = sum(1 for _, attr, _, _ in _NUMERIC_GATES if getattr(cfg, attr, None) is not None)
    if cfg.require_positive_free_cash_flow:
        n += 1
    if (cfg.earnings_blackout_days or 0) > 0 or (cfg.earnings_aftermath_days or 0) > 0:
        n += 1
    return n
