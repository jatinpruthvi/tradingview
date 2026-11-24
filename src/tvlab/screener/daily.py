"""The daily workflow: scan, rank, gate, and write it down.

This is the piece that turns the repo into something a person runs every morning
instead of something they run once to see a backtest. It composes three things that
already exist and does not re-derive any of them:

1. ``screener/scanner.py`` - is there a setup on the last bar?
2. ``screener/quality.py`` - can it be taken at these costs, ranked by
   cost-adjusted expectancy rather than by reward:risk?
3. ``screener/fundamental.py`` + ``data/fundamentals.py`` - is this a company worth
   owning while the setup plays out, and does it report earnings before the stop
   would matter?

It adds two things of its own.

**A journal.** A daily recommendation you cannot look back at is unfalsifiable: in a
month nobody remembers what was suggested, so nothing is ever checked. Every run
appends one JSON line to ``output/daily/journal.jsonl`` and writes the day's full
briefing to ``output/daily/<date>.json``. The journal functions live in this module
(``append_journal``, ``read_journal``, ``journal_path``); the scoring of what was
journaled lives in ``screener/review.py`` and is run with ``tvlab review`` or
``GET /api/review``. Writing the record and reading it back are separate modules
because they have opposite failure modes: a journal that fails to write loses
history, while a scorer that fails to read it invents a verdict.

**The validation verdict, attached to the output.** A study of this strategy on its
default configuration comes back NOT VALIDATED, 3/8 - it is not distinguishable from
random entry on the evidence available here. A daily tool that ranked candidates
without carrying that forward would be laundering an unvalidated strategy into
something that looks like advice, so the briefing states it at the top and in the
machine-readable payload, and ``tvlab daily`` refuses to print a "buy this" framing
when the verdict is not a pass.

Ordering note: fundamentals are applied *after* the cost-aware ranking, and only as
gates. Nothing here produces a weighted composite score, for the reasons set out in
``screener/fundamental.py`` - the arithmetic parts rank, the judgement parts veto.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import AppConfig
from ..data.fundamentals import FundamentalSnapshot, coverage
from . import fundamental as fgates
from .quality import Candidate, suggest

log = logging.getLogger(__name__)

#: The complete verdict vocabulary ``report.py::_verdict`` actually emits, and which
#: of those this module treats as a pass. These are literal strings, not paraphrases:
#: an earlier version of this file guessed "VALIDATED", which silently made every
#: real study look unvalidated. That fails safe but wrong - a properly validated
#: strategy would have been told its edge had never been measured.
_VERDICT_PASS = frozenset({"PASSES ON THIS DATA"})
_VERDICT_FAIL = frozenset({"NOT VALIDATED", "PARTIAL"})
#: Anything outside both sets is unrecognised and must not be read as either.
KNOWN_VERDICTS = _VERDICT_PASS | _VERDICT_FAIL

#: The note recorded when ``fundamentals.enabled`` is true but every threshold is
#: null, so no gate runs. A constant rather than a literal twice: the CLI prints it
#: beside the coverage line, and a message that can drift between the journal and
#: the terminal would be two different explanations of the same run.
NOTE_NO_GATE = ("fundamentals.enabled is true but no threshold is set, "
                "so no fundamental gate ran")


@dataclass
class DailyRow:
    """One ranked idea, with its fundamental context attached.

    Deliberately a superset of ``quality.Candidate`` rather than a wrapper around
    it: the dashboard and the CLI both want one flat record per symbol, and
    reaching through two objects to render a table is how display code and data
    model drift apart.
    """

    symbol: str
    side: str
    last: float
    stop: float
    target: Optional[float]
    rr: float
    risk_pct: float
    cost_in_r: float
    breakeven_win_rate: float
    quality_score: float
    verdict_band: str
    reasons: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    # --- fundamental context (None = not available, which is not "passed")
    fundamental_status: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ttm: Optional[float] = None
    roe: Optional[float] = None
    dividend_yield: Optional[float] = None
    analyst_rating: Optional[float] = None
    days_to_earnings: Optional[int] = None
    fundamental_detail: str = ""
    missing_fundamentals: Tuple[str, ...] = ()

    @property
    def rank_key(self) -> tuple:
        """Cost-adjusted quality first, then the honest tie-breaks.

        Same ordering as ``quality.suggest`` - quality, then lower breakeven win
        rate, then symbol - so a daily list and a ``tvlab suggest`` list on
        identical data agree. Two tools that disagree on the same input are
        unreviewable.
        """
        return (-self.quality_score, self.breakeven_win_rate, self.symbol)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["reasons"] = list(self.reasons)
        d["warnings"] = list(self.warnings)
        d["missing_fundamentals"] = list(self.missing_fundamentals)
        # `rank_key` is a property; __dict__ holds only fields, so it would vanish
        # from serialisation. Same trap Candidate.to_dict already documents.
        return d



@dataclass
class DailyBriefing:
    """One day's output: the shortlist, and everything needed to distrust it."""

    as_of: str
    market: str
    style: str
    strategy: str
    timeframe: str

    #: ranked, fundamentals-gated ideas
    rows: List[DailyRow] = field(default_factory=list)
    #: ideas the technical side found but the cost test rejected
    rejected_technical: List[Dict[str, Any]] = field(default_factory=list)
    #: ideas rejected or unrated on fundamentals, with the reason
    rejected_fundamental: List[Dict[str, Any]] = field(default_factory=list)
    #: symbols the fundamental endpoint did not return
    no_fundamental_data: List[str] = field(default_factory=list)
    #: per-symbol technical failures: a coverage fact, not a market verdict
    data_errors: Dict[str, str] = field(default_factory=dict)

    symbols_scanned: int = 0
    signals_found: int = 0
    #: symbols whose only setup was short, in a market where shorts cannot be taken
    shorts_skipped: List[str] = field(default_factory=list)
    fundamentals_enabled: bool = False
    fundamentals_requested: bool = False
    fundamental_coverage: Dict[str, Any] = field(default_factory=dict)

    #: what the validation study says about the strategy being ranked
    study_verdict: Optional[str] = None
    study_checks: Optional[str] = None
    study_win_rate: Optional[float] = None
    study_source: Optional[str] = None

    #: how this run ended, for the exit code
    outcome: str = "no_signals"
    outcome_detail: str = ""

    #: caveats accumulated from each stage - an unreadable study report, a
    #: fundamental outage, gates configured with no thresholds set. Kept separate
    #: from ``outcome_detail`` because that field is written last, by the outcome
    #: stage; sharing one string meant each stage silently erased the warning the
    #: previous stage had just written.
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ framing
    @property
    def is_validated(self) -> bool:
        """True only for a verdict this repo actually emits as a pass.

        An unrecognised string is not a pass - the safe direction - but it is also
        the direction that would hide a real validation, so an unknown label is
        recorded in ``notes`` rather than merely treated as a failure.
        """
        return (self.study_verdict or "").strip() in _VERDICT_PASS

    @property
    def advisory_line(self) -> str:
        """The sentence that must precede any shortlist.

        An unknown verdict is *not* rendered as a pass - the same rule
        ``_measured_win_rate`` follows in the CLI and dashboard, because a missing
        report is not evidence of validation.
        """
        v = self.study_verdict
        if v is None:
            return ("No validation study on record for this configuration, so the "
                    "edge behind this shortlist has not been measured.")
        if not self.is_validated:
            detail = f" ({self.study_checks})" if self.study_checks else ""
            return (f"The strategy behind this shortlist is {v}{detail} - it was not "
                    f"distinguishable from random entry in the study that produced "
                    f"this verdict. Treat the list as a screen, not as advice.")
        return (f"The strategy behind this shortlist is {v}, so these are ranked "
                f"ideas rather than an unvalidated screen.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of": self.as_of,
            "market": self.market,
            "style": self.style,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "outcome": self.outcome,
            "outcome_detail": self.outcome_detail,
            "notes": list(self.notes),
            "advisory": self.advisory_line,
            "study_verdict": self.study_verdict,
            "study_checks": self.study_checks,
            "study_win_rate": self.study_win_rate,
            "study_source": self.study_source,
            "is_validated": self.is_validated,
            "symbols_scanned": self.symbols_scanned,
            "signals_found": self.signals_found,
            "shorts_skipped": list(self.shorts_skipped),
            "n_shorts_skipped": len(self.shorts_skipped),
            "n_shortlist": len(self.rows),
            "n_rejected_technical": len(self.rejected_technical),
            "n_rejected_fundamental": len(self.rejected_fundamental),
            "no_fundamental_data": list(self.no_fundamental_data),
            "n_data_errors": len(self.data_errors),
            "data_errors": dict(self.data_errors),
            "fundamentals_enabled": self.fundamentals_enabled,
            "fundamentals_requested": self.fundamentals_requested,
            "fundamental_coverage": dict(self.fundamental_coverage),
            "shortlist": [r.to_dict() for r in self.rows],
            "rejected_technical": list(self.rejected_technical),
            "rejected_fundamental": list(self.rejected_fundamental),
        }


# ------------------------------------------------------------------- composing


def build_briefing(
    rows: Sequence[Any],
    errors: Optional[Dict[str, str]],
    cfg: AppConfig,
    snaps: Optional[Dict[str, FundamentalSnapshot]] = None,
    study: Optional[Dict[str, Any]] = None,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    max_breakeven: float = 0.60,
    top: Optional[int] = None,
    as_of: Optional[str] = None,
    fundamentals_error: Optional[str] = None,
) -> DailyBriefing:
    """Compose a briefing from scan rows, fundamental snapshots and a study report.

    Every argument except ``rows`` and ``cfg`` is optional, and each absence is
    reported rather than papered over: no snapshots means the fundamental column
    says "not available", no study means the advisory says the edge is unmeasured.
    A daily tool that quietly dropped one of its three inputs would still look
    complete, which is the failure mode this signature is built to prevent.
    """
    errors = dict(errors or {})
    fcfg = cfg.fundamentals
    fee = cfg.costs.fee_bps if fee_bps is None else float(fee_bps)
    slip = cfg.costs.slippage_bps if slippage_bps is None else float(slippage_bps)

    ranked = suggest(rows, fee_bps=fee, slippage_bps=slip,
                     min_stop_distance_pct=cfg.risk.min_stop_distance_pct,
                     max_breakeven=max_breakeven,
                     allow_short=cfg.risk.allow_short)

    b = DailyBriefing(
        as_of=as_of or _today(),
        market=getattr(cfg, "market", None) or _market_of(cfg),
        style=cfg.style or "default",
        strategy=cfg.strategy.name,
        timeframe=cfg.timeframe.execution,
        symbols_scanned=len(rows) + len(errors),
        signals_found=sum(1 for r in rows if getattr(r, "signal", 0) != 0),
        shorts_skipped=list(ranked["shorts_skipped"]),
        data_errors=errors,
        rejected_technical=ranked["rejected"],
        fundamentals_enabled=bool(fcfg.enabled),
    )
    _apply_study(b, study)

    # ---- fundamentals: gate the survivors, and say what could not be gated
    active = bool(fcfg.enabled) and fgates.active_gate_count(fcfg) > 0

    # Coverage describes the *fetch*, not the gate. It used to be counted over the
    # symbols the gates evaluated, which on a quiet day is the empty set: no setup
    # fired, so nothing reached the gate, and ``tvlab daily`` printed "0 symbol(s)
    # returned" over a full set of snapshots. That reads as a dead endpoint on a run
    # where every symbol answered. Count what came back, and state separately how
    # many names the gates actually looked at.
    if snaps is not None:
        b.fundamental_coverage = _fetch_coverage(snaps, requested=len(rows))

    if not active:
        b.rows = [_to_daily_row(c) for c in ranked["candidates"]]
        if fcfg.enabled and fgates.active_gate_count(fcfg) == 0:
            b.notes.append(NOTE_NO_GATE)
    else:
        b.fundamentals_requested = snaps is not None or fundamentals_error is not None
        if snaps is None and not fundamentals_error:
            # The gates are configured and active, but nobody fetched anything.
            # Treating this as "no data for any symbol" would silently empty the
            # shortlist and look like a market with no qualifying names, so it is
            # handled as an outage: the names survive, flagged as not screened.
            fundamentals_error = ("the fundamental gates are configured but no "
                                  "fundamentals were fetched for this run")
        if fundamentals_error:
            # An endpoint outage must not read as "no fundamental concerns".
            b.notes.append(f"fundamentals unavailable: {fundamentals_error}")
            b.rows = [_to_daily_row(c) for c in ranked["candidates"]]
            for r in b.rows:
                r.warnings = tuple(r.warnings) + (
                    "fundamentals unavailable - this name is NOT fundamentally screened",)
        else:
            outcome = fgates.screen([c.symbol for c in ranked["candidates"]], snaps, fcfg)
            b.fundamental_coverage["gated"] = len(ranked["candidates"])
            b.no_fundamental_data = list(outcome["no_data"])
            by_symbol = {v.symbol: v for v in outcome["approved"]}
            for v in list(outcome["rejected"]) + list(outcome["unrated"]):
                b.rejected_fundamental.append({
                    "symbol": v.symbol,
                    "status": v.status,
                    "failed": v.failed,
                    "could_not_evaluate": v.could_not_evaluate,
                    "detail": v.detail(),
                })
            kept: List[DailyRow] = []
            for c in ranked["candidates"]:
                if c.symbol not in by_symbol:
                    continue
                dr = _to_daily_row(c)
                _attach_fundamentals(dr, c.symbol, by_symbol[c.symbol], snaps.get(c.symbol))
                kept.append(dr)
            b.rows = kept

    # ---- the no-data symbols still belong in the fundamental rejection pile
    for sym in b.no_fundamental_data:
        if not any(x["symbol"] == sym for x in b.rejected_fundamental):
            b.rejected_fundamental.append({
                "symbol": sym, "status": "no_data", "failed": [],
                "could_not_evaluate": ["all"],
                "detail": "not returned by the fundamentals endpoint",
            })

    b.rows.sort(key=lambda r: r.rank_key)
    if top is not None and top > 0:
        b.rows = b.rows[:top]

    _set_outcome(b, ranked, errors)
    return b


def _fetch_coverage(snaps: Dict[str, FundamentalSnapshot], requested: int) -> Dict[str, Any]:
    """What the fundamentals request returned, counted independently of the gate.

    ``coverage()`` counts snapshots, which is the right unit for "did the endpoint
    answer" but says nothing about how many names were asked for. How many names
    reached a gate is the other half of the sentence, and it is added by the caller
    only when a gate actually ran: a ``gated: 0`` invented for a configuration with
    no thresholds set would blame the technical and cost tests for an absence the
    configuration caused.
    """
    cov = dict(coverage(list(snaps.values())))
    cov["requested"] = int(requested)
    return cov


def _to_daily_row(c: Candidate) -> DailyRow:
    return DailyRow(
        symbol=c.symbol, side=c.side, last=c.last, stop=c.stop, target=c.target,
        rr=c.rr, risk_pct=c.risk_pct, cost_in_r=c.cost_in_r,
        breakeven_win_rate=c.breakeven_win_rate, quality_score=c.quality_score,
        verdict_band=c.verdict, reasons=tuple(c.reasons), warnings=tuple(c.warnings),
    )


def _attach_fundamentals(dr: DailyRow, symbol: str, v: Any,
                         snap: Optional[FundamentalSnapshot]) -> None:
    dr.fundamental_status = v.status
    dr.days_to_earnings = v.days_to_earnings
    dr.fundamental_detail = v.detail()
    dr.missing_fundamentals = tuple(v.missing)
    if snap is not None:
        dr.sector = snap.sector
        dr.market_cap = snap.market_cap
        dr.pe_ttm = snap.pe_ttm
        dr.roe = snap.roe
        dr.dividend_yield = snap.dividend_yield
        dr.analyst_rating = snap.analyst_rating
        if v.days_to_earnings is not None and v.days_to_earnings <= 14:
            dr.warnings = tuple(dr.warnings) + (
                f"reports earnings in {v.days_to_earnings} day(s) - outside the "
                f"blackout but within a swing hold",)


def _apply_study(b: DailyBriefing, study: Optional[Dict[str, Any]]) -> None:
    """Copy the study's verdict into the briefing.

    ``study`` is the four-state result from ``cli._measured_win_rate``: ``None``
    means no report exists, and ``status`` distinguishes ok / no_trades / corrupt.
    Treating a missing report as a pass is the exact bug that made an earlier
    version of this repo print PARTIAL when it should have printed NOT VALIDATED.
    """
    if not study:
        return
    if study.get("status") != "ok":
        b.study_verdict = None
        b.study_checks = None
        b.notes.append(f"study report unreadable ({study.get('status')})")
        return
    b.study_verdict = study.get("verdict")
    if b.study_verdict and b.study_verdict not in KNOWN_VERDICTS:
        b.notes.append(f"unrecognised study verdict '{b.study_verdict}' - treated as "
                       f"not validated")
        log.warning("study report carries an unrecognised verdict: %r", b.study_verdict)
    passed, total = study.get("checks_passed"), study.get("checks_total")
    if passed is not None and total is not None:
        b.study_checks = f"{passed}/{total} checks"
    wr = study.get("win_rate")
    b.study_win_rate = float(wr) if wr is not None else None
    b.study_source = study.get("source")


def _set_outcome(b: DailyBriefing, ranked: Dict[str, Any],
                 errors: Dict[str, str]) -> None:
    """Three-way outcome, matching the contract the CLI already uses.

    All-failed is a *data* failure and must not be presented as "nothing to trade
    today" - those have opposite remedies, and conflating them is how an outage
    becomes a market verdict.

    The no-signal branch is checked against ``signals_found``, not against the
    length of the shortlist. Those differ whenever the fundamental gates removed
    something, and reporting "no strategy fired" after two signals fired and were
    screened out sends the user looking for setups that were there.
    """
    scanned = b.symbols_scanned
    if scanned and errors and len(errors) >= scanned:
        b.outcome = "data_failure"
        b.outcome_detail = (f"all {scanned} symbol(s) failed to load; this is a data "
                            f"failure, not a market verdict")
    elif errors and len(errors) < scanned:
        b.outcome = "partial" if b.rows else "no_signals"
        b.outcome_detail = (f"{len(errors)} of {scanned} symbol(s) unavailable - "
                            f"the shortlist covers the rest")
    elif b.rows:
        b.outcome = "shortlist"
        b.outcome_detail = f"{len(b.rows)} idea(s) ranked"
    elif ranked["rejected"]:
        b.outcome = "all_rejected"
        b.outcome_detail = (f"{len(ranked['rejected'])} signal(s) found, all rejected "
                            f"by the cost test")
    elif ranked["shorts_skipped"]:
        # Signals existed, but every one of them was short. That is not a quiet
        # market and not a cost rejection: the market's own rules removed them, and
        # the detail says so rather than sending the user looking for setups.
        b.outcome = "no_signals"
        b.outcome_detail = (f"{len(ranked['shorts_skipped'])} signal(s) fired, all short, "
                            f"and risk.allow_short is false for this market")
    elif b.rejected_fundamental:
        b.outcome = "screened_out"
        b.outcome_detail = (f"{b.signals_found} signal(s) fired, all screened out on "
                            f"fundamentals ({len(b.rejected_fundamental)} name(s))")
    elif b.signals_found:
        # Signals existed but produced no candidates and no rejection record -
        # the cost test returned None for them, i.e. they were unscoreable.
        b.outcome = "unscoreable"
        b.outcome_detail = (f"{b.signals_found} signal(s) fired but none could be "
                            f"scored - no usable stop or price")
    else:
        b.outcome = "no_signals"
        b.outcome_detail = "no strategy fired on the last bar of any symbol"


def _market_of(cfg: AppConfig) -> str:
    """Best-effort market label; ``AppConfig`` carries it on ``universe``."""
    for attr in ("market", "name"):
        v = getattr(getattr(cfg, "universe", None), attr, None)
        if v:
            return str(v)
    return getattr(cfg, "style", None) or "unknown"


def _today() -> str:
    return date.today().isoformat()


# --------------------------------------------------------------------- journal


def journal_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "daily" / "journal.jsonl"


def append_journal(briefing: DailyBriefing, out_dir: str | Path) -> Path:
    """Append one compact line per run, and write the day's full briefing.

    Two files on purpose. The journal is a single append-only line per run - cheap
    to diff, cheap to score in bulk, and it is the record that makes the tool
    falsifiable later. The dated file is the full detail, overwritten if a run is
    repeated the same day so a re-run does not look like two different mornings.

    A failure to write the journal must never fail the briefing: the user asked for
    today's ideas, and losing the audit trail is a warning, not an abort.
    """
    d = Path(out_dir) / "daily"
    jp = journal_path(out_dir)
    line = {
        "as_of": briefing.as_of,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "market": briefing.market,
        "style": briefing.style,
        "strategy": briefing.strategy,
        "timeframe": briefing.timeframe,
        "outcome": briefing.outcome,
        "study_verdict": briefing.study_verdict,
        "is_validated": briefing.is_validated,
        "symbols_scanned": briefing.symbols_scanned,
        "signals_found": briefing.signals_found,
        "n_shorts_skipped": len(briefing.shorts_skipped),
        "n_shortlist": len(briefing.rows),
        "n_rejected_technical": len(briefing.rejected_technical),
        "n_rejected_fundamental": len(briefing.rejected_fundamental),
        "n_data_errors": len(briefing.data_errors),
        "shortlist": [
            {"symbol": r.symbol, "side": r.side, "last": r.last, "stop": r.stop,
             "target": r.target, "rr": r.rr,
             "breakeven_win_rate": r.breakeven_win_rate,
             "quality_score": r.quality_score,
             "verdict_band": r.verdict_band,
             "fundamental_status": r.fundamental_status,
             "days_to_earnings": r.days_to_earnings}
            for r in briefing.rows
        ],
    }
    try:
        # Inside the try, not before it: a bad or unwritable output directory must
        # cost the audit trail, never the briefing the user asked for.
        d.mkdir(parents=True, exist_ok=True)
        with open(jp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        day = d / f"{briefing.as_of}.json"
        with open(day, "w", encoding="utf-8") as fh:
            json.dump(briefing.to_dict(), fh, indent=2, default=str)
    except OSError as exc:
        log.warning("could not write the daily journal to %s: %s", d, exc)
    return jp


def read_journal(out_dir: str | Path, limit: int = 20) -> List[Dict[str, Any]]:
    """The last ``limit`` runs, newest first. Missing file = empty list, not error."""
    jp = journal_path(out_dir)
    if not jp.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for raw in jp.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                log.debug("skipping a corrupt journal line")
    except OSError as exc:
        log.warning("could not read %s: %s", jp, exc)
        return []
    return list(reversed(out[-limit:]))
