"""The quarantine around the look-ahead reference library, enforced.

``smartmoneyconcepts`` publishes a pivot on the pivot bar, before the bars that
confirm it exist. ``VENDOR.md`` promises it never reaches the trading path. This
file makes that a checked property instead of a sentence.

Three things are pinned, from weakest to strongest:

1. **The importer set is exactly what it should be.** An AST scan of ``src/tvlab``
   finds every import of the reference module or the library, in both spelling
   styles. ``from .indicators import reference`` is a different AST shape from
   ``from .indicators.reference import run_all`` and a scan that only looks at
   ``node.module`` misses the first one - which is how ``cli.py`` initially escaped
   this check.

2. **``pipeline.py``'s import is conditional and report-only.** The engine does
   import the reference module, under ``cross_validate``, and its output is attached
   to ``Study.cross_validation``. ``VENDOR.md`` used to say the engine never imports
   it, which was literally false; the honest claim is that the import cannot reach a
   decision, and that is what is asserted here.

3. **A full study completes with the library made unimportable.** The strongest
   form: block ``smartmoneyconcepts`` at ``sys.meta_path``, record every import
   attempt, run a real study, and require that nothing asked for it. If any signal,
   entry, exit or sizing path ever grows a dependency on the reference, this fails
   immediately rather than at the next audit.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set

import pytest

from tvlab.cli import _parse_set
from tvlab.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "tvlab"

#: Every module allowed to reach the quarantined reference, and why.
ALLOWED_IMPORTERS = {
    # the wrapper itself; the only place `smartmoneyconcepts` is named
    "indicators/reference.py",
    # `tvlab validate` - a reporting command, not a decision path
    "cli.py",
    # `run_study(..., cross_validate=True)` - result goes to Study.cross_validation
    "pipeline.py",
}

BLOCKED = {"smartmoneyconcepts", "smartmoneyconcepts.smc"}


def _imported_modules(tree: ast.AST) -> Set[str]:
    """Every module name a file imports, in both AST spellings.

    ``from .indicators import reference`` records module="indicators" and puts the
    interesting name in ``node.names``. Only scanning ``node.module`` silently misses
    it, so both are collected.
    """
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def _reference_importers() -> Set[str]:
    found: Set[str] = set()
    for f in sorted(SRC.rglob("*.py")):
        rel = str(f.relative_to(SRC)).replace("\\", "/")
        mods = _imported_modules(ast.parse(f.read_text()))
        for m in mods:
            if m in BLOCKED or m.endswith(".reference") or m == "reference":
                # `reference` is a common word; only count it where it resolves to
                # the indicators module, i.e. the file imports it from .indicators
                if m == "reference" and not (
                    "indicators" in mods or rel.startswith("indicators/")
                ):
                    continue
                found.add(rel)
    return found


def test_only_the_allowed_modules_import_the_quarantined_reference():
    importers = _reference_importers()
    unexpected = importers - ALLOWED_IMPORTERS
    assert not unexpected, (
        f"the look-ahead reference is now imported by {sorted(unexpected)}. It "
        f"publishes pivots before they are knowable; it must stay out of the "
        f"strategy, engine signal path, scanner and sizer. Allowed: "
        f"{sorted(ALLOWED_IMPORTERS)}")
    # the scan must still be able to see the ones we expect, or it is vacuous
    assert "indicators/reference.py" in importers, (
        "the scan found no importer at all - it is not reading the tree correctly")


def test_pipeline_import_is_conditional_and_report_only():
    """The engine's import is inside `if cross_validate:` and lands in the report."""
    src = (SRC / "pipeline.py").read_text()
    assert "from .indicators.reference import run_all" in src
    tree = ast.parse(src)
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in top_level:
        mods = _imported_modules(ast.Module(body=[node], type_ignores=[]))
        assert not (mods & BLOCKED) and not any(
            m.endswith(".reference") for m in mods), (
            "pipeline.py imports the reference at module level, so it loads even "
            "when cross-validation is off")
    # deferred import inside the guarded branch: the guard must come first in the
    # file, and the import must not be reachable without it
    guard_at = src.find("if cross_validate:")
    import_at = src.find("from .indicators.reference import run_all")
    assert guard_at != -1 and import_at != -1, (
        "could not locate the cross_validate guard or the deferred import")
    assert guard_at < import_at, (
        "the reference import moved out of the cross_validate guard")
    # and its only destination is the report field
    assert "cross_validation=xv" in src


class _ImportBlocker:
    """Meta-path finder that refuses the quarantined library and counts attempts."""

    def __init__(self) -> None:
        self.attempts: List[str] = []

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if fullname in BLOCKED or root in BLOCKED:
            self.attempts.append(fullname)
            raise ImportError(
                f"{fullname} is blocked by test_reference_quarantine")
        return None


def test_a_full_study_never_reaches_for_the_reference_library():
    """The real guarantee: decisions do not depend on it, provably."""
    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)
    # a stale successful import from another test would defeat the blocker
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            del sys.modules[name]
    try:
        from tvlab.pipeline import run_study

        cfg = load_config(market="crypto", style="swing", overrides=_parse_set(
            ("data.synthetic_bars=1200",)))
        study = run_study(cfg, with_benchmark=False, cross_validate=False)
    finally:
        sys.meta_path.remove(blocker)

    assert blocker.attempts == [], (
        f"the trading path tried to import {blocker.attempts}. Every signal, entry, "
        f"exit and sizing decision must be computable without the look-ahead "
        f"reference library.")
    assert study.portfolio_metrics is not None
    assert study.portfolio_metrics.get("n_trades", 0) >= 0


def test_cross_validate_degrades_gracefully_when_the_library_is_missing():
    """--xval must not take the study down with it: the import is already wrapped,
    and this pins that the wrapper really catches a blocked import."""
    blocker = _ImportBlocker()
    sys.meta_path.insert(0, blocker)
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            del sys.modules[name]
    try:
        from tvlab.pipeline import run_study

        cfg = load_config(market="crypto", style="swing", overrides=_parse_set(
            ("data.synthetic_bars=1200",)))
        study = run_study(cfg, with_benchmark=False, cross_validate=True)
    finally:
        sys.meta_path.remove(blocker)

    xv = study.cross_validation
    assert xv is None or "error" in xv, (
        f"with the library blocked, cross_validation should be absent or carry an "
        f"error, got {str(xv)[:200]}")


@pytest.mark.parametrize("seed", [20261123, 42])
def test_the_leaky_convention_does_not_inflate_profit_factor(seed):
    """README.md and VENDOR.md both claimed feeding the reference's non-causal pivot
    convention to a backtester "inflates profit factor to 3-10". Measured across 5
    seeds it did the opposite: PF fell in 4 of 5 and the trade count fell in 5 of 5.

    The reason is structural and worth keeping. This pipeline derives sweep ages,
    structure state and zone ages *from* the swing series, so publishing pivots L
    bars early does not hand the strategy future prices - it re-times every feature
    the strategy was designed around and it sees a different, smaller set of setups.

    Which makes the quarantine more important, not less: a look-ahead bug does not
    reliably make results look better, so a bad-looking backtest is not evidence of
    honesty. The number is meaningless in either direction. This test exists so the
    folklore figure cannot be reinstated without being re-measured.
    """
    from tvlab.indicators import smc
    from tvlab.pipeline import run_study

    ov = _parse_set((f"data.synthetic_bars=2500", f"data.synthetic_seed={seed}"))
    cfg = load_config(market="crypto", style="swing", overrides=ov)

    orig = smc.swing_points

    def leaky(df, swing_length=12, lag=True):
        return orig(df, swing_length, lag=False)   # the reference's convention

    clean = run_study(cfg, with_benchmark=False).portfolio_metrics
    smc.swing_points = leaky
    try:
        dirty = run_study(cfg, with_benchmark=False).portfolio_metrics
    finally:
        smc.swing_points = orig

    ratio = dirty["profit_factor"] / clean["profit_factor"] if clean["profit_factor"] else 0.0
    assert ratio < 3.0, (
        f"seed {seed}: the non-causal convention inflated PF {ratio:.2f}x, which "
        f"would put the documented 3-10x claim back in play - re-measure before "
        f"rewording either doc")
    assert dirty["n_trades"] <= clean["n_trades"], (
        f"seed {seed}: the leak produced more trades ({clean['n_trades']} -> "
        f"{dirty['n_trades']}); the re-timing explanation in this docstring would "
        f"need revisiting")
