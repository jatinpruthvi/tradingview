"""``entry_mode``: the measured case for keeping the default ``sweep``.

``SmcTrendStrategy`` can take a second trigger - a retracement into an unmitigated
FVG or order block - via ``entry_mode: sweep_or_zone``. It is off by default, and
the comment in ``strategy/setups.py`` says why. This file is the evidence behind
that comment, so the comment cannot quietly rot again.

It already rotted once. The previous version claimed zone entries "degraded profit
factor from ~0.9 to ~0.58 over 93 trades". Re-measured across 5 seeds x 2 bar
counts, no sample produced those numbers, and profit factor turned out to be the
wrong instrument entirely: it improved in half the samples while the strategy got
worse in all of them.

What is actually robust, and what is asserted here:

* Zone entries multiply the opportunity set by roughly an order of magnitude.
* Total return, Sharpe and max drawdown all get worse - unanimously.
* Profit factor does not. It moves in both directions, which is exactly why a
  config comment must not rest on it: PF is gross profit over gross loss, blind to
  how often capital is deployed and to fee drag, so 17x more trades at a similar
  PF is a materially worse outcome that PF alone does not report.

Six full studies, so this is the slowest file in the suite by design. It is worth
the seconds: this is the project's central empirical claim about its own default
setup, and the last time it was stated without a test the statement was wrong.

Nothing here tunes toward profitability. Both modes lose on synthetic data; the
point is the *direction* of the difference and which metric reveals it.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from tvlab.cli import _parse_set
from tvlab.config import load_config
from tvlab.pipeline import run_study

N_BARS = 2500

#: Chosen because they disagree with each other. 20261123 is the config default and
#: shows PF degrading; 7 and 99 show PF *improving* while the outcome gets worse.
#: A single seed would have "confirmed" whichever story it happened to tell.
SEEDS = [20261123, 7, 99]


def _portfolio(seed: int, entry_mode: str) -> Dict[str, Any]:
    overrides = [f"data.synthetic_bars={N_BARS}", f"data.synthetic_seed={seed}"]
    if entry_mode != "sweep":
        overrides.append(f"strategy.params.entry_mode={entry_mode}")
    cfg = load_config(market="crypto", style="swing",
                      overrides=_parse_set(tuple(overrides)))
    return run_study(cfg, with_benchmark=False).portfolio_metrics


@pytest.fixture(scope="module")
def pairs() -> Dict[int, Dict[str, Dict[str, Any]]]:
    """{seed: {"sweep": metrics, "sweep_or_zone": metrics}} - computed once."""
    return {s: {m: _portfolio(s, m) for m in ("sweep", "sweep_or_zone")} for s in SEEDS}


def test_the_shipped_default_is_sweep():
    """Every test above sets both modes explicitly, so none of them would notice if
    the default flipped. That flip is the change this evidence exists to prevent.
    """
    from tvlab.strategy.setups import SmcTrendStrategy

    assert SmcTrendStrategy.defaults["entry_mode"] == "sweep", (
        "the strategy default changed - the measured case above is the reason it "
        "should not")
    # and the style configs must not silently override it the other way
    for style in ("swing", "intraday"):
        cfg = load_config(market="crypto", style=style)
        assert cfg.strategy.params.get("entry_mode", "sweep") == "sweep", (
            f"config/styles/{style}.yaml turns zone entries on by default")


def test_zone_entries_multiply_the_opportunity_set(pairs):
    """The whole appeal of the second trigger is more trades. It delivers."""
    for seed, p in pairs.items():
        base, zone = p["sweep"]["n_trades"], p["sweep_or_zone"]["n_trades"]
        assert base > 0, f"seed {seed}: default mode took no trades, nothing to compare"
        assert zone >= 5 * base, (
            f"seed {seed}: zone entries gave {zone} trades vs {base} - expected the "
            f"order-of-magnitude increase this mode exists to produce")


@pytest.mark.parametrize("metric,better_is_higher", [
    ("total_return_pct", True),
    ("sharpe", True),
    ("max_drawdown_pct", False),
])
def test_the_outcome_metrics_all_get_worse(pairs, metric, better_is_higher):
    """Unanimous across every seed. This is the finding that holds."""
    for seed, p in pairs.items():
        base, zone = p["sweep"][metric], p["sweep_or_zone"][metric]
        if better_is_higher:
            assert zone < base, (
                f"seed {seed}: {metric} went {base:+.4f} -> {zone:+.4f}; zone entries "
                f"were expected to make this worse")
        else:
            assert zone > base, (
                f"seed {seed}: max drawdown went {base:.4f} -> {zone:.4f}; zone entries "
                f"were expected to deepen it")


def test_profit_factor_is_not_a_reliable_instrument_here(pairs):
    """The counterexample that killed the old comment.

    Asserted as a *fact about these seeds*: at least one sample where profit factor
    improves while the return gets worse. If that ever stops being true the claim in
    ``setups.py`` needs re-measuring, which is the point - it should not be possible
    to read "PF degraded" in a config comment without a test behind it.
    """
    improved_pf_but_lost_more = [
        seed for seed, p in pairs.items()
        if p["sweep_or_zone"]["profit_factor"] > p["sweep"]["profit_factor"]
        and p["sweep_or_zone"]["total_return_pct"] < p["sweep"]["total_return_pct"]
    ]
    assert improved_pf_but_lost_more, (
        "expected at least one seed where profit factor improves while the return "
        "gets worse - that is the case that makes PF unsafe to judge this by")
    degraded = [seed for seed, p in pairs.items()
                if p["sweep_or_zone"]["profit_factor"] < p["sweep"]["profit_factor"]]
    assert degraded and len(degraded) < len(pairs), (
        f"PF moved in only one direction across the seeds ({degraded}); the "
        f"'both directions' claim in setups.py would need rewording")


def test_profit_factor_is_too_noisy_to_rank_seeds_on(pairs):
    """Why 13 trades cannot support any of this: the default mode's own PF spans an
    order of magnitude across seeds that differ only in their random numbers.

    This is the same fact check 1 of the validity report enforces (enough trades to
    measure), observed from the other end.
    """
    pfs = [p["sweep"]["profit_factor"] for p in pairs.values()]
    assert max(pfs) / max(min(pfs), 1e-9) > 3.0, (
        f"default-mode PF across seeds was {sorted(pfs)} - expected a wide spread, "
        f"since these samples are far too small for PF to be stable")
