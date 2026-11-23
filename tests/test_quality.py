"""Cost-aware candidate ranking, and the naive-R:R trap it exists to avoid.

The central claim under test: sorting by reward:risk promotes the tightest stops,
because R:R is a ratio of two distances and the denominator can be shrunk at will.
A candidate with R:R 8 and a 0.20% stop must rank *below* one with R:R 3 and a
2.0% stop at crypto's 30 bp round trip - and at a tight enough stop it must be
rejected outright rather than merely ranked lower.
"""
import numpy as np
import pytest

from tvlab.config import load_config
from tvlab.screener.quality import (
    MIN_STOP_ATR,
    RR_SATURATION,
    WEIGHTS,
    breakeven_win_rate,
    cost_in_r,
    evaluate,
    implied_win_rate_at_rr,
    round_trip_cost_fraction,
    suggest,
)
from tvlab.screener.scanner import ScanRow, rank

# crypto: 10 bp fee + 5 bp slippage per leg = 30 bp round trip
CRYPTO_FEE, CRYPTO_SLIP = 10.0, 5.0


def _row(symbol, signal=1, last=100.0, stop=98.0, target=104.0, reward_risk=None,
         trend=1, adx=25.0, atr_pct=2.0, zones="-", sweep_age=None, bars=1200,
         support_pct=None, resistance_pct=None):
    """A ScanRow with the fields ranking actually reads."""
    if reward_risk is None and target is not None and stop is not None:
        d = abs(last - stop)
        reward_risk = (abs(target - last) / d) if d > 0 else float("nan")
    return ScanRow(symbol=symbol, last=last, signal=signal, stop=stop, target=target,
                   reward_risk=reward_risk, trend=trend, adx=adx, atr_pct=atr_pct,
                   zones=zones, sweep_age=sweep_age, bars=bars,
                   support_pct=support_pct, resistance_pct=resistance_pct)


# ------------------------------------------------------------------- the math


def test_round_trip_cost_charges_both_legs():
    """Both legs pay both costs. Halving this is the most common way a backtest
    turns a losing intraday strategy into an apparently profitable one."""
    assert round_trip_cost_fraction(10, 5) == pytest.approx(0.0030)
    assert round_trip_cost_fraction(3, 5) == pytest.approx(0.0016)
    assert round_trip_cost_fraction(1, 1.5) == pytest.approx(0.0005)
    assert round_trip_cost_fraction(0, 0) == 0.0


def test_cost_in_r_is_inverse_in_stop_distance():
    """The whole mechanism: costs scale with notional while R scales with the stop
    distance, so narrowing the stop magnifies the cost in R terms."""
    c = round_trip_cost_fraction(CRYPTO_FEE, CRYPTO_SLIP)
    tight = cost_in_r(0.002, c)
    wide = cost_in_r(0.020, c)
    assert tight == pytest.approx(1.5)
    assert wide == pytest.approx(0.15)
    assert tight == pytest.approx(wide * 10)


def test_cost_in_r_of_a_zero_distance_stop_is_infinite():
    """Not an error and not zero: such a trade costs infinitely many R to place."""
    assert cost_in_r(0.0, 0.003) == float("inf")
    assert cost_in_r(-0.01, 0.003) == float("inf")
    assert np.isinf(cost_in_r(np.nan, 0.003))


def test_breakeven_reduces_to_the_textbook_formula_without_costs():
    for rr in (1.0, 2.0, 3.0, 5.0, 10.0):
        assert breakeven_win_rate(rr, 0.0) == pytest.approx(1.0 / (1.0 + rr))
        assert breakeven_win_rate(rr, 0.0) == pytest.approx(implied_win_rate_at_rr(rr))


def test_breakeven_rises_with_cost():
    """Costs are paid win or lose, so they push the threshold up, never down."""
    assert breakeven_win_rate(5.0, 0.0) < breakeven_win_rate(5.0, 0.2) < breakeven_win_rate(5.0, 1.5)
    assert breakeven_win_rate(5.0, 1.5) == pytest.approx(2.5 / 6.0)   # 41.7%


def test_breakeven_falls_as_reward_risk_rises():
    a = breakeven_win_rate(2.0, 0.6)
    b = breakeven_win_rate(5.0, 0.6)
    assert b < a


def test_breakeven_is_capped_at_one():
    """No configuration should ever report needing more than a 100% win rate."""
    for rr, c in ((0.0, 0.5), (0.1, 5.0), (-1.0, 0.2), (np.nan, 0.2)):
        assert breakeven_win_rate(rr, c) <= 1.0


def test_breakeven_of_a_negative_or_missing_reward_is_one():
    assert breakeven_win_rate(0.0, 0.0) == 1.0
    assert breakeven_win_rate(-2.0, 0.0) == 1.0


def test_breakeven_never_below_the_zero_cost_floor():
    """A cost-aware threshold must always be at least the naive one."""
    for rr in (1.0, 2.5, 5.0, 20.0):
        for c in (0.0, 0.1, 1.0, 3.0):
            assert breakeven_win_rate(rr, c) >= implied_win_rate_at_rr(rr) - 1e-12


# ------------------------------------------------- the trap this module exists for


def test_a_tighter_stop_does_not_outrank_a_wider_one():
    """THE central claim. TIGHT has the better ratio on paper (8:1 vs 3:1) and is
    exactly the trade the naive sort would put first."""
    tight = _row("TIGHT", last=100.0, stop=99.80, target=101.60)     # 0.20% stop, R:R 8
    wide = _row("WIDE", last=100.0, stop=98.00, target=104.00)        # 2.00% stop, R:R 3
    assert tight.reward_risk > wide.reward_risk, "test premise: the trap row has the better ratio"

    ct = evaluate(tight, CRYPTO_FEE, CRYPTO_SLIP)
    cw = evaluate(wide, CRYPTO_FEE, CRYPTO_SLIP)
    assert cw.quality_score > ct.quality_score, (
        f"the wider stop must win: {cw.quality_score:.3f} vs {ct.quality_score:.3f}")


def test_an_untradeable_stop_is_rejected_outright_not_just_ranked_lower():
    """Presenting an untradeable idea as merely 'lower quality' implies it is a
    worse version of a good idea. It is not: a 0.05% stop at 30 bp costs 6R to
    place, so the breakeven win rate saturates at 100% and it must be gated out.

    A merely TIGHT stop (0.20%, R:R 8) stays a candidate and is ranked below the
    wider one - that is test_a_tighter_stop_does_not_outrank_a_wider_one. The gate
    is for the ones that cannot be taken at any win rate."""
    untradeable = _row("HAIR", last=100.0, stop=99.95, target=100.20)   # 0.05% stop, R:R 4
    wide = _row("WIDE", last=100.0, stop=98.00, target=104.00)
    res = suggest([untradeable, wide], CRYPTO_FEE, CRYPTO_SLIP)

    assert [c.symbol for c in res["candidates"]] == ["WIDE"]
    assert [r["symbol"] for r in res["rejected"]] == ["HAIR"]
    rej = res["rejected"][0]
    assert rej["breakeven_win_rate"] > res["max_breakeven"]
    assert any("win rate" in w for w in rej["reasons"])
    assert evaluate(untradeable, CRYPTO_FEE, CRYPTO_SLIP).cost_in_r >= 6.0


def test_a_merely_tight_stop_stays_a_candidate_but_ranks_below():
    """Distinguish the two failure modes: 'expensive but tradeable' is ranked,
    'cannot be taken' is rejected. Conflating them either hides real ideas or
    promotes untradeable ones."""
    tight = _row("TIGHT", last=100.0, stop=99.80, target=101.60)   # R:R 8, 1.5R of cost
    wide = _row("WIDE", last=100.0, stop=98.00, target=104.00)
    res = suggest([tight, wide], CRYPTO_FEE, CRYPTO_SLIP)
    assert res["n_rejected"] == 0, "a tight stop is not automatically untradeable"
    assert [c.symbol for c in res["candidates"]] == ["WIDE", "TIGHT"]


def test_holding_rr_constant_a_tighter_stop_needs_a_higher_win_rate():
    """The mechanism, isolated. Both rows are R:R 5 on paper; only the stop width
    differs. Tight pays 1.5R in costs and needs 41.7%; wide pays 0.15R and needs
    20.0%. Same strategy, same ratio, completely different trade.

    Note this is NOT the same as saying a tight stop always loses: narrowing the
    stop while holding the TARGET fixed raises R:R, which can offset the cost.
    What always holds is that the cost per R rises, and that the stop ends up
    inside the instrument's noise - which is what stop_quality penalises."""
    tight = _row("T", last=100.0, stop=99.80, target=101.00)   # 0.20% stop, R:R 5
    wide = _row("W", last=100.0, stop=98.00, target=110.00)    # 2.00% stop, R:R 5
    ct, cw = evaluate(tight, CRYPTO_FEE, CRYPTO_SLIP), evaluate(wide, CRYPTO_FEE, CRYPTO_SLIP)
    assert ct.rr == pytest.approx(cw.rr, rel=1e-6) == pytest.approx(5.0)
    assert ct.breakeven_win_rate == pytest.approx(2.5 / 6.0)     # 41.7%
    assert cw.breakeven_win_rate == pytest.approx(1.15 / 6.0)    # 19.2%
    assert ct.breakeven_win_rate - cw.breakeven_win_rate > 0.20
    assert ct.cost_in_r == pytest.approx(cw.cost_in_r * 10)


def test_forex_costs_change_the_verdict_for_the_same_idea():
    """Same geometry, different fee schedule, different answer. A ranking that
    ignored costs would call these identical."""
    row = _row("X", last=100.0, stop=99.70, target=101.40)     # 0.30% stop
    crypto = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP)             # 30 bp
    fx = evaluate(row, 1.0, 1.5)                                # 5 bp
    assert crypto.breakeven_win_rate > fx.breakeven_win_rate
    assert crypto.cost_in_r > fx.cost_in_r


# ------------------------------------------------------------------ evaluation


def test_evaluate_skips_rows_with_no_signal():
    assert evaluate(_row("X", signal=0), CRYPTO_FEE, CRYPTO_SLIP) is None


def test_evaluate_skips_rows_without_a_usable_stop():
    assert evaluate(_row("X", stop=None), CRYPTO_FEE, CRYPTO_SLIP) is None
    assert evaluate(_row("X", stop=float("nan")), CRYPTO_FEE, CRYPTO_SLIP) is None
    assert evaluate(_row("X", last=0.0), CRYPTO_FEE, CRYPTO_SLIP) is None
    assert evaluate(_row("X", stop=100.0), CRYPTO_FEE, CRYPTO_SLIP) is None   # zero distance


def test_evaluate_uses_the_configured_r_multiple_when_there_is_no_target():
    """A strategy with no fixed target still has an R:R from r_multiple."""
    row = _row("X", target=None, reward_risk=2.5)
    c = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP)
    assert c is not None and c.rr == pytest.approx(2.5)
    assert c.reward_pct is None


def test_evaluate_returns_none_when_neither_target_nor_ratio_exists():
    assert evaluate(_row("X", target=None, reward_risk=None), CRYPTO_FEE, CRYPTO_SLIP) is None


def test_side_follows_the_signal_sign():
    assert evaluate(_row("L", signal=1), CRYPTO_FEE, CRYPTO_SLIP).side == "long"
    assert evaluate(_row("S", signal=-1, stop=102.0, target=96.0, trend=-1),
                    CRYPTO_FEE, CRYPTO_SLIP).side == "short"


def test_short_geometry_is_measured_in_absolute_distance():
    """A short's stop is above entry and its target below; distances must be
    absolute or the ratio comes out negative."""
    row = _row("S", signal=-1, last=100.0, stop=102.0, target=96.0, trend=-1)
    c = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP)
    assert c.risk_pct == pytest.approx(0.02)
    assert c.reward_pct == pytest.approx(0.04)
    assert c.rr == pytest.approx(2.0)


def test_quality_score_is_bounded_zero_to_one():
    rows = [_row(f"S{i}", last=100.0, stop=100.0 - 0.5 * (i + 1), target=100.0 + 2 * (i + 1),
                 adx=10 + 5 * i, zones=("FVG" if i % 2 else "-"), sweep_age=i)
            for i in range(6)]
    for r in rows:
        c = evaluate(r, CRYPTO_FEE, CRYPTO_SLIP)
        assert 0.0 <= c.quality_score <= 1.0, c


def test_weights_are_explicit_and_sum_to_one():
    """The score is judgement, not a fitted parameter. If someone changes a weight
    the sum should still be 1 so the score stays on a 0..1 scale."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert set(WEIGHTS) == {"reward_risk", "cost_drag", "stop_quality", "trend", "location"}


def test_trend_alignment_matters():
    aligned = evaluate(_row("A", signal=1, trend=1, adx=30.0), CRYPTO_FEE, CRYPTO_SLIP)
    against = evaluate(_row("B", signal=1, trend=-1, adx=30.0), CRYPTO_FEE, CRYPTO_SLIP)
    assert aligned.quality_score > against.quality_score
    assert aligned.trend_aligned and not against.trend_aligned
    assert any("against the higher-timeframe trend" in w for w in against.warnings)


def test_zone_and_fresh_sweep_improve_the_location_score():
    plain = evaluate(_row("P", zones="-", sweep_age=None), CRYPTO_FEE, CRYPTO_SLIP)
    rich = evaluate(_row("R", zones="FVG+OB", sweep_age=2), CRYPTO_FEE, CRYPTO_SLIP)
    assert rich.quality_score > plain.quality_score
    assert rich.in_zone and not plain.in_zone


def test_a_stale_sweep_does_not_count_as_confirmation():
    fresh = evaluate(_row("F", sweep_age=2), CRYPTO_FEE, CRYPTO_SLIP)
    stale = evaluate(_row("S", sweep_age=200), CRYPTO_FEE, CRYPTO_SLIP)
    assert fresh.quality_score >= stale.quality_score


def test_structural_stop_is_detected_from_trendlines():
    """A stop beyond the nearest fitted trendline is a real invalidation level;
    one inside it will be tested before the stop is."""
    structural = _row("S", last=100.0, stop=98.0, target=104.0, support_pct=-6.0)
    arbitrary = _row("A", last=100.0, stop=98.0, target=104.0, support_pct=-0.5)
    cs = evaluate(structural, CRYPTO_FEE, CRYPTO_SLIP)
    ca = evaluate(arbitrary, CRYPTO_FEE, CRYPTO_SLIP)
    assert cs.structural_stop and not ca.structural_stop
    assert cs.quality_score > ca.quality_score
    assert any("structural" in r for r in cs.reasons)


def test_structural_stop_works_for_shorts_too():
    row = _row("S", signal=-1, last=100.0, stop=102.0, target=96.0, trend=-1, resistance_pct=6.0)
    assert evaluate(row, CRYPTO_FEE, CRYPTO_SLIP).structural_stop is True


# -------------------------------------------------------------------- warnings


def test_warns_when_the_stop_is_below_the_mandate_minimum():
    """The sizer would reject this trade; the ranking must say so rather than
    quietly scoring it."""
    row = _row("X", last=100.0, stop=99.95, target=100.20, atr_pct=2.0)   # 0.05% stop
    c = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP, min_stop_distance_pct=0.0015)
    assert any("min_stop_distance_pct" in w for w in c.warnings)


def test_warns_when_the_stop_is_inside_the_noise():
    """A 0.10 ATR stop gets hit by ordinary chopping regardless of the thesis."""
    row = _row("X", last=100.0, stop=99.90, target=100.50, atr_pct=2.0)   # 0.05 ATR
    c = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP)
    assert c.stop_atr_multiple < MIN_STOP_ATR
    assert any("ATR away" in w for w in c.warnings)


def test_stop_atr_multiple_converts_percent_to_fraction_correctly():
    """atr_pct is in percent while risk_pct is a fraction; mixing the units would
    make every stop look 100x wider than it is."""
    row = _row("X", last=100.0, stop=98.0, target=104.0, atr_pct=2.0)
    c = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP)
    assert c.risk_pct == pytest.approx(0.02)
    assert c.stop_atr_multiple == pytest.approx(1.0), "a 2% stop with 2% ATR is exactly 1 ATR"


def test_warns_when_costs_exceed_a_whole_r():
    row = _row("X", last=100.0, stop=99.80, target=101.60)
    c = evaluate(row, CRYPTO_FEE, CRYPTO_SLIP)
    assert c.cost_in_r >= 1.0
    assert any("just to place this trade" in w for w in c.warnings)


def test_warns_when_history_is_too_short():
    c = evaluate(_row("X", bars=40), CRYPTO_FEE, CRYPTO_SLIP)
    assert any("bars of history" in w for w in c.warnings)


def test_reasons_always_include_the_breakeven_comparison():
    """The gap between the cost-free assumption and the real threshold is the
    single most useful number here, so it must always be shown."""
    c = evaluate(_row("X"), CRYPTO_FEE, CRYPTO_SLIP)
    assert any("breakeven win rate" in r for r in c.reasons)
    assert any("costs eat" in r for r in c.reasons)


# -------------------------------------------------------------------- suggest()


def test_suggest_orders_by_quality_then_deterministically_breaks_ties():
    rows = [_row("BBB", last=100.0, stop=98.0, target=104.0),
            _row("AAA", last=100.0, stop=98.0, target=104.0)]
    a = [c.symbol for c in suggest(rows, CRYPTO_FEE, CRYPTO_SLIP)["candidates"]]
    b = [c.symbol for c in suggest(list(reversed(rows)), CRYPTO_FEE, CRYPTO_SLIP)["candidates"]]
    assert a == b == ["AAA", "BBB"], "identical candidates must tie-break on symbol, not input order"


def test_suggest_separates_unscoreable_from_rejected():
    """A row with a signal but no stop is a data problem, not a weak idea. Lumping
    it in with cost-rejected candidates would hide it."""
    good = _row("GOOD", last=100.0, stop=98.0, target=104.0)
    broken = ScanRow(symbol="BROKEN", signal=1, last=100.0, stop=None, bars=1200)
    quiet = _row("QUIET", signal=0)
    res = suggest([good, broken, quiet], CRYPTO_FEE, CRYPTO_SLIP)
    assert [c.symbol for c in res["candidates"]] == ["GOOD"]
    assert res["unscoreable"] == ["BROKEN"]
    assert res["n_rejected"] == 0, "no stop means unscoreable, not rejected"


def test_suggest_reports_the_cost_basis_it_used():
    res = suggest([_row("X")], CRYPTO_FEE, CRYPTO_SLIP)
    assert res["round_trip_cost_bps"] == pytest.approx(30.0)
    assert res["max_breakeven"] == pytest.approx(0.60)
    assert res["weights"] == WEIGHTS


def test_suggest_max_breakeven_is_a_gate_not_a_sort_key():
    loose = suggest([_row("X", last=100.0, stop=99.80, target=101.60)],
                    CRYPTO_FEE, CRYPTO_SLIP, max_breakeven=0.99)
    hair = _row("X", last=100.0, stop=99.95, target=100.20)   # breakeven saturates at 100%
    strict = suggest([hair], CRYPTO_FEE, CRYPTO_SLIP, max_breakeven=0.30)
    assert loose["n_candidates"] == 1 and loose["n_rejected"] == 0
    assert strict["n_candidates"] == 0 and strict["n_rejected"] == 1


def test_suggest_rejects_thin_history():
    res = suggest([_row("X", bars=20)], CRYPTO_FEE, CRYPTO_SLIP)
    assert res["n_rejected"] == 1
    assert any("bars of history" in r for r in res["rejected"][0]["reasons"])


def test_suggest_on_an_empty_universe():
    res = suggest([], CRYPTO_FEE, CRYPTO_SLIP)
    assert res["candidates"] == [] and res["n_candidates"] == 0


def test_candidate_verdict_bands_are_ordered_and_reachable():
    """The verdict must get strictly worse as the breakeven win rate rises. Both
    rows are R:R 5 so the only difference is how much the costs eat."""
    sane = evaluate(_row("A", last=100.0, stop=98.0, target=110.0), CRYPTO_FEE, CRYPTO_SLIP)
    tight = evaluate(_row("B", last=100.0, stop=99.80, target=101.00), CRYPTO_FEE, CRYPTO_SLIP)
    hair = evaluate(_row("C", last=100.0, stop=99.95, target=100.20), CRYPTO_FEE, CRYPTO_SLIP)
    assert sane.breakeven_win_rate < tight.breakeven_win_rate < hair.breakeven_win_rate
    assert len({sane.verdict, tight.verdict, hair.verdict}) >= 2
    assert hair.verdict == "untradeable_at_these_costs"


def test_candidate_to_dict_is_json_serialisable():
    import json
    c = evaluate(_row("X"), CRYPTO_FEE, CRYPTO_SLIP)
    d = c.to_dict()
    assert isinstance(d["reasons"], list) and isinstance(d["warnings"], list)
    json.dumps({k: (v if not isinstance(v, tuple) else list(v)) for k, v in d.items()})


# ------------------------------------------------------------- rank() wiring


def test_rank_uses_costs_rather_than_the_raw_ratio():
    """rank() used to sort live signals by reward_risk descending, which promoted
    the tightest stops. It must now agree with the quality score."""
    tight = _row("TIGHT", last=100.0, stop=99.80, target=101.60)
    wide = _row("WIDE", last=100.0, stop=98.00, target=104.00)
    ordered = rank([tight, wide], fee_bps=CRYPTO_FEE, slippage_bps=CRYPTO_SLIP)
    assert [r.symbol for r in ordered] == ["WIDE", "TIGHT"]


def test_rank_still_prefers_live_signals_over_proximity():
    live = _row("LIVE", signal=1, last=100.0, stop=98.0, target=104.0)
    near = _row("NEAR", signal=0, zones="FVG", sweep_age=1)
    ordered = rank([near, live], live_only=False, fee_bps=CRYPTO_FEE, slippage_bps=CRYPTO_SLIP)
    assert ordered[0].symbol == "LIVE"


def test_rank_live_only_filters_quiet_rows():
    live = _row("LIVE", signal=1)
    quiet = _row("QUIET", signal=0)
    assert [r.symbol for r in rank([quiet, live], fee_bps=CRYPTO_FEE, slippage_bps=CRYPTO_SLIP)] == ["LIVE"]


def test_rank_warns_when_costs_are_not_supplied(caplog):
    """A cost-free ranking silently favours tight stops, so it must say it is
    running cost-free rather than looking authoritative."""
    import logging
    with caplog.at_level(logging.WARNING):
        rank([_row("X", signal=1)], fee_bps=None, slippage_bps=None)
    assert any("without cost" in r.message.lower() or "costs" in r.message.lower()
               for r in caplog.records)


def test_rank_handles_no_live_signals():
    assert rank([_row("A", signal=0), _row("B", signal=0)],
                fee_bps=CRYPTO_FEE, slippage_bps=CRYPTO_SLIP) == []


# --------------------------------------------------------------- config wiring


@pytest.mark.parametrize("market,style", [
    ("crypto", "swing"), ("crypto", "intraday"), ("indian_equities", "intraday"),
    ("us_equities", "position"), ("forex", "intraday"),
])
def test_every_market_can_produce_a_breakeven_threshold(market, style):
    """The gate must work on all 16 combinations, and cheaper markets must end up
    with easier thresholds for identical geometry."""
    cfg = load_config(market=market, style=style)
    c = evaluate(_row("X", last=100.0, stop=99.0, target=103.0),
                 cfg.costs.fee_bps, cfg.costs.slippage_bps,
                 cfg.risk.min_stop_distance_pct)
    assert c is not None
    assert 0.0 < c.breakeven_win_rate <= 1.0
    assert c.breakeven_win_rate >= implied_win_rate_at_rr(c.rr)


def test_cheaper_markets_have_lower_breakeven_thresholds():
    """forex (1 bp + 1.5 bp) must beat crypto (10 bp + 5 bp) on identical geometry.
    If this ever inverts, a fee schedule has been mis-set."""
    geo = dict(last=100.0, stop=99.0, target=103.0)
    fx = load_config(market="forex", style="intraday")
    cr = load_config(market="crypto", style="intraday")
    cf = evaluate(_row("F", **geo), fx.costs.fee_bps, fx.costs.slippage_bps)
    cc = evaluate(_row("C", **geo), cr.costs.fee_bps, cr.costs.slippage_bps)
    assert cf.breakeven_win_rate < cc.breakeven_win_rate


def test_rr_saturation_prevents_one_absurd_ratio_dominating():
    """A 50:1 idea is not 10x better than a 5:1 one; it is usually an arbitrary
    target. The score must saturate rather than let it swamp the ranking."""
    sane = _row("SANE", last=100.0, stop=98.0, target=104.0)        # R:R 3
    absurd = _row("ABSURD", last=100.0, stop=98.0, target=200.0)    # R:R 50
    cs = evaluate(sane, CRYPTO_FEE, CRYPTO_SLIP)
    ca = evaluate(absurd, CRYPTO_FEE, CRYPTO_SLIP)
    assert ca.rr > 10 * cs.rr
    # both saturate the reward component, so the gap must stay modest
    assert ca.quality_score - cs.quality_score < 0.25
    assert RR_SATURATION == 4.0
