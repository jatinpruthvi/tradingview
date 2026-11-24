"""Config layering. A wrong default here silently invalidates every backtest."""
import itertools

import pytest

from tvlab.config import (AppConfig, CONFIG_DIR, REPO_ROOT, _deep_merge, load_config)
from tvlab.data.base import bars_per_year


def fingerprint(cfg):
    return cfg.fingerprint()

MARKETS = ("crypto", "indian_equities", "us_equities", "forex")
STYLES = ("intraday", "swing", "position", "hft")


@pytest.mark.parametrize("market,style", list(itertools.product(MARKETS, STYLES)))
def test_every_market_style_combo_loads(market, style):
    cfg = load_config(market=market, style=style)
    assert isinstance(cfg, AppConfig)
    assert cfg.universe.market == market
    assert cfg.style == style, "the config must record which style layer built it"


@pytest.mark.parametrize("market,style", list(itertools.product(MARKETS, STYLES)))
def test_risk_mandate_is_not_overridden_by_any_market(market, style):
    """The conservative mandate must hold in all 16 combos.

    ``risk_per_trade``, ``max_drawdown_pct`` and ``max_leverage`` are hard
    requirements and may not vary. ``daily_loss_limit_pct`` is style-tuned, but
    only ever TIGHTER than the 2% baseline - a style is allowed to be more
    conservative, never less. (The position style used to set 3%.)
    """
    cfg = load_config(market=market, style=style)
    assert cfg.risk.risk_per_trade == 0.005
    assert cfg.risk.max_drawdown_pct == 0.10
    assert cfg.risk.max_leverage == 1.0
    assert cfg.risk.daily_loss_limit_pct <= 0.02, (
        f"{market}/{style} loosens the daily loss limit to "
        f"{cfg.risk.daily_loss_limit_pct:.3f}")
    assert cfg.risk.max_position_pct <= 0.25


def test_timeframe_is_style_driven():
    assert load_config(market="crypto", style="hft").timeframe.execution == "1m"
    assert load_config(market="crypto", style="intraday").timeframe.execution == "5m"
    assert load_config(market="crypto", style="swing").timeframe.execution == "4h"
    assert load_config(market="crypto", style="position").timeframe.execution == "1d"


def test_bias_timeframe_is_higher_than_execution():
    for style in STYLES:
        cfg = load_config(market="crypto", style=style)
        assert cfg.timeframe.bias, style
        assert cfg.timeframe.bias != cfg.timeframe.execution, style


def test_long_only_market_disables_shorts():
    assert load_config(market="indian_equities", style="swing").risk.allow_short is False
    assert load_config(market="crypto", style="swing").risk.allow_short is True


def test_costs_differ_by_market_and_are_nonzero():
    c = {m: load_config(market=m, style="swing").costs for m in MARKETS}
    assert c["crypto"].fee_bps > c["us_equities"].fee_bps
    assert all(v.fee_bps > 0 for v in c.values()), "zero fees flatter every result"
    assert all(v.slippage_bps > 0 for v in c.values())


def test_hft_walk_forward_is_disabled():
    """1m bars carry far less independent information; fitting params there is
    close to pure noise mining, so the style turns it off on purpose."""
    assert load_config(market="crypto", style="hft").walk_forward.enabled is False
    assert load_config(market="crypto", style="swing").walk_forward.enabled is True


def test_dotted_override_and_type_coercion():
    cfg = load_config(market="crypto", style="swing")
    out = cfg.merge({"data.synthetic_bars": 500, "risk.risk_per_trade": 0.01,
                     "strategy.name": "donchian_breakout"})
    assert out.data.synthetic_bars == 500
    assert isinstance(out.data.synthetic_bars, int)
    assert out.risk.risk_per_trade == 0.01
    assert out.strategy.name == "donchian_breakout"


@pytest.mark.parametrize("override,attr,expected", [
    ({"data.synthetic_bars": "777"}, "data.synthetic_bars", 777),
    ({"risk.risk_per_trade": "0.0025"}, "risk.risk_per_trade", 0.0025),
    ({"risk.allow_short": "false"}, "risk.allow_short", False),
    ({"risk.allow_short": "true"}, "risk.allow_short", True),
    ({"backtest.max_bars_in_trade": "null"}, "backtest.max_bars_in_trade", None),
])
def test_overrides_from_strings_are_coerced(override, attr, expected):
    """Both the CLI (--set) and the dashboard (JSON from a browser <input>) can
    deliver strings. A string in an int field explodes later, inside pandas,
    far from the typo that caused it - so coerce at the boundary."""
    cfg = load_config(market="crypto", style="swing")
    section, field = attr.split(".")
    got = getattr(getattr(cfg.merge(override), section), field)
    assert got == expected and type(got) is type(expected), f"{attr}: {got!r}"


def test_unparseable_override_fails_loudly():
    cfg = load_config(market="crypto", style="swing")
    with pytest.raises(ValueError, match="boolean"):
        cfg.merge({"risk.allow_short": "maybe"})
    with pytest.raises(ValueError, match="int"):
        cfg.merge({"data.synthetic_bars": "lots"})


def test_merge_does_not_mutate_the_original():
    cfg = load_config(market="crypto", style="swing")
    before = cfg.data.synthetic_bars
    cfg.merge({"data.synthetic_bars": 42})
    assert cfg.data.synthetic_bars == before


def test_params_for_prevents_cross_strategy_leakage():
    """Regression: a style sets strategy.name and strategy.params together, so
    overriding only the name left the old strategy's params in place.
    buy_and_hold then inherited max_bars_in_trade=90 and stopped being a benchmark."""
    cfg = load_config(market="crypto", style="swing")
    assert cfg.strategy.name == "smc_trend"
    assert cfg.strategy.params_for == "smc_trend"
    out = cfg.merge({"strategy.name": "buy_and_hold"})
    assert out.strategy.name == "buy_and_hold"
    assert out.strategy.params_for != "buy_and_hold"


def test_fingerprint_changes_with_any_meaningful_edit():
    a = load_config(market="crypto", style="swing")
    b = load_config(market="crypto", style="swing")
    assert fingerprint(a) == fingerprint(b), "fingerprint must be deterministic"
    c = a.merge({"risk.risk_per_trade": 0.02})
    assert fingerprint(a) != fingerprint(c)
    d = load_config(market="forex", style="swing")
    assert fingerprint(a) != fingerprint(d)


def test_to_dict_round_trips_through_from_dict():
    from tvlab.config import from_dict
    cfg = load_config(market="us_equities", style="position")
    again = from_dict(cfg.to_dict())
    # style is provenance, not behaviour, so compare everything else
    assert fingerprint(from_dict(again.to_dict())) == fingerprint(again)
    assert again.strategy == cfg.strategy
    assert again.risk == cfg.risk
    assert again.timeframe == cfg.timeframe


def test_to_dict_is_plain_json_types():
    import json
    cfg = load_config(market="crypto", style="swing")
    d = cfg.to_dict()
    json.dumps(d, default=str)              # no dataclass/path leakage
    assert isinstance(d["risk"]["risk_per_trade"], float)
    assert isinstance(d["universe"]["symbols"], list)


def test_periods_per_year_matches_the_market_calendar():
    """Annualisation drives Sharpe/CAGR. Getting this wrong makes a 4h crypto
    strategy look ~6x better or worse than it is."""
    assert bars_per_year("1d", "crypto") == 365.0
    assert bars_per_year("1d", "us") == 252.0
    assert bars_per_year("4h", "crypto") == pytest.approx(2190.0)
    assert bars_per_year("5m", "crypto") > bars_per_year("1d", "crypto") * 100
    # the config's own convenience method must agree with the table
    for style in ("intraday", "swing", "position", "hft"):
        cfg = load_config(market="crypto", style=style)
        assert cfg.periods_per_year() == pytest.approx(
            bars_per_year(cfg.timeframe.execution, cfg.universe.market))


def test_layer_merge_is_deep_not_shallow():
    """A style layer overriding strategy.params must not wipe the market layer's
    other strategy keys - a shallow merge would silently drop them."""
    out = _deep_merge({"strategy": {"name": "a", "params": {"x": 1, "y": 2}}},
                      {"strategy": {"params": {"y": 3}}})
    assert out == {"strategy": {"name": "a", "params": {"x": 1, "y": 3}}}
    # and the inputs must not be mutated
    base = {"strategy": {"params": {"x": 1}}}
    _deep_merge(base, {"strategy": {"params": {"x": 9}}})
    assert base["strategy"]["params"]["x"] == 1


def test_repo_paths_resolve():
    assert (REPO_ROOT / "README.md").exists()
    assert CONFIG_DIR.is_dir()
    assert (CONFIG_DIR / "default.yaml").exists()


def test_unknown_market_or_style_fails_loudly():
    with pytest.raises(Exception):
        load_config(market="martian_futures", style="swing")
    with pytest.raises(Exception):
        load_config(market="crypto", style="yolo")
