"""Typed configuration for tvlab.

Configs are layered: ``config/default.yaml`` -> ``config/markets/<market>.yaml``
-> ``config/styles/<style>.yaml`` -> CLI overrides. Later layers win.

Everything is a frozen dataclass so a backtest can record exactly what it ran
with, and so nothing downstream can silently mutate the risk parameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

#: bars per year, used for annualising Sharpe/Sortino/CAGR
PERIODS_PER_YEAR: Dict[str, float] = {
    "1m": 525_600,   # 24/7 (crypto). Equity markets: use sessions_per_year below.
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "8h": 1_095,
    "12h": 730,
    "1d": 365,
    "1D": 365,
    "1w": 52,
}


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and drawdown governance.

    ``risk_per_trade`` is the fraction of *current* equity placed at risk on a
    single trade (i.e. the loss realised if the stop is hit), not the notional.
    """

    risk_per_trade: float = 0.005          # 0.5% - conservative default
    max_position_pct: float = 0.25         # cap notional at 25% of equity
    max_drawdown_pct: float = 0.10         # halt trading if equity DD exceeds 10%
    daily_loss_limit_pct: float = 0.02     # stop opening new trades for the day at -2%
    max_open_positions: int = 5
    max_correlated_positions: int = 2
    #: pairwise return correlation at or above which two symbols count as ONE bet
    #: for ``max_correlated_positions``. Calibrated per market: crypto alts sit
    #: around 0.6-0.8, FX majors around 0.1-0.3. A single global threshold is
    #: wrong in both directions - too high for crypto (the cap never fires, so
    #: ten alts are treated as ten independent bets) and too low for FX (it
    #: fires on pairs that really are diversifying).
    correlation_threshold: float = 0.60
    max_leverage: float = 1.0
    allow_short: bool = True
    #: scale exposure down linearly once DD passes this fraction of the hard limit
    dd_throttle_start: float = 0.5
    min_stop_distance_pct: float = 0.0015  # ignore absurdly tight stops
    max_stop_distance_pct: float = 0.25    # ignore absurdly wide stops


@dataclass(frozen=True)
class CostConfig:
    """Execution costs. Under-modelling these is the #1 source of fake edges."""

    fee_bps: float = 5.0        # per side, as basis points of notional
    slippage_bps: float = 2.0   # per side, adverse price move on fill
    min_commission: float = 0.0 # absolute floor per order (quote currency)


@dataclass(frozen=True)
class UniverseConfig:
    symbols: tuple = ()
    market: str = "crypto"
    exchange: str = "binance"
    quote_currency: str = "USDT"
    min_quote_volume: float = 1_000_000.0
    max_symbols: int = 20


@dataclass(frozen=True)
class TimeframeConfig:
    #: bar size signals are generated and executed on
    execution: str = "4h"
    #: higher timeframe used for the trend/regime filter (None = disable)
    bias: Optional[str] = "1d"
    #: lower timeframe for entry refinement (None = same as execution)
    entry: Optional[str] = None


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "smc_trend"
    params: Dict[str, Any] = field(default_factory=dict)
    #: which strategy ``params`` were written for. A style config sets both name
    #: and params together; if you then switch ``strategy.name`` the stale params
    #: must NOT follow, or e.g. buy-and-hold inherits a 90-bar time exit.
    #: Defaults to ``name`` at load time when not given explicitly.
    params_for: str = ""


@dataclass(frozen=True)
class BacktestConfig:
    start: Optional[str] = None
    end: Optional[str] = None
    initial_capital: float = 100_000.0
    #: how to resolve a bar that touches both stop and target
    #: "pessimistic" (default) assumes the stop is hit first
    intrabar_fill: str = "pessimistic"
    #: enter at the open of the bar *after* the signal bar (no lookahead)
    entry_on_next_open: bool = True
    max_bars_in_trade: Optional[int] = None
    seed: int = 42


@dataclass(frozen=True)
class WalkForwardConfig:
    enabled: bool = True
    n_splits: int = 5
    train_pct: float = 0.6
    rolling: bool = True
    min_train_bars: int = 500
    min_test_bars: int = 150
    #: optimisation objective: "deflated_sharpe" | "profit_factor" | "expectancy"
    objective: str = "deflated_sharpe"
    max_trials: int = 64


@dataclass(frozen=True)
class MonteCarloConfig:
    enabled: bool = True
    n_sims: int = 2_000
    confidence: float = 0.95
    #: resample whole blocks of trades to preserve short-term autocorrelation
    block_size: int = 20
    seed: int = 7


@dataclass(frozen=True)
class DataConfig:
    #: synthetic | csv | ccxt | yfinance | tvscreener
    source: str = "synthetic"
    cache_dir: str = "data/cache"
    synthetic_bars: int = 4_000
    synthetic_seed: int = 20261123


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "output"
    plots: bool = True
    dpi: int = 130


@dataclass(frozen=True)
class AppConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    timeframe: TimeframeConfig = field(default_factory=TimeframeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    #: Provenance only: which ``config/styles/*.yaml`` layer produced this config.
    #: Not a behaviour switch - every field above is already resolved. Recorded so
    #: reports, artifacts and the dashboard can name the setup they came from.
    style: Optional[str] = None

    # ------------------------------------------------------------------ utils
    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(asdict(self))

    def fingerprint(self) -> str:
        """Stable hash of the config, used to name and de-duplicate runs."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def merge(self, overrides: Dict[str, Any]) -> "AppConfig":
        """Return a new config with ``overrides`` applied (dotted keys allowed).

        Setting ``strategy.name`` without also setting params marks the existing
        params as belonging to the *old* strategy, so they are not reused.
        """
        raw = _to_plain(asdict(self))
        for key, value in (overrides or {}).items():
            _deep_set(raw, key, value)
        strat = raw.get("strategy") or {}
        touched_params = any(k.startswith("strategy.params") for k in (overrides or {}))
        if (strat.get("name") != strat.get("params_for") and not touched_params
                and "strategy.params_for" not in (overrides or {})):
            # keep params_for pointing at the strategy that declared them
            strat.setdefault("params_for", self.strategy.params_for or self.strategy.name)
            raw["strategy"] = strat
        cfg = AppConfig(
            style=raw.get("style", self.style),
            **{n: _coerce(c, raw.get(n) or {}) for n, c in _SECTION_TYPES.items()},
        )
        return cfg

    def periods_per_year(self) -> float:
        return PERIODS_PER_YEAR.get(self.timeframe.execution, 365.0)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _deep_set(raw: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = raw
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


_SECTION_TYPES = {
    "risk": RiskConfig,
    "costs": CostConfig,
    "universe": UniverseConfig,
    "timeframe": TimeframeConfig,
    "strategy": StrategyConfig,
    "backtest": BacktestConfig,
    "walk_forward": WalkForwardConfig,
    "monte_carlo": MonteCarloConfig,
    "data": DataConfig,
    "output": OutputConfig,
}


_SCALAR_CASTS = {"int": int, "float": float, "str": str}
_TRUTHY = {"true", "yes", "y", "1", "on"}
_FALSY = {"false", "no", "n", "0", "off"}
_NULLISH = {"", "none", "null", "~", "nan"}


def _coerce_value(annotation: Any, value: Any) -> Any:
    """Coerce a scalar to its declared field type.

    Overrides arrive as strings from two very different places: the CLI's
    ``--set`` flag and the dashboard's JSON POST body (a browser ``<input>``
    yields ``"777"``, not ``777``). Without coercion a string silently lands in
    an ``int`` field and explodes deep inside pandas - far from the typo that
    caused it. Coercing here makes the failure immediate and legible, and lets
    ``None`` be expressed as ``"null"`` for Optional fields.
    """
    name = str(annotation).replace("typing.", "")
    optional = name.startswith("Optional[")
    inner = name[len("Optional["):-1] if optional else name

    if isinstance(value, str):
        if value.strip().lower() in _NULLISH:
            return None if optional else value
        target = inner if optional else name
        if target == "bool":
            low = value.strip().lower()
            if low in _TRUTHY:
                return True
            if low in _FALSY:
                return False
            raise ValueError(f"cannot interpret {value!r} as a boolean")
        cast = _SCALAR_CASTS.get(target)
        if cast is not None:
            try:
                return cast(value.strip())
            except (TypeError, ValueError):
                raise ValueError(f"cannot interpret {value!r} as {target}") from None
    if name == "bool" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if inner == "float" and isinstance(value, int):
        return float(value)
    return value


def _coerce(cls, payload: Dict[str, Any]):
    """Build a dataclass, ignoring unknown keys and converting sequences."""
    if payload is None:
        return cls()
    valid = {f.name for f in fields(cls)}
    clean = {k: v for k, v in payload.items() if k in valid}
    for f in fields(cls):
        if f.name not in clean:
            continue
        # tuples declared in dataclasses arrive as lists from YAML
        if "tuple" in str(f.type) and isinstance(clean[f.name], list):
            clean[f.name] = tuple(clean[f.name])
        else:
            clean[f.name] = _coerce_value(f.type, clean[f.name])
    unknown = set(payload) - valid
    if unknown:
        import warnings

        warnings.warn(f"{cls.__name__}: ignoring unknown keys {sorted(unknown)}", stacklevel=2)
    return cls(**clean)


def from_dict(raw: Dict[str, Any]) -> AppConfig:
    kwargs = {}
    for name, cls in _SECTION_TYPES.items():
        kwargs[name] = _coerce(cls, raw.get(name) or {})
    strat = kwargs.get("strategy")
    if strat is not None and not strat.params_for:
        # params declared alongside a name belong to that name
        kwargs["strategy"] = StrategyConfig(
            name=strat.name, params=dict(strat.params), params_for=strat.name
        )
    return AppConfig(**kwargs)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    market: Optional[str] = None,
    style: Optional[str] = None,
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Layer default -> market -> style -> explicit file -> overrides."""
    merged: Dict[str, Any] = {}
    layers = [CONFIG_DIR / "default.yaml"]
    if market:
        layers.append(CONFIG_DIR / "markets" / f"{market}.yaml")
    if style:
        layers.append(CONFIG_DIR / "styles" / f"{style}.yaml")
    if config_path:
        layers.append(Path(config_path))

    for layer in layers:
        data = load_yaml(layer)
        if not data and layer not in (CONFIG_DIR / "default.yaml",):
            # a market/style that does not exist is a user typo, not a soft default
            if market and layer.name == f"{market}.yaml":
                raise FileNotFoundError(
                    f"No market config '{market}'. Available: "
                    f"{sorted(p.stem for p in (CONFIG_DIR / 'markets').glob('*.yaml'))}"
                )
            if style and layer.name == f"{style}.yaml":
                raise FileNotFoundError(
                    f"No style config '{style}'. Available: "
                    f"{sorted(p.stem for p in (CONFIG_DIR / 'styles').glob('*.yaml'))}"
                )
        merged = _deep_merge(merged, data)

    cfg = from_dict(merged)
    if style:
        cfg = replace(cfg, style=style)
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
