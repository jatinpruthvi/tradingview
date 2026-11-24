"""tvlab - a multi-market backtest, validation and screening workbench.

Built on top of the vendored upstream projects in ``vendor/`` (see
``vendor/LOCK.json``). The trading path is self-contained: indicators, the
backtester, risk, walk-forward validation and the dashboard are all first-party
code so that every fill and every number in a report can be audited.

Quick start::

    from tvlab.config import load_config
    from tvlab.data.store import DataStore
    from tvlab.pipeline import run_study

    cfg = load_config(market="crypto", style="swing")
    study = run_study(cfg)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
