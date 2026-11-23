"""Strategies and the registry.

Importing this package registers every strategy, so ``get("smc_trend")`` works
without the caller having to know which module it lives in.
"""

from . import setups  # noqa: F401  (side effect: populates REGISTRY)
from .base import REGISTRY, Strategy, available, blank_signals, get, register, trend_filter

__all__ = ["Strategy", "REGISTRY", "get", "register", "available",
           "blank_signals", "trend_filter", "setups"]
