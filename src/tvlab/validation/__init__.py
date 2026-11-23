"""Validation: the checks that decide whether a backtest result means anything.

Run these BEFORE you look at a profit number. A strategy that fails the lookahead
test is not a strategy, it is a bug; a strategy that fails the shuffled-null test
has not been shown to have an edge.
"""

from . import lookahead, null_test, recovery
from .lookahead import (
    LeakReport,
    check_alignment,
    check_function,
    check_smc,
    check_strategy,
    check_technicals,
)
from .null_test import NullResult, interpret, shuffled_null, sign_null
from .recovery import RecoveryResult, run_smc_recovery, score_recovery

__all__ = [
    "lookahead", "null_test", "recovery",
    "LeakReport", "NullResult", "RecoveryResult",
    "check_function", "check_strategy", "check_smc", "check_technicals",
    "check_alignment", "shuffled_null", "sign_null", "interpret",
    "score_recovery", "run_smc_recovery",
]
