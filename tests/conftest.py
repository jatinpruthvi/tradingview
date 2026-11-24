"""Shared fixtures. Everything is deterministic and offline by design."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tvlab.config import load_config          # noqa: E402
from tvlab.data.synthetic import basket       # noqa: E402
from tvlab.data.store import DataStore        # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(market="crypto", style="swing")


@pytest.fixture(scope="session")
def ohlcv():
    return basket(["BTC/USDT"], n_bars=1200, timeframe="4h")["BTC/USDT"]


@pytest.fixture(scope="session")
def store(cfg):
    return DataStore(cfg)
