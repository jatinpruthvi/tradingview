"""Which symbols does "the market" mean? One answer, two front ends.

The CLI learned this the hard way: ``tvlab scan --data-source nse`` screened the
12 hand-typed large caps in ``config/markets/indian_equities.yaml`` while NSE
listed 502, and the table still read like a statement about the market. The
dashboard had the same bug in a different file, because each front end resolved
the universe itself.

So resolution lives here, once, and both call it:

    syms, note = resolve_universe(cfg, mode="auto")

``note`` is returned rather than derived by the caller because it is the honest
part of the answer - "502 symbol(s) listed by 'nse'" versus "12 symbol(s) from
the us_equities config" are different claims, and a caller that reconstructs the
provenance from a length gets it wrong on the day the two happen to agree.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

from ..config import AppConfig
from ..data.store import DataStore

log = logging.getLogger(__name__)

#: Sources whose ``list_symbols`` enumerates the *exchange's* tradeable universe
#: rather than some incidental collection. Auto-resolution is restricted to these:
#: ``ccxt`` can list every pair on a venue (thousands), which is not "the market"
#: in the sense the config means, and ``yfinance`` cannot list at all.
ENUMERABLE_SOURCES = frozenset({"nse"})

MODES = ("auto", "config", "source")


def resolve_universe(cfg: AppConfig,
                     symbols: Optional[Union[str, Sequence[str]]] = None,
                     mode: Optional[str] = "auto",
                     max_symbols: Optional[int] = None) -> Tuple[List[str], str]:
    """``(symbols, note)`` for a scan, a briefing or a chart.

    An explicit list always wins. Otherwise ``mode="auto"`` uses the data
    source's own listing when that source can enumerate the exchange, and the
    configured list everywhere else. ``config`` and ``source`` force either half.

    ``max_symbols=None`` defers to ``cfg.universe.max_symbols`` (0 = no cap);
    an integer overrides it. Raises :class:`ValueError` for an unknown mode -
    the CLI turns that into a usage error, the dashboard into a 400.
    """
    if isinstance(symbols, str):
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if syms:
            return syms, f"{len(syms)} symbol(s) from the explicit list"
    elif symbols:
        syms = [str(s) for s in symbols if s]
        return syms, f"{len(syms)} symbol(s) from the explicit list"

    want = (mode or "auto").lower()
    if want not in MODES:
        raise ValueError(f"universe must be one of {', '.join(MODES)}; got {mode!r}")
    use_source = (want == "source"
                  or (want == "auto" and cfg.data.source.lower() in ENUMERABLE_SOURCES))

    if not use_source:
        syms = list(cfg.universe.symbols)
        return syms, f"{len(syms)} symbol(s) from the {cfg.universe.market} config"

    listed, origin = DataStore(cfg).universe(prefer_source=True)
    if origin != "source":
        return listed, (f"{len(listed)} symbol(s) from the {cfg.universe.market} config "
                        f"(data source '{cfg.data.source}' lists no universe)")

    cap = cfg.universe.max_symbols if max_symbols is None else max_symbols
    capped = listed[:cap] if cap and cap > 0 else listed
    note = f"{len(capped)} symbol(s) listed by '{cfg.data.source}'"
    if len(capped) < len(listed):
        note += f", capped from {len(listed)}"
    return capped, note
