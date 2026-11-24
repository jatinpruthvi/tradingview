"""Data store: one entry point that dispatches to the right adapter and caches.

    store = DataStore(cfg)
    df = store.ohlcv("BTC/USDT", "4h")            # cached parquet under data/cache
    bias = store.bias_frame("BTC/USDT")            # higher timeframe, lookahead-safe

Cache keys include the source, symbol, timeframe, range and adapter options, so
changing config never serves stale bars.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config import AppConfig, REPO_ROOT
from .base import (
    DataError,
    resample,
    slice_range,
    timeframe_seconds,
    validate_ohlcv,
)
from .csv_source import CsvSource, _symbol_filenames
from .synthetic import DEFAULT_BASKET, basket

log = logging.getLogger(__name__)

#: Bump this whenever the *meaning* of a cached bar changes, or stale entries
#: will be served silently. The cache key covers the request parameters but
#: cannot see the code that produced them: after the synthetic generator gained
#: a shared market factor, previously cached frames were still the old
#: independent per-symbol paths, so a 1500-bar run measured ~0.00 cross-asset
#: correlation while a 1800-bar run measured ~0.67 - same config, same code,
#: different answer, with nothing to say which was stale.
#:
#:   v2 - synthetic baskets share a per-asset-class market factor (correlation)
#:   v3 - bias frames are native HTF bars, never forward-filled onto execution
CACHE_SCHEMA_VERSION = "v3"

#: Largest tolerated mean absolute percentage difference between a natively
#: fetched higher-timeframe series and the same series resampled up from the
#: execution bars. Real data agrees to well under 1% (only session-boundary and
#: adjustment conventions differ); two unrelated paths differ by tens of percent.
_MAX_BIAS_DRIFT = 0.02


def _series_drift(bias_df: pd.DataFrame, exec_df: pd.DataFrame,
                  bias_tf: str) -> Optional[float]:
    """Mean absolute percentage difference between two supposedly-same series.

    Returns ``None`` when there is no usable overlap to compare, which callers
    must treat as "cannot verify" rather than "verified identical".
    """
    try:
        ref = resample(exec_df, bias_tf)["close"]
        native = bias_df["close"]
        joined = pd.concat([native, ref], axis=1, join="inner").dropna()
        joined.columns = ["native", "ref"]
        if len(joined) < 10:
            return None
        denom = joined["ref"].abs().replace(0.0, pd.NA)
        rel = ((joined["native"] - joined["ref"]).abs() / denom).dropna()
        if len(rel) < 10:
            return None
        return float(rel.mean())
    except Exception:  # noqa: BLE001 - a comparison failure must never kill the run
        return None


def _csv_root(cfg: "AppConfig") -> Path:
    """Resolve `data.csv_root`, with a fallback to the historical location.

    Relative paths resolve against the repo root rather than the process CWD, so
    the same config works from a script, a cron job and the dashboard. The old
    hardcoded `<cache_dir>/csv` is still honoured if it exists and the configured
    root does not, so an existing setup keeps loading after the upgrade instead of
    failing on a path it never chose.
    """
    configured = Path(cfg.data.csv_root)
    if not configured.is_absolute():
        configured = REPO_ROOT / configured
    if configured.exists():
        return configured

    legacy = Path(cfg.data.cache_dir)
    if not legacy.is_absolute():
        legacy = REPO_ROOT / legacy
    legacy = legacy / "csv"
    if legacy.exists():
        log.warning("data.csv_root=%s does not exist; falling back to the legacy "
                    "location %s. Set data.csv_root to where your bars actually are.",
                    configured, legacy)
        return legacy

    return configured      # let CsvSource raise, naming the path it wanted


class DataStore:
    def __init__(self, cfg: AppConfig, root: Optional[Path] = None):
        self.cfg = cfg
        self.root = Path(root) if root else (REPO_ROOT / cfg.data.cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sources: Dict[str, object] = {}
        self._mem: Dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- sources
    def source(self, name: Optional[str] = None):
        """Lazily build (and memoise) a data source adapter by name."""
        name = (name or self.cfg.data.source).lower()
        if name in self._sources:
            return self._sources[name]

        if name == "synthetic":
            src = _SyntheticAdapter(self.cfg)
        elif name == "csv":
            src = CsvSource(_csv_root(self.cfg))
        elif name in ("ccxt", "exchange"):
            from .network_sources import CcxtSource

            src = CcxtSource(exchange=self.cfg.universe.exchange)
        elif name in ("yfinance", "yahoo"):
            from .network_sources import YFinanceSource

            src = YFinanceSource()
        elif name in ("tvscreener", "tv"):
            from .tv_screener_source import TvScreenerSource

            src = TvScreenerSource(market=self.cfg.universe.market)
        elif name in ("nse", "nseindia", "india"):
            from .nse import NseSource

            src = NseSource(index=self.cfg.universe.nse_index,
                            market=self.cfg.universe.market)
        elif name in ("yahoo_chart", "yahoo-chart", "ychart"):
            from .nse import YahooChartSource

            src = YahooChartSource(market=self.cfg.universe.market)
        else:
            raise DataError(
                f"unknown data source {name!r}. "
                f"Options: synthetic, csv, ccxt, yfinance, tvscreener, nse, yahoo_chart"
            )
        self._sources[name] = src
        return src

    # ------------------------------------------------------------------ fetch
    def ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        timeframe = timeframe or self.cfg.timeframe.execution
        start = start or self.cfg.backtest.start
        end = end or self.cfg.backtest.end

        key = self._cache_key(symbol, timeframe, start, end, limit, source)
        if use_cache and key in self._mem:
            return self._mem[key].copy()

        path = self.root / f"{key}.parquet"
        if use_cache and path.exists():
            try:
                df = pd.read_parquet(path)
                df.index = pd.to_datetime(df.index, utc=True)
                self._mem[key] = df
                log.debug("cache hit %s", path.name)
                return df.copy()
            except Exception as exc:  # noqa: BLE001
                log.warning("unreadable cache %s (%s); refetching", path.name, exc)

        src = self.source(source)
        df = src.fetch_ohlcv(symbol, timeframe, start=start, end=end, limit=limit)
        df = validate_ohlcv(df, symbol)
        df = slice_range(df, start, end, limit)
        if len(df) < 60:
            raise DataError(
                f"{symbol} {timeframe}: only {len(df)} bars returned - not enough to "
                f"compute indicators and backtest meaningfully (need >= 60)"
            )

        df.attrs["symbol"] = symbol
        df.attrs["timeframe"] = timeframe
        df.attrs["source"] = getattr(src, "name", source or self.cfg.data.source)

        self._mem[key] = df
        if use_cache:
            try:
                df.to_parquet(path)
            except Exception as exc:  # noqa: BLE001 - pyarrow optional
                log.debug("parquet cache write skipped (%s)", exc)
        return df.copy()

    def bias_frame(self, symbol: str, source: Optional[str] = None) -> Optional[pd.DataFrame]:
        """NATIVE higher-timeframe bars for the trend filter.

        Returns the HTF frame on its own index - NOT forward-filled onto the
        execution index. Callers compute indicators on these native bars and then
        align the resulting *indicator* with
        :func:`tvlab.data.base.align_indicator`, which shifts by one HTF bar so
        execution bars only ever see a completed HTF bar.

        Forward-filling HTF prices first and computing indicators afterwards is
        the wrong order: it turns a 50-period daily EMA into a 50-execution-bar
        EMA, which is a completely different (and much faster) filter.

        If the venue cannot serve the bias timeframe natively - or the bars it
        does serve do not cover the execution window - we resample up from the
        execution frame instead.
        """
        bias_tf = self.cfg.timeframe.bias
        exec_tf = self.cfg.timeframe.execution
        if not bias_tf:
            return None
        try:
            exec_df = self.ohlcv(symbol, exec_tf, source=source)
        except DataError:
            return None

        src = source or self.cfg.data.source
        bias_df: Optional[pd.DataFrame] = None
        if src == "synthetic":
            # The synthetic generator draws an independent path per (symbol,
            # timeframe): a seeded daily series is NOT a resample of the seeded
            # 4h series. Fetching the bias timeframe natively therefore hands
            # the trend filter a different, uncorrelated market - which gates
            # every entry on noise and quietly invalidates the whole run. For
            # synthetic data the bias frame must always be resampled up from the
            # execution frame so the two are the same path by construction.
            log.debug("%s: synthetic source - resampling %s bias from %s", symbol, bias_tf, exec_tf)
        else:
            try:
                bias_df = self.ohlcv(symbol, bias_tf, source=source)
            except DataError as exc:
                log.info("no native %s bars for %s (%s); resampling from %s",
                         bias_tf, symbol, exc, exec_tf)

        # coverage check: a bias frame that stops short of the execution window
        # would leave the tail of the backtest with a stale trend filter
        if bias_df is not None and len(bias_df):
            covers_end = bias_df.index.max() >= exec_df.index.max() - pd.Timedelta(
                seconds=timeframe_seconds(bias_tf)
            )
            covers_start = bias_df.index.min() <= exec_df.index.min()
            if not (covers_end and covers_start):
                log.info(
                    "%s: native %s range [%s..%s] does not cover execution range [%s..%s]; resampling",
                    symbol, bias_tf, bias_df.index.min().date(), bias_df.index.max().date(),
                    exec_df.index.min().date(), exec_df.index.max().date(),
                )
                bias_df = None

        # consistency check: covering the right DATES is not the same as being
        # the same SERIES. A vendor can return a different instrument, a
        # different session convention or an unadjusted contract for the HTF
        # request. Compare against the resampled execution frame on the overlap
        # and fall back to resampling if they disagree.
        if bias_df is not None and len(bias_df):
            drift = _series_drift(bias_df, exec_df, bias_tf)
            if drift is not None and drift > _MAX_BIAS_DRIFT:
                log.warning(
                    "%s: native %s bars disagree with resampled %s bars by %.1f%% "
                    "on the overlap - not the same series; resampling instead",
                    symbol, bias_tf, exec_tf, drift * 100,
                )
                bias_df = None

        if bias_df is None:
            bias_df = resample(exec_df, bias_tf)
            bias_df = validate_ohlcv(bias_df, symbol)

        if len(bias_df) < 30:
            log.warning("%s: only %d bias bars; trend filter disabled", symbol, len(bias_df))
            return None

        bias_df = bias_df.copy()
        bias_df.attrs.update({"symbol": symbol, "timeframe": bias_tf,
                              "exec_start": exec_df.index.min(), "exec_end": exec_df.index.max()})
        return bias_df

    def basket(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch a whole universe, skipping (and reporting) symbols that fail."""
        syms = list(symbols or self.cfg.universe.symbols) or list(DEFAULT_BASKET)
        out: Dict[str, pd.DataFrame] = {}
        failures: List[Tuple[str, str]] = []
        for s in syms:
            try:
                out[s] = self.ohlcv(s, timeframe, source=source)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the run
                failures.append((s, f"{type(exc).__name__}: {exc}"))
                log.warning("skipping %s: %s", s, exc)
        if not out:
            raise DataError(
                "no symbols could be loaded:\n  " + "\n  ".join(f"{s}: {e}" for s, e in failures)
            )
        if failures:
            log.warning("loaded %d/%d symbols; failed: %s", len(out), len(syms),
                        [s for s, _ in failures])
        return out

    # ------------------------------------------------------------------ cache
    def _source_identity(self, symbol, timeframe, src: str) -> str:
        """Everything about the *source* that can change the bytes it returns.

        The key already names the source, but naming it is not the same as identifying
        it. For a CSV root two things are missing, and both are real rather than
        theoretical:

        * **Which directory.** Two different ``data.csv_root`` values hash to one key,
          so the second silently serves the first's bars. Observed while verifying
          ``tvlab review``: a 120-bar fixture directory was handed a 400-bar frame
          cached from an unrelated directory, because the key said only ``csv``.
        * **Whether the files changed.** Appending newly accumulated history to a file
          that is already cached leaves the key untouched, so the new bars never
          appear. That is precisely the workflow ``tvlab review`` documents - journal
          daily, let the CSV grow, then score - and it would have scored every
          suggestion against the history as it stood the first time it was read.

        Fingerprinting by ``(name, mtime_ns, size)`` rather than by content keeps this
        cheap enough to run on every request while still invalidating on any rewrite.
        """
        if src != "csv":
            return ""
        root = _csv_root(self.cfg)
        parts = [str(root)]
        names = _symbol_filenames(str(symbol))
        cands = [f"{n}_{timeframe}" for n in names] + list(names) if timeframe \
            else list(names)
        for cand in cands:
            for ext in (".csv", ".parquet", ".pq", ".jsonl", ".json"):
                p = root / f"{cand}{ext}"
                try:
                    st = p.stat()
                except OSError:
                    continue
                parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
        return "|".join(parts)

    def _cache_key(self, symbol, timeframe, start, end, limit, source) -> str:
        src = source or self.cfg.data.source
        blob = "|".join([
            CACHE_SCHEMA_VERSION,
            src, str(symbol), str(timeframe), str(start), str(end), str(limit),
            str(self.cfg.universe.exchange), str(self.cfg.universe.market),
            str(self.cfg.data.synthetic_bars), str(self.cfg.data.synthetic_seed),
            self._source_identity(symbol, timeframe, src),
        ])
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(symbol))[:40]
        return f"{safe}_{timeframe}_{src}_{digest}"

    def clear_cache(self) -> int:
        n = 0
        for p in self.root.glob("*.parquet"):
            p.unlink(missing_ok=True)
            n += 1
        self._mem.clear()
        return n


class _SyntheticAdapter:
    """Adapter that makes the synthetic generator look like a network source."""

    name = "synthetic"

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def fetch_ohlcv(self, symbol, timeframe="4h", start=None, end=None, limit=None) -> pd.DataFrame:
        n = limit or self.cfg.data.synthetic_bars
        start = start or "2022-01-01"
        frames = basket(
            symbols=[symbol],
            n_bars=n,
            timeframe=timeframe,
            start=start,
            end=end,
            seed=self.cfg.data.synthetic_seed,
        )
        return frames[symbol]

    def list_symbols(self) -> List[str]:
        syms = list(self.cfg.universe.symbols)
        return syms or list(DEFAULT_BASKET)
