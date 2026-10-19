"""Regime for every instrument that has cached candles (indices, watchlist stocks, MCX commodities).

Reads only the local Parquet cache: no network, no broker call, nothing written. Each instrument is validated by the
same data-quality gate as the pipeline; an instrument whose data fails it shows its status and NO regime (nothing is
substituted). Display only: the regime is not an input to the trading decision.
"""
from __future__ import annotations

import threading
import time

import pandas as pd

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.config.watchlist import COMMODITY_SYMBOLS, WATCHLIST_STOCKS
from quant_intelligence.data.data_manager import CACHE_DIR, _read_cache
from quant_intelligence.data.provenance import classify_mode
from quant_intelligence.data.quality import QUALITY_FAIL, clean_ohlcv, validate_ohlcv
from quant_intelligence.features.feature_engine import compute_features
from quant_intelligence.regimes.regime_engine import classify_series
from quant_intelligence.utils.market_profile import profile_for
from quant_intelligence.utils.timeutil import now_ist

LOOKBACK_DAYS = 60
MIN_BARS = 120
UNIVERSE_TTL_SEC = 300.0


def universe_symbols() -> list[tuple[str, str]]:
    """(symbol, kind) for every configured instrument: option underlyings first, then stocks, then commodities."""
    out = [(s, "index") for s in SETTINGS.option_underlyings]
    out += [(s["symbol"], "stock") for s in WATCHLIST_STOCKS]
    out += [(s, "commodity") for s in sorted(COMMODITY_SYMBOLS)]
    return list(dict.fromkeys(out))


def _timeframe_minutes(timeframe: str) -> int:
    digits = "".join(ch for ch in timeframe if ch.isdigit())
    return int(digits) if digits else 5


def regime_for_instrument(symbol: str, kind: str, timeframe: str = "5min") -> dict:
    row = {"instrument": symbol, "kind": kind, "data_status": "NO DATA", "category": "UNAVAILABLE", "rows": 0,
           "last_bar": None, "mode": None, "source": "cache", "regime": None, "confidence": None, "clarity": None, "direction": None,
           "volatility": None, "volatility_trend": None, "age_bars": None, "note": None}
    path = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        row["note"] = "no cached candles"
        return row
    try:
        raw = _read_cache(path)
        if raw is None or raw.empty:
            row["note"] = "cache unreadable or empty"
            return row
        tf = _timeframe_minutes(timeframe)
        profile = profile_for(symbol)
        # Same order as DataManager._finalize: clean first (drops duplicates, out-of-session prints and a bad opening
        # bar), then judge the cleaned lookback window.
        df = clean_ohlcv(raw, tf, profile)
        df = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=LOOKBACK_DAYS)]
        report = validate_ohlcv(df, tf, profile=profile)
        row["data_status"], row["category"] = report.status, report.category
        row["note"] = "; ".join(report.issues[:2]) or None
        if report.status == QUALITY_FAIL:
            return row  # failed data gets no regime
        row["rows"], row["last_bar"] = int(len(df)), df["timestamp"].max()
        row["mode"] = classify_mode(row["last_bar"], now_ist(), profile, tf)
        if len(df) < MIN_BARS:
            row["note"] = f"only {len(df)} bars (need {MIN_BARS})"
            return row
        series = classify_series(compute_features(df, profile), symbol)
        last = series.iloc[-1]
        row.update({
            "regime": str(last["label"]), "confidence": float(last["confidence"]), "clarity": float(last["clarity"]),
            "direction": str(last["direction"]), "volatility": str(last["volatility"]),
            "volatility_trend": str(last["volatility_trend"]), "age_bars": int(last["age_bars"]),
        })
    except Exception as exc:  # one broken instrument must not take the table down
        row["note"] = f"could not classify: {exc}"
    return row


def scan_universe(timeframe: str = "5min", symbols: list[tuple[str, str]] | None = None) -> list[dict]:
    return [regime_for_instrument(s, k, timeframe) for s, k in (symbols or universe_symbols())]


class UniverseCache:
    """Stale-while-revalidate cache of the universe scan (same idea as research/shared_cache.PipelineCache)."""

    def __init__(self, scanner=None, ttl: float = UNIVERSE_TTL_SEC):
        self._scanner = scanner or scan_universe
        self._ttl = ttl
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._items: dict[str, tuple[float, list[dict]]] = {}
        self._refreshing: set[str] = set()

    def _compute(self, timeframe: str) -> list[dict]:
        with self._run_lock:
            rows = self._scanner(timeframe)
        with self._lock:
            self._items[timeframe] = (time.monotonic(), rows)
        return rows

    def _refresh(self, timeframe: str) -> None:
        with self._lock:
            if timeframe in self._refreshing:
                return
            self._refreshing.add(timeframe)

        def work():
            try:
                self._compute(timeframe)
            except Exception:
                pass  # keep serving the previous scan
            finally:
                with self._lock:
                    self._refreshing.discard(timeframe)

        threading.Thread(target=work, daemon=True).start()

    def get(self, timeframe: str = "5min", force: bool = False) -> list[dict]:
        with self._lock:
            hit = self._items.get(timeframe)
        if hit and not force:
            if time.monotonic() - hit[0] >= self._ttl:
                self._refresh(timeframe)
            return hit[1]
        return self._compute(timeframe)

    def peek(self, timeframe: str = "5min") -> list[dict] | None:
        """The last scan if there is one (refreshing in the background when old), else None after starting a scan.
        Never waits: a page uses this so the first visit does not block on 25 instruments."""
        with self._lock:
            hit = self._items.get(timeframe)
        if hit is None or time.monotonic() - hit[0] >= self._ttl:
            self._refresh(timeframe)
        return hit[1] if hit else None

    def warm(self, timeframe: str = "5min") -> None:
        self._refresh(timeframe)


UNIVERSE = UniverseCache()
