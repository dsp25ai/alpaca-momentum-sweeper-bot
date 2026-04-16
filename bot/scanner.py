"""
bot/scanner.py - Market Scanner

FINAL FIX: Uses Alpaca Market Data API exclusively for bar data.
No yfinance / Yahoo Finance dependency — eliminates JSONDecodeError permanently.
Alpaca Data API v2 is reliable, authenticated, and not rate-limited for paper/live accounts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd
import requests

from config import config

log = logging.getLogger(__name__)

MAX_SYMBOLS = 40   # cap total symbols per scan to keep cycle fast
PER_SYMBOL_DELAY = 0.05  # small courtesy delay between Alpaca requests

# ---------------------------------------------------------------------------
# Alpaca Data API client (no SDK needed — plain REST)
# ---------------------------------------------------------------------------

ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "Accept": "application/json",
    }


def _fetch_bars_alpaca(
    symbol: str, limit: int = 30, timeframe: str = "1Day"
) -> Optional[pd.DataFrame]:
    """
    Fetch up to `limit` daily bars for `symbol` via Alpaca Data API v2.
    Returns a DataFrame with lowercase columns: open, high, low, close, volume.
    Returns None on any error.
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=limit + 10)  # extra buffer for weekends/holidays
        url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": limit,
            "adjustment": "split",
            "feed": "iex",  # IEX feed — free tier compatible
        }
        resp = requests.get(
            url,
            headers=_alpaca_headers(),
            params=params,
            timeout=10,
        )
        if resp.status_code == 422:
            # Symbol not supported on IEX, try sip feed
            params["feed"] = "sip"
            resp = requests.get(
                url, headers=_alpaca_headers(), params=params, timeout=10
            )
        if resp.status_code != 200:
            log.debug(f"Alpaca bars {symbol} -> HTTP {resp.status_code}")
            return None
        data = resp.json()
        bars = data.get("bars", [])
        if not bars:
            log.debug(f"No bars returned for {symbol}")
            return None
        df = pd.DataFrame(bars)
        df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "time"},
            inplace=True,
        )
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.sort_index()
        return df
    except Exception as exc:
        log.debug(f"Alpaca fetch error {symbol}: {exc}")
        return None


def _fetch_snapshot_alpaca(symbols: List[str]) -> dict:
    """
    Fetch latest snapshots (latest trade + prev daily bar) for a list of symbols
    in a single Alpaca request. Returns dict keyed by symbol.
    """
    try:
        url = f"{ALPACA_DATA_BASE}/stocks/snapshots"
        params = {"symbols": ",".join(symbols), "feed": "iex"}
        resp = requests.get(
            url, headers=_alpaca_headers(), params=params, timeout=15
        )
        if resp.status_code != 200:
            log.debug(f"Snapshot request failed: HTTP {resp.status_code}")
            return {}
        return resp.json()
    except Exception as exc:
        log.debug(f"Snapshot error: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    symbol: str
    price: float
    gap_pct: float
    rvol: float
    avg_volume: float
    today_volume: float
    prev_close: float
    open_price: float
    high_price: float
    low_price: float
    atr: float
    score: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.symbol} | Price=${self.price:.2f} | "
            f"Gap={self.gap_pct:+.1f}% | RVOL={self.rvol:.1f}x | "
            f"Score={self.score:.1f} | {', '.join(self.notes)}"
        )


# ---------------------------------------------------------------------------
# Default watchlist
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "AMD", "META", "GOOGL", "AMZN", "MSFT",
    "SPY", "QQQ", "SOXL", "TQQQ",
    "MARA", "RIOT", "COIN", "HOOD", "SOFI", "PLTR",
    "NIO", "RIVN", "GME", "AMC",
]


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

class Scanner:
    def __init__(self, extra_symbols: Optional[List[str]] = None) -> None:
        self.watchlist: List[str] = list(
            dict.fromkeys(DEFAULT_WATCHLIST + (extra_symbols or []))
        )

    def scan(self) -> List[ScanResult]:
        symbols = self.watchlist[:MAX_SYMBOLS]
        log.info(f"Scanning {len(symbols)} symbols via Alpaca Data API...")

        # Batch snapshot for quick RVOL/gap pre-filter (single request)
        snapshots = _fetch_snapshot_alpaca(symbols)

        results = []
        for symbol in symbols:
            df = _fetch_bars_alpaca(symbol, limit=30)
            if df is None or df.empty:
                continue
            result = self._analyze(symbol, df, snapshots.get(symbol))
            if result and self._passes_filters(result):
                result.score = self._score(result)
                results.append(result)
            time.sleep(PER_SYMBOL_DELAY)

        results.sort(key=lambda r: r.score, reverse=True)
        log.info(f"Scan complete: {len(results)} candidates from {len(symbols)} symbols.")
        return results

    def _analyze(
        self, symbol: str, df: pd.DataFrame, snapshot: Optional[dict]
    ) -> Optional[ScanResult]:
        try:
            if len(df) < 3:
                return None

            today = df.iloc[-1]
            prev  = df.iloc[-2]

            open_price  = float(today["open"])
            high_price  = float(today["high"])
            low_price   = float(today["low"])
            close_price = float(today["close"])
            today_vol   = float(today["volume"])
            prev_close  = float(prev["close"])

            if not (config.MIN_PRICE <= close_price <= config.MAX_PRICE):
                return None

            gap_pct = ((open_price - prev_close) / prev_close) * 100 if prev_close else 0

            hist_vol = df["volume"].iloc[:-1].tail(20)
            avg_volume = float(hist_vol.mean()) if len(hist_vol) > 0 else 0

            # Use live volume from snapshot if available for RVOL accuracy
            if snapshot:
                try:
                    daily_bar = snapshot.get("dailyBar", {})
                    if daily_bar.get("v"):
                        today_vol = float(daily_bar["v"])
                except Exception:
                    pass

            now = datetime.now(timezone.utc)
            market_open  = now.replace(hour=13, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=20, minute=0,  second=0, microsecond=0)
            elapsed  = max((now - market_open).total_seconds(), 1)
            day_secs = (market_close - market_open).total_seconds()
            day_frac = min(elapsed / day_secs, 1.0)
            proj_vol = today_vol / day_frac if day_frac > 0.05 else today_vol
            rvol = proj_vol / avg_volume if avg_volume > 0 else 0

            highs  = df["high"].values
            lows   = df["low"].values
            closes = df["close"].values
            tr_list = [
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                for i in range(1, len(highs))
            ]
            atr = float(pd.Series(tr_list).tail(14).mean()) if tr_list else 0

            return ScanResult(
                symbol=symbol,
                price=close_price,
                gap_pct=gap_pct,
                rvol=rvol,
                avg_volume=avg_volume,
                today_volume=today_vol,
                prev_close=prev_close,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                atr=atr,
            )
        except Exception as exc:
            log.debug(f"Analysis error {symbol}: {exc}")
            return None

    def _passes_filters(self, r: ScanResult) -> bool:
        checks = [
            (r.rvol >= config.MIN_RVOL,           f"RVOL {r.rvol:.1f}x < {config.MIN_RVOL}x"),
            (r.gap_pct >= config.MIN_GAP_PCT,      f"Gap {r.gap_pct:.1f}% < {config.MIN_GAP_PCT}%"),
            (r.avg_volume >= config.MIN_AVG_VOLUME, f"AvgVol {r.avg_volume:.0f} < {config.MIN_AVG_VOLUME}"),
            (r.price >= config.MIN_PRICE,           f"Price ${r.price:.2f} < ${config.MIN_PRICE}"),
            (r.price <= config.MAX_PRICE,           f"Price ${r.price:.2f} > ${config.MAX_PRICE}"),
            (r.atr > 0,                             "ATR = 0"),
        ]
        for passed, reason in checks:
            if not passed:
                log.debug(f"  SKIP {r.symbol}: {reason}")
                return False
        return True

    def _score(self, r: ScanResult) -> float:
        rvol_score = min(r.rvol / 10.0, 1.0) * 40
        gap_score  = min(r.gap_pct / 20.0, 1.0) * 30
        intraday   = ((r.price - r.open_price) / r.open_price * 100) if r.open_price else 0
        mom_score  = min(max(intraday, 0) / 10.0, 1.0) * 30
        score = rvol_score + gap_score + mom_score

        notes = []
        if r.rvol >= 5:    notes.append(f"RVOL {r.rvol:.1f}x")
        if r.gap_pct >= 10: notes.append(f"BIG GAP {r.gap_pct:.1f}%")
        if intraday >= 5:  notes.append(f"RUNNING +{intraday:.1f}%")
        r.notes = notes
        return round(score, 2)
