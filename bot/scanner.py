"""
bot/scanner.py - Market Scanner

Uses yfinance for all bar/price data (free, no subscription needed).
Alpaca is used only for order execution, not data.

Filters by: price range, avg volume, gap %, relative volume.
Returns ranked ScanResult list for the strategy engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """A stock that passed all scanner filters."""
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
# Watchlist
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "AMD", "META", "GOOGL", "AMZN", "MSFT",
    "SPY", "QQQ", "SOXL", "TQQQ",
    "MARA", "RIOT", "COIN", "HOOD", "SOFI", "PLTR", "LCID",
    "NIO", "RIVN", "GME", "AMC",
]


def get_top_gainers_yf(n: int = 30) -> List[str]:
    """Fetch today's top gainers from Yahoo Finance screener."""
    try:
        import requests
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "formatted": "false",
            "lang": "en-US",
            "region": "US",
            "scrIds": "day_gainers",
            "count": n,
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        quotes = data["finance"]["result"][0]["quotes"]
        symbols = [q["symbol"] for q in quotes]
        log.info(f"Top-gainers fetched: {symbols[:10]}...")
        return symbols
    except Exception as exc:
        log.warning(f"Could not fetch top-gainers: {exc}")
        return []


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

class Scanner:
    """
    Scans a watchlist using yfinance for bar data.
    Alpaca keys are NOT needed here — data is free via Yahoo Finance.
    """

    def __init__(self, extra_symbols: Optional[List[str]] = None) -> None:
        self.watchlist: List[str] = list(
            dict.fromkeys(DEFAULT_WATCHLIST + (extra_symbols or []))
        )

    def scan(self) -> List[ScanResult]:
        """Run full scan. Returns filtered + ranked results."""
        gainers = get_top_gainers_yf(30)
        symbols = list(dict.fromkeys(self.watchlist + gainers))
        log.info(f"Scanning {len(symbols)} symbols...")

        results = []
        # Batch download all symbols at once — much faster than one-by-one
        try:
            raw = yf.download(
                tickers=symbols,
                period="25d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.error(f"yfinance batch download error: {exc}")
            return []

        if raw.empty:
            log.warning("No bar data returned from yfinance.")
            return []

        for symbol in symbols:
            result = self._analyze(symbol, raw)
            if result and self._passes_filters(result):
                result.score = self._score(result)
                results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        log.info(f"Scan complete: {len(results)} candidates from {len(symbols)} symbols.")
        return results

    def _analyze(self, symbol: str, raw: pd.DataFrame) -> Optional[ScanResult]:
        """Extract metrics for a single symbol from the batch download."""
        try:
            # Handle both single and multi-ticker DataFrame structures
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(1):
                    return None
                df = raw.xs(symbol, axis=1, level=1).dropna(how="all")
            else:
                df = raw.dropna(how="all")

            df = df.copy()
            df.columns = [c.lower() for c in df.columns]

            if len(df) < 3:
                return None

            today = df.iloc[-1]
            prev  = df.iloc[-2]

            open_price  = float(today.get("open",  today["close"]))
            high_price  = float(today["high"])
            low_price   = float(today["low"])
            close_price = float(today["close"])
            today_vol   = float(today["volume"])
            prev_close  = float(prev["close"])

            if not (config.MIN_PRICE <= close_price <= config.MAX_PRICE):
                return None

            # Gap %
            gap_pct = ((open_price - prev_close) / prev_close) * 100

            # 20-day avg volume (excluding today)
            hist_vol = df["volume"].iloc[:-1].tail(20)
            avg_volume = float(hist_vol.mean()) if len(hist_vol) > 0 else 0

            # Projected RVOL
            now = datetime.utcnow()
            market_open  = now.replace(hour=13, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=20, minute=0,  second=0, microsecond=0)
            elapsed   = max((now - market_open).total_seconds(), 1)
            day_secs  = (market_close - market_open).total_seconds()
            day_frac  = min(elapsed / day_secs, 1.0)
            proj_vol  = today_vol / day_frac if day_frac > 0.05 else today_vol
            rvol      = proj_vol / avg_volume if avg_volume > 0 else 0

            # ATR (14-day)
            highs  = df["high"].values
            lows   = df["low"].values
            closes = df["close"].values
            tr_list = [
                max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
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
            log.debug(f"Analysis error for {symbol}: {exc}")
            return None

    def _passes_filters(self, r: ScanResult) -> bool:
        checks = [
            (r.rvol >= config.MIN_RVOL,            f"RVOL {r.rvol:.1f}x < {config.MIN_RVOL}x"),
            (r.gap_pct >= config.MIN_GAP_PCT,       f"Gap {r.gap_pct:.1f}% < {config.MIN_GAP_PCT}%"),
            (r.avg_volume >= config.MIN_AVG_VOLUME,  f"AvgVol {r.avg_volume:.0f} < {config.MIN_AVG_VOLUME}"),
            (r.price >= config.MIN_PRICE,            f"Price ${r.price:.2f} < ${config.MIN_PRICE}"),
            (r.price <= config.MAX_PRICE,            f"Price ${r.price:.2f} > ${config.MAX_PRICE}"),
            (r.atr > 0,                              "ATR = 0 (insufficient history)"),
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
        score      = rvol_score + gap_score + mom_score

        notes = []
        if r.rvol >= 5:      notes.append(f"RVOL {r.rvol:.1f}x")
        if r.gap_pct >= 10:  notes.append(f"BIG GAP {r.gap_pct:.1f}%")
        if intraday >= 5:    notes.append(f"RUNNING +{intraday:.1f}%")
        r.notes = notes

        return round(score, 2)
