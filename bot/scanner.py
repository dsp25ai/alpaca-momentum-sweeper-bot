"""
bot/scanner.py - Market Scanner

Uses yfinance Ticker per-symbol with browser-like headers to bypass
Yahoo Finance datacenter IP blocks (common on Railway/AWS/GCP).
Alpaca is only used for order execution.
"""
from __future__ import annotations

import logging
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf

from config import config

log = logging.getLogger(__name__)

PER_SYMBOL_DELAY = 0.4   # seconds between individual ticker fetches
MAX_SYMBOLS      = 40    # cap total symbols per scan to keep cycle fast

# Rotate user-agents to avoid Yahoo blocking datacenter IPs
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


def _make_session() -> requests.Session:
    """Create a requests session with browser-like headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


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
# Watchlist
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "AMD", "META", "GOOGL", "AMZN", "MSFT",
    "SPY", "QQQ", "SOXL", "TQQQ",
    "MARA", "RIOT", "COIN", "HOOD", "SOFI", "PLTR",
    "NIO", "RIVN", "GME", "AMC",
]


def get_top_gainers_yf(n: int = 15) -> List[str]:
    """Fetch top gainers from Yahoo Finance screener."""
    try:
        session = _make_session()
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "formatted": "false",
            "lang": "en-US",
            "region": "US",
            "scrIds": "day_gainers",
            "count": n,
        }
        resp = session.get(url, params=params, timeout=10)
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

    def __init__(self, extra_symbols: Optional[List[str]] = None) -> None:
        self.watchlist: List[str] = list(
            dict.fromkeys(DEFAULT_WATCHLIST + (extra_symbols or []))
        )

    def scan(self) -> List[ScanResult]:
        gainers = get_top_gainers_yf(15)
        symbols = list(dict.fromkeys(self.watchlist + gainers))[:MAX_SYMBOLS]
        log.info(f"Scanning {len(symbols)} symbols...")

        results = []
        session = _make_session()

        for symbol in symbols:
            df = self._fetch_ticker(symbol, session)
            if df is None or df.empty:
                continue
            result = self._analyze(symbol, df)
            if result and self._passes_filters(result):
                result.score = self._score(result)
                results.append(result)
            time.sleep(PER_SYMBOL_DELAY)

        results.sort(key=lambda r: r.score, reverse=True)
        log.info(f"Scan complete: {len(results)} candidates from {len(symbols)} symbols.")
        return results

    def _fetch_ticker(self, symbol: str, session: requests.Session) -> Optional[pd.DataFrame]:
        """Fetch 25 days of daily bars for a single symbol."""
        try:
            ticker = yf.Ticker(symbol, session=session)
            df = ticker.history(period="25d", interval="1d", auto_adjust=True)
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as exc:
            log.debug(f"Fetch error {symbol}: {exc}")
            return None

    def _analyze(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        try:
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

            gap_pct    = ((open_price - prev_close) / prev_close) * 100
            hist_vol   = df["volume"].iloc[:-1].tail(20)
            avg_volume = float(hist_vol.mean()) if len(hist_vol) > 0 else 0

            now          = datetime.utcnow()
            market_open  = now.replace(hour=13, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=20, minute=0,  second=0, microsecond=0)
            elapsed  = max((now - market_open).total_seconds(), 1)
            day_secs = (market_close - market_open).total_seconds()
            day_frac = min(elapsed / day_secs, 1.0)
            proj_vol = today_vol / day_frac if day_frac > 0.05 else today_vol
            rvol     = proj_vol / avg_volume if avg_volume > 0 else 0

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
            log.debug(f"Analysis error {symbol}: {exc}")
            return None

    def _passes_filters(self, r: ScanResult) -> bool:
        checks = [
            (r.rvol >= config.MIN_RVOL,            f"RVOL {r.rvol:.1f}x < {config.MIN_RVOL}x"),
            (r.gap_pct >= config.MIN_GAP_PCT,       f"Gap {r.gap_pct:.1f}% < {config.MIN_GAP_PCT}%"),
            (r.avg_volume >= config.MIN_AVG_VOLUME,  f"AvgVol {r.avg_volume:.0f} < {config.MIN_AVG_VOLUME}"),
            (r.price >= config.MIN_PRICE,            f"Price ${r.price:.2f} < ${config.MIN_PRICE}"),
            (r.price <= config.MAX_PRICE,            f"Price ${r.price:.2f} > ${config.MAX_PRICE}"),
            (r.atr > 0,                              "ATR = 0"),
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
        if r.rvol >= 5:     notes.append(f"RVOL {r.rvol:.1f}x")
        if r.gap_pct >= 10: notes.append(f"BIG GAP {r.gap_pct:.1f}%")
        if intraday >= 5:   notes.append(f"RUNNING +{intraday:.1f}%")
        r.notes = notes

        return round(score, 2)
