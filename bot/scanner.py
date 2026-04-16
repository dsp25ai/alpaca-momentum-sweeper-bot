"""
bot/scanner.py - Market Scanner

Pulls the top-gainer/high-RVOL watchlist from Alpaca's screener
or a static watchlist, then filters by price, volume, gap, and
relative volume. Returns a list of ScanResult dataclasses for the
strategy engine to evaluate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

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
    gap_pct: float          # % gap from prev close
    rvol: float             # relative volume vs 20-day avg
    avg_volume: float       # 20-day avg daily volume
    today_volume: float     # volume so far today
    prev_close: float
    open_price: float
    high_price: float
    low_price: float
    atr: float              # Average True Range (14-day)
    score: float = 0.0      # composite ranking score
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.symbol} | Price=${self.price:.2f} | "
            f"Gap={self.gap_pct:+.1f}% | RVOL={self.rvol:.1f}x | "
            f"Score={self.score:.1f} | {', '.join(self.notes)}"
        )


# ---------------------------------------------------------------------------
# Watchlist sources
# ---------------------------------------------------------------------------

# Default broad watchlist. The scanner will also try to add top-gainers.
DEFAULT_WATCHLIST = [
    # Large-cap momentum names
    "AAPL", "TSLA", "NVDA", "AMD", "META", "GOOGL", "AMZN", "MSFT",
    "SPY", "QQQ", "SOXL", "TQQQ",
    # Small/mid-cap common runners
    "MARA", "RIOT", "COIN", "HOOD", "SOFI", "PLTR", "LCID",
    "NIO", "RIVN", "MULN", "BBBY", "GME", "AMC",
]


def get_top_gainers_yf(n: int = 30) -> List[str]:
    """
    Use yfinance to fetch the current top-gainers from Yahoo Finance.
    Returns a list of ticker symbols.
    This is a best-effort function; falls back to empty list on error.
    """
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
# Core scanner logic
# ---------------------------------------------------------------------------

class Scanner:
    """
    Scans the market for momentum candidates.

    Usage:
        scanner = Scanner()
        results = scanner.scan()
        for r in results:
            print(r)
    """

    def __init__(self, extra_symbols: Optional[List[str]] = None) -> None:
        self.data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )
        self.watchlist: List[str] = list(
            dict.fromkeys(DEFAULT_WATCHLIST + (extra_symbols or []))
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def scan(self) -> List[ScanResult]:
        """Run full scan. Returns filtered + ranked results."""
        # Optionally augment watchlist with today's top gainers
        gainers = get_top_gainers_yf(30)
        symbols = list(dict.fromkeys(self.watchlist + gainers))
        log.info(f"Scanning {len(symbols)} symbols...")

        raw_data = self._fetch_bars(symbols)
        if raw_data.empty:
            log.warning("No bar data returned from Alpaca.")
            return []

        results = []
        for symbol in symbols:
            result = self._analyze(symbol, raw_data)
            if result and self._passes_filters(result):
                result.score = self._score(result)
                results.append(result)

        results.sort(key=lambda r: r.score, reverse=True)
        log.info(
            f"Scan complete: {len(results)} candidates from {len(symbols)} symbols."
        )
        return results

    # ------------------------------------------------------------------ #
    # Data fetching
    # ------------------------------------------------------------------ #

    def _fetch_bars(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch last 22 daily bars for each symbol via Alpaca."""
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=35)  # extra buffer for weekends/holidays
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=22,
            )
            bars = self.data_client.get_stock_bars(req)
            df = bars.df
            if df.empty:
                return df
            df = df.reset_index()
            return df
        except Exception as exc:
            log.error(f"Alpaca bar fetch error: {exc}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Per-symbol analysis
    # ------------------------------------------------------------------ #

    def _analyze(self, symbol: str, df: pd.DataFrame) -> Optional[ScanResult]:
        """Compute gap %, RVOL, ATR for a single symbol."""
        try:
            sym_df = df[df["symbol"] == symbol].copy()
            if len(sym_df) < 2:
                return None
            sym_df = sym_df.sort_values("timestamp")

            today = sym_df.iloc[-1]
            prev = sym_df.iloc[-2]

            open_price = float(today["open"])
            high_price = float(today["high"])
            low_price = float(today["low"])
            close_price = float(today["close"])
            today_vol = float(today["volume"])
            prev_close = float(prev["close"])

            # Price filter quick exit
            if not (config.MIN_PRICE <= open_price <= config.MAX_PRICE):
                return None

            # Gap %
            gap_pct = ((open_price - prev_close) / prev_close) * 100

            # Average daily volume (last 20 days, excluding today)
            hist_vol = sym_df["volume"].iloc[:-1].tail(20)
            avg_volume = float(hist_vol.mean()) if len(hist_vol) > 0 else 0

            # RVOL (relative volume) vs avg.  Scale today's partial volume.
            # Approximate full-day volume by dividing by fraction of day elapsed.
            now = datetime.utcnow()
            market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)  # UTC
            market_close = now.replace(hour=20, minute=0, second=0, microsecond=0)  # UTC
            elapsed = max((now - market_open).total_seconds(), 1)
            day_secs = (market_close - market_open).total_seconds()
            day_fraction = min(elapsed / day_secs, 1.0)
            projected_vol = today_vol / day_fraction if day_fraction > 0.05 else today_vol
            rvol = projected_vol / avg_volume if avg_volume > 0 else 0

            # ATR (14-day)
            highs = sym_df["high"].values
            lows = sym_df["low"].values
            closes = sym_df["close"].values
            tr_list = []
            for i in range(1, len(highs)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                tr_list.append(tr)
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

    # ------------------------------------------------------------------ #
    # Filters
    # ------------------------------------------------------------------ #

    def _passes_filters(self, r: ScanResult) -> bool:
        """Return True if the scan result meets all entry criteria."""
        checks = [
            (r.rvol >= config.MIN_RVOL,           f"RVOL {r.rvol:.1f}x < {config.MIN_RVOL}x"),
            (r.gap_pct >= config.MIN_GAP_PCT,      f"Gap {r.gap_pct:.1f}% < {config.MIN_GAP_PCT}%"),
            (r.avg_volume >= config.MIN_AVG_VOLUME, f"AvgVol {r.avg_volume:.0f} < {config.MIN_AVG_VOLUME}"),
            (r.price >= config.MIN_PRICE,           f"Price ${r.price:.2f} < ${config.MIN_PRICE}"),
            (r.price <= config.MAX_PRICE,           f"Price ${r.price:.2f} > ${config.MAX_PRICE}"),
            (r.atr > 0,                             "ATR = 0 (insufficient history)"),
        ]
        for passed, reason in checks:
            if not passed:
                log.debug(f"  SKIP {r.symbol}: {reason}")
                return False
        return True

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _score(self, r: ScanResult) -> float:
        """
        Composite momentum score (higher = better).
        Weights: RVOL 40%, Gap 30%, Price momentum 30%.
        """
        rvol_score  = min(r.rvol / 10.0, 1.0) * 40
        gap_score   = min(r.gap_pct / 20.0, 1.0) * 30
        # Intraday move from open
        intraday = ((r.price - r.open_price) / r.open_price * 100) if r.open_price else 0
        mom_score   = min(max(intraday, 0) / 10.0, 1.0) * 30

        score = rvol_score + gap_score + mom_score

        notes = []
        if r.rvol >= 5:  notes.append(f"RVOL {r.rvol:.1f}x")
        if r.gap_pct >= 10: notes.append(f"BIG GAP {r.gap_pct:.1f}%")
        if intraday >= 5:   notes.append(f"RUNNING +{intraday:.1f}%")
        r.notes = notes

        return round(score, 2)
