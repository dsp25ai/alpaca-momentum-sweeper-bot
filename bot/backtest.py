"""
bot/backtest.py - Historical Backtesting Module

FINAL FIX: Uses Alpaca Data API v2 REST directly (same as scanner.py).
No yfinance, no alpaca-trade-api SDK dependency.
Simulates ORB/VWAP entries, bracket stops/targets on daily bars.

Usage:
    python main.py --backtest --symbols TSLA NVDA AMD --days 60
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd
import requests

from config import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alpaca Data REST helper (shared pattern with scanner.py)
# ---------------------------------------------------------------------------

ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "Accept": "application/json",
    }


def _fetch_daily_bars(symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Fetch `days` of daily OHLCV bars for `symbol` via Alpaca Data API v2.
    Returns DataFrame or None.
    """
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 15)  # buffer for holidays
        url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/bars"
        params = {
            "timeframe": "1Day",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": days + 15,
            "adjustment": "split",
            "feed": "iex",
        }
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=15)
        if resp.status_code == 422:
            params["feed"] = "sip"
            resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=15)
        if resp.status_code != 200:
            log.warning(f"Alpaca backtest bars {symbol}: HTTP {resp.status_code}")
            return None
        bars = resp.json().get("bars", [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "time"},
            inplace=True,
        )
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        return df.tail(days)
    except Exception as exc:
        log.warning(f"Backtest fetch error {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    gap_pct: float
    rvol: float


@dataclass
class BacktestResult:
    symbol: str
    trades: List[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    total_trades: int = 0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 60,
        risk_per_trade: float = 0.02,
        rr_ratio: float = 2.0,
        stop_atr_mult: float = 1.5,
    ):
        self.symbols = symbols or ["TSLA", "NVDA", "AMD", "AAPL", "META"]
        self.lookback_days = lookback_days
        self.risk_per_trade = risk_per_trade
        self.rr_ratio = rr_ratio
        self.stop_atr_mult = stop_atr_mult
        self.account_size = 10_000.0  # simulated account

    def run(self, symbols: Optional[List[str]] = None, lookback_days: Optional[int] = None) -> List[BacktestResult]:
        syms = symbols or self.symbols
        days = lookback_days or self.lookback_days
        results = []
        for sym in syms:
            log.info(f"Backtesting {sym} over {days} days...")
            df = _fetch_daily_bars(sym, days=days)
            if df is None or len(df) < 10:
                log.warning(f"Not enough data for {sym}")
                continue
            result = self._backtest_symbol(sym, df)
            results.append(result)
        return results

    def _backtest_symbol(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        result = BacktestResult(symbol=symbol)
        equity = self.account_size
        peak = equity

        for i in range(5, len(df) - 1):
            today = df.iloc[i]
            prev = df.iloc[i - 1]
            hist = df.iloc[i - 20:i]

            open_price = float(today["open"])
            high_price = float(today["high"])
            low_price  = float(today["low"])
            close_price = float(today["close"])
            prev_close = float(prev["close"])
            today_vol  = float(today["volume"])
            avg_vol    = float(hist["volume"].mean()) if len(hist) > 0 else 1

            # Filters
            gap_pct = ((open_price - prev_close) / prev_close) * 100 if prev_close else 0
            rvol    = today_vol / avg_vol if avg_vol > 0 else 0

            if gap_pct < config.MIN_GAP_PCT:
                continue
            if rvol < config.MIN_RVOL:
                continue
            if not (config.MIN_PRICE <= close_price <= config.MAX_PRICE):
                continue

            # ATR for stop sizing
            tr_list = [
                max(
                    float(df.iloc[j]["high"]) - float(df.iloc[j]["low"]),
                    abs(float(df.iloc[j]["high"]) - float(df.iloc[j - 1]["close"])),
                    abs(float(df.iloc[j]["low"]) - float(df.iloc[j - 1]["close"])),
                )
                for j in range(max(i - 14, 1), i)
            ]
            atr = float(pd.Series(tr_list).mean()) if tr_list else 0
            if atr <= 0:
                continue

            # Entry: buy on open of today
            entry_price = open_price
            stop_price  = entry_price - (atr * self.stop_atr_mult)
            target_price = entry_price + (atr * self.stop_atr_mult * self.rr_ratio)

            risk_amount = equity * self.risk_per_trade
            risk_per_share = entry_price - stop_price
            if risk_per_share <= 0:
                continue
            shares = risk_amount / risk_per_share

            # Simulate exit: check same day's high/low for bracket hit
            exit_price = close_price
            exit_reason = "close"

            if high_price >= target_price:
                exit_price = target_price
                exit_reason = "target"
            elif low_price <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"

            pnl = (exit_price - entry_price) * shares
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            equity += pnl
            peak = max(peak, equity)

            trade = BacktestTrade(
                symbol=symbol,
                entry_date=str(df.index[i].date()),
                entry_price=round(entry_price, 2),
                exit_date=str(df.index[i].date()),
                exit_price=round(exit_price, 2),
                shares=round(shares, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                exit_reason=exit_reason,
                gap_pct=round(gap_pct, 2),
                rvol=round(rvol, 2),
            )
            result.trades.append(trade)

        # Aggregate stats
        result.total_trades = len(result.trades)
        if result.trades:
            result.total_pnl = round(sum(t.pnl for t in result.trades), 2)
            winners = [t for t in result.trades if t.pnl > 0]
            losers  = [t for t in result.trades if t.pnl <= 0]
            result.win_rate  = round(len(winners) / result.total_trades * 100, 1)
            result.avg_win   = round(sum(t.pnl for t in winners) / len(winners), 2) if winners else 0
            result.avg_loss  = round(sum(t.pnl for t in losers)  / len(losers),  2) if losers  else 0
            result.max_drawdown = round((peak - equity) / peak * 100, 2) if peak else 0

        return result

    def print_report(self, results: List[BacktestResult]) -> None:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Backtest Results", show_lines=True)
        table.add_column("Symbol",   style="cyan")
        table.add_column("Trades",   justify="right")
        table.add_column("Win%",     justify="right")
        table.add_column("Total P&L", justify="right")
        table.add_column("Avg Win",  justify="right")
        table.add_column("Avg Loss", justify="right")
        table.add_column("MaxDD%",   justify="right")

        for r in results:
            pnl_color = "green" if r.total_pnl >= 0 else "red"
            table.add_row(
                r.symbol,
                str(r.total_trades),
                f"{r.win_rate:.1f}%",
                f"[{pnl_color}]${r.total_pnl:+,.2f}[/{pnl_color}]",
                f"${r.avg_win:,.2f}",
                f"${r.avg_loss:,.2f}",
                f"{r.max_drawdown:.1f}%",
            )
        console.print(table)

        for r in results:
            if r.trades:
                console.print(f"\n[bold]{r.symbol} — last 5 trades:[/bold]")
                for t in r.trades[-5:]:
                    color = "green" if t.pnl > 0 else "red"
                    console.print(
                        f"  {t.entry_date} | entry=${t.entry_price} "
                        f"exit=${t.exit_price} ({t.exit_reason}) "
                        f"[{color}]P&L=${t.pnl:+.2f}[/{color}] "
                        f"gap={t.gap_pct:+.1f}% rvol={t.rvol:.1f}x"
                    )
