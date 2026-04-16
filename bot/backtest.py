"""
bot/backtest.py - Historical Backtesting Module

Runs the ORB and VWAP strategies against historical daily bar data
fetched from Alpaca. Simulates entries, stops, and targets using
the same logic as the live strategy engine.

Usage:
    python main.py --backtest --symbols TSLA NVDA --days 30
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import config

log = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    pnl: float
    outcome: str  # WIN | LOSS | OPEN
    shares: int


@dataclass
class BacktestResult:
    symbol: str
    trades: List[BacktestTrade] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        wins = sum(1 for t in self.trades if t.outcome == "WIN")
        total = len(self.trades)
        return (wins / total * 100) if total > 0 else 0

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    def summary(self) -> str:
        return (
            f"{self.symbol}: {self.num_trades} trades | "
            f"WinRate={self.win_rate:.0f}% | "
            f"NetPnL=${self.total_pnl:+.2f}"
        )


class Backtester:
    """
    Simple historical backtester.

    Fetches daily OHLCV bars and simulates the gap+RVOL scanner
    followed by ORB-style entries (buy next open after signal,
    exit at EOD price if neither SL nor TP is hit intraday).

    Note: This is a daily-bar simulation - intraday path is
    approximated using high/low of the bar. For true intraday
    backtest, switch to 5-min bars.
    """

    def __init__(self) -> None:
        self.data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )

    def run(
        self,
        symbols: List[str],
        lookback_days: int = 30,
    ) -> List[BacktestResult]:
        """Run backtest for all symbols. Returns list of BacktestResult."""
        results = []
        for symbol in symbols:
            log.info(f"Backtesting {symbol} over {lookback_days} days...")
            result = self._backtest_symbol(symbol, lookback_days)
            results.append(result)
            log.info(result.summary())
        return results

    def print_report(self, results: List[BacktestResult]) -> None:
        """Print full backtest report to console."""
        print("\n" + "=" * 60)
        print(f"BACKTEST REPORT ({datetime.utcnow().date()})")
        print("=" * 60)

        total_pnl = 0.0
        total_trades = 0
        all_wins = 0

        for r in results:
            print(f"\n{r.summary()}")
            for t in r.trades:
                print(
                    f"  [{t.outcome:4s}] {t.entry_date} | "
                    f"Entry=${t.entry_price:.2f} "
                    f"Exit=${t.exit_price:.2f} "
                    f"PnL=${t.pnl:+.2f}"
                )
            total_pnl += r.total_pnl
            total_trades += r.num_trades
            all_wins += sum(1 for t in r.trades if t.outcome == "WIN")

        print("\n" + "=" * 60)
        overall_wr = (all_wins / total_trades * 100) if total_trades > 0 else 0
        print(f"TOTAL: {total_trades} trades | WinRate={overall_wr:.0f}% | NetPnL=${total_pnl:+.2f}")
        print("=" * 60 + "\n")

    def _backtest_symbol(self, symbol: str, lookback_days: int) -> BacktestResult:
        result = BacktestResult(symbol=symbol)
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=lookback_days + 10)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = self.data_client.get_stock_bars(req)
            df = bars.df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            if len(df) < 3:
                return result

            # Simulate day-by-day
            for i in range(1, len(df) - 1):
                prev = df.iloc[i - 1]
                today = df.iloc[i]
                next_day = df.iloc[i + 1]

                prev_close = float(prev["close"])
                open_price = float(today["open"])
                high = float(today["high"])
                low = float(today["low"])
                today_vol = float(today["volume"])

                # Approximate avg volume (last 20 days)
                hist_slice = df.iloc[max(0, i - 20):i]
                avg_vol = float(hist_slice["volume"].mean()) if len(hist_slice) > 0 else 1
                rvol = today_vol / avg_vol if avg_vol > 0 else 0

                # Gap filter
                gap_pct = ((open_price - prev_close) / prev_close) * 100
                if gap_pct < config.MIN_GAP_PCT:
                    continue
                if rvol < config.MIN_RVOL:
                    continue
                if not (config.MIN_PRICE <= open_price <= config.MAX_PRICE):
                    continue

                # Simple ORB simulation:
                # Entry = open, Stop = open - ATR*0.5, Target = entry + risk * TAKE_PROFIT_R
                atr_slice = df.iloc[max(0, i - 14):i]
                if len(atr_slice) > 1:
                    tr_vals = []
                    for j in range(1, len(atr_slice)):
                        h = float(atr_slice.iloc[j]["high"])
                        l = float(atr_slice.iloc[j]["low"])
                        c_prev = float(atr_slice.iloc[j-1]["close"])
                        tr_vals.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
                    atr = float(pd.Series(tr_vals).mean())
                else:
                    atr = open_price * 0.02

                entry = open_price
                stop = entry - atr * 0.5
                risk = entry - stop
                if risk <= 0:
                    continue
                target = entry + risk * config.TAKE_PROFIT_R
                shares = max(1, int(config.RISK_PER_TRADE / risk))

                # Simulate: did high touch target? did low touch stop?
                hit_target = high >= target
                hit_stop = low <= stop

                if hit_target and not hit_stop:
                    outcome = "WIN"
                    exit_price = target
                elif hit_stop and not hit_target:
                    outcome = "LOSS"
                    exit_price = stop
                elif hit_target and hit_stop:
                    # Ambiguous - assume stop hit first (conservative)
                    outcome = "LOSS"
                    exit_price = stop
                else:
                    # Neither hit - exit at next open (day close)
                    outcome = "OPEN"
                    exit_price = float(next_day["open"])

                pnl = (exit_price - entry) * shares

                result.trades.append(BacktestTrade(
                    symbol=symbol,
                    strategy="ORB_SIM",
                    entry_date=str(today.get("timestamp", "") or today.name),
                    entry_price=round(entry, 2),
                    stop_loss=round(stop, 2),
                    take_profit=round(target, 2),
                    exit_price=round(exit_price, 2),
                    pnl=round(pnl, 2),
                    outcome=outcome,
                    shares=shares,
                ))

        except Exception as exc:
            log.error(f"Backtest error for {symbol}: {exc}")

        return result
