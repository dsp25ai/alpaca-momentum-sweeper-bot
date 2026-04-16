"""
main.py - Alpaca Momentum Sweeper Bot Entry Point

Usage:
    # Run live bot (paper mode by default)
    python main.py

    # Scan only (no orders)
    python main.py --scan-only

    # Backtest
    python main.py --backtest --symbols TSLA NVDA AMD --days 30

    # Show account status
    python main.py --account
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

import pytz
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from config import config
from bot.scanner import Scanner
from bot.strategy import StrategyEngine
from bot.executor import Executor
from bot.risk import RiskManager
from bot.portfolio import Portfolio
from bot.notifier import Notifier
from bot.backtest import Backtester

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log_path = Path(config.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler(str(log_path)),
    ]
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


log = logging.getLogger("main")
console = Console()


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """Return True if US market is currently open (9:30-16:00 ET, Mon-Fri)."""
    tz = pytz.timezone(config.TIMEZONE)
    now = datetime.now(tz)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = dtime(9, 30)
    market_close = dtime(16, 0)
    return market_open <= now.time() < market_close


def minutes_to_open() -> float:
    """Minutes until market opens (negative if already open)."""
    tz = pytz.timezone(config.TIMEZONE)
    now = datetime.now(tz)
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return (open_time - now).total_seconds() / 60


# ---------------------------------------------------------------------------
# Main bot loop
# ---------------------------------------------------------------------------

class MomentumSweeper:
    """
    Main bot orchestrator.
    Runs the scan -> signal -> risk check -> execute -> track loop.
    """

    def __init__(self) -> None:
        config.validate()
        self.scanner = Scanner()
        self.strategy = StrategyEngine()
        self.executor = Executor()
        self.risk = RiskManager()
        self.portfolio = Portfolio()
        self.notifier = Notifier()
        self._running = True
        self._traded_signals: set = set()  # avoid duplicate signals per session

        # Graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        log.info(f"Bot initialized | {config.summary()}")
        self.notifier.info(f"Bot started | {config.summary()}")

    def run(self, scan_only: bool = False) -> None:
        """Main loop. Runs until stopped."""
        console.print(f"[bold green]Alpaca Momentum Sweeper Bot[/bold green]")
        console.print(f"Mode: [bold]{config.TRADING_MODE}[/bold]")
        console.print(f"Scan interval: {config.SCAN_INTERVAL}s")
        console.print("Press Ctrl+C to stop.\n")

        while self._running:
            try:
                self._cycle(scan_only)
            except Exception as exc:
                log.error(f"Unexpected error in main loop: {exc}", exc_info=True)
                self.notifier.error("Main loop error", str(exc))
            time.sleep(config.SCAN_INTERVAL)

    def _cycle(self, scan_only: bool) -> None:
        """One scan-decide-execute cycle."""
        now_str = datetime.utcnow().strftime("%H:%M:%S UTC")

        # Wait for market hours
        if not is_market_open():
            mins = minutes_to_open()
            if mins > 0:
                log.info(f"Market closed. Opens in {mins:.0f} min.")
            else:
                log.info("Market closed (after hours). Waiting...")
                # Send daily report at EOD
                self._maybe_daily_report()
            return

        log.info(f"--- Scan cycle {now_str} ---")

        # 1. Scan
        candidates = self.scanner.scan()
        if not candidates:
            log.info("No candidates found this cycle.")
            return

        # Print scan table
        self._print_scan_table(candidates[:10])

        if scan_only:
            return

        # 2. Generate signals
        signals = self.strategy.evaluate(candidates)
        if not signals:
            log.info("No trade signals generated.")
            return

        # 3. Risk check + Execute
        open_count = self.portfolio.open_count()
        capital_deployed = self.portfolio.capital_deployed()

        for signal in signals:
            # Deduplicate signals within same session
            sig_key = f"{signal.symbol}_{signal.strategy}_{datetime.utcnow().date()}"
            if sig_key in self._traded_signals:
                log.debug(f"Skipping duplicate signal: {sig_key}")
                continue

            if not self.risk.check(signal, open_count, capital_deployed):
                continue

            # Alert signal
            self.notifier.trade_signal(
                symbol=signal.symbol,
                strategy=signal.strategy,
                entry=signal.entry_price,
                stop=signal.stop_loss,
                target=signal.take_profit,
                shares=signal.shares,
                confidence=signal.confidence,
                rr=signal.risk_reward(),
            )

            # Execute
            result = self.executor.execute(signal)

            if result.success:
                self._traded_signals.add(sig_key)
                self.risk.record_trade(signal)
                self.portfolio.record_open(result)
                open_count += 1
                capital_deployed += signal.shares * signal.entry_price

                self.notifier.trade_filled(
                    symbol=result.symbol,
                    shares=result.shares,
                    price=result.entry_price,
                    stop=result.stop_loss,
                    target=result.take_profit,
                    strategy=result.strategy,
                )
                console.print(
                    f"[green]FILLED[/green]: {result.symbol} x{result.shares} "
                    f"@ ${result.entry_price:.2f} [{result.mode}]"
                )
            else:
                log.error(f"Order failed: {result.error}")

        # Log risk status
        log.info(f"Risk: {self.risk.status_summary()}")
        log.info(f"Portfolio: {self.portfolio.summary()}")

    def _maybe_daily_report(self) -> None:
        """Send daily report once per day (around 4:15 PM ET)."""
        tz = pytz.timezone(config.TIMEZONE)
        now = datetime.now(tz)
        if now.hour == 16 and 15 <= now.minute <= 20:
            report = self.portfolio.daily_report()
            log.info(f"\n{report}")
            self.notifier.daily_summary(report)

    def _print_scan_table(self, candidates) -> None:
        """Print a rich table of top scan candidates."""
        table = Table(title="Top Scanner Candidates", show_lines=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Price", justify="right")
        table.add_column("Gap%", justify="right")
        table.add_column("RVOL", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Notes")

        for c in candidates:
            table.add_row(
                c.symbol,
                f"${c.price:.2f}",
                f"{c.gap_pct:+.1f}%",
                f"{c.rvol:.1f}x",
                f"{c.score:.0f}",
                ", ".join(c.notes) if c.notes else "-",
            )
        console.print(table)

    def _shutdown(self, *args) -> None:
        """Handle graceful shutdown on SIGINT/SIGTERM."""
        log.info("Shutdown signal received. Stopping bot...")
        console.print("\n[yellow]Shutting down...[/yellow]")
        report = self.portfolio.daily_report()
        log.info(f"\n{report}")
        self.notifier.daily_summary(report)
        self.notifier.info("Bot stopped gracefully.")
        self.portfolio.close()
        self._running = False
        sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Alpaca Momentum Sweeper Bot"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Scan and log opportunities without placing orders.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run historical backtest instead of live bot.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["TSLA", "NVDA", "AMD", "AAPL", "META"],
        help="Symbols to backtest (used with --backtest).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback days for backtest.",
    )
    parser.add_argument(
        "--account",
        action="store_true",
        help="Show Alpaca account status and exit.",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    if args.backtest:
        # Backtest mode
        config.validate()
        log.info(f"Running backtest on {args.symbols} for {args.days} days...")
        bt = Backtester()
        results = bt.run(args.symbols, lookback_days=args.days)
        bt.print_report(results)
        return

    if args.account:
        # Account status
        config.validate()
        executor = Executor()
        account = executor.get_account()
        console.print("[bold]Account Status:[/bold]")
        for k, v in account.items():
            console.print(f"  {k}: {v}")
        positions = executor.get_open_positions()
        if positions:
            console.print("\n[bold]Open Positions:[/bold]")
            for p in positions:
                console.print(
                    f"  {p['symbol']}: {p['qty']}sh "
                    f"@ ${p['entry_price']:.2f} "
                    f"(current ${p['current_price']:.2f}) "
                    f"PnL=${p['unrealized_pl']:+.2f}"
                )
        else:
            console.print("No open positions.")
        return

    # Live bot
    bot = MomentumSweeper()
    bot.run(scan_only=args.scan_only)


if __name__ == "__main__":
    main()
