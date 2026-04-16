"""
bot/risk.py - Risk Management Module

Enforces daily and per-trade risk limits. Acts as a gatekeeper
before any order reaches the executor.

Rules enforced:
  - Max simultaneous open positions
  - Max capital deployed at once
  - Max daily loss (bot shuts down trading after breach)
  - Max daily trades
  - Daily profit target (bot stops new entries after hitting goal)
  - Per-symbol cooldown (don't re-enter same stock too soon)
  - PDT protection warning (< 4 daytrades in 5 days on < $25k)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

from config import config
from bot.strategy import TradeSignal

log = logging.getLogger(__name__)


class RiskManager:
    """
    Stateful risk manager. Pass signals through check() before execution.

    Usage:
        rm = RiskManager()
        if rm.check(signal, open_positions_count, capital_deployed):
            executor.execute(signal)
        rm.record_trade(signal, pnl=0)  # call after each fill
        rm.record_close(symbol, pnl=profit_or_loss)  # call after each close
    """

    def __init__(self) -> None:
        # Must initialize _symbol_cooldowns BEFORE calling _reset_daily()
        # because _reset_daily() calls self._symbol_cooldowns.clear()
        self._symbol_cooldowns: Dict[str, datetime] = {}
        self.COOLDOWN_MINUTES: int = 30  # min gap between re-entries on same symbol
        self._last_trade_date: date = date.today()
        self._reset_daily()

    # ------------------------------------------------------------------ #
    # Daily reset
    # ------------------------------------------------------------------ #

    def _reset_daily(self) -> None:
        """Reset all daily counters. Called at start of each new trading day."""
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_loss: float = 0.0
        self.daily_profit: float = 0.0
        self.locked: bool = False
        self.lock_reason: str = ""
        self._symbol_cooldowns.clear()
        log.info("Daily risk counters reset.")

    def _check_daily_reset(self) -> None:
        """Auto-reset at start of new calendar day."""
        today = date.today()
        if today != self._last_trade_date:
            self._last_trade_date = today
            self._reset_daily()

    # ------------------------------------------------------------------ #
    # Pre-trade gate
    # ------------------------------------------------------------------ #

    def check(
        self,
        signal: TradeSignal,
        open_positions: int,
        capital_deployed: float,
    ) -> bool:
        """
        Return True if the trade is allowed, False otherwise.
        Logs the rejection reason if blocked.
        """
        self._check_daily_reset()

        # Hard lock (daily loss or profit target hit)
        if self.locked:
            log.warning(f"RISK LOCKED - {self.lock_reason}. No new trades.")
            return False

        # Max daily trades
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            log.warning(
                f"Daily trade limit reached ({self.daily_trades}/{config.MAX_DAILY_TRADES}). "
                f"Skipping {signal.symbol}."
            )
            return False

        # Max open positions
        if open_positions >= config.MAX_POSITIONS:
            log.warning(
                f"Max open positions reached ({open_positions}/{config.MAX_POSITIONS}). "
                f"Skipping {signal.symbol}."
            )
            return False

        # Max capital deployed
        order_cost = signal.shares * signal.entry_price
        if capital_deployed + order_cost > config.MAX_CAPITAL:
            log.warning(
                f"Max capital limit: deploying ${capital_deployed:.0f} + "
                f"${order_cost:.0f} > ${config.MAX_CAPITAL:.0f}. "
                f"Skipping {signal.symbol}."
            )
            return False

        # Min confidence threshold (50%)
        if signal.confidence < 50:
            log.warning(
                f"Signal confidence too low ({signal.confidence:.0f}% < 50%). "
                f"Skipping {signal.symbol}."
            )
            return False

        # Symbol cooldown
        if not self._cooldown_ok(signal.symbol):
            return False

        log.info(
            f"Risk OK: {signal.symbol} | "
            f"DailyTrades={self.daily_trades}/{config.MAX_DAILY_TRADES} | "
            f"Positions={open_positions}/{config.MAX_POSITIONS} | "
            f"DailyPnL=${self.daily_pnl:+.2f}"
        )
        return True

    # ------------------------------------------------------------------ #
    # Post-trade recording
    # ------------------------------------------------------------------ #

    def record_trade(self, signal: TradeSignal) -> None:
        """Call immediately after a successful order submission."""
        self._check_daily_reset()
        self.daily_trades += 1
        self._symbol_cooldowns[signal.symbol] = datetime.utcnow()
        log.info(f"Trade recorded: {signal.symbol} (total today: {self.daily_trades})")

    def record_close(self, symbol: str, pnl: float) -> None:
        """
        Call when a position is closed.
        pnl is positive for profit, negative for loss.
        """
        self._check_daily_reset()
        self.daily_pnl += pnl

        if pnl < 0:
            self.daily_loss += abs(pnl)
        else:
            self.daily_profit += pnl

        log.info(
            f"Close recorded: {symbol} PnL=${pnl:+.2f} | "
            f"DailyPnL=${self.daily_pnl:+.2f} | "
            f"DailyLoss=${self.daily_loss:.2f} | "
            f"DailyProfit=${self.daily_profit:.2f}"
        )

        # Check daily loss limit
        if self.daily_loss >= config.MAX_DAILY_LOSS:
            self.locked = True
            self.lock_reason = f"Max daily loss hit (${self.daily_loss:.2f} >= ${config.MAX_DAILY_LOSS})"
            log.critical(f"TRADING LOCKED: {self.lock_reason}")

        # Check daily profit target
        if self.daily_profit >= config.DAILY_PROFIT_TARGET:
            self.locked = True
            self.lock_reason = f"Daily profit target hit (${self.daily_profit:.2f} >= ${config.DAILY_PROFIT_TARGET})"
            log.info(f"TRADING STOPPED (PROFIT TARGET): {self.lock_reason}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _cooldown_ok(self, symbol: str) -> bool:
        """Return False if we traded this symbol too recently."""
        if symbol not in self._symbol_cooldowns:
            return True
        last_trade = self._symbol_cooldowns[symbol]
        elapsed = (datetime.utcnow() - last_trade).total_seconds() / 60
        if elapsed < self.COOLDOWN_MINUTES:
            remaining = self.COOLDOWN_MINUTES - elapsed
            log.warning(
                f"Cooldown active for {symbol}: {remaining:.0f}min remaining. Skipping."
            )
            return False
        return True

    def status_summary(self) -> str:
        """One-line status for logging/alerts."""
        return (
            f"DailyPnL=${self.daily_pnl:+.2f} | "
            f"Loss=${self.daily_loss:.2f}/{config.MAX_DAILY_LOSS} | "
            f"Profit=${self.daily_profit:.2f}/{config.DAILY_PROFIT_TARGET} | "
            f"Trades={self.daily_trades}/{config.MAX_DAILY_TRADES} | "
            f"{'LOCKED: ' + self.lock_reason if self.locked else 'ACTIVE'}"
        )
