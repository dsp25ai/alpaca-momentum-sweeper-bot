"""
bot/executor.py - Order Execution Module

Handles all order placement via Alpaca Trading API.
Supports:
  - Paper mode (full simulation via Alpaca paper account)
  - Live mode (real orders)
  - Bracket orders (entry + stop-loss + take-profit in one request)
  - Market and limit order types
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)

from config import config
from bot.strategy import TradeSignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order result model
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str]
    symbol: str
    shares: int
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: str
    mode: str  # PAPER or LIVE
    timestamp: datetime
    error: Optional[str] = None
    client_order_id: Optional[str] = None

    def __str__(self) -> str:
        status = "OK" if self.success else f"FAILED: {self.error}"
        return (
            f"[{self.mode}] {self.symbol} {self.shares}sh "
            f"@ ${self.entry_price:.2f} SL=${self.stop_loss:.2f} "
            f"TP=${self.take_profit:.2f} [{status}]"
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Places bracket orders on Alpaca.

    In PAPER mode: uses paper trading URL (safe).
    In LIVE mode: uses live URL (real money).

    Always uses bracket orders so stop-loss and take-profit are
    attached to the entry order and managed server-side by Alpaca.
    """

    def __init__(self) -> None:
        paper = config.TRADING_MODE == "PAPER"
        self.client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=paper,
        )
        self.mode = config.TRADING_MODE
        log.info(f"Executor initialized in {self.mode} mode.")

    def execute(self, signal: TradeSignal) -> OrderResult:
        """
        Place a bracket order for the given signal.
        Returns OrderResult with success/failure details.
        """
        try:
            client_order_id = f"sweep_{signal.symbol}_{uuid.uuid4().hex[:8]}"

            order_request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=round(signal.take_profit, 2)
                ),
                stop_loss=StopLossRequest(
                    stop_price=round(signal.stop_loss, 2)
                ),
                client_order_id=client_order_id,
            )

            log.info(
                f"Placing {self.mode} bracket order: "
                f"{signal.symbol} x{signal.shares} "
                f"SL=${signal.stop_loss:.2f} TP=${signal.take_profit:.2f}"
            )

            order = self.client.submit_order(order_request)

            result = OrderResult(
                success=True,
                order_id=str(order.id),
                symbol=signal.symbol,
                shares=signal.shares,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy,
                mode=self.mode,
                timestamp=datetime.utcnow(),
                client_order_id=client_order_id,
            )
            log.info(f"Order placed: {result}")
            return result

        except Exception as exc:
            log.error(f"Order execution error for {signal.symbol}: {exc}")
            return OrderResult(
                success=False,
                order_id=None,
                symbol=signal.symbol,
                shares=signal.shares,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy,
                mode=self.mode,
                timestamp=datetime.utcnow(),
                error=str(exc),
            )

    def cancel_all_orders(self) -> None:
        """Emergency: cancel all open orders."""
        try:
            self.client.cancel_orders()
            log.warning("All open orders cancelled.")
        except Exception as exc:
            log.error(f"Error cancelling orders: {exc}")

    def close_position(self, symbol: str) -> bool:
        """Close an open position immediately at market."""
        try:
            self.client.close_position(symbol)
            log.info(f"Position closed: {symbol}")
            return True
        except Exception as exc:
            log.error(f"Error closing position {symbol}: {exc}")
            return False

    def get_account(self) -> dict:
        """Return account info (buying power, equity, etc)."""
        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as exc:
            log.error(f"Error fetching account: {exc}")
            return {}

    def get_open_positions(self) -> list:
        """Return list of current open positions."""
        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc) * 100,
                    "market_value": float(p.market_value),
                }
                for p in positions
            ]
        except Exception as exc:
            log.error(f"Error fetching positions: {exc}")
            return []
