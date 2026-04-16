"""
bot/notifier.py - Telegram & Discord Notification Module

Sends alerts for:
  - New trade signals (pre-execution)
  - Filled orders
  - Stop-loss triggers
  - Daily P&L summary
  - Risk locks / errors

Leave TELEGRAM_BOT_TOKEN and DISCORD_WEBHOOK_URL blank to disable.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import requests

from config import config

log = logging.getLogger(__name__)


class Notifier:
    """
    Sends notifications to Telegram and/or Discord.
    Fails silently if credentials not configured.
    """

    def __init__(self) -> None:
        self.telegram_token = config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = config.TELEGRAM_CHAT_ID
        self.discord_url = config.DISCORD_WEBHOOK_URL
        self.mode = config.TRADING_MODE

        if self.telegram_token:
            log.info("Telegram notifications: ENABLED")
        if self.discord_url:
            log.info("Discord notifications: ENABLED")

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def trade_signal(self, symbol: str, strategy: str, entry: float,
                     stop: float, target: float, shares: int,
                     confidence: float, rr: float) -> None:
        """Alert: new trade signal detected."""
        msg = (
            f"SIGNAL [{self.mode}]\n"
            f"Symbol: {symbol}\n"
            f"Strategy: {strategy}\n"
            f"Entry: ${entry:.2f} | Shares: {shares}\n"
            f"Stop: ${stop:.2f} | Target: ${target:.2f}\n"
            f"R:R = {rr:.1f} | Confidence: {confidence:.0f}%"
        )
        self._send(msg)

    def trade_filled(self, symbol: str, shares: int, price: float,
                      stop: float, target: float, strategy: str) -> None:
        """Alert: order filled."""
        msg = (
            f"FILLED [{self.mode}]\n"
            f"{symbol} x{shares} @ ${price:.2f}\n"
            f"Strategy: {strategy}\n"
            f"SL=${stop:.2f} | TP=${target:.2f}"
        )
        self._send(msg)

    def trade_closed(self, symbol: str, pnl: float, exit_price: float) -> None:
        """Alert: position closed."""
        result = "WIN" if pnl >= 0 else "LOSS"
        msg = (
            f"{result} [{self.mode}]\n"
            f"{symbol} closed @ ${exit_price:.2f}\n"
            f"PnL: ${pnl:+.2f}"
        )
        self._send(msg)

    def daily_summary(self, report: str) -> None:
        """Send end-of-day P&L report."""
        msg = f"DAILY SUMMARY [{self.mode}]\n{report}"
        self._send(msg)

    def risk_lock(self, reason: str) -> None:
        """Alert: risk manager locked trading."""
        msg = f"RISK LOCKED [{self.mode}]\n{reason}"
        self._send(msg)

    def error(self, context: str, err: str) -> None:
        """Alert: unexpected error."""
        msg = f"ERROR [{self.mode}]\n{context}\n{err}"
        self._send(msg)

    def info(self, message: str) -> None:
        """General informational alert."""
        self._send(f"INFO [{self.mode}]\n{message}")

    # ------------------------------------------------------------------ #
    # Internal send
    # ------------------------------------------------------------------ #

    def _send(self, message: str) -> None:
        """Send message to all configured channels."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        full_msg = f"{message}\n[{timestamp}]"

        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(full_msg)

        if self.discord_url:
            self._send_discord(full_msg)

    def _send_telegram(self, message: str) -> None:
        """Send via Telegram Bot API."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                log.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
        except Exception as exc:
            log.warning(f"Telegram error: {exc}")

    def _send_discord(self, message: str) -> None:
        """Send via Discord webhook."""
        try:
            payload = {"content": f"```\n{message}\n```"}
            resp = requests.post(self.discord_url, json=payload, timeout=10)
            if not resp.ok:
                log.warning(f"Discord send failed: {resp.status_code} {resp.text}")
        except Exception as exc:
            log.warning(f"Discord error: {exc}")
