"""
bot/portfolio.py - Portfolio & Position Tracker

Tracks all open and closed positions in SQLite.
Provides unrealized/realized P&L, summary, and daily report.
Also serves as the source of truth for the risk manager's
position count and capital deployed calculations.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

from config import config
from bot.executor import OrderResult

log = logging.getLogger(__name__)


class Portfolio:
    """
    SQLite-backed portfolio tracker.

    Tables:
      - trades: all submitted orders (open + closed)
    """

    def __init__(self) -> None:
        db_path = Path(config.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_tables()
        log.info(f"Portfolio DB: {db_path}")

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id      TEXT,
                symbol        TEXT NOT NULL,
                strategy      TEXT,
                shares        INTEGER,
                entry_price   REAL,
                stop_loss     REAL,
                take_profit   REAL,
                exit_price    REAL,
                realized_pnl  REAL,
                status        TEXT DEFAULT 'OPEN',
                mode          TEXT,
                open_time     TEXT,
                close_time    TEXT
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Open / record position
    # ------------------------------------------------------------------ #

    def record_open(self, result: OrderResult) -> int:
        """Insert a new open trade. Returns the row id."""
        cur = self.conn.execute(
            """
            INSERT INTO trades
              (order_id, symbol, strategy, shares, entry_price,
               stop_loss, take_profit, status, mode, open_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """,
            (
                result.order_id,
                result.symbol,
                result.strategy,
                result.shares,
                result.entry_price,
                result.stop_loss,
                result.take_profit,
                result.mode,
                result.timestamp.isoformat(),
            ),
        )
        self.conn.commit()
        row_id = cur.lastrowid
        log.info(f"Position opened: {result.symbol} (row {row_id})")
        return row_id

    def record_close(
        self, symbol: str, exit_price: float, realized_pnl: float
    ) -> None:
        """Mark the most recent open position for symbol as CLOSED."""
        self.conn.execute(
            """
            UPDATE trades
            SET exit_price = ?, realized_pnl = ?, status = 'CLOSED', close_time = ?
            WHERE symbol = ? AND status = 'OPEN'
            ORDER BY id DESC
            LIMIT 1
            """,
            (exit_price, realized_pnl, datetime.utcnow().isoformat(), symbol),
        )
        self.conn.commit()
        log.info(f"Position closed: {symbol} @ ${exit_price:.2f} PnL=${realized_pnl:+.2f}")

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def open_positions(self) -> List[dict]:
        """Return all currently open positions."""
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN'"
        ).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM trades LIMIT 0").description or []]
        # Simple approach: fetch column names from cursor
        cur = self.conn.execute("SELECT * FROM trades WHERE status = 'OPEN'")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def open_count(self) -> int:
        """Number of open positions."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'OPEN'"
        ).fetchone()[0]

    def capital_deployed(self) -> float:
        """Total capital in open positions (shares * entry_price)."""
        result = self.conn.execute(
            "SELECT SUM(shares * entry_price) FROM trades WHERE status = 'OPEN'"
        ).fetchone()[0]
        return float(result or 0)

    def daily_pnl(self, for_date: Optional[date] = None) -> float:
        """Realized P&L for a given date (default: today)."""
        d = (for_date or date.today()).isoformat()
        result = self.conn.execute(
            "SELECT SUM(realized_pnl) FROM trades WHERE status='CLOSED' AND close_time LIKE ?",
            (f"{d}%",),
        ).fetchone()[0]
        return float(result or 0)

    def all_closed_today(self) -> List[dict]:
        """All trades closed today."""
        d = date.today().isoformat()
        cur = self.conn.execute(
            "SELECT * FROM trades WHERE status='CLOSED' AND close_time LIKE ?",
            (f"{d}%",),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self) -> str:
        """One-line portfolio summary."""
        open_count = self.open_count()
        cap = self.capital_deployed()
        pnl = self.daily_pnl()
        return (
            f"OpenPositions={open_count} | "
            f"CapDeployed=${cap:.2f} | "
            f"DailyRealizedPnL=${pnl:+.2f}"
        )

    def daily_report(self) -> str:
        """Full daily P&L report for Telegram/Discord/log."""
        trades = self.all_closed_today()
        if not trades:
            return "No closed trades today."

        lines = [f"Daily Report - {date.today()}", "-" * 40]
        total_pnl = 0.0
        wins, losses = 0, 0

        for t in trades:
            pnl = t.get("realized_pnl") or 0
            total_pnl += pnl
            symbol = t["symbol"]
            entry = t["entry_price"]
            exit_p = t.get("exit_price") or 0
            shares = t["shares"]
            strat = t.get("strategy", "?")
            emoji = "WIN" if pnl > 0 else "LOSS"
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            lines.append(
                f"  [{emoji}] {symbol} ({strat}) "
                f"{shares}sh ${entry:.2f}->${exit_p:.2f} "
                f"PnL=${pnl:+.2f}"
            )

        lines.append("-" * 40)
        win_rate = wins / max(wins + losses, 1) * 100
        lines.append(
            f"Total: {wins}W / {losses}L ({win_rate:.0f}% win rate) | "
            f"Net PnL=${total_pnl:+.2f}"
        )
        return "\n".join(lines)

    def close(self) -> None:
        self.conn.close()
