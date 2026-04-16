"""
bot/strategy.py - Trading Strategy Engine

Implements two complementary strategies:
  1. Opening Range Breakout (ORB): Buy when price breaks above
     the high of the first N-minute candle after market open.
  2. VWAP Reclaim: Buy when price reclaims VWAP after a brief
     pullback, confirming continuation of upward momentum.

Returns TradeSignal objects for the executor to act on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import config
from bot.scanner import ScanResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------------

@dataclass
class TradeSignal:
    """A confirmed trade signal ready to send to the executor."""
    symbol: str
    strategy: str           # "ORB" | "VWAP_RECLAIM"
    direction: str          # "BUY" (long only for now)
    entry_price: float      # suggested limit/market entry price
    stop_loss: float        # hard stop below
    take_profit: float      # target price
    shares: int             # position size (risk-adjusted)
    confidence: float       # 0-100 signal confidence score
    scan: ScanResult        # originating scan data
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def risk_reward(self) -> float:
        if (self.entry_price - self.stop_loss) == 0:
            return 0
        return (self.take_profit - self.entry_price) / (self.entry_price - self.stop_loss)

    def __str__(self) -> str:
        rr = self.risk_reward()
        return (
            f"[{self.strategy}] {self.symbol} BUY {self.shares}sh @ ${self.entry_price:.2f} "
            f"| SL=${self.stop_loss:.2f} TP=${self.take_profit:.2f} "
            f"| R:R={rr:.1f} | Conf={self.confidence:.0f}%"
        )


# ---------------------------------------------------------------------------
# Strategy engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    Evaluates scanner candidates and generates trade signals.

    Usage:
        engine = StrategyEngine()
        signals = engine.evaluate(scan_results)
    """

    def __init__(self) -> None:
        self.data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )

    def evaluate(self, candidates: List[ScanResult]) -> List[TradeSignal]:
        """Run all strategies on each candidate. Returns valid signals."""
        signals = []
        for candidate in candidates:
            intraday_df = self._fetch_intraday(candidate.symbol)
            if intraday_df is None or intraday_df.empty:
                log.debug(f"No intraday data for {candidate.symbol}")
                continue

            # Try ORB strategy
            orb_signal = self._orb_strategy(candidate, intraday_df)
            if orb_signal:
                signals.append(orb_signal)
                continue  # Don't double-signal same stock

            # Try VWAP reclaim strategy
            vwap_signal = self._vwap_reclaim_strategy(candidate, intraday_df)
            if vwap_signal:
                signals.append(vwap_signal)

        log.info(f"Strategy engine: {len(signals)} signals from {len(candidates)} candidates")
        return signals

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #

    def _fetch_intraday(self, symbol: str, minutes: int = 5) -> Optional[pd.DataFrame]:
        """Fetch today's intraday bars (5-minute by default)."""
        try:
            end = datetime.utcnow()
            start = end.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 ET = 13:30 UTC
            if end < start:
                start = start - timedelta(days=1)

            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                start=start,
                end=end,
            )
            bars = self.data_client.get_stock_bars(req)
            df = bars.df
            if df.empty:
                return None
            df = df.reset_index()
            return df
        except Exception as exc:
            log.error(f"Intraday fetch error for {symbol}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Strategy 1: Opening Range Breakout (ORB)
    # ------------------------------------------------------------------ #

    def _orb_strategy(self, scan: ScanResult, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Buy breakout above the high of the opening range candle(s).
        Opening range = first ORB_MINUTES of trading.

        Entry conditions:
          - Price just broke above ORB high
          - Volume confirms (RVOL >= threshold)
          - Not more than 3% above ORB high (don't chase)
        """
        try:
            orb_end = df["timestamp"].min() + timedelta(minutes=config.ORB_MINUTES)
            orb_df = df[df["timestamp"] <= orb_end]
            post_orb_df = df[df["timestamp"] > orb_end]

            if orb_df.empty or post_orb_df.empty:
                return None

            orb_high = float(orb_df["high"].max())
            orb_low = float(orb_df["low"].min())
            orb_range = orb_high - orb_low

            current_bar = post_orb_df.iloc[-1]
            current_price = float(current_bar["close"])
            prev_bar = post_orb_df.iloc[-2] if len(post_orb_df) >= 2 else orb_df.iloc[-1]
            prev_close = float(prev_bar["close"])

            # Breakout: current bar closes above ORB high, prev bar was below
            broke_out = current_price > orb_high and prev_close <= orb_high
            # Don't chase: entry not more than 3% above ORB high
            not_too_extended = current_price <= orb_high * 1.03

            if not (broke_out and not_too_extended):
                return None

            # Entry, stop, target
            entry = current_price
            stop = orb_low  # Stop below ORB low
            risk = entry - stop
            if risk <= 0:
                return None
            target = entry + (risk * config.TAKE_PROFIT_R)

            # Position sizing: risk $ / (entry - stop)
            shares = max(1, int(config.RISK_PER_TRADE / risk))
            # Cap at max capital
            if shares * entry > config.MAX_CAPITAL:
                shares = max(1, int(config.MAX_CAPITAL / entry))

            confidence = self._orb_confidence(scan, orb_range, current_price, orb_high)

            log.info(f"ORB signal: {scan.symbol} @ ${entry:.2f} | SL=${stop:.2f} | TP=${target:.2f}")
            return TradeSignal(
                symbol=scan.symbol,
                strategy="ORB",
                direction="BUY",
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                take_profit=round(target, 2),
                shares=shares,
                confidence=confidence,
                scan=scan,
            )
        except Exception as exc:
            log.error(f"ORB strategy error for {scan.symbol}: {exc}")
            return None

    def _orb_confidence(self, scan: ScanResult, orb_range: float,
                         current_price: float, orb_high: float) -> float:
        """Score 0-100 for ORB signal quality."""
        score = 50.0
        # High RVOL = more confident
        if scan.rvol >= 5:   score += 15
        elif scan.rvol >= 3: score += 8
        # Strong gap = catalyst-driven
        if scan.gap_pct >= 10: score += 15
        elif scan.gap_pct >= 5: score += 8
        # Tight breakout (not too extended)
        extension = (current_price - orb_high) / orb_high * 100
        if extension <= 0.5: score += 10
        elif extension <= 1.5: score += 5
        else: score -= 10
        return min(score, 100.0)

    # ------------------------------------------------------------------ #
    # Strategy 2: VWAP Reclaim
    # ------------------------------------------------------------------ #

    def _vwap_reclaim_strategy(self, scan: ScanResult, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Buy when price reclaims VWAP after a pullback (bouncing off VWAP).

        Entry conditions:
          - VWAP is calculated for the session
          - Previous bar closed below VWAP
          - Current bar closes above VWAP (reclaim)
          - Stock is still up significantly from open (not a breakdown)
        """
        try:
            if len(df) < 4:
                return None

            # Calculate VWAP
            df = df.copy()
            df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
            df["cum_tp_vol"] = (df["typical_price"] * df["volume"]).cumsum()
            df["cum_vol"] = df["volume"].cumsum()
            df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]

            current = df.iloc[-1]
            prev = df.iloc[-2]

            current_close = float(current["close"])
            prev_close_val = float(prev["close"])
            current_vwap = float(current["vwap"])
            prev_vwap = float(prev["vwap"])

            # Reclaim: prev bar below VWAP, current bar above VWAP
            reclaim = prev_close_val < prev_vwap and current_close > current_vwap

            # Stock must still be up from open (not collapsing)
            still_positive = current_close > scan.open_price * 1.01

            if not (reclaim and still_positive):
                return None

            # Entry, stop, target
            entry = current_close
            # Stop = current VWAP - 0.5 * ATR
            stop = current_vwap - (scan.atr * 0.5)
            risk = entry - stop
            if risk <= 0:
                return None
            target = entry + (risk * config.TAKE_PROFIT_R)

            shares = max(1, int(config.RISK_PER_TRADE / risk))
            if shares * entry > config.MAX_CAPITAL:
                shares = max(1, int(config.MAX_CAPITAL / entry))

            confidence = self._vwap_confidence(scan)

            log.info(f"VWAP signal: {scan.symbol} @ ${entry:.2f} | VWAP=${current_vwap:.2f} | SL=${stop:.2f}")
            return TradeSignal(
                symbol=scan.symbol,
                strategy="VWAP_RECLAIM",
                direction="BUY",
                entry_price=round(entry, 2),
                stop_loss=round(stop, 2),
                take_profit=round(target, 2),
                shares=shares,
                confidence=confidence,
                scan=scan,
            )
        except Exception as exc:
            log.error(f"VWAP strategy error for {scan.symbol}: {exc}")
            return None

    def _vwap_confidence(self, scan: ScanResult) -> float:
        score = 45.0
        if scan.rvol >= 3:   score += 15
        if scan.gap_pct >= 5: score += 20
        if scan.gap_pct >= 10: score += 10
        return min(score, 100.0)
