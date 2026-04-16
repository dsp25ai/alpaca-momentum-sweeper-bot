"""
config.py - Centralized configuration loader.
Reads all settings from environment variables (via .env file).
Import this in every module instead of reading os.environ directly.
"""
import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


class Config:
    # ---- Alpaca Credentials ----
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    TRADING_MODE: str = os.getenv("TRADING_MODE", "PAPER").upper()

    # ---- Strategy ----
    MIN_RVOL: float = float(os.getenv("MIN_RVOL", "2.0"))
    MIN_GAP_PCT: float = float(os.getenv("MIN_GAP_PCT", "3.0"))
    ORB_MINUTES: int = int(os.getenv("ORB_MINUTES", "15"))
    MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "5"))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "50"))
    MAX_CAPITAL: float = float(os.getenv("MAX_CAPITAL", "1000"))
    TAKE_PROFIT_R: float = float(os.getenv("TAKE_PROFIT_R", "2.0"))
    STOP_LOSS_R: float = float(os.getenv("STOP_LOSS_R", "1.0"))

    # ---- Daily Risk Limits ----
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "150"))
    DAILY_PROFIT_TARGET: float = float(os.getenv("DAILY_PROFIT_TARGET", "400"))
    MAX_DAILY_TRADES: int = int(os.getenv("MAX_DAILY_TRADES", "10"))

    # ---- Scanner ----
    SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "60"))
    MIN_PRICE: float = float(os.getenv("MIN_PRICE", "5.0"))
    MAX_PRICE: float = float(os.getenv("MAX_PRICE", "500.0"))
    MIN_AVG_VOLUME: int = int(os.getenv("MIN_AVG_VOLUME", "500000"))

    # ---- Notifications ----
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # ---- Logging / Storage ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/bot.log")
    DB_PATH: str = os.getenv("DB_PATH", "data/trades.db")
    TIMEZONE: str = os.getenv("TIMEZONE", "America/New_York")

    @classmethod
    def validate(cls) -> None:
        """Raise ValueError if required credentials are missing."""
        if not cls.ALPACA_API_KEY or cls.ALPACA_API_KEY == "your_alpaca_api_key_here":
            raise ValueError(
                "ALPACA_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if not cls.ALPACA_SECRET_KEY or cls.ALPACA_SECRET_KEY == "your_alpaca_secret_key_here":
            raise ValueError(
                "ALPACA_SECRET_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if cls.TRADING_MODE not in ("PAPER", "LIVE"):
            raise ValueError("TRADING_MODE must be PAPER or LIVE")

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable config summary (no secrets)."""
        return (
            f"Mode={cls.TRADING_MODE} | "
            f"MaxPositions={cls.MAX_POSITIONS} | "
            f"RiskPerTrade=${cls.RISK_PER_TRADE} | "
            f"MaxCapital=${cls.MAX_CAPITAL} | "
            f"MaxDailyLoss=${cls.MAX_DAILY_LOSS} | "
            f"ProfitTarget=${cls.DAILY_PROFIT_TARGET} | "
            f"MinRVOL={cls.MIN_RVOL}x | "
            f"MinGap={cls.MIN_GAP_PCT}% | "
            f"ORB={cls.ORB_MINUTES}min"
        )


# Singleton instance used across all modules
config = Config()
