import os
from dotenv import load_dotenv

# Load secret environment variables from .env file (if it exists)
load_dotenv()

# ----------------- SECURITY & ENV SETTINGS -----------------
API_KEY = os.getenv("EXCHANGE_API_KEY", "")
SECRET_KEY = os.getenv("EXCHANGE_SECRET_KEY", "")
IS_SANDBOX = True              # Safe mode: True for mock paper-trading, False for live money

# ----------------- DATA SETTINGS -----------------
TICKERS = ["COIN", "BTC-USD", "ETH-USD"] # Focus on highly liquid crypto/crypto-related assets
START_DATE = "2023-01-01"      # Historical data start date
END_DATE = "2026-06-25"        # Historical data end date
INTERVAL = "1d"                # Timeframe interval (e.g., '1d')

# ----------------- LAYER 1: REGIME & TREND FILTER -----------------
VOLATILITY_WINDOW = 20         # Period for volatility calculation
REGIME_PERCENTILE_LIMIT = 75   # Volatility percentile limit above which trading halts
LOOKBACK_PERCENTILE = 200      # Lookback window for computing rolling percentile

USE_TREND_FILTER = True        # Only buy when price is above SMA_TREND_WINDOW
SMA_TREND_WINDOW = 200         # Window for long-term trend filter

# ----------------- LAYER 2: ML ENSEMBLE CLASSIFIER -----------------
FORECAST_HORIZON = 5           # Predictions forecast horizon
TRAIN_TEST_SPLIT_RATIO = 0.7   # Proportion of data used for initial split
ML_MODEL_TYPE = "ensemble"     # Stacking ensemble: RF + GB + CatBoost + LogisticRegression
CONFIDENCE_THRESHOLD = 0.85    # Strict confidence threshold for signals

# ----------------- LAYER 3: EXECUTION & POSITION SIZING -----------------
ATR_WINDOW = 14                # Window for ATR
TP_ATR_MULT = 2.5              # Take-Profit multiplier
SL_ATR_MULT = 1.0              # Stop-Loss multiplier
ENABLE_BREAKEVEN = True        # Move SL to Entry once price moves 1.0 * ATR in our favor

INITIAL_CAPITAL = 10000.0      # Starting backtest capital in USD
MAX_ALLOCATION_PER_TRADE = 0.25# Max portfolio allocation per trade (25% for diversification)

# ----------------- REAL-WORLD RISK PROTECTIONS -----------------
SLIPPAGE_PENALTY_PCT = 0.0015  # 0.15% slippage/fee penalty applied to every trade exit
WEEKLY_DRAWDOWN_LIMIT = 0.05   # 5% weekly drawdown limit. Triggering this halts all trading.

# ----------------- ALERTS & WEBHOOK CHANNELS -----------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ----------------- SYSTEM LOGGING -----------------
LOG_FILE_PATH = "trading_bot.log"
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
