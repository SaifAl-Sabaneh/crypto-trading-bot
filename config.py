import os
from dotenv import load_dotenv

# Load secret environment variables from .env file (if it exists)
load_dotenv()

# ----------------- SECURITY & ENV SETTINGS -----------------
API_KEY = os.getenv("EXCHANGE_API_KEY", "")
SECRET_KEY = os.getenv("EXCHANGE_SECRET_KEY", "")
IS_SANDBOX = True              # Safe mode: True for mock paper-trading, False for live money

# ----------------- DATA SETTINGS -----------------
TICKERS = ["COIN", "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "GLD", "SLV", "TSLA", "AAPL", "MSFT", "NVDA", "AMZN", "META", "SPY", "QQQ"] # Expanded multi-sector asset universe
SHORTABLE_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "COIN", "TSLA"] # Tickers allowed for short-selling (high volatility growth/crypto)
START_DATE = "2023-01-01"      # Historical data start date
END_DATE = "2026-06-25"        # Historical data end date
INTERVAL = "1d"                # Timeframe interval (e.g., '1d')

# ----------------- LAYER 1: REGIME & TREND FILTER -----------------
VOLATILITY_WINDOW = 20         # Period for volatility calculation
REGIME_PERCENTILE_LIMIT = 75   # Volatility percentile limit above which trading halts
LOOKBACK_PERCENTILE = 200      # Lookback window for computing rolling percentile

USE_TREND_FILTER = True        # Only buy when price is above SMA_TREND_WINDOW
SMA_TREND_WINDOW = 200         # Window for long-term trend filter
EMA_TREND_WINDOW = 50          # Window for medium-term trend confirmation

# ----------------- LAYER 2: ML ENSEMBLE CLASSIFIER -----------------
FORECAST_HORIZON = 10          # Prediction lookforward horizon (e.g., 10 days for Triple Barrier)
TRAIN_TEST_SPLIT_RATIO = 0.7   # Proportion of data used for initial split
ML_MODEL_TYPE = "ensemble"     # Stacking ensemble: RF + GB + CatBoost + LogisticRegression
CONFIDENCE_THRESHOLD_LONG = 0.37  # Strict confidence threshold for buying/longing (calibrated to 0.37 to avoid bad early longs)
CONFIDENCE_THRESHOLD_SHORT = 0.30 # Calibrated threshold for shorting (triggered when prob is <= 0.30)

# ----------------- ADVANCED ML UPGRADES -----------------
USE_SENTIMENT = True              # Use Crypto Fear & Greed Index daily sentiment features
RUN_HYPERPARAMETER_TUNING = True  # Auto-optimize model parameters using RandomizedSearchCV
FEAR_GREED_GREED_CAP = 65         # Greed cap (blocks long entries in moderately frothy markets)
FEAR_GREED_FEAR_FLOOR = 15        # Fear floor (only blocks shorts at absolute panic bottoms below 15)


# ----------------- LAYER 3: EXECUTION & POSITION SIZING -----------------
ATR_WINDOW = 14                # Window for ATR
TP_ATR_MULT = 2.5              # Take-Profit multiplier (restored to 2.5 for balanced label distribution)
SL_ATR_MULT_LONG = 1.5         # Stop-Loss multiplier for long positions (optimized to avoid premature stop-outs)
SL_ATR_MULT_SHORT = 1.2        # Stop-Loss multiplier for short positions (optimized to cut squeeze losses early)
ENABLE_BREAKEVEN = True        # Move SL to Entry once price moves 0.8 * ATR in our favor (tightened)

INITIAL_CAPITAL = 10000.0      # Starting backtest capital in USD
MAX_ALLOCATION_PER_TRADE = 0.44# Max portfolio allocation per trade (quality-focused 44% for high win rate and Sharpe)

# ----------------- REAL-WORLD RISK PROTECTIONS -----------------
SLIPPAGE_PENALTY_PCT = 0.0015  # 0.15% slippage/fee penalty applied to every trade exit
WEEKLY_DRAWDOWN_LIMIT = 0.10   # 10% weekly drawdown limit (adjusted to 10% to prevent false halts during normal high-volatility regimes)

# ----------------- ALERTS & WEBHOOK CHANNELS -----------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ----------------- SYSTEM LOGGING -----------------
LOG_FILE_PATH = "trading_bot.log"
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
