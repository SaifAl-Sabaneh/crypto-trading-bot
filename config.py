import os
from dotenv import load_dotenv

# Load secret environment variables from .env file (if it exists)
load_dotenv()

# ----------------- SECURITY & ENV SETTINGS -----------------
API_KEY = os.getenv("EXCHANGE_API_KEY", "")
SECRET_KEY = os.getenv("EXCHANGE_SECRET_KEY", "")
IS_SANDBOX = False            # Safe mode: True for mock paper-trading, False for live money

# ----------------- DATA SETTINGS -----------------
TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "AVAX-USD", "LINK-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "SUI20947-USD", "NEAR-USD", "LTC-USD", "TRX-USD", "OP-USD", "INJ-USD", "BCH-USD", "SHIB-USD", "WIF-USD", "ONDO-USD", "FET-USD", "RENDER-USD", "TON11419-USD", "PEPE24478-USD", "TAO22974-USD", "FTM-USD", "WLD-USD", "APT-USD", "TIA-USD"] # High-momentum crypto universe
SHORTABLE_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "AVAX-USD", "LINK-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "SUI20947-USD", "NEAR-USD", "LTC-USD", "TRX-USD", "OP-USD", "INJ-USD", "BCH-USD", "SHIB-USD", "WIF-USD", "ONDO-USD", "FET-USD", "RENDER-USD", "TON11419-USD", "PEPE24478-USD", "TAO22974-USD", "FTM-USD", "WLD-USD", "APT-USD", "TIA-USD"]
CRYPTO_TICKERS    = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "AVAX-USD", "LINK-USD", "ADA-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "SUI20947-USD", "NEAR-USD", "LTC-USD", "TRX-USD", "OP-USD", "INJ-USD", "BCH-USD", "SHIB-USD", "WIF-USD", "ONDO-USD", "FET-USD", "RENDER-USD", "TON11419-USD", "PEPE24478-USD", "TAO22974-USD", "FTM-USD", "WLD-USD", "APT-USD", "TIA-USD"] # Crypto assets eligible for on-chain features
START_DATE = "2023-01-01"      # Historical data start date
END_DATE = "2026-06-25"        # Historical data end date
INTERVAL = "4h"                # Timeframe interval (resampled from 1h for 4-hour scans)

# ----------------- LAYER 1: REGIME & TREND FILTER -----------------
VOLATILITY_WINDOW = 20         # Period for volatility calculation
REGIME_PERCENTILE_LIMIT = 75   # Volatility percentile limit above which trading halts
LOOKBACK_PERCENTILE = 200      # Lookback window for computing rolling percentile

USE_TREND_FILTER = True        # Only buy when price is above SMA_TREND_WINDOW
SMA_TREND_WINDOW = 50          # Window for long-term trend filter (relaxed from 200 to 50 to capture recovery runs)
EMA_TREND_WINDOW = 50          # Window for medium-term trend confirmation

# ----------------- LAYER 2: ML ENSEMBLE CLASSIFIER -----------------
FORECAST_HORIZON = 10          # Prediction lookforward horizon (e.g., 10 days for Triple Barrier)
TRAIN_TEST_SPLIT_RATIO = 0.7   # Proportion of data used for initial split
ML_MODEL_TYPE = "ensemble"     # Stacking ensemble: RF + GB + CatBoost + LogisticRegression
CONFIDENCE_THRESHOLD_LONG = 0.45  # Strict confidence threshold for buying/longing (calibrated to 0.45 to avoid bad early longs)
CONFIDENCE_THRESHOLD_SHORT = 0.33 # Calibrated threshold for shorting (increased to 0.33 to allow shorts on clean dumps)

# ----------------- ADVANCED ML UPGRADES -----------------
USE_SENTIMENT = True              # Use Crypto Fear & Greed Index daily sentiment features
RUN_HYPERPARAMETER_TUNING = True  # Auto-optimize model parameters using RandomizedSearchCV
FEAR_GREED_GREED_CAP = 65         # Greed cap (blocks long entries in moderately frothy markets)
FEAR_GREED_FEAR_FLOOR = 15        # Fear floor (only blocks shorts at absolute panic bottoms below 15)


# ----------------- LAYER 3: EXECUTION & POSITION SIZING -----------------
ATR_WINDOW = 14                # Window for ATR
TP_ATR_MULT = 2.5              # Take-Profit multiplier (restored to 2.5 for balanced label distribution)
SL_ATR_MULT_LONG = 1.5         # Stop-Loss multiplier for long positions (optimized to avoid premature stop-outs)
SL_ATR_MULT_SHORT = 1.0        # Stop-Loss multiplier for short positions (optimized to cut squeeze losses early)
ENABLE_BREAKEVEN = True        # Move SL to Entry once price moves 0.8 * ATR in our favor (tightened)
ENABLE_TRAILING_TP = True      # Enable dynamic trailing take-profit
TRAILING_TP_ACTIVATION_ATR_MULT = 1.8 # Activate trailing mode once price moves 1.8x ATR in profit
TRAILING_TP_CALLBACK_ATR_MULT = 0.5   # Float the stop-loss exactly 0.5x ATR below the peak

INITIAL_CAPITAL = 10000.0      # Starting backtest capital in USD
MAX_ALLOCATION_PER_TRADE = 0.20# Max portfolio allocation per trade (quality-focused 20% for faster compounded gains)
LEVERAGE = 20                  # Default leverage multiplier
ENABLE_DYNAMIC_LEVERAGE = True  # Enable volatility-adjusted leverage
LEVERAGE_VOL_LOW = 25          # Calm squeezes get 25x leverage
LEVERAGE_VOL_HIGH = 10         # High-volatility panic gets 10x leverage

ENABLE_SENTIMENT_SIZING = True  # Scale position sizes based on global news sentiment
SENTIMENT_SIZE_ALIGNED = 1.0    # 100% allocation if AI signal aligns with news sentiment
SENTIMENT_SIZE_MISALIGNED = 0.25 # 25% allocation if AI signal conflicts with news sentiment (coin-flip trades)
STRICT_TREND_LOCK = True       # Longs only above 200 SMA, Shorts only below 200 SMA
EXTREME_FEAR_BLOCK = True      # Block short positions when Fear & Greed Index drops below FEAR_LIMIT
FEAR_LIMIT = 25                # Extreme Fear threshold for blocking short entries

# ----------------- REAL-WORLD RISK PROTECTIONS -----------------
SLIPPAGE_PENALTY_PCT = 0.0015  # 0.15% slippage/fee penalty applied to every trade exit
WEEKLY_DRAWDOWN_LIMIT = 0.10   # 10% weekly drawdown limit (adjusted to 10% to prevent false halts during normal high-volatility regimes)

# ----------------- ALERTS & WEBHOOK CHANNELS -----------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ----------------- INTELLIGENCE UPGRADES -----------------

# HMM Regime Classifier
USE_HMM_REGIME = True          # Enable HMM regime classifier (filters out trades during choppy/sideways markets)
HMM_N_STATES = 4               # Number of hidden market states: Bull, Bear, Sideways, Crisis
HMM_LOOKBACK = 60              # Minimum bars needed to fit HMM

# Kelly Criterion Dynamic Sizing
USE_KELLY_SIZING = True       # Enable Kelly Criterion (uses dynamic allocation based on model confidence)
KELLY_FRACTION = 0.5           # Half-Kelly for safety (full Kelly is too aggressive)
KELLY_MAX_ALLOC = MAX_ALLOCATION_PER_TRADE  # Cap Kelly sizing at max allocation
KELLY_MIN_ALLOC = 0.05         # Minimum allocation floor (5%) if signal fires

# Correlation Guard
USE_CORRELATION_GUARD = False  # Disable Correlation Guard (restores high trade volume)
CORRELATION_GUARD_THRESHOLD = 0.75  # Correlation threshold above which new trades are blocked
CORRELATION_LOOKBACK = 60     # Rolling window (days) for computing asset correlations

# Macro Features (via free yfinance)
USE_MACRO_FEATURES = True      # Add DXY, VIX, yield curve features to ensemble inputs

# On-Chain Crypto Data (via free CoinMetrics community API)
USE_ONCHAIN_FEATURES = True    # Add NVT, active address, exchange flow features for crypto assets

# News NLP Sentiment (via free RSS feeds)
USE_NEWS_NLP = True            # Add NLP sentiment score from financial RSS feeds
NEWS_RSS_FEEDS = [             # Free RSS feeds for financial news
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://coindesk.com/arc/outboundfeeds/rss/"
]
NEWS_SENTIMENT_WINDOW = 3     # Rolling window (days) for news sentiment averaging

# LSTM Temporal Layer
USE_LSTM_LAYER = True          # Add lightweight NumPy LSTM as additional stacking feature
LSTM_SEQUENCE_LENGTH = 20      # Number of past bars fed into LSTM as context window
LSTM_HIDDEN_SIZE = 32          # Number of hidden units in LSTM cell

# RL Agent
USE_RL_AGENT = True            # Enable RL agent veto layer to protect capital and boost profit margins


# ----------------- SYSTEM LOGGING -----------------
LOG_FILE_PATH = "trading_bot.log"
LOG_LEVEL = "INFO"             # DEBUG, INFO, WARNING, ERROR, CRITICAL
