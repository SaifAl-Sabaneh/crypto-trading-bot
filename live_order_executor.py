"""
live_order_executor.py — Live Binance Futures Order Execution Engine

This script:
1. Connects to your Binance account using API Keys (Isolated Futures Mode).
2. Sets leverage to 10x (config.LEVERAGE).
3. Reads today's trading signals from the Ensemble/LSTM model.
4. Places Market entry orders (Long/Short) on Binance with 10% capital exposure.
5. Places Reduce-Only Take-Profit (TP) and Stop-Loss (SL) limit/stop orders.
6. Manages active positions (moves SL to breakeven once price moves 0.8 * ATR in our favor).
7. Syncs all outcomes to executed_trades.csv and posts real-time alerts to Discord.
"""

import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import EnsembleTradingModel
from features import build_features
from security import logger, send_push_notification, calculate_live_accuracy, generate_ai_commentary, push_to_github

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
api_key = os.getenv("EXCHANGE_API_KEY", "")
secret_key = os.getenv("EXCHANGE_SECRET_KEY", "")

# Ticker mapping between Yahoo Finance (indicators) and Binance Futures (orders)
SYMBOL_MAP = {
    'BTC-USD': 'BTC/USDT',
    'ETH-USD': 'ETH/USDT',
    'SOL-USD': 'SOL/USDT',
    'BNB-USD': 'BNB/USDT',
    'AVAX-USD': 'AVAX/USDT',
    'LINK-USD': 'LINK/USDT',
    'ADA-USD': 'ADA/USDT',
    'XRP-USD': 'XRP/USDT',
    'DOT-USD': 'DOT/USDT',
    'DOGE-USD': 'DOGE/USDT',
    'SUI20947-USD': 'SUI/USDT',
    'NEAR-USD': 'NEAR/USDT',
    'LTC-USD': 'LTC/USDT',
    'TRX-USD': 'TRX/USDT',
    'OP-USD': 'OP/USDT',
    'INJ-USD': 'INJ/USDT',
    'BCH-USD': 'BCH/USDT',
    'SHIB-USD': 'SHIB/USDT',
    'WIF-USD': 'WIF/USDT',
    'ONDO-USD': 'ONDO/USDT',
    'FET-USD': 'FET/USDT',
    'RENDER-USD': 'RENDER/USDT',
    'TON11419-USD': 'TON/USDT',
    'PEPE24478-USD': '1000PEPE/USDT',
    'TAO22974-USD': 'TAO/USDT',
    'FTM-USD': 'FTM/USDT',
    'WLD-USD': 'WLD/USDT',
    'APT-USD': 'APT/USDT',
    'TIA-USD': 'TIA/USDT',
    'RUNE-USD': 'RUNE/USDT',
    'JUP-USD': 'JUP/USDT',
    'LDO-USD': 'LDO/USDT'
}

def get_exchange_connection():
    """Initializes and returns ccxt Binance Futures connection."""
    if not api_key or not secret_key:
        raise ValueError("Exchange API credentials missing in .env file.")
    
    config_dict = {
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'timeout': 15000,  # Set connection timeout to 15 seconds
        'options': {
            'defaultType': 'future',  # Target USDT-M Futures account
        }
    }
    
    # Bypass US IP blocks using secure premium proxy (from environment / secrets)
    proxy = os.getenv("PROXY_URL", "")
    if proxy:
        # Mask credentials in logs for security
        masked_proxy = proxy
        if "@" in proxy:
            parts = proxy.split("@")
            masked_proxy = f"http://***:***@{parts[-1]}"
        logger.info(f"Configured CCXT with premium proxy: {masked_proxy}")
        config_dict['proxies'] = {
            'http': proxy,
            'https': proxy
        }
    else:
        logger.warning("No premium PROXY_URL found in environment secrets. Connecting directly.")
        
    exchange = ccxt.binance(config_dict)
    return exchange

def set_leverage_and_margin(exchange, symbol, leverage=None):
    """Sets isolated margin mode and leverage for the target symbol."""
    if leverage is None:
        leverage = getattr(config, 'LEVERAGE', 20)
    try:
        # 1. Set Isolated Margin Mode
        try:
            exchange.fapiPrivatePostMarginType({
                'symbol': symbol.replace('/', ''),
                'marginType': 'ISOLATED'
            })
            logger.info(f"Set margin type to ISOLATED for {symbol}.")
        except ccxt.ExchangeError as e:
            # Often throws error if already set to ISOLATED, we pass gracefully
            if "No need to change margin type" not in str(e):
                logger.warning(f"Could not set margin type for {symbol}: {e}")
                
        # 2. Set Leverage
        try:
            exchange.set_leverage(int(leverage), symbol)
            logger.info(f"Set leverage to {leverage}x for {symbol}.")
        except Exception as le:
            # Fallback to 20x if account is restricted to maximum 20x (Binance new account rule)
            if "leverage" in str(le).lower() or "-4300" in str(le):
                logger.warning(f"Leverage {leverage}x rejected for {symbol} (likely new account restriction). Retrying with 20x fallback...")
                exchange.set_leverage(20, symbol)
                logger.info(f"Set leverage to 20x fallback for {symbol}.")
            else:
                raise le
    except Exception as e:
        logger.error(f"Failed to configure leverage/margin for {symbol}: {e}")
        raise e

def get_futures_balance(exchange):
    """Returns the free USDT balance in the Futures account."""
    balance = exchange.fetch_balance()
    usdt_balance = balance['free'].get('USDT', 0.0)
    return float(usdt_balance)

def check_active_positions(exchange):
    """Fetches currently open futures positions on Binance."""
    positions = exchange.fetch_positions()
    active_positions = {}
    for pos in positions:
        size = float(pos.get('contracts', 0.0))
        if size > 0:
            symbol = pos['symbol']
            if ':' in symbol:
                symbol = symbol.split(':')[0]
            active_positions[symbol] = {
                'size': size,
                'side': pos['side'].lower(), # 'long' or 'short'
                'entry_price': float(pos['entryPrice']),
                'unrealized_pnl': float(pos['unrealizedPnl'])
            }
    return active_positions

def cancel_all_orders(exchange, symbol):
    """Cancels all open orders for a specific symbol."""
    try:
        exchange.cancel_all_orders(symbol)
        logger.info(f"Cancelled all open orders for {symbol}.")
    except Exception as e:
        logger.warning(f"Could not cancel orders for {symbol}: {e}")

def calculate_weekly_pnl(trade_history):
    """Calculates total PnL USD of trades closed in the last 7 days."""
    if not trade_history:
        return 0.0
    total_pnl = 0.0
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    for t in trade_history:
        try:
            exit_time_str = t.get('ExitTime', '')
            if not exit_time_str:
                continue
            exit_time = datetime.strptime(exit_time_str[:16], "%Y-%m-%d %H:%M")
            if exit_time >= seven_days_ago:
                total_pnl += float(t.get('PnL_USD', 0.0))
        except Exception:
            pass
    return total_pnl

def get_news_sentiment_sizing_multiplier(ticker, signal_direction):
    """
    Downloads latest RSS headlines for the ticker, parses sentiment using VADER,
    and returns a position size multiplier (1.0 or 0.25) depending on alignment.
    """
    if not getattr(config, 'ENABLE_SENTIMENT_SIZING', False):
        return 1.0, 0.0, "Disabled"
        
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        from features import get_vader_analyzer
        import urllib.parse
        
        analyzer = get_vader_analyzer()
        search_term = ticker.split('-')[0].lower() # e.g. btc-usd -> btc
        
        urls = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        ]
        if ticker in getattr(config, 'CRYPTO_TICKERS', []):
            urls.append("https://coindesk.com/arc/outboundfeeds/rss/")
            
        scores = []
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    xml_data = response.read()
                    root = ET.fromstring(xml_data)
                    for item in root.findall('.//item'):
                        title = item.find('title')
                        pub_date = item.find('pubDate')
                        if title is not None and title.text:
                            headline = title.text.strip()
                            if search_term in headline.lower():
                                # Check recency if pubDate is available
                                is_recent = True
                                if pub_date is not None:
                                    try:
                                        dt = pd.to_datetime(pub_date.text).tz_localize(None)
                                        # Limit search to the last 24 hours to keep sentiment fresh
                                        if (datetime.now() - dt).total_seconds() > 86400:
                                            is_recent = False
                                    except Exception:
                                        pass
                                if is_recent:
                                    vs = analyzer.polarity_scores(headline)
                                    scores.append(vs['compound'])
            except Exception:
                continue
                
        if not scores:
            logger.info(f"No recent news headlines found for {ticker}. Defaulting to neutral sentiment multiplier.")
            return 1.0, 0.0, "No News (Neutral)"
            
        avg_score = float(np.mean(scores))
        logger.info(f"News sentiment analysis for {ticker}: {len(scores)} recent articles, average compound score: {avg_score:.3f}")
        
        # Check alignment:
        # Aligned: signal is Long (1) and sentiment is positive (>0.05), OR signal is Short (-1) and sentiment is negative (<-0.05)
        # Misaligned: signal is Long (1) and sentiment is negative (<-0.05), OR signal is Short (-1) and sentiment is positive (>0.05)
        is_aligned = True
        sentiment_label = "Neutral"
        
        if avg_score > 0.05:
            sentiment_label = "Bullish"
            if signal_direction == -1:
                is_aligned = False
        elif avg_score < -0.05:
            sentiment_label = "Bearish"
            if signal_direction == 1:
                is_aligned = False
                
        if is_aligned:
            multiplier = getattr(config, 'SENTIMENT_SIZE_ALIGNED', 1.0)
            status = f"Aligned ({sentiment_label})"
        else:
            multiplier = getattr(config, 'SENTIMENT_SIZE_MISALIGNED', 0.25)
            status = f"MISALIGNED ({sentiment_label})"
            logger.info(f"News Sentiment Veto: Sizing multiplier scaled to {multiplier} for {ticker} due to misalignment.")
            
        return multiplier, avg_score, status
    except Exception as e:
        logger.warning(f"Failed to calculate news sentiment sizing for {ticker}: {e}")
        return 1.0, 0.0, f"Error: {e}"

def execute_live_trading():
    logger.info("Starting Live Order Execution Engine...")
    
    # 1. Connect to Binance
    try:
        exchange = get_exchange_connection()
        usdt_balance = get_futures_balance(exchange)
        logger.info(f"Successfully authenticated with Binance. Futures Balance: ${usdt_balance:,.2f}")
    except Exception as e:
        logger.error(f"Binance connection failed: {e}")
        send_push_notification(f"⚠️ **[CRITICAL]** Live Bot failed to connect to Binance: {e}")
        return

    # 2. Check current open positions
    try:
        active_positions = check_active_positions(exchange)
        logger.info(f"Active Positions: {list(active_positions.keys())}")
    except Exception as e:
        logger.error(f"Failed to fetch active positions: {e}")
        return

    # 3. Load pre-trained models
    crypto_ensemble = EnsembleTradingModel()
    if not os.path.exists('crypto_ensemble.joblib'):
        logger.error("Model files missing. Run training first.")
        return
    crypto_ensemble.load('crypto_ensemble.joblib')
    
    # Load Regime-Switched Models
    regime_models = {}
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_file = f'crypto_ensemble_{regime}.joblib'
        if os.path.exists(regime_file):
            try:
                m = EnsembleTradingModel()
                m.load(regime_file)
                regime_models[regime] = m
                logger.info(f"Successfully loaded regime-specific model for: {regime}")
            except Exception as le:
                logger.warning(f"Failed to load regime model for {regime}: {le}")

    # Load RL Agent if enabled
    rl_agent = None
    if getattr(config, 'USE_RL_AGENT', False):
        try:
            from rl_agent import QLearningAgent
            rl_agent = QLearningAgent()
            logger.info("Live Q-Learning RL Agent loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RL Agent: {e}")

    # Load recent trade log to determine state/drawdowns
    csv_path = 'executed_trades.csv'
    trade_history = []
    if os.path.exists(csv_path):
        try:
            trade_history = pd.read_csv(csv_path).to_dict(orient='records')
        except Exception:
            pass

    trades_triggered = 0
    veto_log = []

    # 4. Generate today's signals and run execution for crypto assets
    for ticker, symbol in SYMBOL_MAP.items():
        try:
            import yfinance as yf
            from features import resample_to_4h
            from datetime import datetime, timedelta
            
            # Download 1h data (from last 75 days for speed and safety) and resample to 4h
            start_date = (datetime.now() - timedelta(days=75)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=start_date, interval="1h", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = resample_to_4h(df)
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient 4h data for {ticker}")
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Build indicators and retrieve latest values
            feature_cols = build_features(df, ticker=ticker)
            df_clean = df.dropna(subset=feature_cols + ['SMA_200']).copy()
            if len(df_clean) < 20:
                continue
                
            X_today = df_clean.iloc[[-1]][feature_cols]
            history = df_clean.iloc[-20:]
            latest_close = float(df_clean.iloc[-1]['Close'])
            atr_val = float(df_clean.iloc[-1]['ATR'])
            
            # Check market regime (HMM) first to swap the correct model ensemble
            from regime_filter import MarketRegimeFilter
            regime_filter = MarketRegimeFilter()
            regime_scales = regime_filter.compute_regime_sizing(df_clean)
            regime_scale = float(regime_scales.iloc[-1])
            current_regime = str(df_clean.iloc[-1]['Regime_Label'])
            
            # Predict signal using active regime-specific model
            active_model = regime_models.get(current_regime, crypto_ensemble)
            sig, probs = active_model.predict_signals(X_today, history=history)
            sig_val = sig[0]
            prob_val = probs[0]
            
            logger.info(f"{ticker} -> Signal: {sig_val}, Prob: {prob_val:.2%}, Regime: {current_regime} (Scale: {regime_scale})")

            # 5. Position Management (Exit logic or Breakeven updates)
            if symbol in active_positions:
                pos = active_positions[symbol]
                
                # Check if model recommends exit
                model_recommends_exit = (pos['side'] == 'long' and sig_val == -1) or (pos['side'] == 'short' and sig_val == 1)
                
                # Check if HMM commands crisis halt
                crisis_halt = (regime_scale == 0.0)
                
                if model_recommends_exit or crisis_halt:
                    # Close position immediately
                    logger.info(f"Closing position on {symbol} due to {'Model Signal reversal' if model_recommends_exit else 'Crisis regime scale'}.")
                    close_side = 'sell' if pos['side'] == 'long' else 'buy'
                    
                    # Cancel all stop/take-profit orders first
                    cancel_all_orders(exchange, symbol)
                    
                    # Place market close order
                    close_order = exchange.create_market_order(
                        symbol=symbol,
                        side=close_side,
                        amount=pos['size'],
                        params={'reduceOnly': True}
                    )
                    
                    exit_price = float(close_order.get('price', latest_close))
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] if pos['side'] == 'long' else (pos['entry_price'] - exit_price) / pos['entry_price']
                    pnl_usd = pnl_pct * pos['entry_price'] * pos['size']
                    
                    # Log trade
                    new_trade = {
                        'Ticker': ticker,
                        'Direction': 'Long' if pos['side'] == 'long' else 'Short',
                        'EntryTime': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'ExitTime': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'EntryPrice': pos['entry_price'],
                        'ExitPrice': exit_price,
                        'PnL_Pct': pnl_pct,
                        'PnL_USD': pnl_usd,
                        'ExitReason': 'Signal_Reversal' if model_recommends_exit else 'Crisis_Regime_Halt'
                    }
                    
                    # Update local database
                    trade_history.append(new_trade)
                    pd.DataFrame(trade_history).to_csv(csv_path, index=False)
                    
                    if rl_agent is not None:
                        rl_agent.update(pnl_pct)
                    
                    # AI review Commentary
                    ai_opinion = generate_ai_commentary(recent_trades=[new_trade], is_live=True, regime=current_regime)
                    
                    send_push_notification(
                        f"🔴 **[EXIT]** Closed {pos['side'].upper()} on **{ticker}** at {exit_price:.2f}.\n"
                        f"• PnL: **{pnl_pct:+.2%}** (${pnl_usd:+.2f})\n"
                        f"🤖 **AI Analyst Review**: *{ai_opinion}*"
                    )
                else:
                    # Update Trailing / Breakeven rules
                    if getattr(config, 'ENABLE_BREAKEVEN', False) or getattr(config, 'ENABLE_TRAILING_TP', False):
                        try:
                            # 1. Fetch current open orders for this symbol
                            open_orders = exchange.fetch_open_orders(symbol)
                            sl_order = None
                            for order in open_orders:
                                order_type = order.get('type', '').lower()
                                if order_type in ['stop_market', 'stop', 'stop_limit'] and order.get('side') == ('sell' if pos['side'] == 'long' else 'buy'):
                                    sl_order = order
                                    break
                            
                            current_sl = None
                            if sl_order:
                                current_sl = float(sl_order.get('stopPrice', sl_order.get('params', {}).get('stopPrice', 0)))
                                if not current_sl and 'info' in sl_order:
                                    current_sl = float(sl_order['info'].get('stopPrice', 0))
                            
                            # 2. Determine new SL price
                            new_sl = None
                            is_trailing = False
                            
                            if pos['side'] == 'long':
                                # Check Trailing TP activation
                                trail_activation = pos['entry_price'] + (config.TRAILING_TP_ACTIVATION_ATR_MULT * atr_val)
                                breakeven_activation = pos['entry_price'] + (0.8 * atr_val)
                                
                                if latest_close >= trail_activation and getattr(config, 'ENABLE_TRAILING_TP', False):
                                    trail_price = latest_close - (config.TRAILING_TP_CALLBACK_ATR_MULT * atr_val)
                                    # Trailing stop only moves up
                                    if current_sl is None or trail_price > current_sl:
                                        new_sl = trail_price
                                        is_trailing = True
                                elif latest_close >= breakeven_activation and getattr(config, 'ENABLE_BREAKEVEN', False):
                                    if current_sl is None or pos['entry_price'] > current_sl:
                                        new_sl = pos['entry_price']
                                        
                            else: # Short position
                                trail_activation = pos['entry_price'] - (config.TRAILING_TP_ACTIVATION_ATR_MULT * atr_val)
                                breakeven_activation = pos['entry_price'] - (0.8 * atr_val)
                                
                                if latest_close <= trail_activation and getattr(config, 'ENABLE_TRAILING_TP', False):
                                    trail_price = latest_close + (config.TRAILING_TP_CALLBACK_ATR_MULT * atr_val)
                                    # Trailing stop only moves down
                                    if current_sl is None or trail_price < current_sl:
                                        new_sl = trail_price
                                        is_trailing = True
                                elif latest_close <= breakeven_activation and getattr(config, 'ENABLE_BREAKEVEN', False):
                                    if current_sl is None or pos['entry_price'] < current_sl:
                                        new_sl = pos['entry_price']
                                        
                            # 2b. Fail-Safe: If no stop-loss order exists on the exchange, recreate the initial SL
                            if sl_order is None and new_sl is None:
                                if pos['side'] == 'long':
                                    initial_sl = pos['entry_price'] - (config.SL_ATR_MULT_LONG * atr_val)
                                    if latest_close <= initial_sl:
                                        new_sl = latest_close * 0.995
                                    else:
                                        new_sl = initial_sl
                                else:
                                    initial_sl = pos['entry_price'] + (config.SL_ATR_MULT_SHORT * atr_val)
                                    if latest_close >= initial_sl:
                                        new_sl = latest_close * 1.005
                                    else:
                                        new_sl = initial_sl
                                logger.warning(f"Fail-Safe: Missing Stop-Loss detected for {symbol}. Recreating at {new_sl:.6f}")
                                        
                            # 3. If new SL is determined and differs from current, update it
                            if new_sl is not None and (current_sl is None or abs(new_sl - current_sl) > 0.00001):
                                # Cancel old SL order
                                if sl_order:
                                    try:
                                        exchange.cancel_order(sl_order['id'], symbol)
                                    except Exception as ce:
                                        logger.warning(f"Failed to cancel old SL order {sl_order['id']} for {symbol}: {ce}")
                                
                                # Create new SL order
                                sl_side = 'sell' if pos['side'] == 'long' else 'buy'
                                exchange.create_order(
                                    symbol=symbol,
                                    type='STOP_MARKET',
                                    side=sl_side,
                                    amount=pos['size'],
                                    price=None,
                                    params={
                                        'stopPrice': new_sl,
                                        'reduceOnly': True
                                    }
                                )
                                logger.info(f"Updated SL for {symbol} to {new_sl:.5f} ({'Trailing' if is_trailing else 'Breakeven'})")
                                send_push_notification(
                                    f"🔄 **[UPDATE]** Moved Stop-Loss for **{ticker}** to **${new_sl:.5f}**\n"
                                    f"• Mode: *{'Trailing Profit Lock' if is_trailing else 'Breakeven Shield'}* (Price: ${latest_close:.5f})"
                                )
                        except Exception as ex:
                            logger.error(f"Failed to update trailing/breakeven stop-loss for {symbol}: {ex}")
                    
            # 6. New Entry Logic
            else:
                # Ensure entry is allowed (no crisis regime, and signal fits confidence trigger)
                allow_entry = (regime_scale > 0.0)
                
                # Weekend Liquidity Filter (Friday 20:00 UTC to Sunday 22:00 UTC)
                now_utc = datetime.utcnow()
                weekday = now_utc.weekday()
                hour = now_utc.hour
                is_weekend = False
                if (weekday == 4 and hour >= 20) or (weekday == 5) or (weekday == 6 and hour < 22):
                    is_weekend = True
                
                if is_weekend:
                    logger.info("Weekend Liquidity Filter: Blocking new entries to avoid low-volume range chop.")
                    allow_entry = False
                
                # Dynamic Threshold Selection based on HMM Regime
                # Adapts entry requirements dynamically to balance compounding speed and capital protection
                thresh_long = getattr(config, 'CONFIDENCE_THRESHOLD_LONG', 0.45)
                thresh_short = getattr(config, 'CONFIDENCE_THRESHOLD_SHORT', 0.33)
                
                if current_regime == 'Bull':
                    thresh_long = 0.41  # Aggressive longs in bull trend
                    thresh_short = 0.20 # High-conviction safety filter for shorts in bull trend
                elif current_regime == 'Bear':
                    thresh_long = 0.47  # High-conviction safety filter for longs in bear trend
                    thresh_short = 0.28 # Adaptive shorts in bear trend
                else: # 'Sideways' or choppy regimes
                    thresh_long = 0.45  # Strict longs to filter chop
                    thresh_short = 0.25 # Strict shorts to filter chop
                
                # Check confidence triggers
                is_long_triggered = (sig_val == 1 and prob_val >= thresh_long)
                is_short_triggered = (sig_val == -1 and prob_val <= thresh_short)
                
                # Regime-Adaptive Correlation Cap
                # In Sideways chop, limit portfolio to max 1 Long and 1 Short across all crypto assets
                if allow_entry and (is_long_triggered or is_short_triggered) and current_regime == 'Sideways':
                    target_direction = 'long' if is_long_triggered else 'short'
                    existing_direction_count = sum(
                        1 for pos_sym, pos_val in active_positions.items() 
                        if pos_val['side'] == target_direction
                    )
                    if existing_direction_count >= 1:
                        logger.info(f"Correlation Cap: VETOED entry on {symbol} — a {target_direction} position is already open during Sideways chop.")
                        veto_log.append(f"• **{ticker}** vetoed: Correlation Cap ({target_direction} already open)")
                        allow_entry = False
                
                if not allow_entry and (is_long_triggered or is_short_triggered):
                    if not is_weekend and not (current_regime == 'Sideways' and existing_direction_count >= 1):
                        logger.info(f"Regime Block: VETOED entry on {symbol} due to Crisis regime halt.")
                        veto_log.append(f"• **{ticker}** vetoed: Crisis Market Regime halt")
                
                if allow_entry and (is_long_triggered or is_short_triggered):
                    # 1. Check Strict Trend Lock (Longs only above SMA, Shorts only below SMA)
                    if getattr(config, 'STRICT_TREND_LOCK', False):
                        sma200 = float(df_clean.iloc[-1]['SMA_200'])
                        if is_long_triggered and latest_close < sma200:
                            logger.info(f"Trend Lock: VETOED LONG on {symbol} — price ({latest_close:.2f}) is below {config.SMA_TREND_WINDOW} SMA ({sma200:.2f}).")
                            veto_log.append(f"• **{ticker}** long vetoed: Price (${latest_close:.2f}) below {config.SMA_TREND_WINDOW} SMA (${sma200:.2f})")
                            continue
                        if is_short_triggered and latest_close > sma200:
                            logger.info(f"Trend Lock: VETOED SHORT on {symbol} — price ({latest_close:.2f}) is above {config.SMA_TREND_WINDOW} SMA ({sma200:.2f}).")
                            veto_log.append(f"• **{ticker}** short vetoed: Price (${latest_close:.2f}) above {config.SMA_TREND_WINDOW} SMA (${sma200:.2f})")
                            continue
                            
                    # 2. Check Extreme Fear Block (No shorting if F&G index < 25)
                    if getattr(config, 'EXTREME_FEAR_BLOCK', False) and is_short_triggered:
                        fng_score = float(df_clean.iloc[-1]['Sentiment_Score'])
                        if fng_score < getattr(config, 'FEAR_LIMIT', 25):
                            logger.info(f"Fear Block: VETOED SHORT on {symbol} — Sentiment Index ({fng_score:.1f}) is in Extreme Fear (< {getattr(config, 'FEAR_LIMIT', 25)}).")
                            veto_log.append(f"• **{ticker}** short vetoed: Extreme Fear sentiment (${fng_score:.1f})")
                            continue

                    # 3. Check RL Agent Veto
                    if rl_agent is not None:
                        confirmed = rl_agent.should_take_action(sig_val, current_regime, prob_val)
                        if not confirmed:
                            logger.info(f"RL Agent: VETOED entry signal on {symbol} due to poor Q-value regime profile.")
                            veto_log.append(f"• **{ticker}** vetoed: Q-Learning Agent veto")
                            continue
                            
                    logger.info(f"Triggering entry for {symbol} ({'LONG' if is_long_triggered else 'SHORT'})...")
                    
                    # Fetch balance and calculate size
                    usdt_balance = get_futures_balance(exchange)
                    if usdt_balance < 5.0:
                        logger.warning("Futures account balance too low to trade.")
                        continue
                        
                    # Calculate Kelly-optimal allocation fraction
                    try:
                        from kelly import KellySizer
                        sizer = KellySizer()
                        direction_val = 1 if is_long_triggered else -1
                        kelly_fraction = sizer.compute(prob_val, direction_val)
                        logger.info(f"Kelly Sizer: Optimal allocation fraction for {symbol} is {kelly_fraction:.2%}")
                    except Exception as ke:
                        logger.warning(f"Failed to calculate Kelly allocation for {symbol}: {ke}. Falling back to default.")
                        kelly_fraction = getattr(config, 'MAX_ALLOCATION_PER_TRADE', 0.20)
                    
                    # Calculate margin cash allocated based on Kelly
                    margin_allocated = usdt_balance * kelly_fraction
                    margin_allocated = max(2.50, margin_allocated) # Minimum $2.50 floor for sandbox
                    
                    # Sizing scale based on HMM
                    margin_allocated = margin_allocated * regime_scale
                    
                    # Apply News Sentiment Sizing
                    sentiment_status = "Not Checked"
                    if getattr(config, 'ENABLE_SENTIMENT_SIZING', False):
                        sent_multiplier, avg_sent, sentiment_status = get_news_sentiment_sizing_multiplier(ticker, sig_val)
                        margin_allocated = margin_allocated * sent_multiplier
                        
                    # Calculate Dynamic Leverage based on ATR volatility percentile
                    target_leverage = getattr(config, 'LEVERAGE', 20)
                    vol_status = "Neutral"
                    if getattr(config, 'ENABLE_DYNAMIC_LEVERAGE', False) and len(df_clean) >= 20:
                        try:
                            atr_pcts = (df_clean['ATR'] / df_clean['Close']).iloc[-100:]
                            low_vol_limit = float(atr_pcts.quantile(0.25))
                            high_vol_limit = float(atr_pcts.quantile(0.75))
                            latest_atr_pct = atr_val / latest_close
                            
                            if latest_atr_pct <= low_vol_limit:
                                target_leverage = getattr(config, 'LEVERAGE_VOL_LOW', 25)
                                vol_status = "Low Volatility Squeeze"
                            elif latest_atr_pct >= high_vol_limit:
                                target_leverage = getattr(config, 'LEVERAGE_VOL_HIGH', 10)
                                vol_status = "High Volatility Panic"
                        except Exception as ve:
                            logger.warning(f"Failed to calculate dynamic leverage metrics for {symbol}: {ve}")
                            
                    logger.info(f"Dynamic Sizing Summary for {symbol} -> Leverage: {target_leverage}x ({vol_status}), News Sizing Status: {sentiment_status}")
                    
                    # Total position value = Margin * Leverage
                    position_value = margin_allocated * target_leverage
                    
                    # Calculate units
                    units = position_value / latest_close
                    
                    # Set exchange leverage and isolated margin mode
                    set_leverage_and_margin(exchange, symbol, leverage=target_leverage)
                    
                    # Execute entry market order
                    order_side = 'buy' if is_long_triggered else 'sell'
                    entry_order = exchange.create_market_order(
                        symbol=symbol,
                        side=order_side,
                        amount=units
                    )
                    
                    entry_price = entry_order.get('price')
                    if entry_price is None:
                        entry_price = latest_close
                    entry_price = float(entry_price)
                    logger.info(f"Entered trade on {symbol} at price {entry_price:.2f}.")
                    trades_triggered += 1
                    
                    # Calculate Stop-Loss and Take-Profit price levels
                    if is_long_triggered:
                        sl_price = entry_price - (config.SL_ATR_MULT_LONG * atr_val)
                        tp_price = entry_price + (config.TP_ATR_MULT * atr_val)
                    else:
                        sl_price = entry_price + (config.SL_ATR_MULT_SHORT * atr_val)
                        tp_price = entry_price - (config.TP_ATR_MULT * atr_val)
                        
                    # Wait 1 second to prevent Binance API race condition on new positions
                    time.sleep(1.0)
                    # Place Stop-Loss and Take-Profit orders on Binance (Reduce-Only)
                    # We cancel any stray orders first
                    cancel_all_orders(exchange, symbol)
                    
                    # Stop Loss Order
                    sl_side = 'sell' if is_long_triggered else 'buy'
                    exchange.create_order(
                        symbol=symbol,
                        type='STOP_MARKET',
                        side=sl_side,
                        amount=units,
                        price=None,
                        params={
                            'stopPrice': sl_price,
                            'reduceOnly': True
                        }
                    )
                    
                    # Take Profit Order
                    exchange.create_order(
                        symbol=symbol,
                        type='TAKE_PROFIT_MARKET',
                        side=sl_side,
                        amount=units,
                        price=None,
                        params={
                            'stopPrice': tp_price,
                            'reduceOnly': True
                        }
                    )
                    
                    logger.info(f"Configured TP/SL orders for {symbol} (SL: {sl_price:.2f}, TP: {tp_price:.2f}).")
                    
                    # Send Discord entry notification
                    send_push_notification(
                        f"🟢 **[ENTRY]** Opened {'LONG' if is_long_triggered else 'SHORT'} on **{ticker}**\n"
                        f"• Entry Price: `${entry_price:,.2f}`\n"
                        f"• Position Size: `${position_value:,.2f}` (Margin: `${margin_allocated:,.2f}` @ 10x leverage)\n"
                        f"• Protection Set: Stop-Loss at `{sl_price:.2f}` | Take-Profit at `{tp_price:.2f}`"
                    )
                    
        except Exception as e:
            logger.error(f"Error executing live trade cycle for {ticker}: {e}")
            
    # Update live dashboard files and push to GitHub
    try:
        update_live_dashboard(usdt_balance)
        push_to_github()
    except Exception as e:
        logger.error(f"Failed to auto-deploy live dashboard: {e}")

    # Send daily status summary to Discord
    try:
        current_active = {}
        pos_msg = ""
        try:
            current_active = check_active_positions(exchange)
            if current_active:
                pos_msg = "💼 **Active Positions Details**:\n"
                for sym, p in current_active.items():
                    pnl_usd = p['unrealized_pnl']
                    side_str = p['side'].upper()
                    display_sym = sym.split('/')[0]
                    pos_msg += f"• **{display_sym}** {side_str} | Entry: ${p['entry_price']:.5f} | PnL: **{pnl_usd:+.2f} USDT**\n"
                pos_msg += "\n"
        except Exception as e:
            logger.warning(f"Failed to fetch active positions for Discord report: {e}")

        # Calculate weekly PnL and formulate acceleration tip
        weekly_pnl = calculate_weekly_pnl(trade_history)
        acceleration_msg = ""
        if weekly_pnl > 0.0:
            suggested_deposit = max(20, int(round(weekly_pnl * 3.0, -1)))
            acceleration_msg = (
                f"🚀 **Roadmap Acceleration Tip**:\n"
                f"• Weekly Profit: `+${weekly_pnl:,.2f} USDT`!\n"
                f"• The bot is on a winning streak. Consider depositing **${suggested_deposit} USDT** to speed up compounding!\n\n"
            )

        reasons_msg = ""
        if trades_triggered > 0:
            reasons_msg = "• Status: Trades executed successfully."
        elif veto_log:
            reasons_msg = f"🔍 **Scan Details & Veto Log**:\n" + "\n".join(veto_log)
        else:
            reasons_msg = "🔍 **Scan Details**: No high-conviction AI signals triggered today (all assets below 45% threshold)."

        summary_msg = (
            f"📊 **Daily Crypto Scan Complete**\n"
            f"• Futures Balance: `${usdt_balance:,.2f}`\n"
            f"• Active Positions: `{len(current_active)}`\n"
            f"• Trades Triggered Today: `{trades_triggered}`\n\n"
            f"{pos_msg}"
            f"{acceleration_msg}"
            f"{reasons_msg}\n\n"
            f"• Status: Active & Monitoring"
        )
        send_push_notification(summary_msg)
        logger.info("Daily summary sent to Discord.")
    except Exception as e:
        logger.error(f"Failed to send daily summary: {e}")

def update_live_dashboard(usdt_balance):
    """
    Updates dashboard files (portfolio_state.js, portfolio_performance.png, report.html)
    based on the live trading history logged in executed_trades.csv.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import json
        
        csv_path = 'executed_trades.csv'
        trades_list = []
        if os.path.exists(csv_path):
            try:
                df_trades = pd.read_csv(csv_path)
                trades_list = df_trades.to_dict(orient='records')
            except Exception:
                pass
                
        # Calculate performance statistics
        initial_capital = 32.33
        final_value = usdt_balance
        return_pct = (final_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        
        # Build equity curve series from trade history
        dates = ['2026-07-01']
        values = [initial_capital]
        
        running_bal = initial_capital
        for tr in trades_list:
            running_bal += float(tr.get('PnL_USD', 0.0))
            dates.append(str(tr.get('ExitTime', datetime.now().strftime("%Y-%m-%d")))[:10])
            values.append(running_bal)
            
        # Draw and save equity curve chart
        plt.figure(figsize=(10, 5))
        plt.plot(dates, values, marker='o', color='#14b8a6', linewidth=2, label='Live Portfolio')
        plt.axhline(initial_capital, color='#4b5563', linestyle='--', alpha=0.5, label='Initial Capital ($32.33)')
        plt.title('Live Bot Portfolio Performance ($)', color='white')
        plt.xlabel('Date', color='white')
        plt.ylabel('Balance (USDT)', color='white')
        plt.grid(True, color='#1f2937', alpha=0.5)
        plt.legend()
        
        # Style chart for dark mode dashboard
        fig = plt.gcf()
        fig.patch.set_facecolor('#0f172a')
        ax = plt.gca()
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        
        plt.savefig('portfolio_performance.png', facecolor='#0f172a', bbox_inches='tight')
        plt.close()
        
        # Generate portfolio_state.js data
        equity_curve_data = [{"date": d, "value": v} for d, v in zip(dates, values)]
        
        # Win Rate
        wins = sum(1 for tr in trades_list if float(tr.get('PnL_USD', 0.0)) > 0)
        total = len(trades_list)
        win_rate = (wins / total) if total > 0 else 0.0
        
        portfolio_state = {
            "ticker_list": config.TICKERS,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "return_pct": return_pct,
            "benchmark_return_pct": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "trades_count": total,
            "win_rate": win_rate,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "circuit_breaker_tripped": False,
            "circuit_breaker_date": "",
            "equity_curve": equity_curve_data,
            "benchmark_curve": [],
            "trades": trades_list
        }
        
        # Write portfolio_state.js
        js_content = f"const PORTFOLIO_STATE = {json.dumps(portfolio_state, indent=2)};"
        with open("portfolio_state.js", "w") as f:
            f.write(js_content)
        logger.info("Live portfolio_state.js written successfully.")
        
    except Exception as e:
        logger.error(f"Failed to update live dashboard files: {e}")

if __name__ == "__main__":
    execute_live_trading()
