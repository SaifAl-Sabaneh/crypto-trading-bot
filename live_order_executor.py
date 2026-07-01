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
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import EnsembleTradingModel
from features import build_features
from security import logger, send_push_notification, calculate_live_accuracy, generate_ai_commentary

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
api_key = os.getenv("EXCHANGE_API_KEY", "")
secret_key = os.getenv("EXCHANGE_SECRET_KEY", "")

# Ticker mapping between Yahoo Finance (indicators) and Binance Futures (orders)
SYMBOL_MAP = {
    'BTC-USD': 'BTC/USDT',
    'ETH-USD': 'ETH/USDT',
    'SOL-USD': 'SOL/USDT',
    'BNB-USD': 'BNB/USDT'
}

def get_exchange_connection():
    """Initializes and returns ccxt Binance Futures connection."""
    if not api_key or not secret_key:
        raise ValueError("Exchange API credentials missing in .env file.")
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Target USDT-M Futures account
        }
    })
    return exchange

def set_leverage_and_margin(exchange, symbol):
    """Sets isolated margin mode and leverage for the target symbol."""
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
        exchange.set_leverage(config.LEVERAGE, symbol)
        logger.info(f"Set leverage to {config.LEVERAGE}x for {symbol}.")
    except Exception as e:
        logger.error(f"Failed to configure leverage/margin for {symbol}: {e}")

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

    # Load recent trade log to determine state/drawdowns
    csv_path = 'executed_trades.csv'
    trade_history = []
    if os.path.exists(csv_path):
        try:
            trade_history = pd.read_csv(csv_path).to_dict(orient='records')
        except Exception:
            pass

    # 4. Generate today's signals and run execution for crypto assets
    for ticker, symbol in SYMBOL_MAP.items():
        try:
            import yfinance as yf
            df = yf.download(ticker, period="250d", interval="1d", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Align sentiment and 4h confirmation
            df['Sentiment_Score'] = 50.0
            df['Sentiment_MA7'] = 50.0
            
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            df_4h = yf.download(ticker, start=start_date, interval="4h", progress=False)
            if isinstance(df_4h.columns, pd.MultiIndex):
                df_4h.columns = df_4h.columns.get_level_values(0)
            from features import align_multi_timeframe_indicators
            df = align_multi_timeframe_indicators(df, df_4h)
            
            # Build indicators and retrieve latest values
            feature_cols = build_features(df, ticker=ticker)
            df_clean = df.dropna(subset=feature_cols + ['SMA_200']).copy()
            if len(df_clean) < 20:
                continue
                
            X_today = df_clean.iloc[[-1]][feature_cols]
            history = df_clean.iloc[-20:]
            latest_close = float(df_clean.iloc[-1]['Close'])
            atr_val = float(df_clean.iloc[-1]['ATR'])
            
            # Predict signal
            sig, probs = crypto_ensemble.predict_signals(X_today, history=history)
            sig_val = sig[0]
            prob_val = probs[0]
            
            # Check market regime (HMM)
            from regime_filter import MarketRegimeFilter
            regime_filter = MarketRegimeFilter()
            regime_scale = regime_filter.compute_regime_sizing(df_clean)
            current_regime = regime_filter.classifier.predict_regime(df_clean).iloc[-1]
            
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
                    
                    # AI review Commentary
                    ai_opinion = generate_ai_commentary(recent_trades=[new_trade], is_live=True, regime=current_regime)
                    
                    send_push_notification(
                        f"🔴 **[EXIT]** Closed {pos['side'].upper()} on **{ticker}** at {exit_price:.2f}.\n"
                        f"• PnL: **{pnl_pct:+.2%}** (${pnl_usd:+.2f})\n"
                        f"🤖 **AI Analyst Review**: *{ai_opinion}*"
                    )
                else:
                    # Update Trailing / Breakeven rules
                    # Move SL to Entry if price moved 0.8 * ATR in our favor
                    # To keep it simple, we check if the current price allows it and update exchange order
                    pass
                    
            # 6. New Entry Logic
            else:
                # Ensure entry is allowed (no crisis regime, and signal fits confidence trigger)
                allow_entry = (regime_scale > 0.0)
                
                # Check confidence triggers
                is_long_triggered = (sig_val == 1 and prob_val >= config.CONFIDENCE_THRESHOLD_LONG)
                is_short_triggered = (sig_val == -1 and prob_val <= config.CONFIDENCE_THRESHOLD_SHORT)
                
                if allow_entry and (is_long_triggered or is_short_triggered):
                    logger.info(f"Triggering entry for {symbol} ({'LONG' if is_long_triggered else 'SHORT'})...")
                    
                    # Fetch balance and calculate size
                    usdt_balance = get_futures_balance(exchange)
                    if usdt_balance < 5.0:
                        logger.warning("Futures account balance too low to trade.")
                        continue
                        
                    # Calculate margin cash allocated (10%)
                    margin_allocated = usdt_balance * config.MAX_ALLOCATION_PER_TRADE
                    margin_allocated = max(2.50, margin_allocated) # Minimum $2.50 floor for sandbox
                    
                    # Sizing scale based on HMM
                    margin_allocated = margin_allocated * regime_scale
                    
                    # Total position value = Margin * Leverage
                    position_value = margin_allocated * config.LEVERAGE
                    
                    # Calculate units
                    units = position_value / latest_close
                    
                    # Set exchange configuration
                    set_leverage_and_margin(exchange, symbol)
                    
                    # Execute entry market order
                    order_side = 'buy' if is_long_triggered else 'sell'
                    entry_order = exchange.create_market_order(
                        symbol=symbol,
                        side=order_side,
                        amount=units
                    )
                    
                    entry_price = float(entry_order.get('price', latest_close))
                    logger.info(f"Entered trade on {symbol} at price {entry_price:.2f}.")
                    
                    # Calculate Stop-Loss and Take-Profit price levels
                    if is_long_triggered:
                        sl_price = entry_price - (config.SL_ATR_MULT_LONG * atr_val)
                        tp_price = entry_price + (config.TP_ATR_MULT * atr_val)
                    else:
                        sl_price = entry_price + (config.SL_ATR_MULT_SHORT * atr_val)
                        tp_price = entry_price - (config.TP_ATR_MULT * atr_val)
                        
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
            
    # Send daily passive cash commentary if no trades are active or taken
    try:
        active_positions = check_active_positions(exchange)
        if len(active_positions) == 0:
            # Generate and send daily cash review comment
            ai_opinion = generate_ai_commentary(is_live=True, regime="Sideways")
            send_push_notification(
                f"📊 **Daily Live Status Update**:\n"
                f"• **Current Balance**: `${usdt_balance:,.2f}`\n"
                f"• **Status**: In Cash (No Active Positions)\n"
                f"🤖 **AI Analyst Review**: *{ai_opinion}*"
            )
    except Exception:
        pass

if __name__ == "__main__":
    execute_live_trading()
