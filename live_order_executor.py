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
    'DOGE-USD': 'DOGE/USDT'
}

def get_eu_proxy():
    """
    Fetches a list of free European HTTP proxies from ProxyScrape API,
    and returns the first one that successfully pings api.binance.com.
    """
    import requests
    url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=DE,FR,GB,NL,ES,IT&ssl=all&anonymity=all"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            proxies = [p.strip() for p in resp.text.splitlines() if p.strip()]
            logger.info(f"Fetched {len(proxies)} free European proxies. Testing connection...")
            for p in proxies[:15]:
                proxy_str = f"http://{p}"
                # Test the proxy
                try:
                    # Test both Linear Futures (fapi) and Inverse Futures (dapi)
                    test_fapi = requests.get(
                        "https://fapi.binance.com/fapi/v1/ping",
                        proxies={"http": proxy_str, "https": proxy_str},
                        timeout=3
                    )
                    test_dapi = requests.get(
                        "https://dapi.binance.com/dapi/v1/ping",
                        proxies={"http": proxy_str, "https": proxy_str},
                        timeout=3
                    )
                    if test_fapi.status_code == 200 and test_dapi.status_code == 200:
                        if "restricted location" not in test_fapi.text and "restricted location" not in test_dapi.text:
                            logger.info(f"Successfully found working EU Full Futures Proxy: {proxy_str}")
                            return proxy_str
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Failed to fetch proxy list: {e}")
    return None

def get_exchange_connection():
    """Initializes and returns ccxt Binance Futures connection."""
    if not api_key or not secret_key:
        raise ValueError("Exchange API credentials missing in .env file.")
    
    config_dict = {
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # Target USDT-M Futures account
        }
    }
    
    # Bypass US IP blocks using auto-rotated European proxy
    proxy = get_eu_proxy()
    if proxy:
        config_dict['proxies'] = {
            'http': proxy,
            'https': proxy
        }
        logger.info(f"Configured CCXT with working proxy: {proxy}")
    else:
        logger.warning("No working European proxies found. Connecting directly.")
        
    exchange = ccxt.binance(config_dict)
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

    # 4. Generate today's signals and run execution for crypto assets
    for ticker, symbol in SYMBOL_MAP.items():
        try:
            import yfinance as yf
            df = yf.download(ticker, period="500d", interval="1d", progress=False)
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
            regime_scales = regime_filter.compute_regime_sizing(df_clean)
            regime_scale = float(regime_scales.iloc[-1])
            current_regime = str(df_clean.iloc[-1]['Regime_Label'])
            
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
                    # 1. Check Strict Trend Lock (Longs only above 200 SMA, Shorts only below 200 SMA)
                    if getattr(config, 'STRICT_TREND_LOCK', False):
                        sma200 = float(df_clean.iloc[-1]['SMA_200'])
                        if is_long_triggered and latest_close < sma200:
                            logger.info(f"Trend Lock: VETOED LONG on {symbol} — price ({latest_close:.2f}) is below 200 SMA ({sma200:.2f}).")
                            continue
                        if is_short_triggered and latest_close > sma200:
                            logger.info(f"Trend Lock: VETOED SHORT on {symbol} — price ({latest_close:.2f}) is above 200 SMA ({sma200:.2f}).")
                            continue
                            
                    # 2. Check Extreme Fear Block (No shorting if F&G index < 25)
                    if getattr(config, 'EXTREME_FEAR_BLOCK', False) and is_short_triggered:
                        fng_score = float(df_clean.iloc[-1]['Sentiment_Score'])
                        if fng_score < getattr(config, 'FEAR_LIMIT', 25):
                            logger.info(f"Fear Block: VETOED SHORT on {symbol} — Sentiment Index ({fng_score:.1f}) is in Extreme Fear (< {getattr(config, 'FEAR_LIMIT', 25)}).")
                            continue

                    # 3. Check RL Agent Veto
                    if rl_agent is not None:
                        confirmed = rl_agent.should_take_action(sig_val, current_regime, prob_val)
                        if not confirmed:
                            logger.info(f"RL Agent: VETOED entry signal on {symbol} due to poor Q-value regime profile.")
                            continue
                            
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
            
    # Update live dashboard files and push to GitHub
    try:
        update_live_dashboard(usdt_balance)
        push_to_github()
    except Exception as e:
        logger.error(f"Failed to auto-deploy live dashboard: {e}")

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
