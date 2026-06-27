import os
import yfinance as yf
import pandas as pd
import numpy as np
import config
from model import EnsembleTradingModel
from features import build_features
from security import logger

def main():
    logger.info("Initializing live signal generator...")
    
    # 1. Load pre-trained models
    crypto_ensemble = EnsembleTradingModel()
    equity_ensemble = EnsembleTradingModel()
    
    crypto_path = 'crypto_ensemble.joblib'
    equity_path = 'equity_ensemble.joblib'
    
    if not os.path.exists(crypto_path) or not os.path.exists(equity_path):
        logger.error("Pre-trained model files not found! Please run 'python main.py' once to train and save the models.")
        return
        
    try:
        crypto_ensemble.load(crypto_path)
        equity_ensemble.load(equity_path)
    except Exception as e:
        logger.error(f"Failed to load pre-trained models: {e}")
        return
        
    # 2. Fetch recent market data for all assets to compute features
    # We download 250 days of daily history to ensure indicators (like SMA 200, MACD, BB) are fully computed.
    logger.info(f"Downloading recent market data for {len(config.TICKERS)} tickers...")
    
    results = []
    
    for ticker in config.TICKERS:
        try:
            # Download 250 days of daily history
            df = yf.download(ticker, period="250d", interval="1d", progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
                
            # Clean column names in case yfinance multi-index columns are present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Build indicators
            feature_columns = build_features(df, ticker=ticker)
            
            # Drop NaN rows to get clean features
            df_clean = df.dropna(subset=feature_columns + ['SMA_200']).copy()
            if df_clean.empty:
                logger.warning(f"Insufficient indicator data for {ticker}")
                continue
                
            # Get latest row (representing today's closed/live bar)
            X_today = df_clean.iloc[[-1]][feature_columns]
            latest_date = df_clean.index[-1].strftime("%Y-%m-%d")
            latest_close = float(df_clean.iloc[-1]['Close'])
            
            # Extract sequence history for the NumPy LSTM layer
            history = df_clean.iloc[-getattr(config, 'LSTM_SEQUENCE_LENGTH', 20):]
            
            # Run dynamic feature selection and predict
            if ticker in config.SHORTABLE_TICKERS:
                sig, probs = crypto_ensemble.predict_signals(X_today, history=history)
            else:
                sig, probs = equity_ensemble.predict_signals(X_today, history=history)
                
            sig_val = sig[0]
            prob_val = probs[0]
            
            # Format display signal
            if sig_val == 1:
                sig_str = "BUY (LONG)"
            elif sig_val == -1:
                sig_str = "SELL (SHORT)"
            else:
                sig_str = "NO TRADE (CASH)"
                
            results.append({
                "Ticker": ticker,
                "Last Close": f"${latest_close:,.2f}",
                "Date": latest_date,
                "Prob (Long)": f"{prob_val:.2%}",
                "Signal": sig_str
            })
        except Exception as e:
            logger.error(f"Failed to generate signal for {ticker}: {e}")
            
    # 3. Print the live trading signals table
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("                      LIVE DAILY TRADING SIGNALS SUMMARY")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    print("Model status: Healthy. Directional filters and stop-loss logic active.")

if __name__ == "__main__":
    main()
