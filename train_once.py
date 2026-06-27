import os
import yfinance as yf
import pandas as pd
import numpy as np
import config
from model import EnsembleTradingModel
from features import build_features, calculate_triple_barrier_labels
from security import logger

def main():
    logger.info("Starting one-time model training script...")
    
    # 1. Initialize models
    crypto_ensemble = EnsembleTradingModel()
    equity_ensemble = EnsembleTradingModel()
    
    # 2. Fetch historical data for all assets
    logger.info("Downloading historical market data to build training sets...")
    
    processed_dfs = {}
    feature_cols = []
    
    for ticker in config.TICKERS:
        try:
            df = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
                
            # Clean column names in case yfinance multi-index columns are present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Calculate features
            feature_cols = build_features(df, ticker=ticker)
            
            # Compute Triple Barrier Method labels (Targets)
            df['Target'] = calculate_triple_barrier_labels(
                df, 
                horizon=config.FORECAST_HORIZON, 
                tp_mult=config.TP_ATR_MULT, 
                sl_mult=config.SL_ATR_MULT_LONG
            )
            
            # Calculate 4h_Bullish placeholder (needed for signal execution)
            df['4h_Bullish'] = 1.0 # Default value
            
            # Drop NaN rows
            df_clean = df.dropna(subset=feature_cols + ['Target', 'SMA_200']).copy()
            processed_dfs[ticker] = df_clean
            
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            
    if not processed_dfs:
        logger.error("No historical data processed successfully. Exiting.")
        return
        
    # 3. Pool data separately into Crypto/High-Beta vs traditional Equity/Commodity
    train_features_crypto = []
    train_targets_crypto = []
    train_features_equity = []
    train_targets_equity = []
    
    for ticker, df in processed_dfs.items():
        if ticker in config.SHORTABLE_TICKERS:
            train_features_crypto.append(df[feature_cols])
            train_targets_crypto.append(df['Target'])
        else:
            train_features_equity.append(df[feature_cols])
            train_targets_equity.append(df['Target'])
            
    # 4. Train the Crypto Ensemble
    if train_features_crypto:
        X_crypto = pd.concat(train_features_crypto, axis=0)
        y_crypto = pd.concat(train_targets_crypto, axis=0)
        logger.info(f"Training Crypto Ensemble on {len(X_crypto)} samples...")
        
        crypto_ensemble.fit(X_crypto, y_crypto)
        # Train meta-labeler
        crypto_ensemble.fit_meta_model(X_crypto, y_crypto)
        
        # Save model
        crypto_ensemble.save('crypto_ensemble.joblib')
    else:
        logger.warning("No crypto training data found.")
        
    # 5. Train the Equity Ensemble
    if train_features_equity:
        X_equity = pd.concat(train_features_equity, axis=0)
        y_equity = pd.concat(train_targets_equity, axis=0)
        logger.info(f"Training Equity Ensemble on {len(X_equity)} samples...")
        
        equity_ensemble.fit(X_equity, y_equity)
        # Train meta-labeler
        equity_ensemble.fit_meta_model(X_equity, y_equity)
        
        # Save model
        equity_ensemble.save('equity_ensemble.joblib')
    else:
        logger.warning("No equity training data found.")
        
    logger.info("One-time training complete! Specialized sector models are ready on disk.")

if __name__ == "__main__":
    main()
