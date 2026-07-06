import os
import yfinance as yf
import pandas as pd
import numpy as np
import config
from model import EnsembleTradingModel
from features import build_features, calculate_triple_barrier_labels, resample_to_4h
from security import logger
from datetime import datetime, timedelta

def main():
    logger.info("Starting one-time model training script...")
    
    # 1. Initialize models
    crypto_ensemble = EnsembleTradingModel()
    equity_ensemble = EnsembleTradingModel()
    
    # 2. Fetch historical data for all assets
    logger.info("Downloading historical market data to build training sets...")
    
    processed_dfs = {}
    feature_cols = []
    
    # If config.INTERVAL is "4h", we download 1h data from the last 725 days and resample
    if config.INTERVAL == "4h":
        start_date = (datetime.now() - timedelta(days=725)).strftime('%Y-%m-%d')
        download_interval = "1h"
    else:
        start_date = config.START_DATE
        download_interval = config.INTERVAL
        
    logger.info(f"Using download parameters: start={start_date}, interval={download_interval}")
    
    for ticker in config.TICKERS:
        try:
            df = yf.download(ticker, start=start_date, end=config.END_DATE, interval=download_interval, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
                
            # Clean column names in case yfinance multi-index columns are present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Resample 1h to 4h if needed
            if config.INTERVAL == "4h":
                df = resample_to_4h(df)
                if df.empty or len(df) < 50:
                    logger.warning(f"Insufficient resampled 4h data for {ticker}")
                    continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
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
            
            # Calculate HMM regimes for switching logic
            try:
                from hmm_regime import HMMRegimeClassifier
                classifier = HMMRegimeClassifier()
                classifier.fit(df)
                df['Regime_Label'] = classifier.predict_regime(df)
            except Exception as hmm_err:
                logger.warning(f"Failed to fit HMM on {ticker}: {hmm_err}")
                df['Regime_Label'] = 'Bull'
            
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
            
    # 4. Train the Crypto Ensemble (Baseline & Regime-Switched Ensembles)
    if train_features_crypto:
        X_crypto = pd.concat(train_features_crypto, axis=0)
        y_crypto = pd.concat(train_targets_crypto, axis=0)
        regimes_crypto = pd.concat([df['Regime_Label'] for tick, df in processed_dfs.items() if tick in config.SHORTABLE_TICKERS], axis=0)
        
        # Save baseline model
        logger.info(f"Training Baseline Crypto Ensemble on {len(X_crypto)} samples...")
        crypto_ensemble.fit(X_crypto, y_crypto)
        crypto_ensemble.fit_meta_model(X_crypto, y_crypto)
        crypto_ensemble.save('crypto_ensemble.joblib')
        
        # Train regime-specific models
        for regime in ['Bull', 'Bear', 'Sideways']:
            mask = regimes_crypto == regime
            if mask.sum() >= 100:
                X_reg = X_crypto[mask]
                y_reg = y_crypto[mask]
                logger.info(f"Training {regime} Crypto Ensemble on {len(X_reg)} samples...")
                
                m = EnsembleTradingModel()
                m.fit(X_reg, y_reg)
                m.fit_meta_model(X_reg, y_reg)
                m.save(f'crypto_ensemble_{regime}.joblib')
            else:
                logger.warning(f"Insufficient samples ({mask.sum()}) to train {regime} Crypto Ensemble. Copying baseline.")
                import shutil
                if os.path.exists('crypto_ensemble.joblib'):
                    shutil.copy('crypto_ensemble.joblib', f'crypto_ensemble_{regime}.joblib')
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
        
    logger.info("One-time training complete! Regime-switched and sector models are ready on disk.")

if __name__ == "__main__":
    main()
