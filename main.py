import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
from security import logger, HealthMonitor, network_retry, safe_atomic_write
from features import build_features, create_labels, fetch_fear_and_greed_data, calculate_triple_barrier_labels
from regime_filter import MarketRegimeFilter
from model import EnsembleTradingModel
from backtester import PortfolioBacktester

@network_retry(retries=3, backoff_factor=2.0)
def fetch_ticker_data(ticker):
    """Downloads historical asset data using yfinance with retry resiliency."""
    logger.info(f"Downloading historical data for {ticker}...")
    df = yf.download(
        tickers=ticker,
        start=config.START_DATE,
        end=config.END_DATE,
        interval=config.INTERVAL,
        progress=False
    )
    if df.empty:
        raise ValueError(f"No price data returned for {ticker} from Yahoo Finance.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Clean volume/prices to make sure we don't have zeros
    df = df[df['Volume'] > 0].copy()
    logger.info(f"Downloaded {len(df)} rows for {ticker}.")
    return df

@network_retry(retries=3, backoff_factor=2.0)
def fetch_ticker_4h_data(ticker):
    """Downloads recent 4-hour asset data using yfinance with retry resiliency."""
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Downloading 4h historical data for {ticker} from {start_date}...")
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="4h",
        progress=False
    )
    if df.empty:
        logger.warning(f"No 4h data returned for {ticker} from Yahoo Finance.")
        return pd.DataFrame()
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[df['Volume'] > 0].copy()
    logger.info(f"Downloaded {len(df)} 4h rows for {ticker}.")
    return df

# Import yfinance inside main wrapper to ensure decorator works
import yfinance as yf

def main():
    # 1. Run Pre-flight Health Checks
    if not HealthMonitor.run_health_checks():
        logger.critical("Pre-flight health checks failed. Halting bot execution.")
        sys.exit(1)
        
    # 1.5 Fetch Fear & Greed Index Sentiment data if configured
    fng_df = pd.DataFrame()
    if config.USE_SENTIMENT:
        try:
            fng_df = fetch_fear_and_greed_data(limit=0)
        except Exception as e:
            logger.error(f"Failed to load Fear & Greed Index: {e}")
        
    # 2. Process all configured assets
    processed_dfs = {}
    feature_columns = None
    
    for ticker in config.TICKERS:
        try:
            df = fetch_ticker_data(ticker)
            
            # Merge Fear & Greed Sentiment data if enabled
            if config.USE_SENTIMENT and not fng_df.empty:
                df = df.join(fng_df, how='left')
                df['Sentiment_Score'] = df['Sentiment_Score'].fillna(50.0)
                df['Sentiment_MA7'] = df['Sentiment_MA7'].fillna(50.0)
            elif config.USE_SENTIMENT:
                df['Sentiment_Score'] = 50.0
                df['Sentiment_MA7'] = 50.0
            
            # Download recent 4h historical data for multi-timeframe trend coherence
            try:
                df_4h = fetch_ticker_4h_data(ticker)
                from features import align_multi_timeframe_indicators
                df = align_multi_timeframe_indicators(df, df_4h)
            except Exception as e:
                logger.error(f"Failed to load 4h data for {ticker}: {e}. Defaulting 4h filter to pass.")
                df['4h_Bullish'] = 1.0
            
            # Feature calculation & Labeling
            df_copy = df.copy()
            feature_cols = build_features(df_copy, ticker=ticker)
            calculate_triple_barrier_labels(
                df_copy, 
                horizon=config.FORECAST_HORIZON, 
                tp_mult=config.TP_ATR_MULT, 
                sl_mult=config.SL_ATR_MULT_LONG
            )
            
            if feature_columns is None:
                feature_columns = feature_cols
                
            # Layer 1: Volatility regime scaling (returns position scale 0.0 to 1.0)
            regime_filter = MarketRegimeFilter()
            regime_scale = regime_filter.compute_regime_sizing(df_copy)
            
            # Setup indicators for dual trend filter & sentiment caps
            df_copy['EMA_50'] = df_copy['Close'].ewm(span=config.EMA_TREND_WINDOW, adjust=False).mean()
            df_copy['Trend_Bullish'] = (df_copy['Close'] > df_copy['SMA_200']) & (df_copy['Close'] > df_copy['EMA_50'])
            df_copy['Trend_Bearish'] = (df_copy['Close'] < df_copy['SMA_200']) & (df_copy['Close'] < df_copy['EMA_50'])
            df_copy['Long_Sentiment'] = df_copy['Sentiment_Score'] <= config.FEAR_GREED_GREED_CAP
            df_copy['Short_Sentiment'] = df_copy['Sentiment_Score'] >= config.FEAR_GREED_FEAR_FLOOR
            df_copy['Regime_Scale'] = regime_scale
            df_copy['Entry_Allowed'] = regime_scale  # default placeholder, will overwrite dynamically
            
            # Remove NaNs
            df_clean = df_copy.dropna(subset=feature_cols + ['SMA_200', 'EMA_50']).copy()
            
            if len(df_clean) < 150:
                logger.warning(f"Ticker {ticker} has insufficient data rows ({len(df_clean)}). Skipping.")
                continue
                
            processed_dfs[ticker] = df_clean
            
        except Exception as e:
            logger.error(f"Failed to process ticker {ticker}: {e}")
            continue
            
    if not processed_dfs:
        logger.critical("No tickers were processed successfully. Exiting.")
        sys.exit(1)
        
    # 3. Setup Daily Walk-Forward Retraining Loop
    # We split chronologically per ticker.
    # Tickers have different length of clean data. We calculate split index per ticker.
    test_dates_set = set()
    test_dfs_dict = {}
    test_allowance_dict = {}
    test_signals_dict = {}
    test_probs_dict = {}
    
    for ticker, df in processed_dfs.items():
        split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
        test_df = df.iloc[split_idx:].copy()
        
        test_dfs_dict[ticker] = test_df
        test_allowance_dict[ticker] = np.zeros(len(test_df))
        
        # Collect test dates for walk-forward execution
        test_dates_set.update(test_df.index)
        
        # Initialize empty signals and probs arrays for test set
        test_signals_dict[ticker] = np.zeros(len(test_df))
        test_probs_dict[ticker] = np.zeros(len(test_df))
        
    # Chronological unique dates in the test partition
    test_dates = sorted(list(test_dates_set))
    
    logger.info(f"Starting daily walk-forward retraining simulation...")
    logger.info(f"Total days to simulate: {len(test_dates)} dates across {len(processed_dfs)} assets.")
    
    crypto_ensemble = EnsembleTradingModel(
        model_type=config.ML_MODEL_TYPE, 
        conf_thresh_long=config.CONFIDENCE_THRESHOLD_LONG, 
        conf_thresh_short=config.CONFIDENCE_THRESHOLD_SHORT
    )
    equity_ensemble = EnsembleTradingModel(
        model_type=config.ML_MODEL_TYPE, 
        conf_thresh_long=config.CONFIDENCE_THRESHOLD_LONG, 
        conf_thresh_short=config.CONFIDENCE_THRESHOLD_SHORT
    )
    
    rl_agent = None
    if getattr(config, 'USE_RL_AGENT', False):
        try:
            from rl_agent import QLearningAgent
            rl_agent = QLearningAgent(
                learning_rate=getattr(config, 'RL_LEARNING_RATE', 0.1),
                discount_factor=getattr(config, 'RL_DISCOUNT_FACTOR', 0.95),
                epsilon=getattr(config, 'RL_EPSILON', 0.05)
            )
            logger.info("QLearning RL Agent initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize RL Agent: {e}")
    
    # 2.5 Tune Hyperparameters on initial pooled training set if enabled
    if config.RUN_HYPERPARAMETER_TUNING and len(test_dates) > 0:
        logger.info("Preparing initial training data for hyperparameter auto-tuning...")
        init_t = test_dates[0]
        train_features_init_crypto = []
        train_targets_init_crypto = []
        train_features_init_equity = []
        train_targets_init_equity = []
        
        for ticker, df in processed_dfs.items():
            historical_data = df.loc[df.index < init_t]
            if len(historical_data) > 50:
                if ticker in config.SHORTABLE_TICKERS:
                    train_features_init_crypto.append(historical_data[feature_columns])
                    train_targets_init_crypto.append(historical_data['Target'])
                else:
                    train_features_init_equity.append(historical_data[feature_columns])
                    train_targets_init_equity.append(historical_data['Target'])
                    
        if train_features_init_crypto:
            X_train_init_c = pd.concat(train_features_init_crypto, axis=0)
            y_train_init_c = pd.concat(train_targets_init_crypto, axis=0)
            crypto_ensemble.tune_hyperparameters(X_train_init_c, y_train_init_c)
        if train_features_init_equity:
            X_train_init_e = pd.concat(train_features_init_equity, axis=0)
            y_train_init_e = pd.concat(train_targets_init_equity, axis=0)
            equity_ensemble.tune_hyperparameters(X_train_init_e, y_train_init_e)
            
    # Run daily walk-forward simulation
    for i, t in enumerate(test_dates):
        if i % 30 == 0 or i == len(test_dates) - 1:
            logger.info(f"Progress: day {i+1}/{len(test_dates)} ({t.date()})...")
            
        # Build pooled training sets separately
        train_features_crypto = []
        train_targets_crypto = []
        train_features_equity = []
        train_targets_equity = []
        
        for ticker, df in processed_dfs.items():
            historical_data = df.loc[df.index < t]
            if len(historical_data) > 50:
                if ticker in config.SHORTABLE_TICKERS:
                    train_features_crypto.append(historical_data[feature_columns])
                    train_targets_crypto.append(historical_data['Target'])
                else:
                    train_features_equity.append(historical_data[feature_columns])
                    train_targets_equity.append(historical_data['Target'])
                    
        # Retrain primary ensembles and secondary meta-labelers monthly
        if i % 30 == 0 or i == 0 or i == len(test_dates) - 1:
            if train_features_crypto:
                X_train_c = pd.concat(train_features_crypto, axis=0)
                y_train_c = pd.concat(train_targets_crypto, axis=0)
                crypto_ensemble.fit(X_train_c, y_train_c)
                crypto_ensemble.fit_meta_model(X_train_c, y_train_c)
            if train_features_equity:
                X_train_e = pd.concat(train_features_equity, axis=0)
                y_train_e = pd.concat(train_targets_equity, axis=0)
                equity_ensemble.fit(X_train_e, y_train_e)
                equity_ensemble.fit_meta_model(X_train_e, y_train_e)
        
        # Generate predictions for today (date 't')
        for ticker in test_dfs_dict.keys():
            df_test = test_dfs_dict[ticker]
            if t in df_test.index:
                row_idx = df_test.index.get_loc(t)
                X_today = df_test.loc[[t], feature_columns]
                history = df_test.iloc[max(0, row_idx - getattr(config, 'LSTM_SEQUENCE_LENGTH', 20) + 1):row_idx + 1]
                
                # Predict signal using the corresponding sector ensemble
                if ticker in config.SHORTABLE_TICKERS:
                    sig, probs = crypto_ensemble.predict_signals(X_today, history=history)
                else:
                    sig, probs = equity_ensemble.predict_signals(X_today, history=history)
                
                test_signals_dict[ticker][row_idx] = sig[0]
                test_probs_dict[ticker][row_idx] = probs[0]
                
                # Dynamic direction-aware and sentiment-filtered entry allowance
                regime_val = df_test.loc[t, 'Regime_Scale']
                mt_bullish_val = df_test.loc[t, '4h_Bullish']
                
                if sig[0] == 1:
                    # Long entry filters
                    if config.USE_TREND_FILTER:
                        trend_ok = df_test.loc[t, 'Trend_Bullish'] and (mt_bullish_val == 1.0)
                    else:
                        trend_ok = (mt_bullish_val == 1.0)
                    sent_ok = df_test.loc[t, 'Long_Sentiment']
                    # Bullish momentum confirmation: RSI > 48 and MACD Histogram is positive
                    mom_ok = (df_test.loc[t, 'RSI'] > 48.0) and (df_test.loc[t, 'MACD_Hist'] > 0.0)
                    allowed = regime_val if (trend_ok and sent_ok and mom_ok) else 0.0
                elif sig[0] == -1 and (ticker in config.SHORTABLE_TICKERS):
                    # Short entry filters
                    if config.USE_TREND_FILTER:
                        trend_ok = df_test.loc[t, 'Trend_Bearish'] and (mt_bullish_val == 0.0)
                    else:
                        trend_ok = (mt_bullish_val == 0.0)
                    sent_ok = df_test.loc[t, 'Short_Sentiment']
                    # Bearish momentum confirmation: RSI < 52 and MACD Histogram is negative
                    mom_ok = (df_test.loc[t, 'RSI'] < 52.0) and (df_test.loc[t, 'MACD_Hist'] < 0.0)
                    allowed = 1.0 if (trend_ok and sent_ok and mom_ok) else 0.0
                else:
                    allowed = 0.0
                    
                test_allowance_dict[ticker][row_idx] = allowed
                
    logger.info("Daily walk-forward retraining loop completed.")
    
    # 4. Layer 3: Run Portfolio Backtester with Slippage & Circuit Breaker
    logger.info("Initializing multi-asset portfolio backtest...")
    backtester = PortfolioBacktester()
    equity_series, trade_log = backtester.run(test_dfs_dict, test_signals_dict, test_allowance_dict, test_probs_dict, rl_agent=rl_agent)
    
    # Analyze portfolio metrics
    metrics = backtester.analyze_performance(equity_series, trade_log, test_dfs_dict)
    
    # 5. Print Trade Log Sample
    if len(trade_log) > 0:
        logger.info("\nExecuted Portfolio Trades (Slippage Adjusted):")
        cols_to_print = ['Ticker', 'Direction', 'EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL_Pct', 'PnL_USD', 'ExitReason']
        logger.info("\n" + trade_log[cols_to_print].to_string())
    else:
        logger.info("\nNo trades were executed. The system protected capital by staying in cash.")
        
    # 6. Generate and Save Portfolio Performance Chart
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    bh_series_list = []
    for ticker, test_df in test_dfs_dict.items():
        norm_bh = (test_df['Close'] / test_df['Close'].iloc[0]) * config.INITIAL_CAPITAL
        bh_series_list.append(norm_bh)
        
    bh_df = pd.concat(bh_series_list, axis=1).ffill().bfill()
    average_bh_equity = bh_df.mean(axis=1)
    # Average Buy & Hold benchmark equity series
    average_bh_equity = pd.Series(average_bh_equity, index=average_bh_equity.index)
    
    plt.plot(equity_series.index, equity_series.values, label=f"Ultimate 3-Layer Strategy (Daily Retrained)", color='#00cfb4', linewidth=2.5)
    plt.plot(average_bh_equity.index, average_bh_equity.values, label="Equal-Weighted Buy & Hold Benchmark", color='#7f8c8d', linestyle='--', linewidth=1.5)
    
    # Highlight Circuit Breaker trigger if it happened
    if backtester.circuit_breaker_tripped:
        cb_date = backtester.circuit_breaker_date
        plt.axvline(cb_date, color='red', linestyle=':', linewidth=2, label="Circuit Breaker Tripped")
        # Shade everything after circuit breaker in red
        plt.fill_between(test_dates, equity_series.min() * 0.9, equity_series.max() * 1.1, 
                         where=[d >= cb_date for d in test_dates], color='red', alpha=0.05)
        
    plt.title("Ultimate Fail-Proof Portfolio Bot Performance (Daily Retrained)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Net Asset Value ($)", fontsize=12)
    plt.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='none')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save the chart to workspace
    workspace_chart_path = "portfolio_performance.png"
    plt.savefig(workspace_chart_path, dpi=150)
    logger.info(f"Portfolio performance chart saved to workspace: {workspace_chart_path}")
    
    # Save to Gemini artifact directory if available
    artifact_dir = "C:/Users/Asus/.gemini/antigravity/brain/f070070a-c3f1-4585-a6e7-0c5f2e92c4dc"
    if os.path.exists(artifact_dir):
        artifact_chart_path = os.path.join(artifact_dir, "portfolio_performance.png")
        plt.savefig(artifact_chart_path, dpi=150)
        logger.info(f"Portfolio performance chart saved to artifact folder: {artifact_chart_path}")
        
    plt.close()
    
    # 7. Write dynamic Javascript file for the dashboard (bypasses browser CORS restrictions)
    import json
    from datetime import datetime
    
    # Format equity curve for javascript plotting
    equity_curve_data = []
    for d, val in zip(equity_series.index, equity_series.values):
        equity_curve_data.append({
            "date": d.strftime("%Y-%m-%d"),
            "val": float(val)
        })
        
    benchmark_curve_data = []
    for d, val in zip(average_bh_equity.index, average_bh_equity.values):
        benchmark_curve_data.append({
            "date": d.strftime("%Y-%m-%d"),
            "val": float(val)
        })
        
    # Format trade history list
    trades_list = []
    if len(trade_log) > 0:
        for _, row in trade_log.iterrows():
            trades_list.append({
                "ticker": str(row['Ticker']),
                "direction": str(row.get('Direction', 'Long')),
                "entry_time": row['EntryTime'].strftime("%Y-%m-%d") if isinstance(row['EntryTime'], pd.Timestamp) else str(row['EntryTime']),
                "exit_time": row['ExitTime'].strftime("%Y-%m-%d") if isinstance(row['ExitTime'], pd.Timestamp) else str(row['ExitTime']),
                "entry_price": float(row['EntryPrice']),
                "exit_price": float(row['ExitPrice']),
                "pnl_pct": float(row['PnL_Pct']),
                "pnl_usd": float(row['PnL_USD']),
                "exit_reason": str(row['ExitReason'])
            })
            
    portfolio_state = {
        "ticker_list": config.TICKERS,
        "initial_capital": float(config.INITIAL_CAPITAL),
        "final_value": float(equity_series.iloc[-1]),
        "return_pct": float(metrics['strategy_return']),
        "benchmark_return_pct": float(metrics['bh_return']),
        "max_drawdown": float(metrics['max_drawdown']),
        "sharpe_ratio": float(metrics['sharpe_ratio']),
        "trades_count": int(metrics['total_trades']),
        "win_rate": float(metrics['win_rate']),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "circuit_breaker_tripped": bool(backtester.circuit_breaker_tripped),
        "circuit_breaker_date": str(backtester.circuit_breaker_date.date()) if backtester.circuit_breaker_tripped else "",
        "equity_curve": equity_curve_data,
        "benchmark_curve": benchmark_curve_data,
        "trades": trades_list
    }
    
    js_content = f"const PORTFOLIO_STATE = {json.dumps(portfolio_state, indent=2)};"
    safe_atomic_write("portfolio_state.js", js_content)
    logger.info("Portfolio state JS variables written to 'portfolio_state.js' atomically.")
    
    # 8. Generate HTML Performance Report (report.html)
    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ultimate Trading Bot - Backtest Performance Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
        }}
    </style>
</head>
<body class="p-8">
    <div class="max-w-6xl mx-auto bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl">
        <div class="flex justify-between items-center border-b border-gray-800 pb-6 mb-8">
            <div>
                <h1 class="text-3xl font-bold text-teal-400">Backtest Performance Report</h1>
                <p class="text-sm text-gray-400 mt-1">Generated dynamically on {portfolio_state['last_updated']}</p>
            </div>
            <div class="text-right">
                <span class="px-4 py-1.5 rounded-full text-xs font-semibold {'bg-red-900/30 text-red-400' if portfolio_state['circuit_breaker_tripped'] else 'bg-green-900/30 text-green-400'}">
                    {'🚨 CIRCUIT BREAKER TRIPPED' if portfolio_state['circuit_breaker_tripped'] else '🛡️ SYSTEM HEALTHY'}
                </span>
            </div>
        </div>
        
        <!-- Metrics Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Final Value</p>
                <p class="text-2xl font-bold text-white mt-1">${portfolio_state['final_value']:,.2f}</p>
            </div>
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Strategy Return</p>
                <p class="text-2xl font-bold text-teal-400 mt-1">{portfolio_state['return_pct']:.2%}</p>
            </div>
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Market Return</p>
                <p class="text-2xl font-bold text-gray-450 mt-1">{portfolio_state['benchmark_return_pct']:.2%}</p>
            </div>
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Max Drawdown</p>
                <p class="text-2xl font-bold text-rose-500 mt-1">{portfolio_state['max_drawdown']:.2%}</p>
            </div>
        </div>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Sharpe Ratio</p>
                <p class="text-2xl font-bold text-white mt-1">{portfolio_state['sharpe_ratio']:.2f}</p>
            </div>
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Total Trades</p>
                <p class="text-2xl font-bold text-white mt-1">{portfolio_state['trades_count']}</p>
            </div>
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Win Rate</p>
                <p class="text-2xl font-bold text-teal-400 mt-1">{portfolio_state['win_rate']:.2%}</p>
            </div>
            <div class="bg-gray-800/40 border border-gray-800 p-6 rounded-xl">
                <p class="text-xs text-gray-450 uppercase tracking-wider">Outperformance</p>
                <p class="text-2xl font-bold text-teal-400 mt-1">{(portfolio_state['return_pct'] - portfolio_state['benchmark_return_pct']):+.2%}</p>
            </div>
        </div>

        <!-- Chart Section -->
        <div class="mb-8 bg-gray-850 border border-gray-800 rounded-xl p-6">
            <h2 class="text-xl font-semibold text-white mb-4">Equity Curve</h2>
            <img src="portfolio_performance.png" alt="Equity Chart" class="w-full rounded-lg border border-gray-800">
        </div>

        <!-- Trades Table -->
        <div class="bg-gray-850 border border-gray-800 rounded-xl p-6">
            <h2 class="text-xl font-semibold text-white mb-4">Trade Logs</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="border-b border-gray-800 text-gray-400 text-sm">
                            <th class="pb-3 font-semibold">Ticker</th>
                            <th class="pb-3 font-semibold">Entry Date</th>
                            <th class="pb-3 font-semibold">Exit Date</th>
                            <th class="pb-3 font-semibold">Entry Price</th>
                            <th class="pb-3 font-semibold">Exit Price</th>
                            <th class="pb-3 font-semibold">PnL %</th>
                            <th class="pb-3 font-semibold">PnL USD</th>
                            <th class="pb-3 font-semibold">Exit Reason</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-800/50 text-sm">
"""
    for tr in trades_list:
        pnl_class = "text-emerald-400" if tr['pnl_pct'] > 0 else ("text-rose-500" if tr['pnl_pct'] < 0 else "text-gray-400")
        html_report += f"""
                        <tr>
                            <td class="py-3 font-semibold text-white">{tr['ticker']}</td>
                            <td class="py-3 text-gray-400">{tr['entry_time']}</td>
                            <td class="py-3 text-gray-400">{tr['exit_time']}</td>
                            <td class="py-3 text-gray-300">${tr['entry_price']:.2f}</td>
                            <td class="py-3 text-gray-300">${tr['exit_price']:.2f}</td>
                            <td class="py-3 font-semibold {pnl_class}">{tr['pnl_pct']:+.2%}</td>
                            <td class="py-3 font-semibold {pnl_class}">${tr['pnl_usd']:+,.2f}</td>
                            <td class="py-3 text-gray-405"><span class="px-2 py-0.5 rounded text-xs bg-gray-850 border border-gray-800">{tr['exit_reason']}</span></td>
                        </tr>"""
    if not trades_list:
        html_report += """
                        <tr>
                            <td colspan="8" class="py-6 text-center text-gray-500">No trades executed during this backtest timeframe.</td>
                        </tr>"""
                        
    html_report += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""
    safe_atomic_write("report.html", html_report)
    logger.info("Performance HTML report written to 'report.html' atomically.")
    
    # 9. Send summary report notification to Discord/Telegram
    summary_msg = (
        f"🚀 **Ultimate Trading Bot Run Summary**\n\n"
        f"💰 **Your Account Balance**:\n"
        f"• **Start Capital**: `${config.INITIAL_CAPITAL:,.2f}` (Initial funds)\n"
        f"• **Final Balance**: `${equity_series.iloc[-1]:,.2f}` (Your total money now)\n"
        f"• **Net Profit (Strategy Return)**: **{metrics['strategy_return']:.2%}** (How much your money grew)\n\n"
        f"📈 **Bot vs. General Market**:\n"
        f"• **Market Benchmark Return**: **{metrics['bh_return']:.2%}** (What you would get by doing nothing/passive investing)\n"
        f"• **Bot Outperformance**: **{metrics['strategy_return'] - metrics['bh_return']:+.2%}** (How much the bot beat the market by)\n\n"
        f"🛡️ **Risk & Safety Controls**:\n"
        f"• **Max Temporary Drop (Drawdown)**: **{metrics['max_drawdown']:.2%}** (The worst-case peak-to-trough paper loss)\n"
        f"• **Circuit Breaker Tripped**: **{backtester.circuit_breaker_tripped}** (Safety halt to prevent major crashes)\n\n"
        f"📊 **Trading Statistics**:\n"
        f"• **Total Trades Executed**: **{metrics['total_trades']}** (Number of buys/shorts made)\n"
        f"• **Win Rate**: **{metrics['win_rate']:.2%}** (Percentage of profitable trades)\n\n"
        f"✨ *The bot models are healthy, synchronized, and ready on disk!*"
    )
    from security import send_push_notification, push_to_github
    send_push_notification(summary_msg)
    
    # 10. Auto-push updated dashboard files to GitHub (persists Render static dashboards)
    push_to_github()
    
    # 11. Save trained sector ensembles to disk
    try:
        crypto_ensemble.save('crypto_ensemble.joblib')
        equity_ensemble.save('equity_ensemble.joblib')
        logger.info("Saved trained sector ensembles to disk (crypto_ensemble.joblib, equity_ensemble.joblib).")
    except Exception as e:
        logger.error(f"Failed to save models to disk: {e}")
        
    logger.info("Bot execution completed successfully.")

if __name__ == "__main__":
    main()
