import pandas as pd
import numpy as np
import config

def calculate_rsi(close_prices, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD, Signal Line, and Histogram."""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist

def calculate_bollinger_bands(close_prices, window=20, num_std=2):
    """Calculates Bollinger Bands and derived metrics."""
    middle_band = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    band_width = (upper_band - lower_band) / (middle_band + 1e-10)
    # Relative position of close price within the band (0 = lower, 1 = upper)
    relative_position = (close_prices - lower_band) / (upper_band - lower_band + 1e-10)
    
    return upper_band, lower_band, band_width, relative_position

def calculate_atr(df, window=14):
    """Calculates the Average True Range (ATR)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr

def build_features(df):
    """
    Calculates features for ML model.
    Modifies df in place and returns the list of feature column names.
    """
    # Technical Indicators
    df['RSI'] = calculate_rsi(df['Close'])
    
    macd_line, signal_line, macd_hist = calculate_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = macd_hist
    
    upper, lower, width, rel_pos = calculate_bollinger_bands(df['Close'])
    df['BB_Width'] = width
    df['BB_RelPos'] = rel_pos
    
    df['ATR'] = calculate_atr(df)
    df['ATR_Pct'] = df['ATR'] / df['Close']  # ATR normalized by Close price
    
    # Volatility features (rolling std of log returns)
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_5'] = log_returns.rolling(5).std()
    df['Vol_10'] = log_returns.rolling(10).std()
    df['Vol_20'] = log_returns.rolling(20).std()
    
    # Momentum (rolling returns)
    df['Ret_1'] = df['Close'].pct_change(1)
    df['Ret_3'] = df['Close'].pct_change(3)
    df['Ret_5'] = df['Close'].pct_change(5)
    df['Ret_10'] = df['Close'].pct_change(10)
    
    # Trend (SMA ratios)
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_Ratio'] = df['SMA_10'] / (df['SMA_50'] + 1e-10)
    
    # Calculate SMA_200 for the trend filter (used in execution, not as a direct ML feature)
    df['SMA_200'] = df['Close'].rolling(config.SMA_TREND_WINDOW).mean()
    
    # Drop intermediate columns that shouldn't be direct features
    feature_cols = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
        'BB_Width', 'BB_RelPos', 'ATR_Pct', 
        'Vol_5', 'Vol_10', 'Vol_20', 
        'Ret_1', 'Ret_3', 'Ret_5', 'Ret_10', 
        'SMA_Ratio'
    ]
    
    if config.USE_SENTIMENT:
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 50.0
            df['Sentiment_MA7'] = 50.0
        feature_cols.extend(['Sentiment_Score', 'Sentiment_MA7'])
        
    return feature_cols

def fetch_fear_and_greed_data(limit=0):
    """
    Downloads Crypto Fear & Greed Index from alternative.me.
    Returns a pandas DataFrame with columns ['Sentiment_Score', 'Sentiment_MA7'] indexed by Date.
    """
    import urllib.request
    import json
    
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            records = data.get("data", [])
            if not records:
                raise ValueError("Empty data returned from Fear & Greed API.")
            
            rows = []
            for rec in records:
                dt = pd.to_datetime(int(rec['timestamp']), unit='s').normalize()
                val = float(rec['value'])
                rows.append({"Date": dt, "Sentiment_Score": val})
                
            fng_df = pd.DataFrame(rows).set_index("Date").sort_index()
            fng_df['Sentiment_MA7'] = fng_df['Sentiment_Score'].rolling(7, min_periods=1).mean()
            return fng_df
    except Exception as e:
        from security import logger
        logger.error(f"Error fetching Fear & Greed data: {e}. Using fallback empty DataFrame.")
        return pd.DataFrame(columns=['Sentiment_Score', 'Sentiment_MA7'])

def calculate_triple_barrier_labels(df, horizon=10, tp_mult=2.5, sl_mult=1.0):
    """
    Computes labels using the Triple Barrier Method.
    Y = 1 if the price hits the Take-Profit barrier before the Stop-Loss barrier.
    Y = 0 if the price hits the Stop-Loss barrier first or if the horizon is reached.
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    n = len(df)
    
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(close[i]):
            labels[i] = 0
            continue
            
        tp_barrier = close[i] + tp_mult * atr[i]
        sl_barrier = close[i] - sl_mult * atr[i]
        
        hit_tp = False
        hit_sl = False
        
        # Look forward
        for j in range(1, horizon + 1):
            if i + j >= n:
                break
                
            curr_high = high[i + j]
            curr_low = low[i + j]
            
            # Stop Loss hit?
            curr_hit_sl = curr_low <= sl_barrier
            # Take Profit hit?
            curr_hit_tp = curr_high >= tp_barrier
            
            if curr_hit_sl and curr_hit_tp:
                # Conservative: assume stop loss hit first
                hit_sl = True
                break
            elif curr_hit_sl:
                hit_sl = True
                break
            elif curr_hit_tp:
                hit_tp = True
                break
                
        if hit_tp and not hit_sl:
            labels[i] = 1
        else:
            labels[i] = 0
            
    df['Target'] = labels
    return labels

def create_labels(df, horizon=5, min_return=0.005):
    """
    Fallback fixed-horizon labeler.
    """
    future_close = df['Close'].shift(-horizon)
    forward_return = (future_close - df['Close']) / df['Close']
    target = (forward_return > min_return).astype(int)
    df['Target_Forward_Ret'] = forward_return
    df['Target'] = target
    return target

