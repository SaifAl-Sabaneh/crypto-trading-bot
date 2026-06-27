import pandas as pd
import numpy as np
import config
import urllib.request
import xml.etree.ElementTree as ET
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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

def calculate_stochastic(df, window=14):
    """Calculates Stochastic Oscillator (%K and %D)."""
    low_min = df['Low'].rolling(window).min()
    high_max = df['High'].rolling(window).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(3).mean()
    return k, d

def calculate_adx(df, window=14):
    """Calculates Average Directional Index (ADX) to capture trend strength."""
    upmove = df['High'] - df['High'].shift(1)
    downmove = df['Low'].shift(1) - df['Low']
    
    plus_dm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minus_dm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    
    # Use 1-day True Range rolling sum for smooth TR
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_smooth = tr.rolling(window).sum()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(window).sum()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(window).sum()
    
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window).mean()
    return adx

def build_features(df, ticker=None):
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
    
    # Stochastic & ADX
    stoch_k, stoch_d = calculate_stochastic(df)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    df['ADX'] = calculate_adx(df)
    
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
        'SMA_Ratio', 'Stoch_K', 'Stoch_D', 'ADX'
    ]
    
    if config.USE_SENTIMENT:
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 50.0
            df['Sentiment_MA7'] = 50.0
        feature_cols.extend(['Sentiment_Score', 'Sentiment_MA7'])
        
    # Merge Macro Features
    if getattr(config, 'USE_MACRO_FEATURES', False):
        try:
            from macro import fetch_macro_features
            start_str = df.index[0].strftime("%Y-%m-%d")
            end_str = df.index[-1].strftime("%Y-%m-%d")
            macro_df = fetch_macro_features(start_str, end_str)
            macro_cols = [
                'Macro_DXY_Return', 'Macro_DXY_Trend', 'Macro_VIX', 
                'Macro_VIX_Zscore', 'Macro_Yield_Spread', 
                'Macro_Yield_Inverted', 'Macro_RiskOff_Ratio'
            ]
            for col in macro_cols:
                if not macro_df.empty and col in macro_df.columns:
                    df[col] = macro_df[col].reindex(df.index).ffill().fillna(0.0)
                else:
                    df[col] = 0.0
            feature_cols.extend(macro_cols)
        except Exception as e:
            from security import logger
            logger.error(f"Error merging macro features: {e}")
            
    # Merge On-Chain Features (for Crypto tickers only)
    if getattr(config, 'USE_ONCHAIN_FEATURES', False):
        onchain_cols = ['OnChain_NVT_Zscore', 'OnChain_ActiveAddr_Change', 'OnChain_ExchangeFlow']
        if ticker in getattr(config, 'CRYPTO_TICKERS', []):
            try:
                from onchain import fetch_onchain_features
                start_str = df.index[0].strftime("%Y-%m-%d")
                end_str = df.index[-1].strftime("%Y-%m-%d")
                onchain_df = fetch_onchain_features(ticker, start_str, end_str)
                for col in onchain_cols:
                    if not onchain_df.empty and col in onchain_df.columns:
                        df[col] = onchain_df[col].reindex(df.index).ffill().fillna(0.0)
                    else:
                        df[col] = 0.0
            except Exception as e:
                from security import logger
                logger.error(f"Error merging on-chain features for {ticker}: {e}")
                for col in onchain_cols:
                    df[col] = 0.0
        else:
            # Non-crypto/equity ticker: populate with neutral zeros to maintain feature symmetry
            for col in onchain_cols:
                df[col] = 0.0
        feature_cols.extend(onchain_cols)
            
    # Merge News NLP Sentiment Features
    if getattr(config, 'USE_NEWS_NLP', False) and ticker:
        try:
            start_str = df.index[0].strftime("%Y-%m-%d")
            end_str = df.index[-1].strftime("%Y-%m-%d")
            df_news = fetch_rss_news_sentiment(ticker, start_str, end_str)
            if not df_news.empty and 'News_Sentiment_MA3' in df_news.columns:
                df['News_Sentiment_MA3'] = df_news['News_Sentiment_MA3'].reindex(df.index).ffill().fillna(50.0)
            else:
                df['News_Sentiment_MA3'] = 50.0
            feature_cols.append('News_Sentiment_MA3')
        except Exception as e:
            from security import logger
            logger.error(f"Error merging news NLP features for {ticker}: {e}")
            
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

def align_multi_timeframe_indicators(df_daily, df_4h):
    """
    Computes a 4-hour trend filter (EMA 50) on 4-hour candles,
    resamples it daily (taking the last candle state of each day),
    and left-joins it back to the daily dataframe.
    Missing entries (e.g. before 730 days ago) default to 1.0 (bullish/pass).
    """
    if df_4h is None or df_4h.empty:
        df_daily['4h_Bullish'] = 1.0
        return df_daily

    df_4h_copy = df_4h.copy()
    
    # Handle multi-index columns if present
    if isinstance(df_4h_copy.columns, pd.MultiIndex):
        df_4h_copy.columns = df_4h_copy.columns.get_level_values(0)
        
    df_4h_copy['EMA_50'] = df_4h_copy['Close'].ewm(span=50, adjust=False).mean()
    df_4h_copy['4h_Bullish'] = (df_4h_copy['Close'] > df_4h_copy['EMA_50']).astype(float)
    
    # Strip timezone if present to prevent join issues
    df_4h_copy.index = df_4h_copy.index.tz_localize(None)
    df_4h_daily = df_4h_copy['4h_Bullish'].resample('D').last().ffill()
    
    # Ensure daily df index is timezone-naive as well
    df_daily_naive = df_daily.copy()
    df_daily_naive.index = df_daily_naive.index.tz_localize(None)
    
    df_daily_naive = df_daily_naive.join(df_4h_daily, how='left')
    df_daily_naive['4h_Bullish'] = df_daily_naive['4h_Bullish'].fillna(1.0)
    
    return df_daily_naive

# Initialize VADER sentiment analyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

_vader_analyzer = None

def get_vader_analyzer():
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

def fetch_rss_news_sentiment(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Scrapes free financial RSS feeds (Yahoo Finance/CoinDesk),
    applies VADER NLP sentiment analysis, and returns a DataFrame
    with daily Sentiment score for the ticker.
    """
    from security import logger
    logger.info(f"Fetching RSS news sentiment for {ticker}...")
    
    # We create a dummy index over the date range
    date_idx = pd.date_range(start_date, end_date)
    result = pd.DataFrame(index=date_idx)
    result['News_Sentiment_Score'] = 0.0  # Neutral default
    
    analyzer = get_vader_analyzer()
    
    # Standardize asset terms to search
    search_term = ticker.split('-')[0].lower() # e.g. btc-usd -> btc
    
    headlines_by_date = {} # {date_str: [headlines]}
    
    # Format Yahoo Finance RSS URL for specific ticker
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    # Also fetch standard CoinDesk feed if it's crypto
    urls = [url]
    if ticker in config.CRYPTO_TICKERS:
        urls.append("https://coindesk.com/arc/outboundfeeds/rss/")
        
    for rss_url in urls:
        try:
            req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                
                for item in root.findall('.//item'):
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    
                    if title is not None and title.text and pub_date is not None:
                        headline = title.text.strip()
                        # Simple case-insensitive match for the asset in the headline
                        if search_term in headline.lower():
                            try:
                                dt = pd.to_datetime(pub_date.text).tz_localize(None).normalize()
                                dt_str = dt.strftime("%Y-%m-%d")
                                if dt_str not in headlines_by_date:
                                    headlines_by_date[dt_str] = []
                                headlines_by_date[dt_str].append(headline)
                            except Exception:
                                continue
        except Exception as e:
            logger.debug(f"RSS fetch failed for {rss_url}: {e}")
            
    # Compute daily scores
    for date_str, headlines in headlines_by_date.items():
        if date_str in result.index.strftime("%Y-%m-%d"):
            scores = []
            for h in headlines:
                vader_score = analyzer.polarity_scores(h)['compound']
                scores.append(vader_score)
            
            # Map VADER compound score [-1.0, 1.0] to a [0, 100] scale
            mean_vader = np.mean(scores) if scores else 0.0
            sentiment_scaled = (mean_vader + 1.0) * 50.0
            dt_index = pd.to_datetime(date_str)
            result.loc[dt_index, 'News_Sentiment_Score'] = sentiment_scaled
            
    # Compute rolling average of news sentiment to smooth noise and handle missing days
    result['News_Sentiment_Score'] = result['News_Sentiment_Score'].replace(0.0, np.nan)
    result['News_Sentiment_Score'] = result['News_Sentiment_Score'].ffill().fillna(50.0)
    result['News_Sentiment_MA3'] = result['News_Sentiment_Score'].rolling(config.NEWS_SENTIMENT_WINDOW, min_periods=1).mean()
    
    return result[['News_Sentiment_MA3']]



