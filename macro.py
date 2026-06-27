"""
macro.py — Macro Market Regime Features

Fetches free macro data via yfinance:
  - DXY (US Dollar Index): Dollar strength
  - VIX (CBOE Volatility Index): Market fear gauge
  - 10Y-2Y Yield Spread: Yield curve shape (recession signal)
  - Gold/SPY ratio: Risk-off vs Risk-on regime signal

Returns a DataFrame indexed by date with macro feature columns
that can be merged into any asset's feature set.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from security import logger

MACRO_TICKERS = {
    'DXY': 'DX-Y.NYB',
    'VIX': '^VIX',
    'TNX': '^TNX',   # 10-Year Treasury Yield
    'IRX': '^IRX',   # 13-Week T-Bill Yield (proxy for 3M)
    'GLD': 'GLD',
    'SPY': 'SPY'
}

_macro_cache = None  # Module-level cache to avoid redundant downloads

def fetch_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads macro data and computes derived features.

    Returns:
        pd.DataFrame with columns:
            - DXY_Return: Daily return of the dollar index
            - DXY_MA20: 20-day MA of DXY (dollar trend)
            - VIX_Level: Raw VIX value
            - VIX_MA10: 10-day MA of VIX (smoothed fear)
            - Yield_Spread: 10Y - 3M yield (negative = inverted = recession risk)
            - RiskOn_Ratio: GLD/SPY ratio (high = risk-off, low = risk-on)
    """
    global _macro_cache
    if _macro_cache is not None:
        logger.info("Using cached macro features.")
        return _macro_cache
    
    logger.info("Fetching macro features (DXY, VIX, Yield Curve, Risk ratio)...")
    dfs = {}
    
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            if df.empty:
                logger.warning(f"No macro data for {name} ({ticker})")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            dfs[name] = df[['Close']].rename(columns={'Close': name})
        except Exception as e:
            logger.warning(f"Failed to fetch macro data for {name}: {e}")
    
    if not dfs:
        logger.warning("No macro data fetched. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Merge all macro series on date index
    macro_df = pd.concat(dfs.values(), axis=1, join='outer').ffill().bfill()
    
    result = pd.DataFrame(index=macro_df.index)
    
    # 1. Dollar Index features
    if 'DXY' in macro_df.columns:
        # Convert series or dataframe close column
        dxy_col = macro_df['DXY']
        if isinstance(dxy_col, pd.DataFrame):
            dxy_col = dxy_col.iloc[:, 0]
        result['Macro_DXY_Return'] = dxy_col.pct_change().fillna(0)
        result['Macro_DXY_Trend'] = (dxy_col / dxy_col.rolling(20).mean() - 1).fillna(0)
    
    # 2. VIX features
    if 'VIX' in macro_df.columns:
        vix_col = macro_df['VIX']
        if isinstance(vix_col, pd.DataFrame):
            vix_col = vix_col.iloc[:, 0]
        result['Macro_VIX'] = vix_col.fillna(20.0)
        result['Macro_VIX_Zscore'] = (
            (vix_col - vix_col.rolling(60).mean()) /
            vix_col.rolling(60).std()
        ).fillna(0)
    
    # 3. Yield curve spread (10Y - 3M)
    if 'TNX' in macro_df.columns and 'IRX' in macro_df.columns:
        tnx_col = macro_df['TNX']
        if isinstance(tnx_col, pd.DataFrame):
            tnx_col = tnx_col.iloc[:, 0]
        irx_col = macro_df['IRX']
        if isinstance(irx_col, pd.DataFrame):
            irx_col = irx_col.iloc[:, 0]
        result['Macro_Yield_Spread'] = (tnx_col - irx_col).fillna(0)
        result['Macro_Yield_Inverted'] = (result['Macro_Yield_Spread'] < 0).astype(float)
    
    # 4. Risk-On/Off ratio (GLD vs SPY)
    if 'GLD' in macro_df.columns and 'SPY' in macro_df.columns:
        gld_col = macro_df['GLD']
        if isinstance(gld_col, pd.DataFrame):
            gld_col = gld_col.iloc[:, 0]
        spy_col = macro_df['SPY']
        if isinstance(spy_col, pd.DataFrame):
            spy_col = spy_col.iloc[:, 0]
        gld_spy = gld_col / spy_col
        result['Macro_RiskOff_Ratio'] = (gld_spy / gld_spy.rolling(30).mean() - 1).fillna(0)
    
    result = result.fillna(0)
    _macro_cache = result
    logger.info(f"Macro features computed: {list(result.columns)} over {len(result)} bars.")
    return result


def merge_macro_into_df(asset_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins macro features into an asset DataFrame by date.
    Missing macro dates are forward-filled from last available value.
    """
    if macro_df.empty:
        # Fill with neutral values if macro unavailable
        for col in ['Macro_DXY_Return', 'Macro_DXY_Trend', 'Macro_VIX',
                    'Macro_VIX_Zscore', 'Macro_Yield_Spread',
                    'Macro_Yield_Inverted', 'Macro_RiskOff_Ratio']:
            asset_df[col] = 0.0
        return asset_df
    
    merged = asset_df.join(macro_df, how='left')
    macro_cols = list(macro_df.columns)
    merged[macro_cols] = merged[macro_cols].ffill().fillna(0)
    return merged
