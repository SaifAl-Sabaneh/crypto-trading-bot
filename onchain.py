"""
onchain.py — On-Chain Crypto Intelligence Features

Fetches free on-chain metrics for crypto assets via the CoinMetrics Community API.
Falls back to neutral (0.0) values gracefully if the API is unavailable.

Metrics fetched per-asset:
  - NVT Signal: Network Value to Transactions — high = overvalued
  - Active Address Count (normalized): Growing wallets = bullish demand
  - Exchange Net Flow (estimated from supply change): Inflows = selling pressure
"""
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from security import logger

COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"
ASSET_MAP = {
    'BTC-USD': 'btc',
    'ETH-USD': 'eth',
    'SOL-USD': 'sol',
    'BNB-USD': 'bnb',
    'COIN': 'btc'  # COIN (Coinbase stock) approximated by BTC on-chain
}

_onchain_cache = {}  # Per-asset cache


def _fetch_coinmetrics(asset_cm: str, metric: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch a single metric from CoinMetrics community API."""
    url = f"{COINMETRICS_BASE}/timeseries/asset-metrics"
    params = {
        'assets': asset_cm,
        'metrics': metric,
        'start_time': start_date,
        'end_time': end_date,
        'frequency': '1d',
        'page_size': 10000
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return pd.Series(dtype=float)
        data = resp.json().get('data', [])
        if not data:
            return pd.Series(dtype=float)
        
        # Clean rows
        rows = []
        for r in data:
            if metric in r and r[metric] is not None:
                val = r[metric]
                # If list/dict from multi-index columns, get raw element
                if isinstance(val, list):
                    val = val[0]
                elif isinstance(val, dict):
                    val = list(val.values())[0]
                rows.append((r['time'][:10], float(val)))
        
        if not rows:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([r[0] for r in rows])
        vals = [r[1] for r in rows]
        return pd.Series(vals, index=idx)
    except Exception as e:
        logger.debug(f"CoinMetrics API error for {asset_cm}/{metric}: {e}")
        return pd.Series(dtype=float)


def fetch_onchain_features(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches on-chain features for a single crypto ticker.

    Returns a DataFrame with columns:
        - OnChain_NVT_Zscore: Normalized NVT signal (high = overbought)
        - OnChain_ActiveAddr_Change: Rate of change in active addresses
        - OnChain_ExchangeFlow: Exchange net flow signal
    Filled with 0.0 if data unavailable.
    """
    asset_cm = ASSET_MAP.get(ticker)
    if asset_cm is None:
        # Non-crypto ticker — return neutral zeros
        return pd.DataFrame(dtype=float)
    
    global _onchain_cache
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cache_key in _onchain_cache:
        return _onchain_cache[cache_key]
    
    logger.info(f"Fetching on-chain features for {ticker} ({asset_cm})...")
    result_parts = {}
    
    # 1. NVT Signal (NVTAdj90 or NVT if available)
    nvt = _fetch_coinmetrics(asset_cm, 'NVTAdj90', start_date, end_date)
    if nvt.empty:
        nvt = _fetch_coinmetrics(asset_cm, 'NVT', start_date, end_date)
    if not nvt.empty:
        nvt_mean = nvt.rolling(30, min_periods=5).mean()
        nvt_std = nvt.rolling(30, min_periods=5).std().replace(0, 1)
        result_parts['OnChain_NVT_Zscore'] = ((nvt - nvt_mean) / nvt_std).clip(-3, 3)
    
    # 2. Active Address Count
    active_addr = _fetch_coinmetrics(asset_cm, 'AdrActCnt', start_date, end_date)
    if not active_addr.empty:
        result_parts['OnChain_ActiveAddr_Change'] = active_addr.pct_change(7).clip(-1, 1).fillna(0)
    
    # 3. Exchange net flow (using SplyCur as supply proxy)
    supply = _fetch_coinmetrics(asset_cm, 'SplyCur', start_date, end_date)
    if not supply.empty:
        supply_change = supply.pct_change(3).clip(-0.1, 0.1).fillna(0)
        result_parts['OnChain_ExchangeFlow'] = supply_change
    
    if not result_parts:
        logger.warning(f"No on-chain data returned for {ticker}. Using neutral values.")
        empty_df = pd.DataFrame(index=pd.date_range(start_date, end_date))
        for col in ['OnChain_NVT_Zscore', 'OnChain_ActiveAddr_Change', 'OnChain_ExchangeFlow']:
            empty_df[col] = 0.0
        _onchain_cache[cache_key] = empty_df
        return empty_df
    
    result = pd.DataFrame(result_parts)
    # Ensure all expected columns exist to prevent KeyError
    for col in ['OnChain_NVT_Zscore', 'OnChain_ActiveAddr_Change', 'OnChain_ExchangeFlow']:
        if col not in result.columns:
            result[col] = 0.0
            
    result = result.fillna(0)
    _onchain_cache[cache_key] = result
    logger.info(f"On-chain features for {ticker}: {list(result.columns)} over {len(result)} bars.")
    return result


def merge_onchain_into_df(asset_df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Merges on-chain features into an asset DataFrame.
    Non-crypto tickers get neutral 0.0 values.
    """
    onchain_cols = ['OnChain_NVT_Zscore', 'OnChain_ActiveAddr_Change', 'OnChain_ExchangeFlow']
    
    onchain_df = fetch_onchain_features(ticker, start_date, end_date)
    
    if onchain_df.empty:
        for col in onchain_cols:
            asset_df[col] = 0.0
        return asset_df
    
    merged = asset_df.join(onchain_df, how='left')
    merged[onchain_cols] = merged[onchain_cols].ffill().fillna(0)
    return merged
