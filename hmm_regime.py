"""
hmm_regime.py — 4-State Market Regime Classifier

Uses Gaussian Mixture Models (sklearn) to approximate a Hidden Markov Model.
Classifies each trading day into one of 4 market states:
  State 0: Bull (low vol, positive trend) — Full long bias
  State 1: Bear (medium vol, negative trend) — Full short bias  
  State 2: Sideways/Choppy (elevated vol, no trend) — Reduced sizing
  State 3: Crisis/Panic (extreme vol spike) — Full halt
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from security import logger

class HMMRegimeClassifier:
    def __init__(self, n_states=4, lookback=60):
        self.n_states = n_states
        self.lookback = lookback
        self.gmm = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42, max_iter=200)
        self.scaler = StandardScaler()
        self.state_labels = {}  # Map GMM cluster IDs to human-readable regime names
        self.is_fitted = False
    
    def _build_regime_features(self, df):
        """Build 5 features that describe market regime: returns, vol, trend, vol_of_vol, drawdown."""
        close = df['Close']
        # Handle multi-index columns if present (from yfinance)
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = close.pct_change()
        
        vol_5 = returns.rolling(5).std()
        vol_20 = returns.rolling(20).std()
        trend = (close / close.rolling(20).mean()) - 1.0  # price vs 20d MA
        vol_of_vol = vol_5.rolling(10).std()  # second-order vol
        rolling_max = close.rolling(20).max()
        drawdown = (close - rolling_max) / rolling_max  # rolling drawdown
        
        features = pd.DataFrame({
            'returns': returns,
            'vol_5': vol_5,
            'vol_20': vol_20,
            'trend': trend,
            'vol_of_vol': vol_of_vol,
            'drawdown': drawdown
        })
        return features.dropna()
    
    def _label_states(self, features, states):
        """Map GMM cluster IDs to Bull/Bear/Sideways/Crisis based on their properties."""
        state_props = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                continue
            avg_return = features.loc[mask, 'returns'].mean()
            avg_vol = features.loc[mask, 'vol_20'].mean()
            state_props[s] = {'return': avg_return, 'vol': avg_vol}
        
        sorted_by_vol = sorted(state_props.items(), key=lambda x: x[1]['vol'])
        self.state_labels = {}
        
        for i, (state_id, _) in enumerate(sorted_by_vol):
            if i == 0:
                if state_props[state_id]['return'] >= 0:
                    self.state_labels[state_id] = 'Bull'
                else:
                    self.state_labels[state_id] = 'Sideways'
            elif i == len(sorted_by_vol) - 1:
                self.state_labels[state_id] = 'Crisis'
            elif state_props[state_id]['return'] < 0:
                self.state_labels[state_id] = 'Bear'
            else:
                self.state_labels[state_id] = 'Sideways'
        logger.info(f"HMM State Labels: {self.state_labels}")
    
    def fit(self, df):
        """Train the GMM on historical regime features."""
        features = self._build_regime_features(df)
        if len(features) < self.lookback:
            logger.warning("Insufficient data to fit HMM regime classifier.")
            return self
        X = self.scaler.fit_transform(features.values)
        self.gmm.fit(X)
        states = self.gmm.predict(X)
        self._label_states(features, states)
        self.is_fitted = True
        logger.info(f"HMM Regime Classifier fitted on {len(features)} bars.")
        return self
    
    def predict_regime(self, df):
        """
        Returns a Series of regime labels aligned with df's index.
        Returns 'Bull' for all bars if model not fitted.
        """
        if not self.is_fitted:
            return pd.Series('Bull', index=df.index)
        
        features = self._build_regime_features(df)
        if features.empty:
            return pd.Series('Bull', index=df.index)
        X = self.scaler.transform(features.values)
        raw_states = self.gmm.predict(X)
        regime_series = pd.Series(
            [self.state_labels.get(s, 'Sideways') for s in raw_states],
            index=features.index
        )
        # Reindex to match full df index, forward-fill gaps
        regime_series = regime_series.reindex(df.index).ffill().fillna('Sideways')
        return regime_series
    
    def get_position_scale(self, regime_label):
        """
        Returns a sizing multiplier (0.0 to 1.0) for a given regime.
        Bull -> 1.0, Sideways -> 0.5, Bear -> 0.75 (short trades ok), Crisis -> 0.0
        """
        scale_map = {
            'Bull': 1.0,
            'Bear': 0.75,
            'Sideways': 0.5,
            'Crisis': 0.0
        }
        return scale_map.get(regime_label, 0.5)
