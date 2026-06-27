import pandas as pd
import numpy as np
import config
from security import logger

class MarketRegimeFilter:
    """
    Layer 1 Volatility Filter (Dynamic Position Sizing).
    Adjusts position scaling or halts trading based on historical volatility percentiles.
    """
    def __init__(self, volatility_window=config.VOLATILITY_WINDOW, 
                 percentile_limit=config.REGIME_PERCENTILE_LIMIT, 
                 lookback=config.LOOKBACK_PERCENTILE):
        self.volatility_window = volatility_window
        self.percentile_limit = percentile_limit  # e.g., 75th percentile
        self.lookback = lookback

    def calculate_volatility_metric(self, df):
        """Calculates rolling standard deviation of log returns."""
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        rolling_vol = log_returns.rolling(self.volatility_window).std()
        return rolling_vol

    def compute_regime_sizing(self, df):
        """
        Determines the capital allocation scale factor for each bar.
        Returns:
            position_scale: Series of float sizing factors:
                            - 1.0: Calm (Full Allocation)
                            - 0.5: Moderate Volatility (Half Allocation)
                            - 0.0: Hyper-Volatile / Bear (Halt Trading)
        """
        if getattr(config, 'USE_HMM_REGIME', False):
            try:
                from hmm_regime import HMMRegimeClassifier
                classifier = HMMRegimeClassifier(
                    n_states=getattr(config, 'HMM_N_STATES', 4),
                    lookback=getattr(config, 'HMM_LOOKBACK', 60)
                )
                classifier.fit(df)
                regimes = classifier.predict_regime(df)
                
                # Compute position scales from regimes
                position_scale = regimes.apply(classifier.get_position_scale)
                
                # Diagnostic columns
                df['Regime_Label'] = regimes
                df['Regime_Scale'] = position_scale
                
                # Also set default Regime_Position_Scale to make sure main.py and reports work
                df['Regime_Position_Scale'] = position_scale
                
                return position_scale
            except Exception as e:
                logger.error(f"HMM Regime fit/predict failed: {e}. Falling back to Volatility Percentile.")
                
        # Fill default columns if HMM not used to prevent KeyError in reporting
        df['Regime_Label'] = 'Bull'
        df['Regime_Scale'] = 1.0
        
        vol = self.calculate_volatility_metric(df)
        
        # Calculate two thresholds:
        # 1. Calm Limit (e.g., 75th percentile)
        # 2. Halt Limit (e.g., 90th percentile)
        halt_percentile = min(95.0, self.percentile_limit + 15.0) # e.g. 75 + 15 = 90
        
        def get_percentile(window_series, p):
            valid = window_series.dropna()
            if len(valid) < self.volatility_window:
                return np.nan
            return np.percentile(valid, p)
            
        # Compute rolling percentiles
        calm_thresholds = vol.rolling(window=self.lookback, min_periods=self.volatility_window).apply(
            lambda x: get_percentile(x, self.percentile_limit), raw=False
        )
        halt_thresholds = vol.rolling(window=self.lookback, min_periods=self.volatility_window).apply(
            lambda x: get_percentile(x, halt_percentile), raw=False
        )
        
        # Determine position scaling factors
        position_scale = pd.Series(1.0, index=df.index)
        
        # Moderate Volatility: Scale to 50% capital size
        mod_mask = (vol > calm_thresholds) & (vol <= halt_thresholds)
        position_scale[mod_mask] = 0.5
        
        # Hyper Volatile: Halt trading (0%)
        halt_mask = (vol > halt_thresholds)
        position_scale[halt_mask] = 0.0
        
        # Fill missing data at start of series with 0.0 (halt) for security
        position_scale = position_scale.fillna(0.0)
        
        # Also, check for structural alignment: if volatility calculation is NaN, scale to 0.0
        nan_mask = vol.isna() | calm_thresholds.isna()
        position_scale[nan_mask] = 0.0
        
        # Diagnostic columns
        df['Regime_Volatility'] = vol
        df['Regime_Calm_Threshold'] = calm_thresholds
        df['Regime_Halt_Threshold'] = halt_thresholds
        df['Regime_Position_Scale'] = position_scale
        df['Regime_Scale'] = position_scale
        
        return position_scale
