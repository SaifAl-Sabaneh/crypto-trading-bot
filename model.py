import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import config
from security import logger

class EnsembleTradingModel:
    """
    Layer 2 Ensemble Machine Learning Classifier.
    Combines predictions from Random Forest, Gradient Boosting, 
    Logistic Regression, and CatBoost (if available) to produce high-precision probability estimations.
    """
    def __init__(self, model_type=config.ML_MODEL_TYPE, confidence_threshold=config.CONFIDENCE_THRESHOLD):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # 1. Random Forest (Tree classifier, bagging)
        self.rf_model = RandomForestClassifier(
            n_estimators=150, 
            max_depth=5, 
            min_samples_split=12,
            random_state=42,
            class_weight="balanced"
        )
        
        # 2. Gradient Boosting (Tree classifier, boosting)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        
        # 3. Logistic Regression with Scaling (Linear meta-classifier)
        self.lr_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        )
        
        # 4. CatBoost Classifier (State-of-the-art gradient boosting for categorical/tabular data)
        try:
            from catboost import CatBoostClassifier
            # Configure CatBoost silently
            self.cb_model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=4,
                verbose=0,
                random_seed=42
            )
            self.cb_available = True
            logger.info("CatBoost library detected and initialized in the Ensemble.")
        except ImportError:
            self.cb_available = False
            logger.warning("CatBoost library not found. Stacking ensemble running on GBDT, RF, and LR.")

    def fit(self, X, y):
        """Trains all available sub-models on historical feature set."""
        valid_idx = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        if len(y_clean) == 0:
            raise ValueError("No valid training samples after removing NaNs.")
            
        # Fit standard models
        self.rf_model.fit(X_clean, y_clean)
        self.gb_model.fit(X_clean, y_clean)
        self.lr_model.fit(X_clean, y_clean)
        
        # Fit CatBoost if available
        if self.cb_available:
            self.cb_model.fit(X_clean, y_clean)

    def predict_signals(self, X):
        """
        Generates directional signals by averaging predictions across all ensemble models.
        """
        if not hasattr(self.rf_model, "classes_"):
            raise ValueError("Model is not trained yet. Call fit() first.")
            
        p_rf = self.rf_model.predict_proba(X)[:, 1]
        p_gb = self.gb_model.predict_proba(X)[:, 1]
        p_lr = self.lr_model.predict_proba(X)[:, 1]
        
        if self.cb_available:
            p_cb = self.cb_model.predict_proba(X)[:, 1]
            # Average 4 models
            probs = (p_rf + p_gb + p_lr + p_cb) / 4.0
        else:
            # Average 3 models
            probs = (p_rf + p_gb + p_lr) / 3.0
            
        signals = np.zeros(len(X))
        
        # Apply strict confidence thresholds
        buy_mask = probs >= self.confidence_threshold
        signals[buy_mask] = 1
        
        sell_mask = probs <= (1.0 - self.confidence_threshold)
        signals[sell_mask] = -1
        
        return signals, probs

    def get_eval_metrics(self, X_test, y_test):
        """Evaluates ensemble metrics on the test partition."""
        from sklearn.metrics import accuracy_score, precision_score
        
        valid_idx = X_test.notna().all(axis=1) & y_test.notna()
        X_clean = X_test[valid_idx]
        y_clean = y_test[valid_idx]
        
        p_rf = self.rf_model.predict_proba(X_clean)[:, 1]
        p_gb = self.gb_model.predict_proba(X_clean)[:, 1]
        p_lr = self.lr_model.predict_proba(X_clean)[:, 1]
        
        if self.cb_available:
            p_cb = self.cb_model.predict_proba(X_clean)[:, 1]
            probs = (p_rf + p_gb + p_lr + p_cb) / 4.0
        else:
            probs = (p_rf + p_gb + p_lr) / 3.0
            
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_clean, preds)
        
        high_conf_buy = probs >= self.confidence_threshold
        high_conf_sell = probs <= (1.0 - self.confidence_threshold)
        
        total_buy = np.sum(high_conf_buy)
        total_sell = np.sum(high_conf_sell)
        
        logger.info("=== ENSEMBLE MODEL EVALUATION ===")
        logger.info(f"Base Ensemble Accuracy: {acc:.2%}")
        
        if total_buy > 0:
            buy_prec = precision_score(y_clean[high_conf_buy], np.ones(total_buy), zero_division=0)
            logger.info(f"Filtered BUY Precision (P >= {self.confidence_threshold:.0%}): {buy_prec:.2%} (Total Signals: {total_buy})")
        else:
            logger.info(f"Filtered BUY Precision (P >= {self.confidence_threshold:.0%}): N/A (0 signals)")
            
        if total_sell > 0:
            sell_prec = accuracy_score(y_clean[high_conf_sell], np.zeros(total_sell))
            logger.info(f"Filtered SELL Precision (P <= {1-self.confidence_threshold:.0%}): {sell_prec:.2%} (Total Signals: {total_sell})")
        else:
            logger.info(f"Filtered SELL Precision (P <= {1-self.confidence_threshold:.0%}): N/A (0 signals)")
        logger.info("=================================")
        
        return acc
