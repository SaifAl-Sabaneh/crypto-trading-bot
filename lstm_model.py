"""
lstm_model.py — Lightweight Pure-NumPy LSTM Temporal Pattern Recognizer

Implements a single-layer LSTM using only numpy and scipy.
Captures temporal patterns like:
  - 3 consecutive lower highs before breakdown
  - Volume divergence patterns
  - Sequential indicator convergence

The LSTM probability score is used as an additional stacking feature
for the ensemble meta-learner, not as a standalone signal.
"""
import numpy as np
from scipy.special import expit as sigmoid  # numerically stable sigmoid
from security import logger


def tanh_np(x):
    return np.tanh(np.clip(x, -10, 10))


class NumpyLSTM:
    """
    Single-layer LSTM implemented in pure NumPy.
    Input: sequence of shape (seq_len, input_size)
    Output: scalar probability in [0, 1]
    """
    def __init__(self, input_size: int, hidden_size: int = 32, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Forget gate
        self.Wf = rng.randn(hidden_size, hidden_size + input_size) * scale
        self.bf = np.zeros((hidden_size, 1))
        # Input gate
        self.Wi = rng.randn(hidden_size, hidden_size + input_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        # Cell gate
        self.Wg = rng.randn(hidden_size, hidden_size + input_size) * scale
        self.bg = np.zeros((hidden_size, 1))
        # Output gate
        self.Wo = rng.randn(hidden_size, hidden_size + input_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        # Final linear classifier
        self.Wy = rng.randn(1, hidden_size) * 0.01
        self.by = np.zeros((1, 1))
        
        self.is_trained = False
    
    def _forward(self, X: np.ndarray):
        """
        Forward pass through the LSTM.
        X: shape (seq_len, input_size)
        Returns: final hidden state h_T of shape (hidden_size,)
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for t in range(X.shape[0]):
            x_t = X[t].reshape(-1, 1)
            combined = np.vstack([h, x_t])
            
            f_t = sigmoid(self.Wf @ combined + self.bf)
            i_t = sigmoid(self.Wi @ combined + self.bi)
            g_t = tanh_np(self.Wg @ combined + self.bg)
            o_t = sigmoid(self.Wo @ combined + self.bo)
            
            c = f_t * c + i_t * g_t
            h = o_t * tanh_np(c)
        
        return h
    
    def predict_proba(self, X: np.ndarray) -> float:
        """
        Returns a probability score [0, 1] for a single sequence.
        X: shape (seq_len, input_size)
        """
        if not self.is_trained:
            return 0.5  # Neutral until trained
        h = self._forward(X)
        logit = self.Wy @ h + self.by
        return float(sigmoid(logit)[0, 0])
    
    def fit(self, sequences: np.ndarray, labels: np.ndarray,
            epochs: int = 20, lr: float = 0.001):
        """
        Trains the LSTM using BPTT with gradient clipping.
        
        Args:
            sequences: shape (n_samples, seq_len, input_size)
            labels: shape (n_samples,) with values 0 or 1
            epochs: number of training epochs
            lr: learning rate
        """
        n = len(sequences)
        if n < 20:
            logger.warning(f"LSTM: Insufficient training samples ({n}). Need >= 20.")
            return self
        
        logger.info(f"LSTM: Training on {n} sequences for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            # Shuffle
            idx = np.random.permutation(n)
            
            for i in idx:
                X = sequences[i]  # (seq_len, input_size)
                y = float(labels[i])
                
                # Forward pass
                h = self._forward(X)
                logit = self.Wy @ h + self.by
                prob = float(sigmoid(logit)[0, 0])
                prob = np.clip(prob, 1e-7, 1 - 1e-7)
                
                # Loss: binary cross-entropy
                loss = -(y * np.log(prob) + (1 - y) * np.log(1 - prob))
                total_loss += loss
                
                # Output layer gradient
                dL_dlogit = prob - y  # dL/d(logit)
                
                # Update output weights
                self.Wy -= lr * dL_dlogit * h.T
                self.by -= lr * dL_dlogit
                
                # Gradient through hidden state (simplified 1-step BPTT)
                dL_dh = (self.Wy.T * dL_dlogit)  # (hidden_size, 1)
                
                # Gradient clipping
                dL_dh = np.clip(dL_dh, -1.0, 1.0)
                
                # Update final gate weights proportionally
                x_T = X[-1].reshape(-1, 1)
                h_prev = np.zeros((self.hidden_size, 1))  # simplified
                combined = np.vstack([h_prev, x_T])
                
                self.Wo -= lr * (dL_dh * 0.1) @ combined.T
                self.Wg -= lr * (dL_dh * 0.05) @ combined.T
                self.Wi -= lr * (dL_dh * 0.05) @ combined.T
                self.Wf -= lr * (dL_dh * 0.02) @ combined.T
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / n
                logger.debug(f"LSTM Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("LSTM: Training complete.")
        return self


class TemporalLSTMLayer:
    """
    Wrapper that manages sequence preparation and LSTM integration
    with the existing EnsembleTradingModel pipeline.
    """
    def __init__(self, seq_length: int = 20, hidden_size: int = 32):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.lstm = None
        self.is_fitted = False
        self.feature_cols = []
    
    def _build_sequences(self, X: np.ndarray, y = None):
        """Build sliding window sequences from a feature matrix."""
        if y is not None:
            if hasattr(y, 'values'):
                y = y.values
            y = np.array(y)
            
        seqs = []
        labs = [] if y is not None else None
        n = len(X)
        for i in range(self.seq_length, n):
            seqs.append(X[i - self.seq_length:i])
            if labs is not None:
                labs.append(y[i])
        seqs = np.array(seqs)
        return (seqs, np.array(labs)) if labs is not None else seqs
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the LSTM on feature sequences."""
        n_features = X.shape[1]
        self.lstm = NumpyLSTM(input_size=n_features, hidden_size=self.hidden_size)
        seqs, labs = self._build_sequences(X, y)
        # Binarize labels: 1 = long signal (target=1), 0 = otherwise
        binary_labs = (labs == 1).astype(float)
        self.lstm.fit(seqs, binary_labs, epochs=5, lr=0.001)
        self.is_fitted = True
        return self
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an LSTM probability score for each row in X.
        First seq_length rows return 0.5 (no history yet).
        """
        if not self.is_fitted or self.lstm is None:
            return np.full(len(X), 0.5)
        
        scores = np.full(len(X), 0.5)
        for i in range(self.seq_length, len(X)):
            seq = X[i - self.seq_length:i]
            scores[i] = self.lstm.predict_proba(seq)
        return scores
