"""
rl_agent.py — Q-Learning Reinforcement Learning Agent

A lightweight tabular Q-Learning agent that acts as a
veto/confirmation layer on top of the ensemble model's signals.

State space (discretized):
  - Market regime: [Bull, Bear, Sideways, Crisis] -> [0, 1, 2, 3]
  - Model confidence: [Low <0.4, Medium 0.4-0.55, High >0.55] -> [0, 1, 2]
  - Macro environment: [Bearish, Neutral, Bullish] -> [0, 1, 2]

Actions: [Hold (0), Long (1), Short (-1)] -> mapped to [0, 1, 2] internally

The agent learns from real trade PnL rewards after each trade closes.
It only vetoes signals — it never overrides a Hold with a trade.
"""
import numpy as np
import os
import joblib
from security import logger


REGIME_MAP = {'Bull': 0, 'Bear': 1, 'Sideways': 2, 'Crisis': 3}
ACTION_MAP = {0: 0, 1: 1, -1: 2}  # Hold, Long, Short
ACTION_REVERSE = {0: 0, 1: 1, 2: -1}

N_REGIMES = 4
N_CONFIDENCE_BINS = 3
N_MACRO_BINS = 3
N_ACTIONS = 3


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.05,
        save_path: str = 'rl_agent_qtable.joblib'
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Small exploration rate
        self.save_path = save_path
        
        # Q-table: shape [n_regimes, n_conf, n_macro, n_actions]
        self.q_table = np.zeros((N_REGIMES, N_CONFIDENCE_BINS, N_MACRO_BINS, N_ACTIONS))
        
        # Initialize with slight long bias
        self.q_table[:, :, :, 1] += 0.01  # Tiny long preference
        
        self.trade_count = 0
        self.last_state = None
        self.last_action = None
        
        # Load existing Q-table if available
        self._load_if_exists()
    
    def _get_confidence_bin(self, prob: float) -> int:
        if prob < 0.40:
            return 0  # Low confidence
        elif prob < 0.55:
            return 1  # Medium confidence
        else:
            return 2  # High confidence
    
    def _get_macro_bin(self, vix_zscore: float, yield_spread: float) -> int:
        """Classify macro environment as Bearish/Neutral/Bullish."""
        if vix_zscore > 1.5 or yield_spread < -0.5:
            return 0  # Bearish macro
        elif vix_zscore < -0.5 and yield_spread > 0.5:
            return 2  # Bullish macro
        else:
            return 1  # Neutral macro
    
    def _get_state(self, regime: str, prob: float, vix_zscore: float, yield_spread: float) -> tuple:
        return (
            REGIME_MAP.get(regime, 2),
            self._get_confidence_bin(prob),
            self._get_macro_bin(vix_zscore, yield_spread)
        )
    
    def should_take_action(self, signal: int, regime: str, prob: float,
                           vix_zscore: float = 0.0, yield_spread: float = 0.0) -> bool:
        """
        Given the ensemble signal, returns True if the RL agent confirms
        the trade, False if it vetoes it.
        
        In Crisis regime, always veto.
        In early learning phase (< 10 trades), always confirm to gather data.
        """
        if signal == 0:
            return False  # Never trade if ensemble says Hold
        
        if regime == 'Crisis':
            logger.info("RL Agent: Vetoing trade — Crisis regime detected.")
            return False
        
        state = self._get_state(regime, prob, vix_zscore, yield_spread)
        self.last_state = state
        action_idx = ACTION_MAP.get(signal, 0)
        self.last_action = action_idx
        
        # Early learning phase: confirm all trades to collect experience
        if self.trade_count < 10:
            return True
        
        # Epsilon-greedy: occasionally allow trades even if Q-value is low
        if np.random.random() < self.epsilon:
            return True
        
        # Confirm only if the proposed action has a non-negative Q-value
        q_val_action = self.q_table[state[0], state[1], state[2], action_idx]
        q_val_hold = self.q_table[state[0], state[1], state[2], 0]
        
        if q_val_action >= q_val_hold:
            return True
        else:
            logger.info(f"RL Agent: Vetoing trade (Q={q_val_action:.4f} < Hold Q={q_val_hold:.4f})")
            return False
    
    def update(self, pnl_pct: float):
        """
        Update Q-table after a trade closes with its realized PnL.
        reward = pnl_pct (positive for profit, negative for loss).
        """
        if self.last_state is None or self.last_action is None:
            return
        
        s = self.last_state
        a = self.last_action
        reward = float(pnl_pct) * 100  # Scale to make updates meaningful
        
        # Q-learning update rule: Q(s,a) += lr * (reward + gamma * max(Q(s')) - Q(s,a))
        # Since we don't track next state precisely, use reward as terminal signal
        current_q = self.q_table[s[0], s[1], s[2], a]
        max_next_q = np.max(self.q_table[s[0], s[1], s[2]])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[s[0], s[1], s[2], a] = new_q
        
        self.trade_count += 1
        self.last_state = None
        self.last_action = None
        
        # Persist Q-table periodically
        if self.trade_count % 5 == 0:
            self.save()
    
    def save(self):
        """Persist Q-table to disk."""
        try:
            joblib.dump({'q_table': self.q_table, 'trade_count': self.trade_count}, self.save_path)
        except Exception as e:
            logger.warning(f"RL Agent: Failed to save Q-table: {e}")
    
    def _load_if_exists(self):
        """Load existing Q-table from disk if available."""
        if os.path.exists(self.save_path):
            try:
                state = joblib.load(self.save_path)
                self.q_table = state['q_table']
                self.trade_count = state.get('trade_count', 0)
                logger.info(f"RL Agent: Loaded Q-table with {self.trade_count} historical trades.")
            except Exception as e:
                logger.warning(f"RL Agent: Failed to load Q-table: {e}. Starting fresh.")
