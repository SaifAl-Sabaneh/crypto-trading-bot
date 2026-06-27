"""
kelly.py — Kelly Criterion Dynamic Position Sizer

Calculates the mathematically optimal bet size using the Kelly formula:
  f* = (p * b - q) / b
where p = win probability, q = 1 - p, b = reward/risk ratio.

Uses half-Kelly for safety and caps at MAX_ALLOCATION_PER_TRADE.
"""
import numpy as np
import config
from security import logger

class KellySizer:
    def __init__(
        self,
        kelly_fraction=0.5,
        max_alloc=config.MAX_ALLOCATION_PER_TRADE,
        min_alloc=0.05
    ):
        self.kelly_fraction = kelly_fraction  # Half-Kelly for safety
        self.max_alloc = max_alloc
        self.min_alloc = min_alloc
        self.tp_mult = config.TP_ATR_MULT
        self.sl_mult_long = config.SL_ATR_MULT_LONG
        self.sl_mult_short = config.SL_ATR_MULT_SHORT
    
    def compute(self, win_prob: float, direction: int) -> float:
        """
        Computes the Kelly-optimal fraction of capital to allocate.

        Args:
            win_prob: Model's predicted probability of a winning trade (0.0 to 1.0)
            direction: 1 for long, -1 for short

        Returns:
            float: Fraction of capital to allocate (between min_alloc and max_alloc)
        """
        p = float(np.clip(win_prob, 0.01, 0.99))
        q = 1.0 - p
        
        # Reward-to-risk ratio based on ATR multiples
        if direction == 1:
            b = self.tp_mult / self.sl_mult_long  # e.g. 2.5 / 1.5 = 1.667
        else:
            b = self.tp_mult / self.sl_mult_short  # e.g. 2.5 / 1.2 = 2.083
        
        if b <= 0:
            return self.min_alloc
        
        # Full Kelly formula
        kelly_full = (p * b - q) / b
        
        # Apply Kelly fraction (half-Kelly by default)
        kelly_frac = kelly_full * self.kelly_fraction
        
        # Clamp to [min_alloc, max_alloc]
        allocation = float(np.clip(kelly_frac, self.min_alloc, self.max_alloc))
        
        return allocation
    
    def compute_batch(self, win_probs, directions):
        """Vectorized Kelly computation for arrays of predictions."""
        return [self.compute(p, d) for p, d in zip(win_probs, directions)]
