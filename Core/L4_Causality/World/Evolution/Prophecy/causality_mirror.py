"""
Causality Mirror (The Reflection)
=================================

"The future is a hypothesis; the present is the test."

Role: Compares Predicted Future vs Actual Present to calculate 'Surprise' (Loss).
      Updates the Prophet's intuition based on this feedback.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
import time

# Integration
from Core.L4_Causality.World.Evolution.Prophecy.prophet_engine import ProphetEngine

logger = logging.getLogger("CausalityMirror")

@dataclass
class PredictionRecord:
    timestamp: float
    action: str
    predicted_state: Dict[str, float]
    actual_state: Dict[str, float]
    error: float

class CausalityMirror:
    def __init__(self, prophet: ProphetEngine):
        self.prophet = prophet
        self.history: List[PredictionRecord] = []
        self.learning_rate = 0.05
        
    def reflect(self, action: str, predicted_future: Dict[str, float], actual_now: Dict[str, float]):
        """
        Compare the prophecy with reality.
        """
        # Calculate Error (MSE-like)
        error = 0.0
        comparable_keys = ['Energy', 'Inspiration', 'Joy']
        
        for key in comparable_keys:
            pred = predicted_future.get(key, 0.0)
            actual = actual_now.get(key, 0.0)
            error += (pred - actual) ** 2
            
        mean_error = error / len(comparable_keys)
        
        logger.info(f"  [REFLECTION] Action: {action} | Prediction Error: {mean_error:.4f}")
        
        # Record
        record = PredictionRecord(
            timestamp=time.time(),
            action=action,
            predicted_state=predicted_future,
            actual_state=actual_now,
            error=mean_error
        )
        self.history.append(record)
        
        # Feedback Loop (Simple Reinforcement)
        if mean_error > 0.1:
            self._calibrate_prophet(mean_error)
            
    def _calibrate_prophet(self, error: float):
        """
        Adjusts the simulation physics if predictions are consistently off.
        """
        logger.info(f"  [CALIBRATION] Tuning ProphetEngine Physics (Error: {error:.4f})...")
        
        # Simple heuristic: If error is high, increase 'uncertainty' or adjust decay rates
        # Here we just nudge the energy decay as a demo of learning
        self.prophet.energy_decay *= (1.0 + self.learning_rate)
        logger.info(f"   -> Prophet Energy Decay Adjusted to: {self.prophet.energy_decay:.4f}")