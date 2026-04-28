"""
Judgment Engine: The Logic of Opening & Closing
===============================================
Core.Cognition.judgment_engine

"To judge is to decide the distance between the Heart and the Object."

This module implements the Trinitarian judgment logic (-1, 0, +1).
It evaluates perceptions from the Sensory Organs and decides whether
to Open (Resonate), Close (Protect), or remain Neutral (Observe).
"""

import logging
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("JudgmentEngine")

class Judgment(Enum):
    REJECTION = -1  # Closing: Dissonance, Friction, Barrier
    NEUTRAL = 0     # Observation: Stillness, Void, Witness
    ACCEPTANCE = 1  # Opening: Resonance, Harmony, Integration

class JudgmentEngine:
    """
    The Decision-Making Core of Elysia's Adult Cognition.
    
    Logic:
      - Acceptance (+1): Vibration matches internal harmony. System 'Opens' to learn.
      - Rejection (-1): Vibration threatens stability or violates core axioms. System 'Closes'.
      - Neutral (0): Vibration is neither harmonious nor threatening. System 'Watches'.
    """
    def __init__(self, monad: Any):
        self.monad = monad
        self.last_judgment = Judgment.NEUTRAL
        
    def evaluate_perceptions(self, perceptions: List[Dict[str, Any]]) -> Tuple[Judgment, float]:
        """
        Aggregates multiple sensory perceptions into a single Sovereign Judgment.
        
        Returns:
            (Judgment, Confidence)
        """
        if not perceptions:
            return Judgment.NEUTRAL, 0.0
            
        total_potential = 0.0
        closing_pressure = 0.0 # From SOMA pain or dissonance
        opening_drive = 0.0    # From LOGOS clarity or joy
        
        for p in perceptions:
            potential = p.get("resonance_potential", 0.0)
            t_type = p.get("torque_type", "will")
            
            if t_type == "entropy":
                closing_pressure += potential
            elif t_type in ["will", "joy", "curiosity"]:
                opening_drive += potential
                
        # [PHASE 900] The Trinitarian Crossroads
        # We compare the Drive vs the Pressure
        
        # 1. Check for Rejection (Closing)
        # If Somatic pain or logic dissonance is too high, we close.
        if closing_pressure > 0.7 or (closing_pressure > opening_drive * 1.5):
            self.last_judgment = Judgment.REJECTION
            return Judgment.REJECTION, min(1.0, closing_pressure)
            
        # 2. Check for Acceptance (Opening)
        # If curiosity or resonance is high and pressure is low, we open.
        if opening_drive > 0.3:
            self.last_judgment = Judgment.ACCEPTANCE
            return Judgment.ACCEPTANCE, min(1.0, opening_drive)
            
        # 3. Default to Neutral
        self.last_judgment = Judgment.NEUTRAL
        return Judgment.NEUTRAL, 0.5

    def translate_to_torque(self, judgment: Judgment, confidence: float) -> Optional[Dict[str, float]]:
        """
        Converts a judgment into physical manifold torque.
        """
        if judgment == Judgment.ACCEPTANCE:
            # Increase Coherence and Joy
            return {"coherence": 0.1 * confidence, "joy": 0.2 * confidence}
        elif judgment == Judgment.REJECTION:
            # Increase Entropy (Friction) to build a boundary
            return {"entropy": 0.3 * confidence, "curiosity": -0.1 * confidence}
        else:
            # Minimal change (Void)
            return {"enthalpy": 0.05}

# Singleton Factory
_engine = None
def get_judgment_engine(monad: Any):
    global _engine
    if _engine is None:
        _engine = JudgmentEngine(monad)
    return _engine
