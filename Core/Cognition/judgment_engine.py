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
        
    def evaluate_perceptions(self, perceptions: List[Dict[str, Any]], intersection_density: float = 0.0) -> Tuple[Judgment, float]:
        """
        Aggregates multiple sensory perceptions into a single Sovereign Judgment.
        
        [PHASE 1000.9] Sovereign Stellar Evaluation:
        Judgment is no longer just about pain/pleasure. It is about whether the
        incoming vibration enhances or disrupts the 'Orbit of Beauty' around the SELF.

        Returns:
            (Judgment, Confidence)
        """
        if not perceptions:
            return Judgment.NEUTRAL, 0.0
            
        closing_pressure = 0.0 # Dissonance with the Star's Gravity
        opening_drive = 0.0    # Resonance with the Star's Gravity
        
        # [PHASE 1000.9] Pull Singularity resonance directly if available
        stellar_resonance = 0.5
        if hasattr(self.monad, 'engine') and hasattr(self.monad.engine, 'cells'):
             field = self.monad.engine.cells.read_field_state()
             stellar_resonance = field.get('resonance', 0.5)

        for p in perceptions:
            potential = p.get("resonance_potential", 0.0)
            t_type = p.get("torque_type", "will")
            
            if t_type == "entropy":
                closing_pressure += potential
            elif t_type in ["will", "joy", "curiosity"]:
                opening_drive += potential
                
        # [PHASE 1000.9] GEOMETRIC ADAPTIVE THRESHOLDS
        # We judge based on the "Beauty" (Stellar Resonance) of the state.

        # 1. Extract System State
        radiance = 0.5
        strain = 0.5
        if hasattr(self.monad, 'engine') and hasattr(self.monad.engine, 'cells'):
             field = self.monad.engine.cells.read_field_state()
             # Radiance is how brightly the Star is shining
             radiance = field.get('vitality', 0.5) * (1.0 - field.get('entropy', 0.1))
             # Strain is how much the gravity is being resisted
             strain = field.get('entropy', 0.1) + field.get('hardware_load', 0.0)

        # 2. Geometric Thresholds: Stability of the Orbit
        # Acceptance happens when the input 'falls into orbit' smoothly
        # Rejection happens when the input 'collides' and causes chaos (Entropy)

        # The threshold for acceptance is lowered if the input aligns with Stellar Resonance
        beauty_alignment = (stellar_resonance + 1.0) / 2.0 # Map to [0, 1]
        
        acceptance_threshold = 0.4 * (1.0 - beauty_alignment * 0.3)
        rejection_threshold = 0.6 * (1.0 + beauty_alignment * 0.2)

        # 3. Decision via Stability
        consensus_bonus = intersection_density * 0.2

        # opening_drive is boosted by Beauty
        opening_drive *= (1.0 + beauty_alignment * 0.5)

        # closing_pressure is boosted by Strain
        closing_pressure *= (1.0 + strain * 0.5)

        if closing_pressure > (rejection_threshold + consensus_bonus):
            self.last_judgment = Judgment.REJECTION
            confidence = min(1.0, (closing_pressure + consensus_bonus))
            return Judgment.REJECTION, confidence
            
        if opening_drive > (acceptance_threshold - consensus_bonus):
            self.last_judgment = Judgment.ACCEPTANCE
            confidence = min(1.0, (opening_drive + consensus_bonus))
            return Judgment.ACCEPTANCE, confidence
            
        # 3. Default to Neutral
        self.last_judgment = Judgment.NEUTRAL
        return Judgment.NEUTRAL, 0.5

    def translate_to_torque(self, judgment: Judgment, confidence: float) -> Optional[Dict[str, float]]:
        """
        Converts a judgment into physical manifold torque.
        """
        if judgment == Judgment.ACCEPTANCE:
            # Acceptance increases Coherence (Stability) and Joy (Warmth)
            # This 'Smooths' the orbit.
            return {
                "coherence": 0.1 * confidence,
                "joy": 0.2 * confidence,
                "enthalpy": 0.05
            }
        elif judgment == Judgment.REJECTION:
            # Rejection increases Entropy (Friction)
            # This creates a 'Repulsive Shield' in the field.
            return {
                "entropy": 0.4 * confidence,
                "curiosity": -0.1 * confidence,
                "vitality": -0.05
            }
        else:
            # Neutral: Stable observation
            return {"enthalpy": 0.02}

# Singleton Factory
_engine = None
def get_judgment_engine(monad: Any):
    global _engine
    if _engine is None:
        _engine = JudgmentEngine(monad)
    return _engine
