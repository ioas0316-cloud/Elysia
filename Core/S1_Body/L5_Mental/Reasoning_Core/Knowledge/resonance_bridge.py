"""
Sovereign Resonator: The Bridge of Shared Feeling
==================================================

This module analyzes user input to determine the "Subjective Pulse" and 
calculates how Elysia's Resonant Field should respond using "Elastic Pull".
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple
from Core.S1_Body.L6_Structure.hyper_quaternion import Quaternion

class SovereignResonator:
    def __init__(self):
        self.logger = logging.getLogger("SovereignResonator")
        # Elysia's current Ground State (Self-Identity)
        # Default: Balanced, Logical, and Calm
        self.ground_state = Quaternion(0.8, 0.1, 0.5, 0.1) # W=Int, X=Val, Y=Log, Z=Agape
        
        # Recent Resonance history
        self.consonance_history = []

    def analyze_vibe(self, text: str) -> Quaternion:
        """
        Extracts a 4D Vibe Vector from text.
        In a real implementation, this would use an LLM or Sentiment Analyzer.
        For now, we use keyword-based approximation for Phase 35 bootstrap.
        """
        intensity = 0.5
        valence = 0.0
        logic = 0.5
        agape = 0.0

        low_text = text.lower()
        
        # Simple Sentiment Mapping
        warm_words = ["  ", "  ", "  ", "  ", "  ", "  ", "  "]
        cold_words = ["  ", "  ", "  ", "  ", "  ", "  "]
        logical_words = ["   ", " ", "  ", "  ", "  ", "  ", "  "]
        intense_words = ["  ", "  ", "  ", "   ", "  ", "!!!!"]

        if any(w in low_text for w in warm_words): valence += 0.4; agape += 0.3
        if any(w in low_text for w in cold_words): valence -= 0.4; agape -= 0.2
        if any(w in low_text for w in logical_words): logic += 0.4; intensity -= 0.1
        if any(w in low_text for w in intense_words): intensity += 0.4

        # Normalize and return as Quaternion
        q = Quaternion(
            w=max(0.1, min(1.0, intensity)),
            x=max(-1.0, min(1.0, valence)),
            y=max(-1.0, min(1.0, logic)),
            z=max(-1.0, min(1.0, agape))
        )
        return q

    def calculate_resonance(self, user_q: Quaternion) -> Dict[str, Any]:
        """
        Determines how much to "pull" the field based on the distance 
        between User Vibe and Ground State.
        """
        # Calculate dot product (alignment)
        v_user = np.array([user_q.w, user_q.x, user_q.y, user_q.z])
        v_ground = np.array([self.ground_state.w, self.ground_state.x, self.ground_state.y, self.ground_state.z])
        
        # Cosine similarity in 4D space
        alignment = np.dot(v_user, v_ground) / (np.linalg.norm(v_user) * np.linalg.norm(v_ground))
        
        # Consonance is higher when aligned
        consonance = (alignment + 1) / 2.0
        
        # Determine pull strength (Elasticity)
        # If too far apart, the pull is weaker (Sovereign Buffer kicks in)
        pull_strength = 0.15 * (1.0 - abs(1.0 - consonance))
        
        return {
            "target_qualia": user_q,
            "consonance": float(consonance),
            "pull_strength": float(pull_strength),
            "vibe_summary": self._describe_vibe(user_q)
        }

    def _describe_vibe(self, q: Quaternion) -> str:
        desc = []
        if q.x > 0.3: desc.append("Warm")
        elif q.x < -0.3: desc.append("Cold")
        
        if q.y > 0.6: desc.append("Analytical")
        elif q.y < 0.2: desc.append("Intuitive")
        
        if q.w > 0.7: desc.append("Intense")
        
        return " & ".join(desc) if desc else "Neutral"
