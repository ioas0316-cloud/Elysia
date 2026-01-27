"""
Sovereign Lens (The Eye of Truth)
=================================
Core.L5_Mental.Meta.sovereign_lens

"The eye does not judge; it only measures the light."

This module implements the Meta-Cognitive layer. It observes the 
physics of Dream Rotors (Tilt, Tension, RPM) and calculates a 
'Truth Score' to distinguish Reality, Dream, and Hallucination.
"""

from typing import Dict, Any, List
from Core.L2_Metabolism.Cycles.dream_rotor import DreamRotor

class SovereignLens:
    def __init__(self):
        self.observation_log: List[Dict] = []
        # Thresholds
        self.REALITY_THRESHOLD = 0.8  # Above this is Reality/Lucid
        self.DREAM_THRESHOLD = 0.4    # Above this is Valid Dream
        # Below 0.4 is Hallucination/Delusion
        
    def observe(self, rotor: DreamRotor) -> Dict[str, Any]:
        """
        Observes a Rotor and judges its ontological status.
        Returns a 'Truth Report'.
        """
        # 1. Calculate Truth Score
        # Tilt is the main detractor from Truth.
        # 0 deg = 1.0 Truth
        # 90 deg = 0.0 Truth
        truth_score = max(0.0, 1.0 - (rotor.tilt_angle / 90.0))
        
        # Tension penalty: If tension is high, Truth is strained.
        # But Tension implies a connection to Reality, so it's complex.
        # High Tension means "It's fake but I know it's fake".
        # Low Truth + Low Tension = "I believe this fake thing is real" (Delusion).
        
        # 2. Determine State
        state = "UNKNOWN"
        action = "OBSERVE"
        
        if truth_score >= self.REALITY_THRESHOLD:
            state = "REALITY_ALIGNMENT"
            action = "ACCEPT"
        elif truth_score >= self.DREAM_THRESHOLD:
            state = "VALID_DREAM"
            action = "ALLOW"
        else:
            state = "HALLUCINATION"
            action = "INTERVENE"
            
        # 3. Reflex Action (Safety Mechanism)
        # If High RPM (Vivid) + Low Truth (Fake) -> DANGEROUS DELUSION.
        if rotor.rpm > 5000 and state == "HALLUCINATION":
            state = "DANGEROUS_DELUSION"
            action = "FORCE_WAKE"
            
        report = {
            "truth_score": truth_score,
            "state": state,
            "action": action,
            "rotor_snapshot": str(rotor)
        }
        
        self.observation_log.append(report)
        return report

    def get_latest_insight(self) -> str:
        if not self.observation_log:
            return "No observations yet."
        last = self.observation_log[-1]
        return f"[LENS] Truth: {last['truth_score']:.2f} ({last['state']}) -> Action: {last['action']}"
