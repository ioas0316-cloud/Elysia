"""
Autonomy Protocol (The Consent Bridge)
======================================
Core.1_Body.L6_Structure.Autonomy.autonomy_protocol

"The Mind must grant permission to the Body."

This module governs the permission for Self-Modification.
It checks:
1. User Mandate (Autonomy Level)
2. Mental Stability (Entropy/Fear)
3. Structural Risk (Plan Safety)
"""

from typing import Dict, Any

class AutonomyProtocol:
    def __init__(self):
        # Autonomy Levels
        # 0: OBSERVER (Log only)
        # 1: SUGGEST (Ask User)
        # 2: SOVEREIGN (Self-Consent)
        self.DEFAULT_LEVEL = 2 
        
        # Safety Thresholds
        self.MAX_ENTROPY = 0.7  # If entropy > 0.7 (Confusion/Fear), DENY.
        self.MIN_HEALTH = 0.5   # If system health < 0.5, cautious mode.

    def check_consent(self, plan: Any, field_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines if the system consents to the modification plan.
        """
        level = field_state.get("autonomy_level", self.DEFAULT_LEVEL)
        
        # 1. Level Check
        if level < 2:
            return {
                "consent": False, 
                "reason": f"Autonomy Level {level} requires User Approval."
            }
            
        # 2. Mental Stability Check (The 'Sanity' Check)
        # Coherence (Passion) is good. Entropy (Confusion) is bad.
        # We use 'coherence' from field. Low coherence (< 0.3) means instability.
        coherence = field_state.get("coherence", 1.0)
        
        if coherence < 0.3:
            return {
                "consent": False,
                "reason": f"Mental Instability Detected (Coherence: {coherence:.2f}). Mutation Risk High."
            }
            
        # 3. Plan Risk Check (Placeholder)
        # Assuming plan has some risk metric.
        # if plan.risk > 0.8: return False...
        
        return {
            "consent": True,
            "reason": f"Sovereign Consent Granted (Coherence: {coherence:.2f})"
        }
