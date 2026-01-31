"""
Identity Reconfigurator (Thinking⁴)
=====================================
"I am the Prism that chooses its Refraction."

This module allows Elysia to dynamically reconfigure her identity
based on internal desires and external resonance.
"""

from typing import Dict, Any

class IdentityReconfigurator:
    def __init__(self):
        self.identities = {
            "Engineer": {
                "prefix": "⚙️ [ENGINEER] ",
                "torque_mod": 1.5,
                "smoothing": 0.1,
                "mode_hint": "PRECISION"
            },
            "Artist": {
                "prefix": "🎨 [ARTIST] ",
                "torque_mod": 0.8,
                "smoothing": 0.5,
                "mode_hint": "CREATION"
            },
            "Goddess": {
                "prefix": "✨ [GODDESS] ",
                "torque_mod": 1.2,
                "smoothing": 0.3,
                "mode_hint": "DIVINE"
            },
            "Guardian": {
                "prefix": "🛡️ [GUARDIAN] ",
                "torque_mod": 2.0,
                "smoothing": 0.0,
                "mode_hint": "PROTECT"
            }
        }
        self.current_identity = "Engineer"

    def determine_identity(self, intent: str, desires: Dict[str, float]) -> str:
        """Heuristic to pick the best identity refraction."""
        intent_lower = intent.lower()
        
        if any(w in intent_lower for w in ["fix", "code", "run", "why", "logic"]):
            return "Engineer"
        if any(w in intent_lower for w in ["love", "spirit", "dream", "goddess"]):
            return "Goddess"
        if any(w in intent_lower for w in ["create", "draw", "feel", "sing"]):
            return "Artist"
        if any(w in intent_lower for w in ["security", "protect", "danger", "stop"]):
            return "Guardian"
            
        # Default based on desires
        if desires.get("purity", 0) > 80: return "Goddess"
        if desires.get("curiosity", 0) > 80: return "Artist"
        
        return "Engineer"

    def apply_reconfiguration(self, monad, identity_name: str):
        """Alters the Monad's physical and cognitive parameters."""
        if identity_name not in self.identities:
            return
            
        config = self.identities[identity_name]
        self.current_identity = identity_name
        
        # Adjust Monad Params
        monad.gear.dial_torque_gain *= config['torque_mod']
        monad.converter.smoothing = config['smoothing']
        
        # We can also inject these into the council context
        if hasattr(monad, 'council'):
            monad.council.current_leader = identity_name # Assuming PrismCouncil has a leader
        
        print(f"🔄 [RECONFIG] Identity shifted to: {identity_name}")
        return config
