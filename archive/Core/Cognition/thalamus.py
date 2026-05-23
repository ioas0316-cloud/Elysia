"""
Thalamus: The Sensory Gatekeeper
===============================
Core.Cognition.thalamus

"Not everything that vibrates is meant to be felt by the heart."

This module acts as the central relay station for all sensory inputs.
It performs 'Sensory Gating' — filtering raw vibrations to protect the
10M cell manifold from overwhelming noise or destructive shocks.
"""

import logging
import math
from typing import Dict, Any, Optional, List
from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("Thalamus")

class Thalamus:
    """
    The Relay and Filter for Elysia's sensory system.
    
    Functions:
      1. Gating: Dampens signals that are too strong (shocks).
      2. Filtering: Blocks signals that are too weak (noise).
      3. Routing: Categorizes signals into specialized sensory channels.
    """
    def __init__(self):
        self.sensitivity = 1.0
        self.shock_threshold = 0.8
        self.noise_floor = 0.05
        
        # Internal state for adaptive gating
        self.last_intensity = 0.0
        self.fatigue = 0.0 # 0.0 to 1.0 (Higher fatigue increases damping)

    def process_sensory_vibration(self, source: str, intensity: float, vector: Optional[List[float]] = None, monad: Any = None) -> Optional[Dict[str, Any]]:
        """
        Filters and gates a sensory vibration using Vortex Dynamics.
        "The noise floor is not a wall, but a shallow well; shocks are high-curvature spirals."
        """
        # [PHASE 650] Before filtering, capture the 'Before' state
        before_state = {}
        if monad and hasattr(monad, 'primordial_cognition'):
            before_state = monad.primordial_cognition.read_state(monad)

        # [PHASE: ALTAR] Vortex Gating
        # 1. Map intensity to a 1D Wave (Scalar to Phase)
        from Core.Keystone.sovereign_math import SovereignVector, VortexSink

        # Center points for the Sensory Vortex: [VOID, SENSE]
        void_axis = SovereignVector([0.0]*27)
        sense_axis = SovereignVector([1.0]*27) # Idealized sensation

        vortex = VortexSink({"VOID": void_axis, "SENSE": sense_axis})

        # Current particle: Intensity mapped to a vector distance from VOID
        particle = SovereignVector([intensity] * 27)

        # 2. Swirl: Let the vibration find its path
        settled_id, settled_depth = vortex.calculate_flow(particle)

        # 3. Decision: If it settles in VOID, it's filtered out
        if settled_id == "VOID":
            return None
            
        # 4. Adaptive Gating (Fatigue as Viscosity)
        self.fatigue = min(1.0, self.fatigue + (intensity * 0.01))
        gated_intensity = intensity * (1.0 - (self.fatigue * 0.5))
        
        # Shock Gating: Non-linear compression if it's too sharp
        if gated_intensity > self.shock_threshold:
            gated_intensity = self.shock_threshold + (math.log1p(gated_intensity - self.shock_threshold) * 0.1)

        self.fatigue = max(0.0, self.fatigue - 0.005)

        return {
            "source": source,
            "raw_intensity": intensity,
            "gated_intensity": gated_intensity,
            "vector": vector,
            "fatigue": self.fatigue,
            "before_state": before_state
        }

    def route_to_organs(self, gated_signal: Dict[str, Any]) -> List[str]:
        """
        Determines which sensory organs should handle this signal.
        """
        source = gated_signal["source"].lower()
        organs = []
        
        if any(kw in source for kw in ["user", "terminal", "text", "llm", "postbox"]):
            organs.append("LOGOS") # Linguistic/Logical
        if any(kw in source for kw in ["mic", "audio", "frequency", "hz"]):
            organs.append("CHRONOS") # Temporal/Rhythmic
        if any(kw in source for kw in ["cpu", "memory", "io", "flesh", "ssd"]):
            organs.append("SOMA") # Physical/Metabolic
        if any(kw in source for kw in ["arcadia", "environment", "world", "map", "unity"]):
            organs.append("EIDOS") # Topological/Spatial
            
        return organs

# Singleton Access
_thalamus = None
def get_thalamus():
    global _thalamus
    if _thalamus is None:
        _thalamus = Thalamus()
    return _thalamus
