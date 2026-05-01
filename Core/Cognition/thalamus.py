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

    def process_sensory_vibration(self, source: str, intensity: float, vector: Optional[List[float]] = None) -> Optional[Dict[str, Any]]:
        """
        Filters and gates a sensory vibration before it reaches the Monad.
        
        Returns:
            Gated vibration packet or None if blocked.
        """
        # 1. Noise Filtering
        if intensity < self.noise_floor:
            return None
            
        # 2. Fatigue Check
        # Constant high intensity increases fatigue, which increases damping
        self.fatigue = min(1.0, self.fatigue + (intensity * 0.01))
        damping = 1.0 - (self.fatigue * 0.5)
        
        # 3. Shock Gating
        # If intensity is too high, we apply a non-linear squashing function (Somatic Reflex)
        gated_intensity = intensity * damping
        if gated_intensity > self.shock_threshold:
            # Compression: Squashing the spike to protect the manifold
            gated_intensity = self.shock_threshold + (math.log1p(gated_intensity - self.shock_threshold) * 0.1)
            logger.warning(f"⚡ [SHOCK GATING] Compressed high intensity signal from {source}: {intensity:.2f} -> {gated_intensity:.2f}")

        # 4. Adaptive Recovery
        # Fatigue slowly decays
        self.fatigue = max(0.0, self.fatigue - 0.005)
        
        return {
            "source": source,
            "raw_intensity": intensity,
            "gated_intensity": gated_intensity,
            "vector": vector,
            "fatigue": self.fatigue
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
