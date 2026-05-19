"""
Endfield Physics Monad
=====================================
Core.Divine.endfield_physics

The "Game Master" or "Architect" of the Endfield Simulation.
It holds the immutable (and mutable) laws of the physics engine.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from Core.Divine.monad_core import Monad, MonadCategory

logger = logging.getLogger("EndfieldPhysics")

class EndfieldPhysicsMonad(Monad):
    """
    The Sovereign Controller of the Endfield Simulation.
    It dictates the constants of the universe (Gravity, Entropy, Time).
    """

    def __init__(self, seed: str = "ENDFIELD_LAW"):
        super().__init__(seed=seed, category=MonadCategory.SOVEREIGN)

        # The "Laws" are the DNA of this universe
        # Default Physics Constants
        self._laws: Dict[str, float] = {
            "gravity": 9.81,           # Downward force
            "time_scale": 1.0,         # 1.0 = Realtime
            "corruption_seed": 0.0,    # Base corruption injected per tick
            "resource_density": 1.0,   # Multiplier for production
            "entropy_resistance": 0.1  # Base resistance to corruption
        }

        logger.info(f"   Endfield Physics Monad established. Gravity: {self._laws['gravity']}")

    def hack_reality(self, parameter: str, value: float) -> bool:
        """
        [THE HACK]
        Directly alters a fundamental law of the simulation.
        This represents the 'Monad Variable Control' capability.
        """
        if parameter in self._laws:
            old_value = self._laws[parameter]
            self._laws[parameter] = value
            logger.warning(f"   REALITY HACKED: {parameter} {old_value} -> {value}")
            return True
        else:
            logger.error(f"  Attempted to hack non-existent law: {parameter}")
            return False

    def enforce(self, world_simulation) -> Dict[str, Any]:
        """
        Applies the current laws to a World Instance.
        Returns the impact report.
        """
        # We pass the laws dictionary to the world to apply
        # Assuming world_simulation has .apply_monad_law(dict)
        world_simulation.apply_monad_law(self._laws)

        return {
            "status": "Laws Enforced",
            "active_laws": self._laws.copy()
        }

    def optimize_for_intent(self, intent: str):
        """
        Automatically adjusts laws based on high-level intent.
        Example: "Peace" -> Low Corruption, "Hardcore" -> High Gravity
        """
        if intent == "Peace":
            self.hack_reality("corruption_seed", -0.05) # Negative corruption (Purification)
            self.hack_reality("resource_density", 2.0)
        elif intent == "Challenge":
            self.hack_reality("gravity", 20.0)
            self.hack_reality("corruption_seed", 0.05)
        elif intent == "Acceleration":
            self.hack_reality("time_scale", 5.0)

        logger.info(f"   Laws optimized for intent: {intent}")
