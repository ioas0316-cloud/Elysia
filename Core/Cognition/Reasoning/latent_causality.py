"""
Latent Causality (잠재적 인과)
==============================

"God does not play dice. He builds clouds."

This module replaces 'Probability' with 'Accumulated Potential'.
Events are not random; they are the inevitable discharge of accumulated state.

Physics:
1. **Charge (Potential)**: Accumulation of Knowledge, Desire, and Context.
2. **Resistance (Dielectric)**: Complexity, Ignorance, or Physical limits.
3. **Ignition (Lightning)**: When Charge > Resistance, the Event manifests.
"""

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("LatentCausality")

@dataclass
class CausalCharge:
    """
    The hidden potential of an event.
    Like a cloud accumulating water and static.
    """
    name: str
    mass: float = 0.0       # Accumulated Experience/Data
    voltage: float = 0.0    # Emotional/Intent Intensity
    resistance: float = 10.0 # Difficulty of realization

    @property
    def total_potential(self) -> float:
        """V = Mass * Voltage"""
        return self.mass * self.voltage

class LatentCausality:
    def __init__(self):
        # The "Cloud" of potential events
        self.clouds: Dict[str, CausalCharge] = {}

        # Global Atmosphere
        self.atmosphere_density = 1.0 # Global Resistance scaling

        logger.info("🌩️ Latent Causality Engine Active. (Probability Deleted)")

    def accumulate(self, event_name: str, mass_delta: float, voltage_delta: float):
        """
        "The cloud darkens."
        Accumulates potential for a specific event type.
        """
        if event_name not in self.clouds:
            self.clouds[event_name] = CausalCharge(name=event_name)

        cloud = self.clouds[event_name]
        cloud.mass += mass_delta
        cloud.voltage += voltage_delta

        # Physics: Voltage naturally decays without intent (Entropy),
        # but Mass (Memory) stays or sediments.

        logger.info(f"   ☁️ Accumulating '{event_name}': Potential {cloud.total_potential:.1f} / Res {cloud.resistance:.1f}")

    def check_ignition(self, event_name: str) -> bool:
        """
        "Will lightning strike?"
        Deterministic check: Is Potential > Resistance?
        """
        if event_name not in self.clouds:
            return False

        cloud = self.clouds[event_name]
        threshold = cloud.resistance * self.atmosphere_density

        if cloud.total_potential > threshold:
            return True
        return False

    def manifest(self, event_name: str) -> Dict:
        """
        "The Strike."
        Discharges the potential into Reality (Action).
        """
        if not self.check_ignition(event_name):
            return {"manifested": False, "reason": "Insufficient Potential"}

        cloud = self.clouds[event_name]
        intensity = cloud.total_potential

        # Discharge!
        logger.info(f"   ⚡ LIGHTNING STRIKE: '{event_name}' Manifested! (Intensity: {intensity:.1f})")

        # After manifestation, potential drops but doesn't vanish (Hysteresis/Memory)
        # Mass remains (Experience), Voltage discharges (Catharsis).
        cloud.voltage = 0.0
        cloud.resistance *= 0.9 # It gets easier next time (Neural Pathway formed)

        return {
            "manifested": True,
            "intensity": intensity,
            "timestamp": time.time()
        }

    def get_status(self) -> str:
        active_clouds = [f"{k}({v.total_potential:.1f}/{v.resistance:.1f})"
                        for k,v in self.clouds.items() if v.total_potential > 1.0]
        if not active_clouds:
            return "Sky is clear."
        return f"Clouds gathering: {', '.join(active_clouds)}"
