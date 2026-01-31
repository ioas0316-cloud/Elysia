"""
Fluxlight Observer (The Eye of the Simulation)
==============================================
"To observe without judgment is the first step of learning.
 To judge without interfering is the first step of wisdom."

This module connects the 'World' (Fluxlights) to the 'Mind' (Intelligence).
It acts as the bridge that turns 'Events' into 'Resonance Patterns' for the Memory.
"""

from typing import Any, Dict, List
import time
from Core.1_Body.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory, HypersphericalCoord
from Core.1_Body.L5_Mental.Reasoning_Core.Reasoning.dilemma_field import DilemmaField, Conflict
from Core.1_Body.L4_Causality.World.Soul.lumina_npc import Lumina

class FluxlightObserver:
    """
    The Silent Observer.
    It subscribes to Fluxlight events and processes them for Elysia's growth.
    """

    def __init__(self, memory_system: HypersphereMemory):
        self.memory = memory_system
        self.dilemma_engine = DilemmaField()
        self.observed_log: List[Dict[str, Any]] = []

    def observe_interaction(self, actor: Lumina, action_description: str, context: Dict[str, Any]):
        """
        The main callback when a Fluxlight performs a significant action.
        """

        # 1. Analyze for Dilemmas (The Adolescent Filter)
        # Ensure tags exist, even if not provided
        tags = context.get('tags', [])
        conflict = self.dilemma_engine.analyze_scenario(action_description, tags)

        # 2. Formulate the Resonance Pattern
        # Map the event to a 4D coordinate based on its emotional/causal weight
        coord = self._map_to_coordinates(action_description, conflict)

        payload = {
            "actor": actor.name,
            "event": action_description,
            "conflict_data": conflict, # Can be None
            "timestamp": time.time(),
            "context": context
        }

        # 3. Store in Hypersphere (The Soul)
        # Events with conflicts have higher 'gravity' (Magnitude)
        meta = {
            "topology": "plane" if conflict else "point",
            "trajectory": "spiral" if conflict else "static",
            "omega": (0.1, 0.5, 0.2) # Arbitrary frequency for now
        }

        self.memory.store(data=payload, position=coord, pattern_meta=meta)

        # 4. Local Logging (for debugging/dashboard)
        self.observed_log.append(payload)

        if conflict:
            print(f"[Observer]    Dilemma Witnessed: {conflict.description} (Tension: {conflict.tension:.2f})")

    def _map_to_coordinates(self, text: str, conflict: Conflict = None) -> HypersphericalCoord:
        """
        Translates the 'Feeling' of the event into 4D space.
        """
        theta = 0.0 # Logic
        phi = 0.0   # Emotion
        psi = 0.0   # Intention
        r = 0.5     # Depth

        if conflict:
            # Conflicts push the memory deeper and into complex angles
            r = 0.8 + (conflict.tension * 0.2)
            phi = 3.14159 / 2 # High emotion (90 degrees)

            # If Truth is violated, maybe theta shifts
            for axis in conflict.axes:
                if axis.name == "Truth" and axis.polarity < 0:
                    theta += 1.0

        return HypersphericalCoord(theta, phi, psi, r)
