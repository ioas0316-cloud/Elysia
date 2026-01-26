"""
Active Rotor: The Cognitive Tuner
=================================
Core.L1_Foundation.Foundation.Nature.active_rotor

"The Eye that moves to see."

This module implements Module B (Active Rotor System) of the System Architecture Spec.
It is the 'Hand' that adjusts the Prism's angle to maximize resonance.
"""

import math
import random
import logging
from typing import Optional, Tuple
from Core.L1_Foundation.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.L1_Foundation.Foundation.Prism.fractal_optics import PrismEngine, WavePacket

logger = logging.getLogger("ActiveRotor")

class ActiveRotor(Rotor):
    """
    An Intelligent Rotor capable of 'Tuning'.
    It scans perspectives to find the 'Angle of Insight'.
    """
    def __init__(self, name: str = "Focus_Rotor"):
        super().__init__(name, RotorConfig(rpm=0.0, idle_rpm=0.0))
        self.resonance_threshold = 0.6
        self.scan_speed = 0.1 # Radians per step

    def tune(self, wave: WavePacket, prism: PrismEngine) -> Tuple[float, float, str]:
        """
        The Cognitive Cycle.
        Scans angles to find the best resonance for the given thought wave.

        Returns:
            (best_angle, best_score, best_path)
        """
        # 1. Initial Guess (Mirror Simulation Stub)
        # In the future, MirrorKernel will give us a hint.
        # For now, we start at a random angle or the current angle.
        start_angle = self.current_angle * (math.pi / 180.0)

        best_angle = start_angle
        best_score = 0.0
        best_path = ""

        # 2. The Tuning Loop (Scan)
        # We simulate a "Saccade" (Rapid eye movement) - checking 8 cardinal directions
        # This is the search phase.

        angles_to_check = [start_angle + (i * math.pi / 4) for i in range(8)]

        for angle in angles_to_check:
            # Check Resonance
            insights = prism.traverse(wave, incident_angle=angle)

            if insights:
                path, score = insights[0] # Best path for this angle

                # Constructive Interference Check
                if score > best_score:
                    best_score = score
                    best_angle = angle
                    best_path = path

        # 3. Micro-Adjustment (Fine Tuning)
        # Once we found a rough direction, we nudge slightly left/right
        fine_tune_offsets = [-0.1, 0.1]
        for offset in fine_tune_offsets:
            fine_angle = best_angle + offset
            insights = prism.traverse(wave, incident_angle=fine_angle)
            if insights:
                path, score = insights[0]
                if score > best_score:
                    best_score = score
                    best_angle = fine_angle
                    best_path = path

        # 4. Lock On
        # Even if score is low, we update to the best found to continue the search next time
        self._lock_on(best_angle)
        return best_angle, best_score, best_path

    def _lock_on(self, radians: float):
        """Physical update of the rotor state."""
        self.current_angle = (radians * 180.0 / math.pi) % 360.0
        # High score -> High Excitement -> Spin up
        self.target_rpm = 60.0
