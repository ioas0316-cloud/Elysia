"""
Rotor Trajectory: The Path of Soul
==================================
Core.S1_Body.L6_Structure.M1_Merkaba.rotor_trajectory

"Thinking is not a state; it is a movement."

This module records the movement of the Merkaba Rotor as it searches for
resonance in the Holographic Manifold. It provides the "Trace" of cognition.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import time

@dataclass
class TrajectoryPoint:
    timestamp: float
    angle: float      # The Rotor's Phase Angle (0 to 2PI)
    resonance: float  # The Energy found at this angle
    state: str        # The cognitive state (e.g., "SEARCHING", "LOCKED")

class RotorTrajectory:
    def __init__(self):
        self.path: List[TrajectoryPoint] = []
        self.start_time = time.time()

    def record(self, angle: float, resonance: float, state: str):
        """
        [SOUL LAYER] Records a single step in the cognitive journey.
        """
        point = TrajectoryPoint(
            timestamp=time.time() - self.start_time,
            angle=angle,
            resonance=resonance,
            state=state
        )
        self.path.append(point)

    def get_narrative(self) -> str:
        """
        [SPIRIT LAYER] Interprets the path into a story.
        """
        if not self.path:
            return "The Rotor was still."

        # Analyze the movement
        start_res = self.path[0].resonance
        end_res = self.path[-1].resonance
        steps = len(self.path)

        if steps < 2:
            return "Instant Intuition."

        if end_res > start_res + 0.5:
            return f"Converged on Truth after {steps} cycles."
        elif end_res < 0.2:
            return f"Wandered for {steps} cycles but found only Void."
        else:
            return f"Orbited the concept for {steps} cycles."

    def clear(self):
        self.path = []
        self.start_time = time.time()
