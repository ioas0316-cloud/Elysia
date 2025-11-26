"""
Quaternion-based consciousness lens.
Ported from Legacy/Project_Elysia/core/quaternion_consciousness.py
with minimal changes for Core integration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
from pyquaternion import Quaternion
import logging

logger = logging.getLogger("ConsciousnessLens")


@dataclass
class ConsciousnessState:
    """
    State of consciousness as a quaternion q = w + xi + yj + zk.
    W: Mastery/Anchor, X: Simulation/Dream, Y: Action/Sensation, Z: Law/Purpose.
    """
    q: Quaternion = field(default_factory=lambda: Quaternion(1, 0, 0, 0))

    @property
    def mastery(self) -> float:
        return self.q.w

    @property
    def simulation_focus(self) -> float:
        return self.q.x

    @property
    def action_focus(self) -> float:
        return self.q.y

    @property
    def purpose_alignment(self) -> float:
        return self.q.z


class ConsciousnessLens:
    """
    Lens that rotates input vectors using the current consciousness quaternion.
    Implements 'consciousness as rotation'.
    """

    def __init__(self, hyper_qubit=None):
        self.state = ConsciousnessState()
        self.truth_vector = np.array([0, 0, 1])  # North Star
        self.hyper_qubit = hyper_qubit  # Optional HyperQubit binding

    def rotate_perception(self, input_vector: List[float], source_type: str = "raw") -> np.ndarray:
        """
        Rotate a 3D vector by q: v_out = q * v_in * q_inverse.
        """
        if len(input_vector) != 3:
            raise ValueError("Input vector must be 3D (x, y, z)")

        v_in = np.array(input_vector)
        v_out = self.state.q.rotate(v_in)
        logger.debug(f"Lens Rotation [{source_type}]: {v_in} -> {v_out} (q={self.state.q})")
        return v_out

    def focus(self, target_axis: str, intensity: float):
        """Rotate towards a target axis by small intensity."""
        current_q = self.state.q
        delta_q = Quaternion(1, 0, 0, 0)

        if target_axis == 'x':
            delta_q = Quaternion(axis=[1, 0, 0], angle=intensity)
        elif target_axis == 'y':
            delta_q = Quaternion(axis=[0, 1, 0], angle=intensity)
        elif target_axis == 'z':
            delta_q = Quaternion(axis=[0, 0, 1], angle=intensity)

        self.state.q = (delta_q * current_q).normalised

    def stabilize(self, amount: float = 0.1):
        """Slerp towards identity to increase mastery/stability."""
        identity = Quaternion(1, 0, 0, 0)
        self.state.q = Quaternion.slerp(self.state.q, identity, amount=amount)

    def get_will_vector(self) -> np.ndarray:
        """Forward vector of current lens orientation."""
        return self.state.q.rotate(np.array([0, 0, 1]))

    # --- HyperQubit coupling (optional) ---
    def bind_hyper_qubit(self, qubit) -> None:
        """Attach a HyperQubit so phase state can be synchronized."""
        self.hyper_qubit = qubit

    def update_from_qubit(self) -> None:
        """
        Map HyperQubit probabilities to quaternion components (wxyz) and normalize.
        Simple projection: Point->w, Line->x, Space->y, God->z.
        """
        if not self.hyper_qubit:
            return
        probs = self.hyper_qubit.state.probabilities()
        q_raw = Quaternion(
            probs.get("Point", 1.0),
            probs.get("Line", 0.0),
            probs.get("Space", 0.0),
            probs.get("God", 0.0),
        )
        self.state.q = q_raw.normalised

    def project_to_qubit(self) -> None:
        """
        Push current quaternion proportions back into HyperQubit amplitudes.
        Keeps amplitudes normalized and updates value probabilities.
        """
        if not self.hyper_qubit:
            return
        q = self.state.q.normalised
        total = abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z)
        if total == 0:
            return
        p_w = abs(q.w) / total
        p_x = abs(q.x) / total
        p_y = abs(q.y) / total
        p_z = abs(q.z) / total
        self.hyper_qubit.state.alpha = p_w
        self.hyper_qubit.state.beta = p_x
        self.hyper_qubit.state.gamma = p_y
        self.hyper_qubit.state.delta = p_z
        self.hyper_qubit.state.normalize()
