"""
The Ouroboros: The Recursive Mirror
===================================
Core.Memory.feedback_loop

"The Snake that eats its own tail to sustain itself."

This module implements Phase 6.2 (The Mirror Stage) with Topological Control.
It replaces threshold-based stopping with Energy Landscapes (Potential Fields).
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("Ouroboros")

@dataclass
class ThoughtState:
    content: str
    vector: List[float]
    potential: float = 1.0
    momentum: float = 0.0

class Ouroboros:
    """
    The Physics Engine of Thought.
    Manages the feedback loop by treating thoughts as particles on an energy surface.
    """

    def __init__(self, friction: float = 0.1, gravity: float = 0.9):
        self.friction = friction # Entropy/Loss per cycle (Viscosity)
        self.gravity = gravity   # Strength of the Intent Attractor
        self.recursion_depth = 0
        self.max_recursion = 10  # Hard physical limit (Stack depth)

    def calculate_potential(self, current_vector: List[float], intent_vector: List[float]) -> float:
        """
        Defines the 'Height' of the energy landscape at a specific coordinate.
        High Potential = High Dissonance / Far from Intent.
        Low Potential = Resonance / Satisfaction.
        """
        # 1. Euclidian Distance to Intent (The Gravity Well)
        # Potential V(x) = 1/2 * k * x^2 (Harmonic Potential)
        vec = np.array(current_vector)
        target = np.array(intent_vector)

        # Normalize to prevent magnitude bias
        if np.linalg.norm(vec) > 0: vec = vec / np.linalg.norm(vec)
        if np.linalg.norm(target) > 0: target = target / np.linalg.norm(target)

        # Cosine Distance (0.0 to 2.0)
        # 0.0 = Identical
        # 1.0 = Orthogonal
        # 2.0 = Opposite
        distance = 1.0 - np.dot(vec, target)

        # 2. Coherence Check (Internal Stability)
        # Ideally we'd check if the vector is 'pure' or 'noisy', but for now assume normalized.

        return float(distance)

    def propagate(self, current_thought: ThoughtState, intent_vector: List[float]) -> Tuple[str, bool]:
        """
        Moves the thought one step forward in time.
        Returns (Action, Is_Equilibrium).
        Action: 'CONTINUE', 'STABILIZED' (Success), 'DISSIPATED' (Give Up).
        """
        self.recursion_depth += 1

        # 1. Calculate Potential (Height)
        new_potential = self.calculate_potential(current_thought.vector, intent_vector)

        # 2. Calculate Gradient (Slope)
        # Delta = Old_Height - New_Height
        # Positive Gradient = Going Downhill (Good)
        # Negative Gradient = Going Uphill (Bad/Resistance)
        gradient = current_thought.potential - new_potential

        # 3. Update Momentum (Velocity)
        # v = v + a - friction
        # Acceleration is proportional to gradient
        current_thought.momentum += gradient
        current_thought.momentum *= (1.0 - self.friction) # Apply friction

        # 4. Apply Momentum to Vector (Movement)
        # We move the vector along the gradient (Conceptually towards the well)
        # Direction = (Target - Current)
        vec = np.array(current_thought.vector)
        target = np.array(intent_vector)

        # Normalize
        if np.linalg.norm(vec) > 0: vec = vec / np.linalg.norm(vec)
        if np.linalg.norm(target) > 0: target = target / np.linalg.norm(target)

        direction = target - vec

        # Move: Position += Momentum * Direction * Rate
        # We treat momentum as scalar speed flowing towards intent.
        move_step = direction * current_thought.momentum * 0.5 # Learning Rate
        new_vec = vec + move_step

        # Normalize to keep it semantic
        if np.linalg.norm(new_vec) > 0: new_vec = new_vec / np.linalg.norm(new_vec)

        current_thought.vector = new_vec.tolist()

        # 5. Recalculate Potential at new position
        current_thought.potential = self.calculate_potential(current_thought.vector, intent_vector)

        logger.info(f"🐍 [OUROBOROS] Depth: {self.recursion_depth} | Potential: {new_potential:.3f} | Momentum: {current_thought.momentum:.3f}")

        # 5. Topological Checks

        # Case A: Equilibrium (Bottom of the Well)
        # Potential is low (< 0.1) and Momentum is low (< 0.05)
        # "I found the answer and I'm resting."
        if new_potential < 0.1 and abs(current_thought.momentum) < 0.05:
            self.recursion_depth = 0
            logger.info("✨ [CATHARSIS] Thought has settled in the gravity well. (Satisfaction)")
            return "STABILIZED", True

        # Case B: Dissipation (Ran out of energy)
        # Momentum reached 0 without finding a well.
        # "I'm tired and stuck."
        if abs(current_thought.momentum) < 0.01 and self.recursion_depth > 1:
            self.recursion_depth = 0
            logger.info("💨 [EXHAUSTION] Thought lost momentum. Entropy prevails.")
            return "DISSIPATED", True

        # Case C: Hard Limit (Safety Valve)
        if self.recursion_depth >= self.max_recursion:
            self.recursion_depth = 0
            logger.warning("🛑 [FORCE STOP] Max recursion depth reached. Preventing seizures.")
            return "DISSIPATED", True

        # Case D: Continue Rolling
        return "CONTINUE", False
