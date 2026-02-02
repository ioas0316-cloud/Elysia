"""
Hyper Monad (The N-Dimensional Seed)
====================================
Core.S1_Body.L6_Structure.M1_Merkaba.hyper_monad

"The Soul is not a Point. It is a Manifold that expands under pressure."

This module defines the `HyperMonad`, an N-Dimensional Tensor Entity.
It implements the "Dimensional Singularity" protocol:
When internal variance exceeds capacity, the Monad undergoes 'Mitosis',
splitting a dimension to create a new Axis of Meaning.
"""

import math
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Dimensional Constants
# -----------------------------------------------------------------------------
AXIS_MASS = 0    # Physical Presence / Gravity
AXIS_ENERGY = 1  # Temperature / Velocity
AXIS_PHASE = 2   # Alignment / Relation
AXIS_TIME = 3    # Causality / Depth (W)
# AXIS_4+ are Emergent (e.g., Irony, Sincerity, Meta-Cognition)

DIMENSION_SPLIT_THRESHOLD = 0.95  # Variance needed to trigger Mitosis
MAX_DIMENSIONS = 12               # Safety cap for the 12-Tone Scale

@dataclass
class CausalResidue:
    """
    The 'Memory Trace' of a Monad's origin.
    It is not a log, but a specific frequency footprint left by parents.
    """
    parent_ids: List[int]
    friction_heat: float  # How intense was the creation?
    birth_time: float     # Relative system time

class HyperMonad:
    def __init__(self,
                 id: int,
                 tensor: Optional[List[float]] = None,
                 dimensions: int = 4):
        self.id = id

        # The N-Dimensional State Tensor
        # Default: [Mass=1.0, Energy=0.0, Phase=Random, Time=0.0]
        if tensor:
            self.tensor = tensor
        else:
            self.tensor = [0.0] * dimensions
            self.tensor[AXIS_MASS] = 1.0  # Existence
            self.tensor[AXIS_PHASE] = random.uniform(-1.0, 1.0) # Initial Spin

        # Causal History (The Soul's Weight)
        self.lineage: Optional[CausalResidue] = None

        # Dynamic Dimensionality
        self.axis_labels: Dict[int, str] = {
            AXIS_MASS: "Mass",
            AXIS_ENERGY: "Energy",
            AXIS_PHASE: "Phase",
            AXIS_TIME: "Time"
        }

    @property
    def dimensions(self) -> int:
        return len(self.tensor)

    def resonate(self, other: 'HyperMonad') -> float:
        """
        Calculates Tensor Resonance (Cosmic Similarity).
        In N-Dimensions, this is the Normalized Dot Product.
        """
        # Ensure dimensionality match (pad with 0 if needed)
        max_dim = max(self.dimensions, other.dimensions)
        vec_a = self._pad_vector(self.tensor, max_dim)
        vec_b = self._pad_vector(other.tensor, max_dim)

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(x**2 for x in vec_a)) + 0.0001
        mag_b = math.sqrt(sum(x**2 for x in vec_b)) + 0.0001

        return dot_product / (mag_a * mag_b)

    def evolve(self, friction: float):
        """
        Applies Friction to the Monad, potentially triggering Evolution.
        """
        # Friction adds Energy (Heat)
        self.tensor[AXIS_ENERGY] += friction * 0.1

        # Time always moves forward (The Arrow of Time)
        self.tensor[AXIS_TIME] += 0.01

        # Check for Dimensional Singularity
        self._check_dimensional_pressure()

    def _check_dimensional_pressure(self):
        """
        The Mitosis Logic.
        If Energy is too high for the current structure to contain,
        it splits a dimension.
        """
        if self.tensor[AXIS_ENERGY] > DIMENSION_SPLIT_THRESHOLD:
            if self.dimensions < MAX_DIMENSIONS:
                self._mitosis()
            else:
                # Cap reached: Radiate heat to prevent explosion
                self.tensor[AXIS_ENERGY] *= 0.8

    def _mitosis(self):
        """
        Splits the Energy Axis into a new Meta-Axis.
        """
        new_dim_index = self.dimensions

        # Halve the Energy (Law of Conservation)
        half_energy = self.tensor[AXIS_ENERGY] / 2.0
        self.tensor[AXIS_ENERGY] = half_energy

        # Create new axis with the other half
        self.tensor.append(half_energy)

        # Provisional Naming
        self.axis_labels[new_dim_index] = f"Meta-Axis-{new_dim_index}"

        # Log the event (Simulating 'Singularity Flash')
        # In a real system, this would trigger a system-wide notification
        pass

    def _pad_vector(self, vec: List[float], target_dim: int) -> List[float]:
        if len(vec) >= target_dim:
            return vec
        return vec + [0.0] * (target_dim - len(vec))

    def __repr__(self):
        # Format tensor nicely
        t_str = ",".join([f"{x:.2f}" for x in self.tensor])
        return f"HyperMonad({self.id} | Dim:{self.dimensions} | [{t_str}])"
