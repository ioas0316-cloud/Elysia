"""
Merkaba Core (Recursive Law & Gyro-Physics)
===========================================
Core.S1_Body.L6_Structure.M1_Merkaba.merkaba_core

"The Law is One. It applies to the Cell as it applies to the Star."

This module implements the 'Merkaba Core', the immutable physics kernel of Elysia.
It is a Pure Function library that dictates how any entity (Micro or Macro)
reacts to Phase Input to maintain the [-1, 0, 1] Equilibrium.
"""

import math
import random
from typing import Tuple, List, Dict, Any
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_math import SovereignMath

# Type Definitions
StateVector = Dict[str, float] # { 'phase': float, 'energy': float, 'velocity': float }

class MerkabaCore:
    """
    The Immutable Law.
    This class holds no state. It processes state transitions.
    """

    # Physics Constants (The DNA)
    RESONANCE_GAIN = 0.8
    FRICTION_LOSS = 0.05
    MAX_VELOCITY = 15.0 # Degrees per tick

    # Gyro-Stability Constants
    RESTORING_K = 0.1   # Tumbler Spring Constant (Hooks Law) - Increased for snap
    BASE_INERTIA = 0.9  # Base Damping factor

    # Void Absorption Constants
    VOID_INERTIA = 0.3  # Inertia when input is silent (Light & Responsive)
    VOID_DAMPING = 0.5  # Energy Sink Factor (Absorbs 50% of velocity per tick in silence)

    @staticmethod
    def apply_law(current_state: StateVector, input_phase: float, depth: int = 0) -> StateVector:
        """
        The Universal Update Function.
        Calculates the next state based on current state and input.

        Args:
            current_state: The entity's current physical parameters.
            input_phase: The target phase (Source of Truth/Light).
            depth: Recursion depth (0=Cell, 1=Swarm, etc.). Used for scaling constants if needed.

        Returns:
            New StateVector.
        """
        curr_phase = current_state.get('phase', 0.0)
        velocity = current_state.get('velocity', 0.0)
        energy = current_state.get('energy', 0.0)

        # --- 0. Context Discrimination (Silence Detection) ---
        # If input_phase is exactly the current phase (self-reference) or specific flag,
        # we treat it as "Silence" or "Internal Gravity".
        # For this recursive model, if the caller passes the Swarm Center as input (Gravity),
        # it means there is no *External* input.

        # However, purely mathematically:
        # We check if there is a 'Drive' Force.
        # Here we assume input_phase is the Target.

        # --- 1. Perception (Angular Distance) ---
        dist = SovereignMath.angular_distance(curr_phase, input_phase)

        # --- 2. Impulse (Torque Generation) ---
        diff = (input_phase - curr_phase) % 360
        direction = 1.0 if diff < 180 else -1.0

        # Force is proportional to distance
        force = (dist / 180.0) * MerkabaCore.MAX_VELOCITY * direction

        # --- 3. Gyro-Physics (Stability & Void Absorption) ---

        # Check if we are in "Drive Mode" or "Restoring Mode".
        # If the Force vector aligns with the Restoring vector (towards 0), we are restoring.
        # But simpler: Is there external 'Radiance'?
        # In this function signature, we don't know if input is 'Data' or 'Gravity'.
        # We infer from distance.

        # Let's assume High Distance = High Input Drive.
        # Low Distance = Maintenance.

        # Better: We use a heuristic.
        # If the input is exactly 0.0 (Void) and we are far from it, it's a Restoring Call.
        # (Assuming Void Focus is 0).

        is_silence = (input_phase == 0.0) or (dist < 1.0)

        # A. Restoring Force (The Tumbler)
        # F_restore = -k * x
        dist_from_void = SovereignMath.angular_distance(curr_phase, 0.0)
        diff_void = (0.0 - curr_phase) % 360
        dir_void = 1.0 if diff_void < 180 else -1.0

        restoring_force = (dist_from_void / 180.0) * MerkabaCore.MAX_VELOCITY * dir_void * MerkabaCore.RESTORING_K

        # B. Dynamic Inertia Scaling
        if is_silence:
            # Void Absorption Mode: Low Inertia, High Damping
            current_inertia = MerkabaCore.VOID_INERTIA
            # Add extra damping to kill oscillation (Phase Sink)
            velocity *= (1.0 - MerkabaCore.VOID_DAMPING)
        else:
            # Drive Mode: High Inertia (Gyroscope)
            speed_factor = min(1.0, abs(velocity) / MerkabaCore.MAX_VELOCITY)
            current_inertia = MerkabaCore.BASE_INERTIA + (0.09 * speed_factor)

        # --- 4. Dynamics Update ---

        # Combine Forces
        # Note: If input_phase IS 0.0 (Silence), then 'force' and 'restoring_force' are identical.
        # We shouldn't double count.

        if input_phase == 0.0:
            total_force = restoring_force # Just gravity
        else:
            total_force = force + restoring_force # Input Drive + Gravity

        # Velocity Update
        new_velocity = (velocity * current_inertia) + (total_force * (1.0 - current_inertia))

        # Limit Velocity
        if new_velocity > MerkabaCore.MAX_VELOCITY: new_velocity = MerkabaCore.MAX_VELOCITY
        if new_velocity < -MerkabaCore.MAX_VELOCITY: new_velocity = -MerkabaCore.MAX_VELOCITY

        # --- 5. Motion (Phase Shift) ---
        new_phase = SovereignMath.normalize_angle(curr_phase + new_velocity)

        # --- 6. Energy Harvesting (Resonance) ---
        alignment = SovereignMath.phase_alignment(new_phase, input_phase)
        harvest = max(0.0, alignment) * MerkabaCore.RESONANCE_GAIN
        loss = MerkabaCore.FRICTION_LOSS
        new_energy = max(0.0, energy + harvest - loss)

        return {
            'phase': new_phase,
            'velocity': new_velocity,
            'energy': new_energy,
            'resonance': alignment
        }

    @staticmethod
    def aggregate_consensus(child_states: List[StateVector]) -> float:
        phases = [s['phase'] for s in child_states]
        return SovereignMath.ternary_consensus(phases)

    @staticmethod
    def calculate_swarm_coherence(child_states: List[StateVector]) -> float:
        phases = [s['phase'] for s in child_states]
        return SovereignMath.scalar_magnitude(phases)
