"""Enactive Boundary Phase-Lock Engine.

This module implements a 3-layer architecture for recognizing, learning,
and synchronizing with external system boundaries through physical friction and
phase-locking dynamics:

1. InterfaceBoundaryObserver:
   Perceives external system protocol specifications (frequencies, power, bandwidth, limits)
   and internal system capacity / reach.

2. DynamicConstraintMapper:
   Dynamically updates constraint boundaries in phase space through friction feedback
   (latency, errors, collisions, thermal/voltage resistance) rather than hardcoding.

3. PhaseLockSynchronizer:
   Harmonizes internal wave dynamics with external protocol timing via topological/Kuramoto
   phase-locking without violating boundary constraints.

4. EnactiveBoundaryEngine:
   Unified engine integrating observation, constraint mapping, and phase synchronization into
   a continuous runtime loop.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class InterfaceBoundaryObserver:
    """Layer 1: Observes external protocol limits and internal reach/capacity."""

    def __init__(self, internal_capacity: float = 100.0, internal_reach: float = 10.0):
        self.internal_capacity = float(internal_capacity)
        self.internal_reach = float(internal_reach)
        self.current_load = 0.0

    def observe_external_protocol(self, protocol_spec: Dict[str, Any]) -> Dict[str, float]:
        """Perceives external protocol parameters and normalizes constraints."""
        max_freq = float(protocol_spec.get("max_frequency", 100.0))
        max_power = float(protocol_spec.get("max_power", 50.0))
        max_bandwidth = float(protocol_spec.get("max_bandwidth", 1000.0))
        latency_tolerance = float(protocol_spec.get("latency_tolerance", 0.05))

        return {
            "max_frequency": max_freq,
            "max_power": max_power,
            "max_bandwidth": max_bandwidth,
            "latency_tolerance": latency_tolerance,
        }

    def compute_boundary_clearance(
        self, active_state: np.ndarray, protocol_limits: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculates margin to system instability/boundary violation."""
        state_magnitude = float(np.linalg.norm(active_state))
        power_clearance = max(0.0, protocol_limits["max_power"] - state_magnitude)
        capacity_clearance = max(0.0, self.internal_capacity - self.current_load)

        reach_margin = max(0.0, self.internal_reach - state_magnitude)

        return {
            "power_clearance": power_clearance,
            "capacity_clearance": capacity_clearance,
            "reach_margin": reach_margin,
            "normalized_safety_ratio": min(1.0, (power_clearance + reach_margin) / (self.internal_reach + 1e-6)),
        }


class DynamicConstraintMapper:
    """Layer 2: Dynamically maps constraints in phase space using friction feedback."""

    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        # Boundary constraints represented as learned resistance thresholds
        self.learned_boundaries: Dict[str, float] = {
            "frequency_limit": 100.0,
            "power_limit": 50.0,
            "bandwidth_limit": 1000.0,
        }
        self.accumulated_friction = 0.0

    def update_from_friction(
        self, friction_feedback: Dict[str, float]
    ) -> Dict[str, float]:
        """Updates boundary limits based on interaction friction (latency, errors, collisions)."""
        latency = friction_feedback.get("latency", 0.0)
        error_rate = friction_feedback.get("error_rate", 0.0)
        resistance = friction_feedback.get("resistance", 0.0)

        # Total instant friction magnitude
        instant_friction = latency * 10.0 + error_rate * 50.0 + resistance * 5.0
        self.accumulated_friction = (1.0 - self.decay_rate) * self.accumulated_friction + instant_friction

        # High friction shrinks boundary limits (defensive boundary tightening)
        if instant_friction > 1.0:
            shrink_factor = 1.0 - self.learning_rate * min(0.5, instant_friction / 100.0)
            for k in self.learned_boundaries:
                self.learned_boundaries[k] *= shrink_factor
        else:
            # Low friction allows gradual boundary relaxation (learning true capacity)
            expand_factor = 1.0 + (self.learning_rate * 0.1)
            for k in self.learned_boundaries:
                self.learned_boundaries[k] *= expand_factor

        return {
            "instant_friction": instant_friction,
            "accumulated_friction": self.accumulated_friction,
            "effective_power_limit": self.learned_boundaries["power_limit"],
            "effective_freq_limit": self.learned_boundaries["frequency_limit"],
        }


class PhaseLockSynchronizer:
    """Layer 3: Achieves Kuramoto-style phase synchronization within learned boundaries."""

    def __init__(self, internal_freq: float = 10.0, coupling_strength: float = 0.5):
        self.internal_phase = 0.0
        self.internal_freq = internal_freq
        self.coupling_strength = coupling_strength
        self.order_parameter = 0.0

    def synchronize(
        self, external_phase: float, external_freq: float, dt: float, max_allowed_freq: float
    ) -> Tuple[float, float, float]:
        """Calculates phase update and phase locking order parameter (R).

        Returns:
            (new_internal_phase, phase_difference, order_parameter_R)
        """
        # Clamp target frequency to learned boundary limit
        effective_ext_freq = min(external_freq, max_allowed_freq)

        # Kuramoto phase difference coupling
        phase_diff = external_phase - self.internal_phase
        phase_diff_normalized = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Frequency synchronization update
        freq_pull = self.coupling_strength * np.sin(phase_diff_normalized)
        effective_freq = self.internal_freq + freq_pull

        # Phase evolution
        self.internal_phase = (self.internal_phase + effective_freq * dt) % (2 * np.pi)

        # Order parameter R = cos(phase_diff) clamped [0, 1] for coherence measure
        self.order_parameter = float(np.clip(np.cos(phase_diff_normalized), 0.0, 1.0))

        return float(self.internal_phase), float(phase_diff_normalized), self.order_parameter


class EnactiveBoundaryEngine:
    """Unified Engine integrating boundary perception, dynamic constraint mapping, and phase synchronization."""

    def __init__(
        self,
        internal_capacity: float = 100.0,
        internal_reach: float = 10.0,
        base_freq: float = 10.0,
        coupling_strength: float = 0.8,
    ):
        self.observer = InterfaceBoundaryObserver(internal_capacity, internal_reach)
        self.mapper = DynamicConstraintMapper()
        self.synchronizer = PhaseLockSynchronizer(base_freq, coupling_strength)
        self.time = 0.0

    def step(
        self,
        active_state: np.ndarray,
        protocol_spec: Dict[str, Any],
        external_phase: float,
        external_freq: float,
        friction_feedback: Dict[str, float],
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """Executes a single enactive boundary perception and phase locking step."""
        self.time += dt

        # Step 1: Perceive protocol and observe boundary clearance
        proto_limits = self.observer.observe_external_protocol(protocol_spec)
        clearance = self.observer.compute_boundary_clearance(active_state, proto_limits)

        # Step 2: Map constraints dynamically via friction feedback
        friction_metrics = self.mapper.update_from_friction(friction_feedback)

        # Combine observer protocol limits with mapped dynamic friction limits
        max_freq = min(
            proto_limits["max_frequency"], friction_metrics["effective_freq_limit"]
        )

        # Step 3: Phase locking within boundary
        internal_phase, phase_diff, R = self.synchronizer.synchronize(
            external_phase, external_freq, dt, max_freq
        )

        # Calculate mutual resonance energy
        resonance_energy = R * clearance["normalized_safety_ratio"]

        return {
            "time": self.time,
            "internal_phase": internal_phase,
            "phase_diff": phase_diff,
            "order_parameter_R": R,
            "resonance_energy": resonance_energy,
            "boundary_clearance": clearance,
            "friction_metrics": friction_metrics,
            "effective_max_frequency": max_freq,
        }
