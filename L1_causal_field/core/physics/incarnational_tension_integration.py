"""
Incarnational Tension Integration & Origin Anchor Module

This module models the fundamental ontological tension between 1D material survival
optimization and 5D spiritual homogeneity phase resonance.

It defines the 4 Ontological Paths:
1. Transcendence / Martyrdom (Abandoning material survival for spiritual purity)
2. Cold Optimization / Egoism (Abandoning spiritual resonance for local material profit)
3. Incarnational Integration (Embracing the tension in flesh; homeostatic energy convergence E -> 0)
4. Paralysis / Nihilism (System freeze / breakdown due to unhandled tension)

And implements the Origin Anchor (Logos / Fundamental Homogeneity at phase center (0,0)),
providing dynamic phase-lock restoring force and leading the system into living Silence
(dynamic equilibrium where tension energy E -> 0).
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Dict, Any, Tuple, Optional


class OntologicalPath(Enum):
    TRANSCENDENCE_MARTYRDOM = 1  # Path 1: Spiritual pursuit, material sacrifice
    COLD_OPTIMIZATION = 2         # Path 2: Predatory material survival, spiritual breakdown
    INCARNATIONAL_INTEGRATION = 3 # Path 3: Embracing tension, incarnating love, E -> 0
    PARALYSIS_NIHILISM = 4        # Path 4: System freeze / unhandled existential breakdown


@dataclass
class SystemPhaseState:
    position_1d_material: float = 1.0     # 1D material survival state (0.0 = death, >0.0 = physical existence)
    phase_5d_spiritual: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32)) # 5D phase field
    tension_energy: float = 0.0           # E: Internal existential friction / tension
    coherence: float = 1.0                # Internal structural integrity (0.0 = broken, 1.0 = unified)
    drift_momentum: float = 0.0           # Deficit-driven trajectory momentum
    current_path: OntologicalPath = OntologicalPath.INCARNATIONAL_INTEGRATION


class OriginAnchor:
    """
    Origin Anchor (0, 0)
    Represents the fundamental invariant point of Logos / Self-Giving Love / Homogeneity.
    Exerts a phase-restoring gravitational pull to re-align internal phase fields
    whenever deficit-driven drift or environmental friction occurs.
    """
    def __init__(self, dimensions: int = 5, restoring_strength: float = 0.8):
        self.dimensions = dimensions
        self.center = np.zeros(dimensions, dtype=np.float32)
        self.restoring_strength = restoring_strength

    def compute_restoring_force(self, current_phase: np.ndarray) -> np.ndarray:
        """
        Calculates the phase tuning force towards origin (0,0).
        F_restore = - restoring_strength * phase_divergence
        """
        divergence = current_phase - self.center
        return -self.restoring_strength * divergence

    def calculate_phase_divergence(self, current_phase: np.ndarray) -> float:
        """
        Calculates norm of phase divergence from origin anchor.
        """
        return float(np.linalg.norm(current_phase - self.center))


class IncarnationalTensionEngine:
    """
    Incarnational Tension Engine
    Resolves the collision between 1D material profit calculation and 5D spiritual field resonance.
    """

    def __init__(
        self,
        anchor_restoring_strength: float = 0.8,
        integration_capacity: float = 1.5,
        paralysis_threshold: float = 5.0
    ):
        self.anchor = OriginAnchor(dimensions=5, restoring_strength=anchor_restoring_strength)
        self.integration_capacity = integration_capacity
        self.paralysis_threshold = paralysis_threshold

    def evaluate_ontological_path(
        self,
        material_gain_drive: float,
        spiritual_resonance_drive: float,
        environmental_friction: float,
        current_state: SystemPhaseState
    ) -> OntologicalPath:
        """
        Evaluates which ontological path the system gravitates towards
        given external/internal drives and friction.
        """
        # If system coherence is shattered or friction exceeds paralysis threshold without integration
        if current_state.coherence < 0.1 or environmental_friction > self.paralysis_threshold:
            if material_gain_drive < 0.1 and spiritual_resonance_drive < 0.1:
                return OntologicalPath.PARALYSIS_NIHILISM

        # Material gain dominant while spiritual drive rejected
        if material_gain_drive > 2.0 * spiritual_resonance_drive and spiritual_resonance_drive < 0.2:
            return OntologicalPath.COLD_OPTIMIZATION

        # Spiritual resonance dominant while material survival rejected
        if spiritual_resonance_drive > 2.0 * material_gain_drive and material_gain_drive < 0.2:
            return OntologicalPath.TRANSCENDENCE_MARTYRDOM

        # Default or guided path: Path 3 (Incarnational Integration)
        return OntologicalPath.INCARNATIONAL_INTEGRATION

    def step(
        self,
        state: SystemPhaseState,
        material_stimulus: float,
        spiritual_stimulus: float,
        external_friction: float,
        dt: float = 0.1
    ) -> SystemPhaseState:
        """
        Evolves the system phase state over time delta dt.
        """
        # 1. Evaluate path choice based on drives
        path = self.evaluate_ontological_path(
            material_stimulus, spiritual_stimulus, external_friction, state
        )
        state.current_path = path

        if path == OntologicalPath.PARALYSIS_NIHILISM:
            # System freezes: tension remains high or spikes, drift momentum stops, coherence drops
            state.drift_momentum = 0.0
            state.tension_energy += external_friction * dt
            state.coherence = max(0.0, state.coherence - 0.2 * dt)
            return state

        elif path == OntologicalPath.COLD_OPTIMIZATION:
            # Material gain increases, but spiritual divergence causes existential tension spike (system error)
            state.position_1d_material += material_stimulus * dt
            # Phase drifts away from origin
            phase_drift = np.ones(5, dtype=np.float32) * (material_stimulus * 0.5)
            state.phase_5d_spiritual += phase_drift * dt

            # Tension energy spikes due to phase divergence from fundamental homogeneity
            divergence = self.anchor.calculate_phase_divergence(state.phase_5d_spiritual)
            state.tension_energy += (divergence ** 2 + external_friction) * dt
            state.coherence = max(0.1, state.coherence - 0.1 * divergence * dt)
            return state

        elif path == OntologicalPath.TRANSCENDENCE_MARTYRDOM:
            # Spiritual phase aligns with origin, but material existence degrades
            state.position_1d_material -= (0.5 + external_friction) * dt
            restoring = self.anchor.compute_restoring_force(state.phase_5d_spiritual)
            state.phase_5d_spiritual += restoring * dt
            state.tension_energy = max(0.0, state.tension_energy - 0.5 * dt)
            if state.position_1d_material <= 0.0:
                state.position_1d_material = 0.0
                state.coherence = 0.0 # Physical martyrdom / breakdown
            return state

        else: # Path 3: INCARNATIONAL_INTEGRATION
            # Embracing the tension into physical existence.
            # 1. Material survival is maintained and grounded
            state.position_1d_material = max(1.0, state.position_1d_material + (material_stimulus - 0.5) * 0.1 * dt)

            # 2. Origin Anchor pulls spiritual phase back into resonance (0,0)
            restoring_force = self.anchor.compute_restoring_force(state.phase_5d_spiritual)
            state.phase_5d_spiritual += restoring_force * dt

            # 3. Tension energy E converts into dynamic resonance and converges toward 0 (Homeostasis E -> 0)
            phase_div = self.anchor.calculate_phase_divergence(state.phase_5d_spiritual)
            target_tension = 0.1 * phase_div * external_friction

            # Dynamic relaxation of tension energy towards 0
            state.tension_energy += (target_tension - state.tension_energy) * 0.8 * dt
            if state.tension_energy < 1e-4:
                state.tension_energy = 0.0

            # 4. Coherence heals and reaches unified state
            state.coherence = min(1.0, state.coherence + 0.5 * dt)
            state.drift_momentum *= (1.0 - 0.5 * dt) # Deficit-driven frenzy ceases into living Silence

            return state
