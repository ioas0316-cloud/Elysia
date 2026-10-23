import numpy as np
from typing import Dict, Any, List, Optional
from synaptic_architecture.machine_internal_world import MachineInternalWorld

class EmergentMacroAxiom:
    """
    Represents an emergent macro-invariant (Self-Emergent Axiom) captured by the Scale Lens.
    """
    def __init__(self, name: str, curvature_threshold: float, reluctance_modifier: float, boundary_cap: float):
        self.name = name
        self.curvature_threshold = curvature_threshold
        self.reluctance_modifier = reluctance_modifier
        self.boundary_cap = boundary_cap
        self.resonance_count = 0

class ScaleLensEngine:
    """
    [Scale Lens & Emergent Macro Engine]
    Transforms micro-friction fluctuations from fast-clock state transitions into emergent macro-invariants
    (Self-Emergent Axioms) via coarse-graining and temporal damping, and enforces top-down constraints.
    """
    def __init__(self, internal_world: MachineInternalWorld, damping_factor: float = 0.85, window_size: int = 20):
        self.internal_world = internal_world
        self.damping_factor = damping_factor
        self.window_size = window_size

        # Damped accumulators for coarse-graining
        self.damped_friction = 0.0
        self.damped_impedance = 0.0
        self.recent_curvatures: List[float] = []

        # Self-Emergent Axioms catalog
        self.emergent_axioms: List[EmergentMacroAxiom] = []

    def observe_and_coarse_grain(self, micro_step_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observes micro-step metrics and applies slow-scale coarse-graining / temporal damping.
        """
        instant_friction = micro_step_result.get("instant_friction", 0.0)
        impedance = micro_step_result.get("impedance", 0.0)
        velocity = micro_step_result.get("velocity", np.zeros(2))

        # Damping update
        self.damped_friction = self.damping_factor * self.damped_friction + (1.0 - self.damping_factor) * instant_friction
        self.damped_impedance = self.damping_factor * self.damped_impedance + (1.0 - self.damping_factor) * impedance

        # Calculate trajectory curvature angle (micro fluctuation indicator)
        if len(self.internal_world.history) >= 2:
            prev_v = self.internal_world.history[-2]["velocity"]
            norm_curr = np.linalg.norm(velocity)
            norm_prev = np.linalg.norm(prev_v)
            if norm_curr > 1e-5 and norm_prev > 1e-5:
                cos_angle = np.clip(np.dot(velocity, prev_v) / (norm_curr * norm_prev), -1.0, 1.0)
                curvature = float(np.arccos(cos_angle))
            else:
                curvature = 0.0
        else:
            curvature = 0.0

        self.recent_curvatures.append(curvature)
        if len(self.recent_curvatures) > self.window_size:
            self.recent_curvatures.pop(0)

        mean_curvature = float(np.mean(self.recent_curvatures)) if self.recent_curvatures else 0.0

        # Check for macro axiom emergence
        self._check_axiom_emergence(mean_curvature)

        # Apply top-down constraints back to internal world
        self.apply_top_down_constraints()

        return {
            "damped_friction": self.damped_friction,
            "damped_impedance": self.damped_impedance,
            "mean_curvature": mean_curvature,
            "axiom_count": len(self.emergent_axioms)
        }

    def _check_axiom_emergence(self, mean_curvature: float):
        """
        Spontaneously crystallizes a new Self-Emergent Macro Axiom if coarse-grained curvature/impedance passes threshold.
        """
        if self.damped_impedance > 0.1 and (mean_curvature > 0.1 or self.damped_friction > 0.1):
            axiom_name = f"MacroConstraint_ImpedanceCap_{len(self.emergent_axioms)+1}"
            # Avoid duplicate creation if similar axiom already exists
            if not any(a.name == axiom_name for a in self.emergent_axioms):
                new_axiom = EmergentMacroAxiom(
                    name=axiom_name,
                    curvature_threshold=mean_curvature,
                    reluctance_modifier=1.25,
                    boundary_cap=float(np.mean(self.internal_world.boundary_limits) * 0.9)
                )
                self.emergent_axioms.append(new_axiom)

    def apply_top_down_constraints(self):
        """
        Top-Down Constraint Mechanism: Mutates reluctance field and boundary limits of the micro internal world.
        """
        for axiom in self.emergent_axioms:
            axiom.resonance_count += 1
            # Scale up reluctance to restrict chaotic micro movements
            self.internal_world.reluctance_field *= (1.0 + 0.02 * (axiom.reluctance_modifier - 1.0))
            # Constrain boundary limits
            self.internal_world.boundary_limits = np.minimum(
                self.internal_world.boundary_limits,
                axiom.boundary_cap
            )
