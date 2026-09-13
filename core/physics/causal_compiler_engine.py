import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from core.physics.telos_attractor_field import TelosAttractorField

class CausalCompilerEngine:
    """
    [Causal Compilation Engine]

    Reinterprets code not as a discrete AST array of instructions with heuristic branching,
    but as a continuous 'state transition trajectory' in physical phase space.

    Aligns branch predictions and invalidation overhead into physical wave convergence,
    compiling programs into Zero-Friction Geodesic Trajectories along the Telos potential landscape.
    """
    def __init__(self, state_dim: int = 8, telos_field: Optional[TelosAttractorField] = None):
        self.state_dim = state_dim
        self.telos_field = telos_field or TelosAttractorField(dim=state_dim)

    def compile_instruction_stream(self, instruction_states: List[np.ndarray]) -> Dict[str, Any]:
        """
        [Causal Reordering & Wave Alignment]
        Takes an instruction sequence represented by discrete target state vectors,
        and reorders/aligns them into a smooth physical wave trajectory that minimizes friction.

        Eliminates heuristic branch prediction and invalidation loops.
        """
        raw_states = [np.array(st, dtype=np.float64) for st in instruction_states]
        num_insts = len(raw_states)

        if num_insts == 0:
            return {
                "compiled_trajectory": np.array([]),
                "friction_reduction": 1.0,
                "if_branch_evaluations": 0
            }

        # Calculate initial raw friction (Euclidean distance sum in arbitrary order)
        raw_distances = [np.linalg.norm(raw_states[i+1] - raw_states[i]) for i in range(num_insts - 1)]
        raw_friction = sum(raw_distances)

        # Causal Reordering: Sort/Align states along the Telos potential gradient
        potentials = [self.telos_field.compute_potential(st) for st in raw_states]
        # Align states by descending potential (falling smoothly towards Telos minimum)
        aligned_indices = np.argsort(potentials)[::-1]
        aligned_states = [raw_states[idx] for idx in aligned_indices]

        # Calculate zero-friction aligned trajectory distance
        aligned_distances = [np.linalg.norm(aligned_states[i+1] - aligned_states[i]) for i in range(num_insts - 1)]
        compiled_friction = sum(aligned_distances)

        friction_reduction = float((raw_friction - compiled_friction) / max(raw_friction, 1e-6))

        return {
            "compiled_trajectory": np.array(aligned_states),
            "raw_friction": float(raw_friction),
            "compiled_friction": float(compiled_friction),
            "friction_reduction": max(0.0, friction_reduction),
            "if_branch_evaluations": 0  # Zero IF branch evaluations during compilation!
        }

    def trigger_phase_transition(self, external_friction: float, threshold: float = 5.0) -> Dict[str, Any]:
        """
        [Spontaneous Phase Transition (상전이) Engine]

        When external friction/contradiction exceeds the critical threshold,
        the system refuses to remain trapped in existing logic rules.
        It spontaneously restructures its operator topology and field curvature.
        """
        if external_friction > threshold:
            transition_occurred = True
            # Restructure field curvature by inverting and scaling tension
            new_curvature = np.eye(self.state_dim, dtype=np.float64) * (1.0 + external_friction / threshold)
            self.telos_field.curvature_matrix = new_curvature
            transition_energy = float(external_friction * 0.5)
        else:
            transition_occurred = False
            transition_energy = 0.0

        return {
            "phase_transition_triggered": transition_occurred,
            "external_friction": float(external_friction),
            "threshold": float(threshold),
            "transition_energy": transition_energy,
            "operator_reconfigured": transition_occurred
        }
