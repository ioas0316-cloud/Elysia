import numpy as np
from typing import Dict, List, Any
from .causal_field import CausalField, InformationVoxel, ConnectivityBeam

class CausalDynamicsEngine(CausalField):
    """
    [Causal Dynamics Engine: Bridging the Missing Link]
    Implements the "Recrystallization" feedback loop where spatial tension
    actually modifies the internal state (tensor) of information.

    "Time is Causality (Force -> State Change),
     Space is Arrangement (State -> Force)."
    """

    def __init__(self, dimensions: int = 3, crystallization_rate: float = 0.01):
        super().__init__(dimensions)
        self.crystallization_rate = crystallization_rate

    def step(self, dt: float = 0.1):
        """
        Overridden step to include State Recrystallization.
        """
        # 1. Physical Phase: State -> Force -> Arrangement (Space)
        self._update_connectivity_and_tension(dt)
        self._flow_potential(dt)
        self._preserve_mobility(dt)

        # 2. Causal Phase: Arrangement (Tension) -> State Change (Time/Causality)
        self._recrystallize_states(dt)

        self._enforce_informational_continuity(dt)

    def _recrystallize_states(self, dt: float):
        """
        [The Missing Link]
        Tension in the field 'warps' the internal tensor of the information.
        Information is not static; it is molded by its context.
        """
        for beam in self.beams:
            if beam.is_broken: continue

            v_a = self.voxels[beam.source_id]
            v_b = self.voxels[beam.target_id]

            # Tension represents the 'contradiction' or 'strain' between
            # the current arrangement and the internal state.
            # We use this strain to 'pull' the tensors towards each other.

            # Direction of state change: v_b.tensor - v_a.tensor
            state_diff = v_b.tensor - v_a.tensor

            # The more tension there is, the more the state is forced to change.
            # But if tension is TOO high, it might break the beam (handled in super).
            # Here, we allow the state to 'flow' to relieve tension.

            strain_factor = beam.current_tension * self.crystallization_rate * dt

            # Update tensors (State change)
            v_a.tensor += state_diff * (strain_factor / v_a.mass)
            v_b.tensor -= state_diff * (strain_factor / v_b.mass)

            # Normalize tensors if they represent unit-norm meaning vectors
            norm_a = np.linalg.norm(v_a.tensor)
            norm_b = np.linalg.norm(v_b.tensor)
            if norm_a > 1e-9: v_a.tensor /= norm_a
            if norm_b > 1e-9: v_b.tensor /= norm_b

    def apply_causal_interference(self, target_id: str, field_distortion: np.ndarray):
        """
        Directly distorts the 'Space' around a voxel,
        which will then cause the 'State' to change in the next step.
        """
        if target_id in self.voxels:
            # We don't change the state directly, we change the position
            # and let the resulting tension drive the state change.
            self.voxels[target_id].position += field_distortion


    def _flow_potential(self, dt: float):
        """
        [Enhanced Potential Flow]
        Now potential is not just a metric, it's a 'Resonance Pressure'
        that pushes states towards equilibrium.
        """
        super()._flow_potential(dt)

        # Pressure-based state refinement
        for vid, v in self.voxels.items():
            # If potential is high, the state is 'unstable' in its current position.
            # We add a small noise/jitter to the tensor to allow for 'stochastic exploration'
            # of better states (Annealing).
            if v.potential > 0.7:
                v.tensor += np.random.normal(0, 0.05 * v.potential, size=v.tensor.shape).astype(np.float32)
                v.tensor /= (np.linalg.norm(v.tensor) + 1e-9)
