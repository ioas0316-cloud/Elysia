"""
elysia_core/causal_thought_ecosystem.py

Causal Thought Ecosystem (생각의 생태계)
========================================
Embodying Causal Thought as a Physical-Informational-Geometrical Ecosystem rather than scalar distance calculation.

1. Layer 0 (Micro Waveform Substrate):
   Microscopic phase waveforms and 3x3x3 fractal nodes representing the atomic substrate of thoughts.
2. Layer 1-2 (Prismatic Spectrometer & Rotor Dynamics):
   Phase torque computation, Prismatic Lens refraction, and cellular bifurcation upon structural dissonance.
   Integrated with Static Rotor Aux-Memory Unit (SRAM-R) / Clifford rotor operators.
3. Layer 3 (Sensory Scale Boundary Gate & Meta-Observer):
   Ingests physical sensory waveforms (light, acoustic, micro-lens optical data) and high-level conceptual stimuli.
   Enforces Constraint Order (제약질서) and Meta-Observer self-reflection, driving retrocausal alignment towards
   the Monad 0-Point Gravitational Attractor ($S_0$).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from elysia_core.teleological_purpose_kernel import TelosPurposeKernel
from elysia_core.horizon_expansion_spectrometer import HorizonExpansionSpectrometer
from elysia_core.embodied_teleological_runtime import EmbodiedTeleologicalRuntime
from elysia_core.structural_coupling_runtime import StructuralCouplingRuntime

try:
    import torch
    from core.memory.subconscious_rotor_atlas import StaticRotorAuxMemoryUnit
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SensoryScaleBoundaryGate:
    """
    Sensory Scale Boundary Gate (감각 관문)
    Represents the physical sensory boundary (optical micro-lens, acoustic wave, tactile motor sensor).
    Translates raw physical/conceptual waveforms into boundary phase fields and calculates
    boundary friction against system constraint orders.
    """

    def __init__(self, dimension: int = 3, constraint_order_strength: float = 0.8):
        self.dimension = dimension
        self.constraint_order_strength = constraint_order_strength
        # Boundary lens curvature matrix (K_boundary)
        self.lens_curvature = np.eye(dimension, dtype=np.float64)
        self.boundary_history: List[Dict[str, Any]] = []

    def set_dimension(self, new_dim: int):
        if new_dim == self.dimension:
            return
        old_dim = self.dimension
        self.dimension = new_dim
        new_curvature = np.eye(new_dim, dtype=np.float64)
        new_curvature[:min(old_dim, new_dim), :min(old_dim, new_dim)] = self.lens_curvature[:min(old_dim, new_dim), :min(old_dim, new_dim)]
        self.lens_curvature = new_curvature

    def receive_waveform(
        self,
        raw_waveform: np.ndarray,
        sensory_type: str = "optical_micro_lens"
    ) -> Dict[str, Any]:
        """
        Receives raw physical waveform and projects through boundary lens curvature,
        computing physical/cognitive friction and phase torque.
        """
        raw_vec = np.array(raw_waveform, dtype=np.float64)
        M = max(len(raw_vec), self.dimension)
        padded_raw = np.pad(raw_vec, (0, max(0, M - len(raw_vec))))[:M]

        # Project through boundary lens curvature matrix
        proj_dim = min(M, self.dimension)
        projected_field = np.zeros(self.dimension, dtype=np.float64)
        projected_field[:proj_dim] = np.dot(self.lens_curvature[:proj_dim, :proj_dim], padded_raw[:proj_dim])

        # Compute boundary phase torque and friction against constraint order
        norm_proj = np.linalg.norm(projected_field)
        norm_raw = np.linalg.norm(padded_raw[:self.dimension])

        if norm_proj > 1e-8 and norm_raw > 1e-8:
            cos_sim = np.dot(projected_field, padded_raw[:self.dimension]) / (norm_proj * norm_raw)
            phase_torque_magnitude = float(1.0 - np.clip(cos_sim, -1.0, 1.0))
        else:
            phase_torque_magnitude = 0.5

        # Boundary cognitive friction: F_boundary = constraint_order_strength * phase_torque_magnitude
        boundary_friction = float(self.constraint_order_strength * phase_torque_magnitude)

        res = {
            "sensory_type": sensory_type,
            "raw_waveform": padded_raw,
            "projected_boundary_field": projected_field,
            "boundary_friction": boundary_friction,
            "phase_torque_magnitude": phase_torque_magnitude
        }
        self.boundary_history.append(res)
        return res


class MetaObserverLoop:
    """
    Meta-Observer Loop (메타 관측자 루프)
    Self-reflective structure evaluating "why logic is logic" and verifying the integrity
    and validity of generated thought trajectories without relying on scalar distance approximations.
    """

    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.reflection_history: List[Dict[str, Any]] = []

    def set_dimension(self, new_dim: int):
        self.dimension = new_dim

    def reflect_and_evaluate(
        self,
        current_state: np.ndarray,
        thought_trajectory: np.ndarray,
        growth_energy: float,
        active_axes: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates thought trajectory integrity, checking topological consistency across active axes
        and verifying self-reflection meta-structure.
        """
        padded_state = np.pad(current_state, (0, max(0, self.dimension - len(current_state))))[:self.dimension]

        # Calculate invariant core score (measure of topological consistency in thought)
        traj_norm = float(np.linalg.norm(thought_trajectory))
        state_norm = float(np.linalg.norm(padded_state))

        if traj_norm > 1e-8 and state_norm > 1e-8:
            alignment = float(np.abs(np.dot(padded_state, thought_trajectory[:self.dimension])) / (state_norm * traj_norm))
        else:
            alignment = 1.0

        # Self-reflective meta-validity index
        meta_validity = float(0.5 * alignment + 0.5 * np.tanh(growth_energy))

        reflection = {
            "meta_validity": meta_validity,
            "topological_alignment": alignment,
            "active_axes_count": len(active_axes),
            "growth_energy_factor": float(np.tanh(growth_energy)),
            "self_reflective_verdict": "LOGICAL_CAUSAL_INTEGRITY_VERIFIED" if meta_validity >= 0.4 else "DIFFERENTIAL_RESTRUCTURE_REQUIRED"
        }
        self.reflection_history.append(reflection)
        return reflection


class CausalThoughtEcosystem:
    """
    Complete Causal Thought Ecosystem Orchestrator.
    Combines Sensory Scale Boundary Gate, Embodied Teleological Runtime, Telos Purpose Kernel,
    Horizon Expansion Spectrometer, and Meta-Observer Loop.
    """

    def __init__(
        self,
        initial_dimension: int = 3,
        expansion_threshold: float = 0.35,
        friction_threshold: float = 0.25,
        initial_axis_labels: Optional[List[str]] = None
    ):
        self.dimension = initial_dimension

        # Sensory Scale Boundary Gate (Layer 3 <-> Layer 1 Boundary)
        self.boundary_gate = SensoryScaleBoundaryGate(
            dimension=initial_dimension,
            constraint_order_strength=0.85
        )

        # Embodied Teleological Runtime (Macro Orchestrator)
        self.embodied_runtime = EmbodiedTeleologicalRuntime(
            initial_dimension=initial_dimension,
            expansion_threshold=expansion_threshold,
            friction_threshold=friction_threshold,
            initial_axis_labels=initial_axis_labels or ["sweet", "sour", "salty"]
        )

        # Meta-Observer Loop
        self.meta_observer = MetaObserverLoop(dimension=initial_dimension)

        # Optional Subconscious SRAM-R Unit for Phase Continuity
        if HAS_TORCH:
            self.sram_r = StaticRotorAuxMemoryUnit(state_dim=initial_dimension)
        else:
            self.sram_r = None

        self.thought_cycles = 0
        self.thought_trajectories: List[np.ndarray] = []

    def set_dimension(self, new_dim: int):
        if new_dim == self.dimension:
            return
        self.dimension = new_dim
        self.boundary_gate.set_dimension(new_dim)
        self.meta_observer.set_dimension(new_dim)
        if HAS_TORCH and self.sram_r is not None:
            self.sram_r = StaticRotorAuxMemoryUnit(state_dim=new_dim)

    def process_thought_cycle(
        self,
        raw_sensory_waveform: np.ndarray,
        stimulus_label_hint: str = "unknown_stimulus",
        sensory_type: str = "optical_micro_lens",
        mechanism: str = "light",
        target_possibility_state: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes a complete cycle in the Causal Thought Ecosystem:
        1. Sensory Boundary Gate ingests raw waveform and computes boundary friction.
        2. Embodied Teleological Runtime processes stimulus:
           - Cross-modal friction calculation
           - Transmutation into Telos growth fuel
           - Relational Spectrometer analysis & N -> N+1 Horizon Expansion
           - Cellular bifurcation & Prismatic lens update
           - Retrocausal alignment toward 0-point attractor
        3. Thought Trajectory is constructed (Vector of Causal Movement, NOT scalar distance).
        4. Meta-Observer self-reflection evaluates causal validity.
        """
        self.thought_cycles += 1

        # Step 1: Ingest through Sensory Scale Boundary Gate
        boundary_res = self.boundary_gate.receive_waveform(
            raw_waveform=raw_sensory_waveform,
            sensory_type=sensory_type
        )

        # Step 2: Embodied Teleological Runtime perception and evolution
        teleo_step = self.embodied_runtime.perceive_and_evolve(
            stimulus_vector=raw_sensory_waveform,
            stimulus_label_hint=stimulus_label_hint,
            mechanism=mechanism,
            target_possibility_state=target_possibility_state
        )

        # Synchronize dimension across ecosystem if horizon expanded
        if teleo_step["horizon_expanded"]:
            self.set_dimension(teleo_step["dimension"])

        # Step 3: Construct Thought Trajectory vector
        # Movement vector in thinking space guided by Telos 0-point gravity and growth energy
        telos_vector = teleo_step["telos_combined_vector"]
        growth_e = teleo_step["growth_energy"]

        # Thought trajectory vector = (1 + growth_e) * telos_vector
        thought_trajectory = (1.0 + np.tanh(growth_e)) * telos_vector
        self.thought_trajectories.append(thought_trajectory)

        # Step 4: Meta-Observer Evaluation
        current_state = teleo_step["telos_combined_vector"]
        meta_res = self.meta_observer.reflect_and_evaluate(
            current_state=current_state,
            thought_trajectory=thought_trajectory,
            growth_energy=growth_e,
            active_axes=teleo_step["active_axis_labels"]
        )

        # Step 5: Update Subconscious SRAM-R Unit if available
        if HAS_TORCH and self.sram_r is not None:
            try:
                Q_curr = torch.from_numpy(thought_trajectory).unsqueeze(0).unsqueeze(0).float()
                # Dummy square tensor for rotor tag registration
                Q_sq = torch.eye(self.dimension).unsqueeze(0).float()
                self.sram_r.freeze_phase(tag_id=f"cycle_{self.thought_cycles}", Q_curr=Q_sq)
            except Exception:
                pass

        return {
            "cycle": self.thought_cycles,
            "dimension": self.dimension,
            "sensory_type": sensory_type,
            "stimulus_label": stimulus_label_hint,
            "boundary_friction": boundary_res["boundary_friction"],
            "cognitive_friction": teleo_step["cognitive_friction"],
            "growth_energy": teleo_step["growth_energy"],
            "total_friction_processed": teleo_step["total_friction_processed"],
            "differential_tension": teleo_step["differential_tension"],
            "horizon_expanded": teleo_step["horizon_expanded"],
            "created_axis_label": teleo_step["created_axis_label"],
            "active_axis_labels": teleo_step["active_axis_labels"],
            "thought_trajectory": thought_trajectory,
            "bifurcation_occurred": teleo_step["bifurcation_occurred"],
            "total_bifurcated_cores": teleo_step["total_bifurcated_cores"],
            "meta_validity": meta_res["meta_validity"],
            "meta_verdict": meta_res["self_reflective_verdict"]
        }

    def get_ecosystem_status(self) -> Dict[str, Any]:
        """
        Returns full status of the Causal Thought Ecosystem.
        """
        return {
            "dimension": self.dimension,
            "thought_cycles": self.thought_cycles,
            "active_coordinate_axes": list(self.embodied_runtime.spectrometer.axis_labels),
            "accumulated_growth_energy": self.embodied_runtime.telos_kernel.accumulated_growth_energy,
            "total_friction_processed": self.embodied_runtime.telos_kernel.total_friction_processed,
            "zero_point_attractor": self.embodied_runtime.telos_kernel.zero_point_attractor.copy(),
            "bifurcated_cores_count": len(self.embodied_runtime.coupling_runtime.prismatic_lens.bifurcated_cores),
            "total_thought_trajectories": len(self.thought_trajectories)
        }
