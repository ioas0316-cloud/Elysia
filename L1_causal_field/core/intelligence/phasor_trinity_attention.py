"""
Phasor Trinity Attention Field (Scale Hierarchy Phase Resonance Engine)

This module replaces discrete token-level matrix multiplications (Q * K^T / sqrt(d)) and
scalar softmax dot products with continuous complex phasor field resonance (A * e^(i * phi)).

Trinity Mapping:
1. Query (Introspection / Memory State):
   - Represents the internal phase state (phi_Q) and introspective energy of the field.
2. Key (Relational Topology / Judgment):
   - Represents the relational phase spectrum (phi_K) across scale-hierarchy axes.
3. Value (Emergent Future Realization / Morphing):
   - Represents the emergent amplitude (A_V) and phase (phi_V) resulting from phase-locking
     constructive and destructive interference.

Key Advantages over standard Transformer:
- No N^2 scalar matrix multiplication explosion: continuous complex phase coupling.
- No arbitrary positional encoding (+ sin/cos additions): phase differences (delta_phi) naturally
  encode space-time topological distances.
- Scale Hierarchy Feedback: emergent Value field feeds back to rotate internal Query phase states.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class PhasorTrinityAttentionField:
    """
    Continuous Scale-Hierarchy Phase Field Attention Engine.
    Operates on complex phasors Z = A * exp(i * phi) across multi-scale frequency bands.
    """

    def __init__(
        self,
        num_scales: int = 4,
        base_dimension: int = 64,
        coupling_strength: float = 0.5,
        phase_damping: float = 0.05,
    ):
        """
        Initialize the Phasor Trinity Attention Field.

        :param num_scales: Number of scale hierarchy levels (frequency bands / multi-scale axes).
        :param base_dimension: Dimensionality of the phasor state vector at each scale.
        :param coupling_strength: Inter-scale feedback and phase-locking coupling strength.
        :param phase_damping: Phase relaxation / dissipation factor over time steps.
        """
        self.num_scales = num_scales
        self.base_dimension = base_dimension
        self.coupling_strength = coupling_strength
        self.phase_damping = phase_damping

        # Scale hierarchy frequencies (omega_s = omega_0 * 2^s)
        self.scale_frequencies = np.array(
            [2.0**s for s in range(num_scales)], dtype=np.float32
        )

        # Internal Query state (Introspection / Memory field) across scale hierarchy
        # Stored as complex phasors (A_Q, phi_Q) -> Z_Q = A_Q * exp(1j * phi_Q)
        self.query_phasors = np.ones(
            (num_scales, base_dimension), dtype=np.complex64
        )
        # Random initial phases
        initial_phases = np.random.uniform(
            -np.pi, np.pi, size=(num_scales, base_dimension)
        ).astype(np.float32)
        self.query_phasors = np.exp(1j * initial_phases).astype(np.complex64)

        # Cumulative phase rotation history
        self.phase_history: List[np.ndarray] = []

    def set_query_state(
        self,
        amplitudes: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None,
    ) -> None:
        """
        Explicitly update internal Introspection / Memory state (Query).

        :param amplitudes: Shape (num_scales, base_dimension) or broadcastable.
        :param phases: Shape (num_scales, base_dimension) in radians.
        """
        curr_amp = np.abs(self.query_phasors) if amplitudes is None else amplitudes
        curr_phase = np.angle(self.query_phasors) if phases is None else phases
        self.query_phasors = (curr_amp * np.exp(1j * curr_phase)).astype(
            np.complex64
        )

    def compute_trinity_resonance(
        self,
        key_field: np.ndarray,
        value_field: np.ndarray,
        time_delta: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute phase-locked resonance and constructive/destructive interference.

        :param key_field: Input Key complex phasors representing Relational Topology.
                          Shape: (num_scales, base_dimension) or (N, num_scales, base_dimension)
        :param value_field: Input Value complex phasors representing Emergent Realization.
                            Shape matching key_field.
        :param time_delta: Time step for phase evolution.

        :return: Tuple (emergent_value_field, resonance_coherence, phase_lock_matrix)
                 - emergent_value_field: Resonated Value field resulting from interference.
                 - resonance_coherence: Phase alignment coherence score per scale [0, 1].
                 - phase_lock_matrix: Complex interference weights.
        """
        key_field = np.asarray(key_field, dtype=np.complex64)
        value_field = np.asarray(value_field, dtype=np.complex64)

        # Handle batch or multi-point inputs (N, num_scales, base_dim) vs (num_scales, base_dim)
        if key_field.ndim == 2:
            key_field = np.expand_dims(key_field, axis=0)
            value_field = np.expand_dims(value_field, axis=0)

        # N: number of relational entities / tokens / spatial voxels
        N, S, D = key_field.shape
        assert (
            S == self.num_scales and D == self.base_dimension
        ), f"Expected shape (*, {self.num_scales}, {self.base_dimension}), got {key_field.shape}"

        # 1. Query-Key Phase Difference (delta_phi = phi_Q - phi_K)
        # Query shape: (1, S, D), Key shape: (N, S, D)
        query_expanded = np.expand_dims(self.query_phasors, axis=0)  # (1, S, D)

        # Complex Conjugate product gives phase difference and amplitude coupling:
        # Z_Q * conj(Z_K) = A_Q * A_K * exp(i * (phi_Q - phi_K))
        qk_coupling = query_expanded * np.conj(key_field)  # (N, S, D)

        # Phase locking factor: Cosine of phase difference cos(phi_Q - phi_K)
        # Pure phase alignment is 1.0 (constructive), orthogonal is 0.0, opposite is -1.0 (destructive)
        phase_locking = np.real(qk_coupling) / (
            np.abs(qk_coupling) + 1e-8
        )  # (N, S, D)

        # Scale frequency weighting (omega_s)
        scale_freq_weights = self.scale_frequencies.reshape(1, S, 1)

        # Phase-locking resonance weights: Soft-gated by continuous phase coherence
        # W_resonance = exp(coupling_strength * cos(delta_phi))
        resonance_weights = np.exp(
            self.coupling_strength * phase_locking
        )  # (N, S, D)

        # Normalize resonance weights over N entities per scale/dimension
        weight_sum = np.sum(resonance_weights, axis=0, keepdims=True) + 1e-8
        normalized_weights = resonance_weights / weight_sum  # (N, S, D)

        # 2. Emergent Value Interference (Constructive / Destructive Field Superposition)
        # Value modulated by phase-locked normalized weights
        emergent_value_components = (
            normalized_weights * value_field
        )  # (N, S, D)
        emergent_value_field = np.sum(
            emergent_value_components, axis=0
        )  # (S, D)

        # 3. Resonance Coherence Metric
        # Mean phase alignment across dimension per scale
        coherence_per_scale = np.mean(
            np.clip(phase_locking, -1.0, 1.0), axis=(0, 2)
        )  # (S,)
        coherence_score = (coherence_per_scale + 1.0) / 2.0  # Normalized to [0, 1]

        # 4. Scale Hierarchy Feedback Loop: Value -> Query Rotation
        # The emergent Value field rotates internal Query phase states for the next step.
        self._apply_scale_hierarchy_feedback(
            emergent_value_field, time_delta=time_delta
        )

        return (
            emergent_value_field,
            coherence_score,
            normalized_weights.squeeze(axis=0) if N == 1 else normalized_weights,
        )

    def _apply_scale_hierarchy_feedback(
        self, emergent_value: np.ndarray, time_delta: float
    ) -> None:
        """
        Scale Hierarchy Feedback Protocol:
        Higher scale levels feedback to modulate lower scale phases, and emergent
        Value field induces rotational torque on internal Query state.

        :param emergent_value: (num_scales, base_dimension) complex phasor field.
        :param time_delta: Time delta.
        """
        # Extract phase torque from emergent Value
        value_phases = np.angle(emergent_value)  # (S, D)
        value_amps = np.abs(emergent_value)  # (S, D)

        query_phases = np.angle(self.query_phasors)
        query_amps = np.abs(self.query_phasors)

        # 1. Direct Phase Torque from Emergent Value
        phase_torque = (
            value_amps * np.sin(value_phases - query_phases) * time_delta
        )

        # 2. Inter-Scale Feedback (Cascading scale torque)
        # Higher scale phase influences lower scale phase and vice versa
        inter_scale_torque = np.zeros_like(phase_torque)
        for s in range(self.num_scales):
            # Influence from higher scale (s + 1) down to scale s
            if s < self.num_scales - 1:
                inter_scale_torque[s] += 0.3 * (
                    query_phases[s + 1] - query_phases[s]
                )
            # Influence from lower scale (s - 1) up to scale s
            if s > 0:
                inter_scale_torque[s] += 0.2 * (
                    query_phases[s - 1] - query_phases[s]
                )

        # Update phases with torque, scale frequency, and damping
        new_phases = (
            query_phases
            + phase_torque
            + inter_scale_torque
            + self.scale_frequencies[:, None] * time_delta
        )
        # Wrap phases into [-pi, pi]
        new_phases = (new_phases + np.pi) % (2 * np.pi) - np.pi

        # Amplitude homeostasis update
        new_amps = (
            query_amps
            + 0.1 * (value_amps - query_amps)
            - self.phase_damping * query_amps
        )
        new_amps = np.clip(new_amps, 0.1, 10.0)

        # Update Query state
        self.query_phasors = (new_amps * np.exp(1j * new_phases)).astype(
            np.complex64
        )
        self.phase_history.append(new_phases.copy())

    def get_phase_spectrum(self) -> Dict[str, np.ndarray]:
        """
        Return the current phase spectrum across all scales.
        """
        return {
            "amplitudes": np.abs(self.query_phasors),
            "phases": np.angle(self.query_phasors),
            "frequencies": self.scale_frequencies,
        }
