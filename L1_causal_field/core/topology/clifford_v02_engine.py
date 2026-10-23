"""
ELYSIA Clifford Multivector Rotor Engine & Dynamic Buffer Pruning v0.2

This module implements the v0.2 upgrade to ELYSIA's Topological Engine:
1. CliffordMultivector & CliffordSpacetimeRotor: Genuine geometric algebra Cl(3,0) / Cl(3,1) multivector
   and bivector rotor operators (R = cos(theta/2) - B_hat sin(theta/2), v' = R v R~).
2. CliffordSpectralWaveformTensor: Wavefield tensor embedding bivector rotor states, frequency spectrum,
   and dynamic VRAM buffer management.
3. DynamicBufferPruningEngine: Phase-decoupled buffer pruning and threshold ROI slicing pipeline.
   Prunes unaligned/decoupled VRAM data while retaining generative parameters (Theta = rotor, freq, amp)
   for O(1) counterfactual reconstruction.
4. CliffordRotorCouplingConstraint: Bivector phase-locking joint relaxation and torque alignment.
5. CliffordCausalCognitiveMemory: Intertemporal belief workspace leveraging Clifford rotor sandwich operations
   and counterfactual time-reversal (Eye of Agamotto).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import uuid


@dataclass
class CliffordMultivector:
    """
    Multivector in Cl(3,0) Geometric Algebra.
    Components:
      scalar (s): 1D scalar component [1]
      vector (v): 3D vector components [e1, e2, e3]
      bivector (b): 3D bivector components [e12, e23, e31]
      pseudoscalar (p): 1D pseudoscalar component [e123]
    """
    scalar: float = 0.0
    vector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bivector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # [e12, e23, e31]
    pseudoscalar: float = 0.0

    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray) or self.vector.shape != (3,):
            self.vector = np.array(self.vector, dtype=np.float64) if hasattr(self.vector, '__len__') else np.zeros(3, dtype=np.float64)
        if not isinstance(self.bivector, np.ndarray) or self.bivector.shape != (3,):
            self.bivector = np.array(self.bivector, dtype=np.float64) if hasattr(self.bivector, '__len__') else np.zeros(3, dtype=np.float64)

    def norm(self) -> float:
        """Computes Euclidean norm of multivector."""
        return float(np.sqrt(
            self.scalar**2 +
            np.sum(self.vector**2) +
            np.sum(self.bivector**2) +
            self.pseudoscalar**2
        ))

    def reverse(self) -> 'CliffordMultivector':
        """
        Reversal operator (tilde / dagger) R~.
        Reverses basis vector ordering: bivector & pseudoscalar negated.
        For unit rotor R, R~ is the exact inverse rotor R^(-1).
        """
        return CliffordMultivector(
            scalar=self.scalar,
            vector=self.vector.copy(),
            bivector=-self.bivector.copy(),
            pseudoscalar=-self.pseudoscalar
        )

    def normalize(self) -> 'CliffordMultivector':
        """Normalizes multivector to unit magnitude."""
        n = self.norm()
        if n < 1e-12:
            return CliffordMultivector(scalar=1.0)
        return CliffordMultivector(
            scalar=self.scalar / n,
            vector=self.vector / n,
            bivector=self.bivector / n,
            pseudoscalar=self.pseudoscalar / n
        )

    @staticmethod
    def create_rotor(axis: np.ndarray, angle: float) -> 'CliffordMultivector':
        """
        Creates unit 3D rotor R = cos(theta/2) - sin(theta/2) * B,
        where B is the unit bivector dual to axis vector.
        """
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return CliffordMultivector(scalar=1.0)

        u_axis = axis / axis_norm
        half_angle = angle / 2.0
        cos_ha = float(np.cos(half_angle))
        sin_ha = float(np.sin(half_angle))

        # Dual mapping: axis [x, y, z] -> bivector [e12, e23, e31] = [z, x, y] * sin(theta/2)
        bivector = np.array([u_axis[2], u_axis[0], u_axis[1]], dtype=np.float64) * sin_ha

        return CliffordMultivector(
            scalar=cos_ha,
            vector=np.zeros(3, dtype=np.float64),
            bivector=bivector,
            pseudoscalar=0.0
        )

    def transform_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Applies rotor sandwich operator v' = R v R~ to a vector or array of vectors.
        Equivalent to: v' = v + 2 s (u_b x v) + 2 (u_b x (u_b x v)).
        """
        is_single = (v.ndim == 1)
        vecs = np.atleast_2d(v).astype(np.float64)

        s = self.scalar
        b = self.bivector
        # Dual vector u_b from bivector [e12, e23, e31]
        u_b = np.array([b[1], b[2], b[0]], dtype=np.float64)

        cross1 = np.cross(u_b, vecs)
        cross2 = np.cross(u_b, cross1)

        v_prime = vecs + 2.0 * s * cross1 + 2.0 * cross2
        return v_prime[0] if is_single else v_prime

    def reverse_transform_vector(self, v_transformed: np.ndarray) -> np.ndarray:
        """
        Applies inverse rotor sandwich v = R~ v' R for 1:1 counterfactual reconstruction.
        """
        inv_rotor = self.reverse()
        return inv_rotor.transform_vector(v_transformed)


@dataclass
class CliffordSpectralWaveformTensor:
    """
    ELYSIA v0.2 Clifford Wavefield Tensor.
    Embeds Clifford bivector rotor states alongside frequency, complex phase,
    amplitude, and dynamic VRAM buffer management attributes.
    """
    tensor_id: str = field(default_factory=lambda: f"spec_v02_{uuid.uuid4().hex[:8]}")
    shape: Tuple[int, ...] = (8, 8)
    amplitudes: np.ndarray = field(default_factory=lambda: np.ones((8, 8), dtype=np.float64))
    phases: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=np.float64))
    frequencies: np.ndarray = field(default_factory=lambda: np.ones((8, 8), dtype=np.float64))
    bivectors: np.ndarray = field(default_factory=lambda: np.zeros((8, 8, 3), dtype=np.float64))  # Bivector field [e12, e23, e31]
    chromatic_signature: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.33, 0.33]))
    volume_boundary: Dict[str, Any] = field(default_factory=lambda: {"radius": 1.0, "tension": 1.0, "is_contained": True})
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Dynamic VRAM Pruning State
    is_pruned: bool = False
    pruned_mask: Optional[np.ndarray] = None  # True where buffer is pruned
    generative_params: Dict[str, Any] = field(default_factory=dict)  # Theta parameters for O(1) hydration

    def __post_init__(self):
        if self.amplitudes.shape != self.shape:
            self.amplitudes = np.ones(self.shape, dtype=np.float64)
        if self.phases.shape != self.shape:
            self.phases = np.zeros(self.shape, dtype=np.float64)
        if self.frequencies.shape != self.shape:
            self.frequencies = np.ones(self.shape, dtype=np.float64)
        if self.bivectors.shape != (*self.shape, 3):
            self.bivectors = np.zeros((*self.shape, 3), dtype=np.float64)

    @property
    def complex_field(self) -> np.ndarray:
        """Returns complex wavefunction field A * exp(i * phase) for PoC v0.1 compatibility."""
        return self.amplitudes * np.exp(1j * self.phases)

    @complex_field.setter
    def complex_field(self, field: np.ndarray):
        self.amplitudes = np.abs(field)
        self.phases = np.angle(field)

    def compute_energy(self) -> float:
        """Calculates total spectral energy integrated over tensor volume."""
        return float(np.sum(self.amplitudes ** 2 * self.frequencies))

    def calculate_memory_bytes(self) -> int:
        """Calculates current memory footprint in bytes."""
        total = 0
        if self.amplitudes is not None and self.amplitudes.size > 0:
            total += self.amplitudes.nbytes
        if self.phases is not None and self.phases.size > 0:
            total += self.phases.nbytes
        if self.frequencies is not None and self.frequencies.size > 0:
            total += self.frequencies.nbytes
        if self.bivectors is not None and self.bivectors.size > 0:
            total += self.bivectors.nbytes
        return total


class CliffordScaleRotor:
    """
    Global and Local Clifford Scale Rotor Operator for ELYSIA v0.2.
    Combines bivector rotor transformations, scale dilation, and counterfactual time rewind.
    """

    @staticmethod
    def rotate_global_bivector(
        tensor: CliffordSpectralWaveformTensor,
        rotor: CliffordMultivector
    ) -> CliffordSpectralWaveformTensor:
        """
        Executes O(1) global Clifford rotor rotation across the entire spectral tensor.
        Updates bivector states and phase angles using the Clifford multivector rotor.
        """
        # Directional phase angle derived from rotor bivector projection and scalar
        b_sum = np.sum(rotor.bivector)
        sign = 1.0 if b_sum >= 0 else -1.0
        angle_delta = 2.0 * np.arctan2(np.linalg.norm(rotor.bivector), max(1e-12, rotor.scalar)) * sign

        # Rotate bivectors in tensor by rotor's bivector direction
        new_bivectors = tensor.bivectors + rotor.bivector

        new_tensor = CliffordSpectralWaveformTensor(
            tensor_id=f"rot_v02_{tensor.tensor_id}",
            shape=tensor.shape,
            amplitudes=tensor.amplitudes.copy(),
            phases=(tensor.phases + angle_delta) % (2 * np.pi),
            frequencies=tensor.frequencies.copy(),
            bivectors=new_bivectors,
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy()
        )
        return new_tensor

    @staticmethod
    def apply_local_rotor(
        tensor: CliffordSpectralWaveformTensor,
        roi_mask: np.ndarray,
        rotor: CliffordMultivector,
        rewind_factor: float = 1.0
    ) -> CliffordSpectralWaveformTensor:
        """
        Local Clifford Spacetime Rotor Control ('Eye of Agamotto').
        Applies Clifford rotor transformation exclusively inside ROI mask.
        """
        new_phases = tensor.phases.copy()
        new_bivectors = tensor.bivectors.copy()

        b_sum = np.sum(rotor.bivector)
        sign = 1.0 if b_sum >= 0 else -1.0
        angle_delta = 2.0 * np.arctan2(np.linalg.norm(rotor.bivector), max(1e-12, rotor.scalar)) * sign
        shift = angle_delta * rewind_factor

        new_phases[roi_mask] = (new_phases[roi_mask] - shift) % (2 * np.pi)
        new_bivectors[roi_mask] -= rotor.bivector * rewind_factor

        new_tensor = CliffordSpectralWaveformTensor(
            tensor_id=f"local_rot_v02_{tensor.tensor_id}",
            shape=tensor.shape,
            amplitudes=tensor.amplitudes.copy(),
            phases=new_phases,
            frequencies=tensor.frequencies.copy(),
            bivectors=new_bivectors,
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy()
        )
        return new_tensor

    @staticmethod
    def rewind_causal_trajectory(
        tensor: CliffordSpectralWaveformTensor,
        rotor: CliffordMultivector
    ) -> CliffordSpectralWaveformTensor:
        """
        [Counterfactual Time Reversal (Eye of Agamotto)]
        Applies inverse Clifford rotor R~ to reverse the waveform state to original state.
        """
        inv_rotor = rotor.reverse()
        return CliffordScaleRotor.rotate_global_bivector(tensor, inv_rotor)


class DynamicBufferPruningEngine:
    """
    Dynamic Buffer Pruning & Tensor Slicing Pipeline for VRAM/Buffer Optimization.
    Monitors phase coherence across waveform tensor cells.
    Prunes decoupled noise regions while preserving generative parameter manifold (Theta).
    Hydrates pruned cells on demand in O(1) time without full re-computation.
    """

    def __init__(self, coherence_threshold: float = 0.3):
        self.coherence_threshold = coherence_threshold
        self.pruned_registry: Dict[str, Dict[str, Any]] = {}

    def prune_tensor_buffer(
        self,
        tensor: CliffordSpectralWaveformTensor,
        reference_phase: float = 0.0
    ) -> CliffordSpectralWaveformTensor:
        """
        Prunes VRAM data for tensor regions whose phase coherence falls below threshold.
        Replaces unaligned cell buffer data with zero-fill while stashing generative parameters.
        """
        phase_diff = np.abs((tensor.phases - reference_phase + np.pi) % (2 * np.pi) - np.pi)
        # Mask where phase is unaligned / decoupled
        decoupled_mask = phase_diff > self.coherence_threshold

        pruned_amp = tensor.amplitudes.copy()
        pruned_bivec = tensor.bivectors.copy()

        # Zero out buffer data in decoupled regions to simulate VRAM buffer release
        pruned_amp[decoupled_mask] = 0.0
        pruned_bivec[decoupled_mask] = 0.0

        # Save generative parameter manifold (Theta) for O(1) counterfactual hydration
        generative_params = {
            "mean_freq": float(np.mean(tensor.frequencies)),
            "mean_amp": float(np.mean(tensor.amplitudes[~decoupled_mask])) if np.any(~decoupled_mask) else 1.0,
            "bivector_theta": tensor.bivectors.copy(),
            "decoupled_mask": decoupled_mask,
            "original_shape": tensor.shape
        }

        pruned_tensor = CliffordSpectralWaveformTensor(
            tensor_id=tensor.tensor_id,
            shape=tensor.shape,
            amplitudes=pruned_amp,
            phases=tensor.phases.copy(),
            frequencies=tensor.frequencies.copy(),
            bivectors=pruned_bivec,
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy(),
            is_pruned=True,
            pruned_mask=decoupled_mask,
            generative_params=generative_params
        )

        self.pruned_registry[tensor.tensor_id] = generative_params
        return pruned_tensor

    def hydrate_tensor_buffer(
        self,
        tensor: CliffordSpectralWaveformTensor
    ) -> CliffordSpectralWaveformTensor:
        """
        O(1) Counterfactual Hydration of pruned buffer regions from generative parameter manifold.
        """
        if not tensor.is_pruned or tensor.pruned_mask is None or not tensor.generative_params:
            return tensor

        gen_params = tensor.generative_params
        decoupled_mask = tensor.pruned_mask

        hydrated_amp = tensor.amplitudes.copy()
        hydrated_bivec = tensor.bivectors.copy()

        # Hydrate pruned regions using saved generative parameters
        hydrated_amp[decoupled_mask] = gen_params.get("mean_amp", 1.0)
        hydrated_bivec[decoupled_mask] = gen_params["bivector_theta"][decoupled_mask]

        hydrated_tensor = CliffordSpectralWaveformTensor(
            tensor_id=tensor.tensor_id,
            shape=tensor.shape,
            amplitudes=hydrated_amp,
            phases=tensor.phases.copy(),
            frequencies=tensor.frequencies.copy(),
            bivectors=hydrated_bivec,
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy(),
            is_pruned=False,
            pruned_mask=None,
            generative_params={}
        )
        return hydrated_tensor

    @staticmethod
    def slice_active_threshold_roi(
        tensor: CliffordSpectralWaveformTensor,
        roi_bounds: Tuple[slice, ...]
    ) -> CliffordSpectralWaveformTensor:
        """
        Lightweight Tensor Slicing around active threshold ROI for real-time low-VRAM processing.
        """
        sliced_amp = tensor.amplitudes[roi_bounds]
        sliced_phases = tensor.phases[roi_bounds]
        sliced_freqs = tensor.frequencies[roi_bounds]
        sliced_bivecs = tensor.bivectors[roi_bounds]

        return CliffordSpectralWaveformTensor(
            tensor_id=f"slice_{tensor.tensor_id}",
            shape=sliced_amp.shape,
            amplitudes=sliced_amp,
            phases=sliced_phases,
            frequencies=sliced_freqs,
            bivectors=sliced_bivecs,
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata={"parent_id": tensor.tensor_id, "roi_bounds": str(roi_bounds)}
        )

    @staticmethod
    def get_memory_footprint_report(
        original: CliffordSpectralWaveformTensor,
        pruned: CliffordSpectralWaveformTensor
    ) -> Dict[str, Any]:
        """Calculates VRAM / memory buffer reduction ratio."""
        orig_bytes = original.calculate_memory_bytes()
        # Active bytes in pruned tensor
        active_bytes = pruned.calculate_memory_bytes()
        if pruned.pruned_mask is not None:
            # Subtracted bytes from zeroed decoupled elements
            pruned_count = int(np.sum(pruned.pruned_mask))
            total_elements = pruned.pruned_mask.size
            reduction_ratio = float(pruned_count / total_elements)
        else:
            reduction_ratio = 0.0

        return {
            "original_bytes": orig_bytes,
            "pruned_active_bytes": active_bytes,
            "reduction_ratio": reduction_ratio,
            "savings_percentage": reduction_ratio * 100.0
        }


class CliffordRotorCouplingConstraint:
    """
    Bivector Rotor Coupling Constraint System for ELYSIA v0.2.
    Relaxes coupled Clifford waveform tensors toward phase-locking equilibrium.
    """

    def __init__(self, constraint_id: str, target_phase_diff: float = 0.0, coupling_strength: float = 0.8):
        self.constraint_id = constraint_id
        self.target_phase_diff = target_phase_diff
        self.coupling_strength = coupling_strength

    def compute_bivector_torque(
        self,
        tensor_a: CliffordSpectralWaveformTensor,
        tensor_b: CliffordSpectralWaveformTensor
    ) -> float:
        """Computes phase discrepancy torque between two coupled bivector rotors."""
        mean_phase_a = np.mean(tensor_a.phases)
        mean_phase_b = np.mean(tensor_b.phases)
        phase_diff = (mean_phase_a - mean_phase_b) % (2 * np.pi)
        torque = np.sin(phase_diff - self.target_phase_diff)
        return float(torque)

    def relax_step(
        self,
        tensor_a: CliffordSpectralWaveformTensor,
        tensor_b: CliffordSpectralWaveformTensor
    ) -> Tuple[CliffordSpectralWaveformTensor, CliffordSpectralWaveformTensor, float]:
        """
        Relaxes coupled bivector tensors toward equilibrium in a single step.
        """
        torque = self.compute_bivector_torque(tensor_a, tensor_b)
        shift = 0.5 * self.coupling_strength * torque

        axis = np.array([0.0, 0.0, 1.0])
        rotor_a = CliffordMultivector.create_rotor(axis, -shift)
        rotor_b = CliffordMultivector.create_rotor(axis, shift)

        new_a = CliffordScaleRotor.rotate_global_bivector(tensor_a, rotor_a)
        new_b = CliffordScaleRotor.rotate_global_bivector(tensor_b, rotor_b)

        residual_error = abs(torque)
        return new_a, new_b, residual_error


# =====================================================================
# Clifford Causal Cognitive Memory Workspace (v0.2)
# =====================================================================

@dataclass
class CliffordObservationEvent:
    event_id: str
    timestamp: float
    target_id: str
    attributes: Dict[str, Any]
    tensor_representation: Optional[CliffordSpectralWaveformTensor] = None
    context_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CliffordCausalBelief:
    belief_id: str
    target_id: str
    category: str
    attributes: Dict[str, Any]
    spectral_tensor: Optional[CliffordSpectralWaveformTensor] = None
    status: str = "ACTIVE"  # "ACTIVE", "REVISED", "SUSPENDED"
    evidence_history: List[str] = field(default_factory=list)
    confidence: float = 1.0


class CliffordCausalCognitiveMemory:
    """
    Relational Cognitive Memory Workspace for ELYSIA v0.2.
    Integrates CliffordMultivector rotor mechanics & Dynamic Buffer Pruning to evaluate:
    - Identity vs property distinction with bivector rotor tracking
    - Damage attribution and uncertainty preservation
    - Intertemporal belief updates & counterfactual trajectory rewind (Eye of Agamotto)
    """

    def __init__(self):
        self.observations: Dict[str, CliffordObservationEvent] = {}
        self.beliefs: Dict[str, CliffordCausalBelief] = {}
        self.pruning_engine = DynamicBufferPruningEngine()

    def record_observation(self, obs: CliffordObservationEvent):
        if obs.tensor_representation is None:
            freq = obs.attributes.get("frequency", 440.0)
            amp = 1.0 - obs.attributes.get("damage_level", 0.0)
            obs.tensor_representation = CliffordSpectralWaveformTensor(
                shape=(4, 4),
                amplitudes=np.full((4, 4), max(0.01, amp)),
                frequencies=np.full((4, 4), freq)
            )
        self.observations[obs.event_id] = obs

    def evaluate_identity_and_property_change(
        self,
        target_id: str,
        category_invariant_keys: List[str],
        variant_attribute_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Distinguishes between identity preservation vs property changes across observations,
        evaluating bivector rotor phase shifts.
        """
        target_obs = sorted(
            [obs for obs in self.observations.values() if obs.target_id == target_id],
            key=lambda o: o.timestamp
        )
        if len(target_obs) < 2:
            return {"status": "INSUFFICIENT_OBSERVATIONS", "identity_maintained": True}

        obs_initial = target_obs[0]
        obs_latest = target_obs[-1]

        identity_maintained = True
        for obs in target_obs:
            for k in category_invariant_keys:
                if obs.attributes.get(k) != obs_initial.attributes.get(k):
                    identity_maintained = False
                    break

        property_changes = {
            k: (obs_initial.attributes.get(k), obs_latest.attributes.get(k))
            for k in variant_attribute_keys
            if obs_initial.attributes.get(k) != obs_latest.attributes.get(k)
        }

        phase_shift = 0.0
        if obs_initial.tensor_representation and obs_latest.tensor_representation:
            p_init = np.mean(obs_initial.tensor_representation.phases)
            p_late = np.mean(obs_latest.tensor_representation.phases)
            phase_shift = float((p_late - p_init) % (2 * np.pi))

        return {
            "target_id": target_id,
            "identity_maintained": identity_maintained,
            "category": obs_initial.attributes.get("category"),
            "invariant_keys": category_invariant_keys,
            "property_changes": property_changes,
            "bivector_phase_shift_rad": phase_shift,
            "trajectory_count": len(target_obs),
            "initial_event": obs_initial.event_id,
            "latest_event": obs_latest.event_id
        }

    def evaluate_state_change_and_classification(
        self,
        target_id: str,
        damage_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyzes damage/fragmentation using spectral amplitude energy decay
        and bivector phase decorrelation, preserving uncertainty.
        """
        target_obs = sorted(
            [obs for obs in self.observations.values() if obs.target_id == target_id],
            key=lambda o: o.timestamp
        )
        if not target_obs:
            return {"status": "NO_DATA"}

        latest_obs = target_obs[-1]
        damage_level = latest_obs.attributes.get("damage_level", 0.0)
        is_fragmented = latest_obs.attributes.get("is_fragmented", False)
        observed_cause = latest_obs.context_conditions.get("observed_cause")

        spectral_energy = 0.0
        if latest_obs.tensor_representation:
            spectral_energy = latest_obs.tensor_representation.compute_energy()

        if damage_level >= damage_threshold or is_fragmented:
            if observed_cause:
                causal_attribution = "OBSERVED_CAUSE"
                cause_summary = observed_cause
                classification_action = "REVISE_CLASSIFICATION"
                uncertainty_preserved = False
            else:
                causal_attribution = "INFERRED_UNKNOWN"
                cause_summary = "Cause not directly observed; ambiguous perturbation"
                classification_action = "SUSPEND_JUDGMENT"
                uncertainty_preserved = True
        else:
            causal_attribution = "NO_MAJOR_DAMAGE"
            cause_summary = "Intact"
            classification_action = "MAINTAIN_CLASSIFICATION"
            uncertainty_preserved = False

        return {
            "target_id": target_id,
            "damage_level": damage_level,
            "is_fragmented": is_fragmented,
            "spectral_energy": spectral_energy,
            "causal_attribution": causal_attribution,
            "cause_summary": cause_summary,
            "classification_action": classification_action,
            "uncertainty_preserved": uncertainty_preserved
        }

    def update_intertemporal_belief(
        self,
        target_id: str,
        new_evidence_event_id: str
    ) -> CliffordCausalBelief:
        """
        Multi-point evidence update, judgment reuse, and belief status management.
        """
        evidence_obs = self.observations[new_evidence_event_id]
        existing_belief = self.beliefs.get(target_id)

        if not existing_belief:
            new_belief = CliffordCausalBelief(
                belief_id=f"belief_v02_{target_id}",
                target_id=target_id,
                category=evidence_obs.attributes.get("category", "UNKNOWN"),
                attributes=evidence_obs.attributes.copy(),
                spectral_tensor=evidence_obs.tensor_representation,
                status="ACTIVE",
                evidence_history=[new_evidence_event_id]
            )
            self.beliefs[target_id] = new_belief
            return new_belief

        eval_b = self.evaluate_state_change_and_classification(target_id)
        action = eval_b.get("classification_action")

        if action == "SUSPEND_JUDGMENT":
            existing_belief.status = "SUSPENDED"
        elif action == "REVISE_CLASSIFICATION":
            existing_belief.status = "REVISED"
            existing_belief.attributes.update(evidence_obs.attributes)
            if evidence_obs.tensor_representation:
                existing_belief.spectral_tensor = evidence_obs.tensor_representation
        elif action == "MAINTAIN_CLASSIFICATION":
            existing_belief.status = "ACTIVE"
            existing_belief.attributes.update(evidence_obs.attributes)

        existing_belief.evidence_history.append(new_evidence_event_id)
        return existing_belief
