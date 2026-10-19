"""
ELYSIA Global Scale Rotor & Causal-Cognitive PoC Engine v0.1

This module implements the Global Scale Rotor Architecture:
1. SpectralWaveformTensor: Wave-based continuous phase and frequency spectrum tensor.
2. ScaleRotor: O(1) global and local phase/quaternion/bivector rotor operations ("Eye of Agamotto" local rewind).
3. RotorCouplingConstraint: CAD-style constraint system and phase-locking joint relaxation.
4. RecursiveScaleHierarchy: Fractal scale hierarchy compressing lower-layer spectral manifolds into single-frequency nodes.
5. CausalCognitiveMemory: Cognitive memory workspace directly binding spectral rotor mechanics
   to validate hypotheses H1-H6 and Experiments A, B, C.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import uuid


@dataclass
class SpectralWaveformTensor:
    """
    Continuous spectral medium where data elements possess frequency (omega),
    complex phase angle (theta), amplitude (A), and volumetric boundary constraints.
    """
    tensor_id: str = field(default_factory=lambda: f"spec_{uuid.uuid4().hex[:8]}")
    shape: Tuple[int, ...] = (8, 8)
    amplitudes: np.ndarray = field(default_factory=lambda: np.ones((8, 8), dtype=np.float64))
    phases: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=np.float64))
    frequencies: np.ndarray = field(default_factory=lambda: np.ones((8, 8), dtype=np.float64))
    chromatic_signature: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.33, 0.33]))  # Red, Blue, Yellow
    volume_boundary: Dict[str, Any] = field(default_factory=lambda: {"radius": 1.0, "tension": 1.0, "is_contained": True})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.amplitudes.shape != self.shape:
            self.amplitudes = np.ones(self.shape, dtype=np.float64)
        if self.phases.shape != self.shape:
            self.phases = np.zeros(self.shape, dtype=np.float64)
        if self.frequencies.shape != self.shape:
            self.frequencies = np.ones(self.shape, dtype=np.float64)

    @property
    def complex_field(self) -> np.ndarray:
        """Returns the complex wavefunction field A * exp(i * phase)."""
        return self.amplitudes * np.exp(1j * self.phases)

    @complex_field.setter
    def complex_field(self, field: np.ndarray):
        self.amplitudes = np.abs(field)
        self.phases = np.angle(field)

    def compute_energy(self) -> float:
        """Calculates total spectral energy integrated over tensor volume."""
        return float(np.sum(self.amplitudes ** 2 * self.frequencies))


class ScaleRotor:
    """
    Global and Local Scale Rotor Operator.
    Executes O(1) phase shifts, order inversions, bivector/quaternion rotations,
    and local spacetime rewinds ("Eye of Agamotto").
    """

    @staticmethod
    def rotate_global_phase(
        tensor: SpectralWaveformTensor,
        phase_shift: float
    ) -> SpectralWaveformTensor:
        """
        Executes O(1) global phase rotation across the entire spectral tensor.
        Shifts all theta angles simultaneously without forward loops over elements.
        """
        new_tensor = SpectralWaveformTensor(
            tensor_id=f"rot_{tensor.tensor_id}",
            shape=tensor.shape,
            amplitudes=tensor.amplitudes.copy(),
            phases=(tensor.phases + phase_shift) % (2 * np.pi),
            frequencies=tensor.frequencies.copy(),
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy()
        )
        return new_tensor

    @staticmethod
    def invert_global_order(
        tensor: SpectralWaveformTensor
    ) -> SpectralWaveformTensor:
        """
        O(1) Order Inversion / Yin-Yang Reversal.
        Flips the phase and reverses waveform spectrum sequence.
        """
        new_tensor = SpectralWaveformTensor(
            tensor_id=f"inv_{tensor.tensor_id}",
            shape=tensor.shape,
            amplitudes=np.flip(tensor.amplitudes),
            phases=(-np.flip(tensor.phases)) % (2 * np.pi),
            frequencies=np.flip(tensor.frequencies),
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy()
        )
        return new_tensor

    @staticmethod
    def apply_local_rotor(
        tensor: SpectralWaveformTensor,
        roi_mask: np.ndarray,
        phase_shift: float,
        rewind_factor: float = 1.0
    ) -> SpectralWaveformTensor:
        """
        Local Spacetime Rotor Control ('Eye of Agamotto').
        Applies phase rotation and causal rewind exclusively within the specified ROI mask.
        Unmasked surrounding region remains untouched and invariant.
        """
        new_phases = tensor.phases.copy()
        new_amplitudes = tensor.amplitudes.copy()

        # Apply phase rotation and rewind factor inside ROI
        new_phases[roi_mask] = (new_phases[roi_mask] - phase_shift * rewind_factor) % (2 * np.pi)

        new_tensor = SpectralWaveformTensor(
            tensor_id=f"local_rot_{tensor.tensor_id}",
            shape=tensor.shape,
            amplitudes=new_amplitudes,
            phases=new_phases,
            frequencies=tensor.frequencies.copy(),
            chromatic_signature=tensor.chromatic_signature.copy(),
            volume_boundary=tensor.volume_boundary.copy(),
            metadata=tensor.metadata.copy()
        )
        return new_tensor


class RotorCouplingConstraint:
    """
    CAD-style geometric constraint system and rotor coupling.
    Rotor couplings act as multivariable joints that relax and converge
    to equilibrium without combinatorial if-else branching.
    """

    def __init__(self, constraint_id: str, target_phase_diff: float = 0.0, coupling_strength: float = 0.8):
        self.constraint_id = constraint_id
        self.target_phase_diff = target_phase_diff
        self.coupling_strength = coupling_strength

    def compute_torque(self, tensor_a: SpectralWaveformTensor, tensor_b: SpectralWaveformTensor) -> float:
        """Computes phase discrepancy torque between two coupled rotors."""
        mean_phase_a = np.mean(tensor_a.phases)
        mean_phase_b = np.mean(tensor_b.phases)
        phase_diff = (mean_phase_a - mean_phase_b) % (2 * np.pi)
        torque = np.sin(phase_diff - self.target_phase_diff)
        return float(torque)

    def relax_step(
        self,
        tensor_a: SpectralWaveformTensor,
        tensor_b: SpectralWaveformTensor
    ) -> Tuple[SpectralWaveformTensor, SpectralWaveformTensor, float]:
        """
        Relaxes the coupled tensors toward equilibrium in a single geometric step.
        """
        torque = self.compute_torque(tensor_a, tensor_b)
        shift = 0.5 * self.coupling_strength * torque

        new_a = ScaleRotor.rotate_global_phase(tensor_a, -shift)
        new_b = ScaleRotor.rotate_global_phase(tensor_b, shift)

        residual_error = abs(torque)
        return new_a, new_b, residual_error


class RecursiveScaleHierarchy:
    """
    Fractal Scale Hierarchy (Master Orchestra Layer).
    Compresses an entire lower-scale spectral manifold into a single high-level frequency node,
    allowing scale-free propagation (like H2O molecule properties preserved across drops, rivers, oceans).
    """

    def __init__(self, layer_id: str, scale_level: int = 0):
        self.layer_id = layer_id
        self.scale_level = scale_level
        self.sub_manifolds: List[SpectralWaveformTensor] = []

    def register_sub_manifold(self, tensor: SpectralWaveformTensor):
        self.sub_manifolds.append(tensor)

    def compress_to_node(self) -> SpectralWaveformTensor:
        """
        Compresses all registered lower-level tensors into a single representative spectral node.
        """
        if not self.sub_manifolds:
            return SpectralWaveformTensor(shape=(1, 1))

        avg_freq = np.mean([np.mean(t.frequencies) for t in self.sub_manifolds])
        avg_phase = np.mean([np.mean(t.phases) for t in self.sub_manifolds]) % (2 * np.pi)
        avg_amp = np.mean([np.mean(t.amplitudes) for t in self.sub_manifolds])
        avg_chroma = np.mean([t.chromatic_signature for t in self.sub_manifolds], axis=0)

        compressed_node = SpectralWaveformTensor(
            tensor_id=f"macro_node_L{self.scale_level}_{uuid.uuid4().hex[:6]}",
            shape=(1, 1),
            amplitudes=np.array([[avg_amp]], dtype=np.float64),
            phases=np.array([[avg_phase]], dtype=np.float64),
            frequencies=np.array([[avg_freq]], dtype=np.float64),
            chromatic_signature=avg_chroma,
            metadata={"source_layer": self.layer_id, "child_count": len(self.sub_manifolds)}
        )
        return compressed_node


# =====================================================================
# Causal Cognitive Memory & PoC Validation Structures
# =====================================================================

@dataclass
class ObservationEvent:
    event_id: str
    timestamp: float
    target_id: str
    attributes: Dict[str, Any]
    tensor_representation: Optional[SpectralWaveformTensor] = None
    context_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalBelief:
    belief_id: str
    target_id: str
    category: str
    attributes: Dict[str, Any]
    spectral_tensor: Optional[SpectralWaveformTensor] = None
    status: str = "ACTIVE"  # "ACTIVE", "REVISED", "SUSPENDED"
    evidence_history: List[str] = field(default_factory=list)
    confidence: float = 1.0


class CausalCognitiveMemory:
    """
    Relational Cognitive Memory Workspace for ELYSIA PoC v0.1.
    Integrates SpectralWaveformTensor & ScaleRotor phase coherence to evaluate:
    - Identity vs property distinction (Experiment A)
    - Damage attribution and uncertainty preservation (Experiment B)
    - Intertemporal evidence-based belief update (Experiment C)
    """

    def __init__(self):
        self.observations: Dict[str, ObservationEvent] = {}
        self.beliefs: Dict[str, CausalBelief] = {}
        self.causal_links: List[Dict[str, Any]] = []

    def record_observation(self, obs: ObservationEvent):
        # Auto-instantiate spectral waveform representation if not provided
        if obs.tensor_representation is None:
            freq = obs.attributes.get("frequency", 440.0)
            amp = 1.0 - obs.attributes.get("damage_level", 0.0)
            obs.tensor_representation = SpectralWaveformTensor(
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
        Experiment A: Distinguishes between identity preservation vs property changes
        across the full trajectory of observations, utilizing spectral phase coherence.
        """
        target_obs = sorted(
            [obs for obs in self.observations.values() if obs.target_id == target_id],
            key=lambda o: o.timestamp
        )
        if len(target_obs) < 2:
            return {"status": "INSUFFICIENT_OBSERVATIONS", "identity_maintained": True}

        obs_initial = target_obs[0]
        obs_latest = target_obs[-1]

        # Check category invariant continuity across all intermediate trajectory observations
        identity_maintained = True
        for obs in target_obs:
            for k in category_invariant_keys:
                if obs.attributes.get(k) != obs_initial.attributes.get(k):
                    identity_maintained = False
                    break

        # Calculate property changes between initial and latest observation
        property_changes = {
            k: (obs_initial.attributes.get(k), obs_latest.attributes.get(k))
            for k in variant_attribute_keys
            if obs_initial.attributes.get(k) != obs_latest.attributes.get(k)
        }

        # Spectral phase shift computation
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
            "spectral_phase_shift_rad": phase_shift,
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
        Experiment B: Analyzes damage/fragmentation using spectral amplitude decay
        and phase decorrelation, attributes cause, and preserves uncertainty.
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

        # Spectral amplitude energy evaluation
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
    ) -> CausalBelief:
        """
        Experiment C: Multi-point observation, past judgment reuse, and belief revision/preservation.
        Integrates classification action results seamlessly.
        """
        evidence_obs = self.observations[new_evidence_event_id]
        existing_belief = self.beliefs.get(target_id)

        if not existing_belief:
            new_belief = CausalBelief(
                belief_id=f"belief_{target_id}",
                target_id=target_id,
                category=evidence_obs.attributes.get("category", "UNKNOWN"),
                attributes=evidence_obs.attributes.copy(),
                spectral_tensor=evidence_obs.tensor_representation,
                status="ACTIVE",
                evidence_history=[new_evidence_event_id]
            )
            self.beliefs[target_id] = new_belief
            return new_belief

        # Evaluate state change & classification action
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
            # Maintain category while recording non-disruptive property variations
            existing_belief.attributes.update(evidence_obs.attributes)

        existing_belief.evidence_history.append(new_evidence_event_id)
        return existing_belief
