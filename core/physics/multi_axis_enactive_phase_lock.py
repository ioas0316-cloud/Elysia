"""Multi-Axis Enactive Boundary & Phase-Lock Engine with Resonance Density.

This module expands Elysia's 3-layer enactive boundary architecture to N-dimensional
heterogeneous spectrum axes (Acoustic, Semantic, Logical, Physical).

Key Features:
1. Multi-Axis Protocol Boundary Observation (Acoustic, Semantic, Logical, Physical)
2. Dynamic Constraint Mapping with Inter-Axis Tension Redistribution
3. Multi-Axis Kuramoto Phase Synchronization & Free Energy Principle Active Inference
4. Quantitative Resonance Density Calculation:
   rho_res(t) = sum_{i < j} w_{ij} * cos(theta_i - theta_j) / (1 + alpha * Friction + beta * F_free)
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from core.physics.enactive_boundary_phase_lock import (
    InterfaceBoundaryObserver,
    DynamicConstraintMapper,
    PhaseLockSynchronizer,
    EnactiveBoundaryEngine,
)

# Standard 4 Primary Modality Axes
AXIS_ACOUSTIC = "acoustic"   # 소리: 주파수/파동 스펙트럼
AXIS_SEMANTIC = "semantic"   # 언어: 의미/맥락 위상 구조
AXIS_LOGICAL  = "logical"    # 수학/논리: 엄밀한 인과적 불변량 (Invariant)
AXIS_PHYSICAL = "physical"   # 물리: 장력, 변위, 마찰 및 하드웨어 한계

DEFAULT_AXES = [AXIS_ACOUSTIC, AXIS_SEMANTIC, AXIS_LOGICAL, AXIS_PHYSICAL]


class MultiAxisBoundaryObserver(InterfaceBoundaryObserver):
    """Layer 1: Observes protocol limits and boundary clearances across multiple spectrum axes."""

    def __init__(
        self,
        axes: Optional[List[str]] = None,
        internal_capacity: float = 100.0,
        internal_reach: float = 10.0,
    ):
        super().__init__(internal_capacity, internal_reach)
        self.axes = axes if axes is not None else list(DEFAULT_AXES)

    def observe_multi_axis_protocol(
        self, protocol_spec: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Extracts and normalizes external protocol parameters for each axis.

        `protocol_spec` can be nested by axis name, or flat.
        """
        parsed_specs: Dict[str, Dict[str, float]] = {}

        for axis in self.axes:
            spec = protocol_spec.get(axis, {}) if isinstance(protocol_spec.get(axis), dict) else protocol_spec
            max_freq = float(spec.get("max_frequency", 100.0))
            max_power = float(spec.get("max_power", 50.0))
            max_bandwidth = float(spec.get("max_bandwidth", 1000.0))
            latency_tolerance = float(spec.get("latency_tolerance", 0.05))

            parsed_specs[axis] = {
                "max_frequency": max_freq,
                "max_power": max_power,
                "max_bandwidth": max_bandwidth,
                "latency_tolerance": latency_tolerance,
            }

        return parsed_specs

    def compute_multi_axis_clearance(
        self,
        active_states: Dict[str, np.ndarray],
        protocol_limits: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Calculates safety clearance margin for each axis."""
        clearances: Dict[str, Dict[str, float]] = {}

        for axis in self.axes:
            state = active_states.get(axis, np.array([1.0, 1.0, 1.0]))
            limits = protocol_limits.get(axis, {"max_power": 50.0})
            cl = self.compute_boundary_clearance(state, limits)
            clearances[axis] = cl

        return clearances


class MultiAxisDynamicConstraintMapper(DynamicConstraintMapper):
    """Layer 2: Learns dynamic boundary thresholds and redistributes tension across axes."""

    def __init__(
        self,
        axes: Optional[List[str]] = None,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
    ):
        super().__init__(learning_rate, decay_rate)
        self.axes = axes if axes is not None else list(DEFAULT_AXES)

        # Per-axis learned boundary limits & tension vectors
        self.axis_boundaries: Dict[str, Dict[str, float]] = {
            axis: {
                "frequency_limit": 100.0,
                "power_limit": 50.0,
                "bandwidth_limit": 1000.0,
            }
            for axis in self.axes
        }
        self.axis_friction: Dict[str, float] = {axis: 0.0 for axis in self.axes}
        self.axis_tension: Dict[str, float] = {axis: 1.0 / len(self.axes) for axis in self.axes}

    def update_multi_axis_friction(
        self, multi_friction_feedback: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Updates friction and redistributes tension dynamically across all axes.

        When one axis experiences severe friction (e.g. Physical stress),
        other axes dynamically absorb tension to maintain overall balance.
        """
        total_instant_friction = 0.0
        instant_frictions: Dict[str, float] = {}

        for axis in self.axes:
            feedback = multi_friction_feedback.get(axis, {})
            latency = feedback.get("latency", 0.0)
            error_rate = feedback.get("error_rate", 0.0)
            resistance = feedback.get("resistance", 0.0)

            instant = latency * 10.0 + error_rate * 50.0 + resistance * 5.0
            instant_frictions[axis] = instant
            self.axis_friction[axis] = (
                1.0 - self.decay_rate
            ) * self.axis_friction[axis] + instant

            # Boundary update per axis
            if instant > 1.0:
                shrink = 1.0 - self.learning_rate * min(0.5, instant / 100.0)
                for k in self.axis_boundaries[axis]:
                    self.axis_boundaries[axis][k] *= shrink
            else:
                expand = 1.0 + (self.learning_rate * 0.1)
                for k in self.axis_boundaries[axis]:
                    self.axis_boundaries[axis][k] *= expand

            total_instant_friction += instant

        # Dynamic Constraint Mapping: Tension Redistribution
        # High friction on axis i shifts operational tension load to unconstrained axes
        friction_array = np.array([instant_frictions[a] for a in self.axes])
        inv_friction = 1.0 / (1.0 + friction_array)
        redistributed_tension = inv_friction / np.sum(inv_friction)

        for idx, axis in enumerate(self.axes):
            self.axis_tension[axis] = float(redistributed_tension[idx])

        self.accumulated_friction = (
            1.0 - self.decay_rate
        ) * self.accumulated_friction + total_instant_friction

        return {
            "total_instant_friction": total_instant_friction,
            "accumulated_friction": self.accumulated_friction,
            "per_axis_instant_friction": instant_frictions,
            "per_axis_accumulated_friction": dict(self.axis_friction),
            "per_axis_tension": dict(self.axis_tension),
            "per_axis_boundaries": self.axis_boundaries,
        }


class MultiAxisPhaseLockSynchronizer:
    """Layer 3: Multi-axis Kuramoto phase-locking and Free Energy principle active inference."""

    def __init__(
        self,
        axes: Optional[List[str]] = None,
        base_freqs: Optional[Dict[str, float]] = None,
        coupling_matrix: Optional[np.ndarray] = None,
    ):
        self.axes = axes if axes is not None else list(DEFAULT_AXES)
        self.num_axes = len(self.axes)

        # Internal phases and base frequencies
        self.phases: Dict[str, float] = {axis: 0.0 for axis in self.axes}
        self.base_freqs: Dict[str, float] = {
            axis: base_freqs.get(axis, 10.0) if base_freqs else 10.0
            for axis in self.axes
        }

        # Inter-axis coupling matrix (NxN)
        if coupling_matrix is not None and coupling_matrix.shape == (self.num_axes, self.num_axes):
            self.coupling_matrix = coupling_matrix
        else:
            # Default uniform pairwise coupling strength
            self.coupling_matrix = np.full((self.num_axes, self.num_axes), 1.5)
            np.fill_diagonal(self.coupling_matrix, 0.0)

        self.free_energy = 0.0
        self.multi_order_parameter = 0.0

    def synchronize_multi_axis(
        self,
        external_phases: Dict[str, float],
        external_freqs: Dict[str, float],
        tensions: Dict[str, float],
        effective_freq_limits: Dict[str, float],
        dt: float,
    ) -> Dict[str, Any]:
        """Performs multi-axis Kuramoto coupled phase evolution and updates internal phases.

        Calculates inter-axis order parameters, phase differences, and Free Energy error.
        """
        new_phases: Dict[str, float] = {}
        phase_diffs: Dict[str, float] = {}
        order_parameters: Dict[str, float] = {}

        phase_vec = np.array([self.phases[a] for a in self.axes])
        tension_vec = np.array([tensions.get(a, 1.0 / self.num_axes) for a in self.axes])

        # Active Inference Free Energy prediction error scan
        # F = sum_i tension_i * (external_phase_i - internal_phase_i)^2
        ext_phase_vec = np.array([external_phases.get(a, 0.0) for a in self.axes])
        phase_errors = np.arctan2(np.sin(ext_phase_vec - phase_vec), np.cos(ext_phase_vec - phase_vec))
        self.free_energy = float(np.sum(tension_vec * (phase_errors ** 2)))

        # N-dimensional Kuramoto interaction
        # d(theta_i)/dt = omega_i + sum_j (K_ij * tension_j * sin(theta_j - theta_i))
        for i, axis_i in enumerate(self.axes):
            ext_freq = external_freqs.get(axis_i, self.base_freqs[axis_i])
            max_allowed = effective_freq_limits.get(axis_i, 100.0)
            eff_ext_freq = min(ext_freq, max_allowed)

            # External phase coupling
            ext_diff = external_phases.get(axis_i, self.phases[axis_i]) - self.phases[axis_i]
            ext_diff_norm = np.arctan2(np.sin(ext_diff), np.cos(ext_diff))

            # Internal inter-axis coupling
            inter_coupling = 0.0
            for j, axis_j in enumerate(self.axes):
                if i != j:
                    diff_ij = phase_vec[j] - phase_vec[i]
                    inter_coupling += (
                        self.coupling_matrix[i, j]
                        * tension_vec[j]
                        * np.sin(diff_ij)
                    )

            # Combined frequency update with active inference correction
            freq_pull = 1.0 * np.sin(ext_diff_norm) + inter_coupling
            eff_freq = eff_ext_freq + freq_pull

            # Phase update
            updated_phase = (self.phases[axis_i] + eff_freq * dt) % (2 * np.pi)
            self.phases[axis_i] = updated_phase
            new_phases[axis_i] = updated_phase

            phase_diffs[axis_i] = float(ext_diff_norm)
            order_parameters[axis_i] = float(np.clip(np.cos(ext_diff_norm), 0.0, 1.0))

        # Overall Multi-Axis Kuramoto Order Parameter (R_M)
        cos_sum = 0.0
        pair_count = 0
        current_phase_vec = np.array([self.phases[a] for a in self.axes])
        for i in range(self.num_axes):
            for j in range(i + 1, self.num_axes):
                cos_sum += np.cos(current_phase_vec[i] - current_phase_vec[j])
                pair_count += 1

        self.multi_order_parameter = float(cos_sum / pair_count) if pair_count > 0 else 1.0

        return {
            "per_axis_phases": new_phases,
            "per_axis_phase_diffs": phase_diffs,
            "per_axis_order_parameters": order_parameters,
            "multi_axis_order_parameter": self.multi_order_parameter,
            "free_energy": self.free_energy,
        }


class MultiAxisEnactiveBoundaryEngine(EnactiveBoundaryEngine):
    """Unified Universal Runtime Engine for Multi-Axis Enactive Phase-Locking & Resonance Density.

    Fully backwards compatible with single-axis `EnactiveBoundaryEngine`.
    """

    def __init__(
        self,
        axes: Optional[List[str]] = None,
        internal_capacity: float = 100.0,
        internal_reach: float = 10.0,
        base_freqs: Optional[Dict[str, float]] = None,
        alpha_friction: float = 0.1,
        beta_free_energy: float = 0.2,
    ):
        super().__init__(
            internal_capacity=internal_capacity,
            internal_reach=internal_reach,
            base_freq=(base_freqs.get(AXIS_ACOUSTIC, 10.0) if base_freqs else 10.0),
        )

        self.axes = axes if axes is not None else list(DEFAULT_AXES)
        self.multi_observer = MultiAxisBoundaryObserver(self.axes, internal_capacity, internal_reach)
        self.multi_mapper = MultiAxisDynamicConstraintMapper(self.axes)
        self.multi_synchronizer = MultiAxisPhaseLockSynchronizer(self.axes, base_freqs)

        self.alpha_friction = alpha_friction
        self.beta_free_energy = beta_free_energy
        self.resonance_density = 0.0

    def compute_resonance_density(
        self,
        phases: Dict[str, float],
        tensions: Dict[str, float],
        friction: float,
        free_energy: float,
    ) -> float:
        """Computes the Resonance Density (공명밀도, rho_res):

        rho_res(t) = sum_{i < j} w_{ij} * cos(theta_i - theta_j) / (1 + alpha * Friction + beta * F_free)

        Where w_{ij} = tension_i * tension_j (inter-axis tension coupling weight).
        High phase alignment between modalities increases numerator;
        unresolved friction and free energy errors increase denominator.
        """
        phase_vec = np.array([phases[a] for a in self.axes])
        tension_vec = np.array([tensions.get(a, 1.0 / len(self.axes)) for a in self.axes])

        weighted_alignment_sum = 0.0
        weight_norm = 0.0

        num_axes = len(self.axes)
        for i in range(num_axes):
            for j in range(i + 1, num_axes):
                w_ij = tension_vec[i] * tension_vec[j]
                cos_ij = np.cos(phase_vec[i] - phase_vec[j])
                weighted_alignment_sum += w_ij * cos_ij
                weight_norm += w_ij

        normalized_alignment = (
            weighted_alignment_sum / weight_norm if weight_norm > 1e-8 else 0.0
        )

        # Shift [-1, 1] cosine alignment to [0, 1] range for clean physical density
        coherence_numerator = max(0.0, (normalized_alignment + 1.0) / 2.0)

        # Friction and Free Energy penalty denominator
        denominator = 1.0 + self.alpha_friction * friction + self.beta_free_energy * free_energy

        self.resonance_density = float(coherence_numerator / denominator)
        return self.resonance_density

    def step_multi_axis(
        self,
        active_states: Dict[str, np.ndarray],
        protocol_specs: Dict[str, Dict[str, Any]],
        external_phases: Dict[str, float],
        external_freqs: Dict[str, float],
        multi_friction_feedback: Dict[str, Dict[str, float]],
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """Executes a single multi-axis enactive boundary and resonance density step."""
        self.time += dt

        # Step 1: Multi-Axis Protocol Boundary Observation
        proto_limits = self.multi_observer.observe_multi_axis_protocol(protocol_specs)
        clearances = self.multi_observer.compute_multi_axis_clearance(active_states, proto_limits)

        # Step 2: Multi-Axis Dynamic Constraint Mapping & Tension Redistribution
        friction_metrics = self.multi_mapper.update_multi_axis_friction(multi_friction_feedback)
        tensions = friction_metrics["per_axis_tension"]

        # Extract effective frequency limits per axis
        effective_limits: Dict[str, float] = {}
        for axis in self.axes:
            proto_max = proto_limits[axis]["max_frequency"]
            learned_max = friction_metrics["per_axis_boundaries"][axis]["frequency_limit"]
            effective_limits[axis] = min(proto_max, learned_max)

        # Step 3: Multi-Axis Phase Locking & Active Inference Free Energy Scan
        sync_res = self.multi_synchronizer.synchronize_multi_axis(
            external_phases,
            external_freqs,
            tensions,
            effective_limits,
            dt,
        )

        # Step 4: Calculate Resonance Density (공명밀도)
        rho_res = self.compute_resonance_density(
            phases=sync_res["per_axis_phases"],
            tensions=tensions,
            friction=friction_metrics["total_instant_friction"],
            free_energy=sync_res["free_energy"],
        )

        # Average normalized safety clearance across axes
        avg_safety_ratio = float(
            np.mean([cl["normalized_safety_ratio"] for cl in clearances.values()])
        )

        # Also update legacy single-axis engine state for backwards compatibility
        self.synchronizer.internal_phase = sync_res["per_axis_phases"].get(AXIS_ACOUSTIC, 0.0)
        self.synchronizer.order_parameter = sync_res["per_axis_order_parameters"].get(AXIS_ACOUSTIC, 0.0)

        return {
            "time": self.time,
            "resonance_density": rho_res,
            "multi_axis_order_parameter": sync_res["multi_axis_order_parameter"],
            "free_energy": sync_res["free_energy"],
            "per_axis_phases": sync_res["per_axis_phases"],
            "per_axis_phase_diffs": sync_res["per_axis_phase_diffs"],
            "per_axis_order_parameters": sync_res["per_axis_order_parameters"],
            "per_axis_tension": tensions,
            "friction_metrics": friction_metrics,
            "clearances": clearances,
            "average_safety_ratio": avg_safety_ratio,
        }
