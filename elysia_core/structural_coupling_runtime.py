"""
Elysia Structural Coupling Runtime (구조결합 런타임)
=====================================================
Embodying the 3 Structural Coupling Principles:
1. Resonant Attractor Coupling (공명 어트랙터 결합):
   Conditionless phase-frequency basin convergence replacing hardcoded if-else branching.
2. Meta-Plastic Prismatic Lens (메타-가소성 프리즘 렌즈):
   Dynamic refractive matrix (PrismMatrix) re-tuning driven by trajectory spectral feedback.
3. Cross-Modal Manifold Projection (비동기 다중모달 교차 투영):
   Cross-modal mapping onto a unified Clifford Rotor field.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from elysia_core.unified_topological_phase_runtime import UnifiedTopologicalPhaseRuntime


class ResonantAttractorCoupling:
    """
    Resonant Attractor Coupling (공명 어트랙터 결합)
    Replaces discrete condition trees (if-else logic) with continuous phase-frequency
    attractor basins (Resonant Basins). System state is naturally pulled into the most
    phase-coherent basin via phase-locking dynamics.
    """

    def __init__(self, dimension: int = 5, num_basins: int = 4):
        self.dimension = dimension
        self.num_basins = num_basins

        # Define resonant basins with natural frequencies and characteristic phase vectors
        self.basin_frequencies = np.linspace(1.0, 4.0, num_basins)
        self.basin_phases = np.zeros((num_basins, dimension))
        for k in range(num_basins):
            angle = (2.0 * np.pi * k) / num_basins
            self.basin_phases[k] = np.array([
                np.cos(angle),
                np.sin(angle),
                np.cos(2 * angle),
                np.sin(2 * angle),
                1.0 / (k + 1.0)
            ])[:dimension]
            norm = np.linalg.norm(self.basin_phases[k])
            if norm > 1e-6:
                self.basin_phases[k] /= norm

        self.current_state = np.zeros(dimension)
        self.active_basin_idx = 0
        self.coupling_strength = 0.8

    def evaluate_resonance(self, input_wave: np.ndarray, input_frequency: float) -> Tuple[np.ndarray, int, float]:
        """
        Calculates resonance energy between the incoming wave signal and each attractor basin.
        Pulls the current state toward the dominant resonant basin without hardcoded branching.
        """
        padded_wave = np.pad(input_wave, (0, max(0, self.dimension - len(input_wave))))[:self.dimension]
        wave_norm = np.linalg.norm(padded_wave)
        if wave_norm > 1e-6:
            padded_wave /= wave_norm

        resonance_energies = np.zeros(self.num_basins)
        for k in range(self.num_basins):
            # Frequency tuning match (Lorentzian resonance profile)
            freq_diff = np.abs(input_frequency - self.basin_frequencies[k])
            freq_resonance = 1.0 / (1.0 + 4.0 * (freq_diff ** 2))

            # Phase alignment match
            phase_resonance = np.abs(np.dot(padded_wave, self.basin_phases[k]))

            # Total resonance energy
            resonance_energies[k] = freq_resonance * phase_resonance

        # Softmax-style resonance distribution (continuous, smooth attraction field)
        exp_energies = np.exp(4.0 * (resonance_energies - np.max(resonance_energies)))
        resonance_weights = exp_energies / np.sum(exp_energies)

        # Gradient pull toward resonant basin centers
        target_phase = np.dot(resonance_weights, self.basin_phases)
        self.current_state = (1.0 - self.coupling_strength) * self.current_state + self.coupling_strength * target_phase

        self.active_basin_idx = int(np.argmax(resonance_weights))
        peak_resonance = float(np.max(resonance_energies))

        return self.current_state.copy(), self.active_basin_idx, peak_resonance


class MetaPlasticPrismaticLens:
    """
    Meta-Plastic Prismatic Lens (메타-가소성 프리즘 렌즈)
    Dynamically warps and re-tunes its internal refractive matrix (PrismMatrix) and optical axis
    based on the spectral fingerprint (Fourier/Wavelet breakdown) of accumulated interaction trajectories.
    """

    def __init__(self, dimension: int = 5, plasticity_rate: float = 0.05):
        self.dimension = dimension
        self.plasticity_rate = plasticity_rate
        # Initial isotropic refraction matrix
        self.prism_matrix = np.eye(dimension, dtype=float)
        self.spectral_history: List[np.ndarray] = []
        self.refractive_index = 1.5

    def refract(self, incoming_tensor: np.ndarray) -> np.ndarray:
        """
        Passes incoming tensor through the current Meta-Plastic Prism Lens.
        """
        padded = np.pad(incoming_tensor, (0, max(0, self.dimension - len(incoming_tensor))))[:self.dimension]
        refracted = np.dot(self.prism_matrix, padded) / self.refractive_index
        return refracted

    def adapt_lens_from_trajectory(self, trajectory_samples: np.ndarray):
        """
        Computes the spectral fingerprint of the trajectory (frequency components across time)
        and re-tunes the refractive prism matrix via meta-plastic feedback.
        """
        if len(trajectory_samples) < 2:
            return

        # Spectral breakdown (FFT along temporal axis)
        spectral_fft = np.fft.rfft(trajectory_samples, axis=0)
        spectral_power = np.abs(spectral_fft)  # Power spectrum

        # Dominant spatial-frequency covariance matrix
        cov_spectrum = np.cov(spectral_power.T)
        if cov_spectrum.shape != (self.dimension, self.dimension):
            cov_spectrum = np.eye(self.dimension) * np.mean(spectral_power)

        # Normalize covariance
        cov_norm = np.linalg.norm(cov_spectrum)
        if cov_norm > 1e-6:
            cov_spectrum /= cov_norm

        # Meta-Plastic update rule: PrismMatrix deforms along principle spectral covariance directions
        self.prism_matrix = (1.0 - self.plasticity_rate) * self.prism_matrix + self.plasticity_rate * (np.eye(self.dimension) + cov_spectrum)

        # Maintain matrix conditioning
        u, s, vh = np.linalg.svd(self.prism_matrix)
        s_clamped = np.clip(s, 0.2, 5.0)
        self.prism_matrix = np.dot(u * s_clamped, vh)

        # Retune refractive index based on total spectral energy variance
        spectral_energy = float(np.sum(spectral_power))
        self.refractive_index = 1.0 + 0.5 * np.tanh(spectral_energy / (len(trajectory_samples) + 1e-5))

        self.spectral_history.append(spectral_power.flatten()[:10])
        if len(self.spectral_history) > 50:
            self.spectral_history.pop(0)


class CrossModalManifoldProjection:
    """
    Cross-Modal Manifold Projection (비동기 다중모달 교차 투영)
    Projects heterogeneous modal signals (tactile/physical, visual/spatial, acoustic/vibrational, chromatic)
    onto a unified Clifford Rotor topological manifold.
    """

    def __init__(self, target_dim: int = 5):
        self.target_dim = target_dim
        # Modal projection operators (orthonormal / random projection matrices)
        rng = np.random.default_rng(42)
        self.projection_operators = {
            "physical": np.eye(target_dim),
            "sensory": np.eye(target_dim),
            "chromatic": np.eye(target_dim),
            "symbolic": np.eye(target_dim),
        }

    def project_modality(self, raw_signal: np.ndarray, modality_type: str) -> np.ndarray:
        """
        Projects a raw modality tensor onto the target manifold dimension using the modal operator.
        """
        if modality_type not in self.projection_operators:
            in_dim = len(raw_signal)
            proj = np.eye(self.target_dim, in_dim)
            self.projection_operators[modality_type] = proj

        proj_matrix = self.projection_operators[modality_type]
        if proj_matrix.shape[1] != len(raw_signal):
            padded_signal = np.pad(raw_signal, (0, max(0, proj_matrix.shape[1] - len(raw_signal))))[:proj_matrix.shape[1]]
        else:
            padded_signal = raw_signal

        manifold_vector = np.dot(proj_matrix, padded_signal)
        norm = np.linalg.norm(manifold_vector)
        if norm > 1e-6:
            manifold_vector /= norm
        return manifold_vector

    def cross_resonate(self, modal_signals: Dict[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Computes cross-modal phase resonance across all active modality projections.
        """
        projected_vectors = []
        for mod_type, signal in modal_signals.items():
            proj_vec = self.project_modality(signal, mod_type)
            projected_vectors.append(proj_vec)

        if not projected_vectors:
            return np.zeros(self.target_dim), 1.0

        stacked = np.array(projected_vectors)  # (N_modalities, target_dim)
        unified_field = np.mean(stacked, axis=0)
        u_norm = np.linalg.norm(unified_field)
        if u_norm > 1e-6:
            unified_field /= u_norm

        # Cross-modal resonance coherence score (average pairwise cosine similarity)
        dots = np.dot(stacked, stacked.T)
        num_mods = len(projected_vectors)
        if num_mods > 1:
            coherence = float((np.sum(dots) - num_mods) / (num_mods * (num_mods - 1)))
        else:
            coherence = 1.0

        return unified_field, coherence


class StructuralCouplingRuntime:
    """
    Structural Coupling Runtime (구조결합 런타임)
    Macro runtime orchestrator linking Resonant Attractor Coupling, Meta-Plastic Prismatic Lens,
    and Cross-Modal Manifold Projection with ELYSIA's Topological Phase Runtime.
    """

    def __init__(self, target_dim: int = 5):
        self.target_dim = target_dim
        self.attractor_coupling = ResonantAttractorCoupling(dimension=target_dim)
        self.prismatic_lens = MetaPlasticPrismaticLens(dimension=target_dim)
        self.cross_modal_projection = CrossModalManifoldProjection(target_dim=target_dim)
        self.macro_phase_runtime = UnifiedTopologicalPhaseRuntime()

        self.trajectory_buffer: List[np.ndarray] = []
        self.step_count = 0

    def process_step(
        self,
        multimodal_inputs: Dict[str, np.ndarray],
        input_frequency: float = 2.0,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        Processes a single unified execution step across all 3 structural coupling principles.
        """
        self.step_count += 1

        # 1. Cross-Modal Manifold Projection
        unified_manifold_vec, modal_coherence = self.cross_modal_projection.cross_resonate(multimodal_inputs)

        # 2. Meta-Plastic Prismatic Lens Refraction
        refracted_wave = self.prismatic_lens.refract(unified_manifold_vec)

        # 3. Resonant Attractor Coupling (Conditionless attraction)
        attracted_state, active_basin_idx, peak_resonance = self.attractor_coupling.evaluate_resonance(
            input_wave=refracted_wave,
            input_frequency=input_frequency
        )

        # 4. Propagate into ELYSIA Macro Topological Phase Runtime
        macro_step_record = self.macro_phase_runtime.trigger_wave_propagation(
            external_trigger=attracted_state,
            dt=dt
        )

        # 5. Record trajectory and re-tune Meta-Plastic Lens periodically
        self.trajectory_buffer.append(attracted_state.copy())
        if len(self.trajectory_buffer) > 20:
            self.trajectory_buffer.pop(0)

        if self.step_count % 5 == 0 and len(self.trajectory_buffer) >= 5:
            traj_array = np.array(self.trajectory_buffer)
            self.prismatic_lens.adapt_lens_from_trajectory(traj_array)

        # 6. Evaluate Invariant Core Stability
        invariant_diff = float(np.linalg.norm(np.dot(self.prismatic_lens.prism_matrix, attracted_state) - attracted_state))

        return {
            "step": self.step_count,
            "modal_coherence": modal_coherence,
            "active_basin_idx": active_basin_idx,
            "peak_resonance": peak_resonance,
            "prism_refractive_index": self.prismatic_lens.refractive_index,
            "macro_coherence_score": macro_step_record["coherence_score"],
            "macro_boundary_tension": macro_step_record["boundary_tension"],
            "invariant_core_stability": 1.0 / (1.0 + invariant_diff),
            "attracted_state": attracted_state,
        }
