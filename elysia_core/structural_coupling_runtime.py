"""
Elysia Structural Coupling Runtime (구조결합 런타임)
=====================================================
Embodying the 3 Structural Coupling Principles & Generative Mechanism Differentiation:
1. Resonant Attractor Coupling (공명 어트랙터 결합):
   Conditionless phase-frequency basin convergence replacing hardcoded if-else branching.
2. Meta-Plastic Prismatic Lens (메타-가소성 프리즘 렌즈):
   Dynamic refractive matrix (PrismMatrix) re-tuning & cellular bifurcation driven by cognitive friction.
3. Cross-Modal Manifold Projection (비동기 다중모달 교차 투영):
   Cross-modal mapping onto a unified Clifford Rotor field.
4. Generative Mechanism Prismatic Differentiation (생성 메커니즘 프리즘 분별):
   Phase torque (tau_phase) and cognitive friction (F_cog) driven cellular bifurcation,
   differentiating identical surface profiles (e.g., Light vs. Crayon vs. Pastel yellow)
   into invariant generative cores (Theta).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from elysia_core.unified_topological_phase_runtime import UnifiedTopologicalPhaseRuntime


class GenerativeMechanismSynthesizer:
    """
    Synthesizes multi-scale sensory waveforms for identical surface profiles (e.g. Hex #FFFF00),
    embodying 3 distinct physical generative mechanisms:
    1. 'light': Photon emission - pure, smooth frequency wave with zero surface friction.
    2. 'crayon': Wax pressing/drag - heavy low-frequency harmonics, structural viscosity, high inertia.
    3. 'pastel': Powder particle scattering - chaotic high-frequency phase jitter, micro-groove torque.
    """

    @staticmethod
    def synthesize_waveform(
        mechanism: str,
        surface_color: np.ndarray = np.array([1.0, 1.0, 0.0]),
        num_samples: int = 128
    ) -> Dict[str, Any]:
        t = np.linspace(0, 4 * np.pi, num_samples)
        base_freq = 2.0

        if mechanism == "light":
            # Pure photon emission
            waveform = np.sin(base_freq * t) + 0.5 * np.cos(base_freq * 2 * t)
            viscosity = 0.01
            micro_roughness = 0.02
            phase_jitter = 0.01
        elif mechanism == "crayon":
            # Wax pressing & drag: low frequency heavy harmonics, viscous resistance
            waveform = (
                1.2 * np.sin(base_freq * 0.5 * t)
                + 0.8 * np.sin(base_freq * t)
                + 0.4 * np.tanh(np.sin(base_freq * 3 * t))
            )
            viscosity = 0.85
            micro_roughness = 0.45
            phase_jitter = 0.12
        elif mechanism == "pastel":
            # Powder particle scattering in paper grooves: high frequency chaotic jitter & torque spikes
            rng = np.random.default_rng(12345)
            scattering_noise = 0.35 * rng.standard_normal(num_samples)
            waveform = (
                np.sin(base_freq * t)
                + 0.6 * np.sin(base_freq * 5 * t)
                + scattering_noise
            )
            viscosity = 0.25
            micro_roughness = 0.92
            phase_jitter = 0.78
        else:
            waveform = np.sin(base_freq * t)
            viscosity = 0.1
            micro_roughness = 0.1
            phase_jitter = 0.1

        # Calculate FFT spectral fingerprint
        fft_vals = np.fft.rfft(waveform)
        spectral_power = np.abs(fft_vals)
        phases = np.angle(fft_vals)

        return {
            "mechanism": mechanism,
            "surface_color": surface_color,
            "waveform": waveform,
            "viscosity": viscosity,
            "micro_roughness": micro_roughness,
            "phase_jitter": phase_jitter,
            "spectral_power": spectral_power,
            "phases": phases,
        }


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

    def set_dimension(self, new_dimension: int):
        if new_dimension == self.dimension:
            return
        old_dim = self.dimension
        self.dimension = new_dimension
        new_phases = np.zeros((self.num_basins, new_dimension))
        for k in range(self.num_basins):
            m_k = min(old_dim, len(self.basin_phases[k]))
            new_phases[k, :m_k] = self.basin_phases[k][:m_k]
        self.basin_phases = new_phases
        self.current_state = np.pad(self.current_state, (0, max(0, new_dimension - len(self.current_state))))[:new_dimension]

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
    based on the spectral fingerprint of accumulated interaction trajectories and cognitive friction.
    Includes cellular bifurcation (세포 분열적 분화) into invariant generative cores (Theta).
    """

    def __init__(self, dimension: int = 5, plasticity_rate: float = 0.1, friction_threshold: float = 0.25):
        self.dimension = dimension
        self.plasticity_rate = plasticity_rate
        self.friction_threshold = friction_threshold

        # Initial un-differentiated isotropic refraction matrix
        self.prism_matrix = np.eye(dimension, dtype=float)
        self.spectral_history: List[np.ndarray] = []
        self.refractive_index = 1.5

        # Generative Invariant Core Registry (Theta) resulting from cellular bifurcation
        # Maps mechanism_key -> (Sub-Prism Matrix, Invariant Vector)
        self.bifurcated_cores: Dict[str, Dict[str, np.ndarray]] = {}
        self.bifurcation_count = 0

    def set_dimension(self, new_dimension: int):
        """
        Adapts lens matrices and bifurcated core matrices to new dimension during horizon expansion.
        """
        if new_dimension == self.dimension:
            return

        old_dim = self.dimension
        self.dimension = new_dimension

        new_prism = np.eye(new_dimension, dtype=float)
        new_prism[:old_dim, :old_dim] = self.prism_matrix[:old_dim, :old_dim]
        self.prism_matrix = new_prism

        for key, core in self.bifurcated_cores.items():
            old_mat = core["prism_matrix"]
            m_dim = old_mat.shape[0]
            new_mat = np.eye(new_dimension, dtype=float)
            new_mat[:m_dim, :m_dim] = old_mat
            core["prism_matrix"] = new_mat
            core["invariant_vector"] = np.pad(core["invariant_vector"], (0, max(0, new_dimension - len(core["invariant_vector"]))))[:new_dimension]

    def refract(self, incoming_tensor: np.ndarray, mechanism_hint: Optional[str] = None) -> np.ndarray:
        """
        Passes incoming tensor through the current Meta-Plastic Prism Lens or its bifurcated core.
        """
        padded = np.pad(incoming_tensor, (0, max(0, self.dimension - len(incoming_tensor))))[:self.dimension]

        if mechanism_hint and mechanism_hint in self.bifurcated_cores:
            matrix = self.bifurcated_cores[mechanism_hint]["prism_matrix"]
        else:
            matrix = self.prism_matrix

        if matrix.shape[0] != self.dimension:
            padded_matrix = np.eye(self.dimension, dtype=float)
            m_dim = min(matrix.shape[0], self.dimension)
            padded_matrix[:m_dim, :m_dim] = matrix[:m_dim, :m_dim]
            matrix = padded_matrix

        refracted = np.dot(matrix, padded) / self.refractive_index
        return refracted

    def compute_phase_torque_and_friction(
        self,
        sensory_profile: Dict[str, Any],
        observed_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Computes Phase Torque (tau_phase) and Cognitive Friction Index (F_cog).
        Phase torque measures the angular/phase discrepancy between the lens expectation
        and the actual micro-waveform texture (viscosity, micro-roughness, phase-jitter).
        """
        padded_obs = np.pad(observed_vector, (0, max(0, self.dimension - len(observed_vector))))[:self.dimension]
        norm_obs = np.linalg.norm(padded_obs)
        if norm_obs > 1e-6:
            padded_obs /= norm_obs

        mechanism = sensory_profile.get("mechanism", "unknown")

        # If already bifurcated and aligned with mechanism, expectation matches reality
        if mechanism in self.bifurcated_cores:
            expected_core = self.bifurcated_cores[mechanism]["invariant_vector"]
            expected_core = np.pad(expected_core, (0, max(0, self.dimension - len(expected_core))))[:self.dimension]
            phase_diff = padded_obs - expected_core
            phase_torque = np.cross(padded_obs[:3], expected_core[:3]) if self.dimension >= 3 else phase_diff[:3]
            torque_magnitude = float(np.linalg.norm(phase_torque)) + float(np.linalg.norm(phase_diff))

            # Residual cognitive friction approaches 0 after reorganization
            cognitive_friction = 0.15 * torque_magnitude
            return phase_torque, float(cognitive_friction)

        # Un-differentiated state: surface label matches (e.g. Yellow), but waveform micro-texture collides
        roughness = sensory_profile.get("micro_roughness", 0.1)
        viscosity = sensory_profile.get("viscosity", 0.1)
        jitter = sensory_profile.get("phase_jitter", 0.1)

        # Micro-texture mismatch creates phase torque vector
        torque_components = np.array([
            viscosity * np.sin(padded_obs[0] * np.pi),
            roughness * np.cos(padded_obs[1] * np.pi),
            jitter * np.sin(padded_obs[2] * np.pi),
            0.5 * roughness * viscosity,
            0.5 * jitter * roughness
        ])[:self.dimension]

        phase_torque = torque_components
        torque_norm = float(np.linalg.norm(phase_torque))

        # Cognitive Friction Index: F_cog = ||tau_phase|| * (1 + TextureMismatch)
        texture_mismatch = roughness + viscosity + jitter
        cognitive_friction = torque_norm * (1.0 + texture_mismatch)

        return phase_torque, float(cognitive_friction)

    def trigger_cellular_bifurcation(
        self,
        sensory_profile: Dict[str, Any],
        observed_vector: np.ndarray,
        phase_torque: np.ndarray
    ) -> Dict[str, Any]:
        """
        Triggers cellular bifurcation (세포 분열적 분화) of the prism lens matrix
        when cognitive friction exceeds threshold, carving out a dedicated Invariant Core (Theta).
        """
        mechanism = sensory_profile.get("mechanism", f"core_{self.bifurcation_count + 1}")
        padded_obs = np.pad(observed_vector, (0, max(0, self.dimension - len(observed_vector))))[:self.dimension]
        norm_obs = np.linalg.norm(padded_obs)
        if norm_obs > 1e-6:
            padded_obs /= norm_obs

        # Construct specialized prism matrix for this generative mechanism
        torque_outer = np.outer(phase_torque, phase_torque)
        norm_to = np.linalg.norm(torque_outer)
        if norm_to > 1e-6:
            torque_outer /= norm_to

        # Bifurcated prism matrix aligns with physical constraint profile
        specialized_matrix = np.eye(self.dimension) + 0.35 * torque_outer + 0.25 * np.outer(padded_obs, padded_obs)

        # Normalize and condition matrix
        u, s, vh = np.linalg.svd(specialized_matrix)
        specialized_matrix = np.dot(u * np.clip(s, 0.5, 3.0), vh)

        # Store invariant core
        self.bifurcated_cores[mechanism] = {
            "prism_matrix": specialized_matrix,
            "invariant_vector": padded_obs.copy(),
            "viscosity": sensory_profile.get("viscosity", 0.0),
            "micro_roughness": sensory_profile.get("micro_roughness", 0.0),
        }
        self.bifurcation_count += 1

        # Also refine global prism matrix
        self.prism_matrix = 0.8 * self.prism_matrix + 0.2 * specialized_matrix

        return {
            "mechanism": mechanism,
            "bifurcation_index": self.bifurcation_count,
            "invariant_core_vector": padded_obs,
            "total_cores": len(self.bifurcated_cores)
        }

    def adapt_lens_from_trajectory(self, trajectory_samples: np.ndarray):
        """
        Computes the spectral fingerprint of the trajectory (frequency components across time)
        and re-tunes the refractive prism matrix via meta-plastic feedback.
        """
        if len(trajectory_samples) < 2:
            return

        spectral_fft = np.fft.rfft(trajectory_samples, axis=0)
        spectral_power = np.abs(spectral_fft)

        cov_spectrum = np.cov(spectral_power.T)
        if cov_spectrum.shape != (self.dimension, self.dimension):
            cov_spectrum = np.eye(self.dimension) * np.mean(spectral_power)

        cov_norm = np.linalg.norm(cov_spectrum)
        if cov_norm > 1e-6:
            cov_spectrum /= cov_norm

        self.prism_matrix = (1.0 - self.plasticity_rate) * self.prism_matrix + self.plasticity_rate * (np.eye(self.dimension) + cov_spectrum)

        u, s, vh = np.linalg.svd(self.prism_matrix)
        s_clamped = np.clip(s, 0.2, 5.0)
        self.prism_matrix = np.dot(u * s_clamped, vh)

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
        rng = np.random.default_rng(42)
        self.projection_operators = {
            "physical": np.eye(target_dim),
            "sensory": np.eye(target_dim),
            "chromatic": np.eye(target_dim),
            "symbolic": np.eye(target_dim),
        }

    def set_dimension(self, new_dimension: int):
        self.target_dim = new_dimension
        for mod in list(self.projection_operators.keys()):
            self.projection_operators[mod] = np.eye(new_dimension)

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

        stacked = np.array(projected_vectors)
        unified_field = np.mean(stacked, axis=0)
        u_norm = np.linalg.norm(unified_field)
        if u_norm > 1e-6:
            unified_field /= u_norm

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
    Includes Generative Mechanism Prismatic Differentiation (Light vs. Crayon vs. Pastel Yellow).
    """

    def __init__(self, target_dim: int = 5, friction_threshold: float = 0.25):
        self.target_dim = target_dim
        self.friction_threshold = friction_threshold
        self.attractor_coupling = ResonantAttractorCoupling(dimension=target_dim)
        self.prismatic_lens = MetaPlasticPrismaticLens(dimension=target_dim, friction_threshold=friction_threshold)
        self.cross_modal_projection = CrossModalManifoldProjection(target_dim=target_dim)
        self.macro_phase_runtime = UnifiedTopologicalPhaseRuntime()

        self.trajectory_buffer: List[np.ndarray] = []
        self.step_count = 0
        self.learning_episodes: List[Dict[str, Any]] = []

    def set_dimension(self, new_dimension: int):
        self.target_dim = new_dimension
        self.attractor_coupling.set_dimension(new_dimension)
        self.prismatic_lens.set_dimension(new_dimension)
        self.cross_modal_projection.set_dimension(new_dimension)
        self.trajectory_buffer = [
            np.pad(vec, (0, max(0, new_dimension - len(vec))))[:new_dimension]
            for vec in self.trajectory_buffer
        ]

    def perceive_generative_stimulus(
        self,
        mechanism: str,
        surface_color: np.ndarray = np.array([1.0, 1.0, 0.0])
    ) -> Dict[str, Any]:
        """
        Perceives a sensory stimulus with explicit generative mechanism (light, crayon, pastel)
        and processes it through the cognitive friction and cellular bifurcation loop.
        """
        sensory_profile = GenerativeMechanismSynthesizer.synthesize_waveform(
            mechanism=mechanism,
            surface_color=surface_color
        )

        modal_inputs = {
            "chromatic": surface_color,
            "physical": sensory_profile["waveform"][:self.target_dim],
            "sensory": np.array([
                sensory_profile["viscosity"],
                sensory_profile["micro_roughness"],
                sensory_profile["phase_jitter"],
                1.0,
                0.5
            ])[:self.target_dim]
        }

        return self.process_step(
            multimodal_inputs=modal_inputs,
            sensory_profile=sensory_profile
        )

    def process_step(
        self,
        multimodal_inputs: Dict[str, np.ndarray],
        input_frequency: float = 2.0,
        dt: float = 0.1,
        sensory_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes a single unified execution step across all structural coupling principles.
        """
        self.step_count += 1

        # 1. Cross-Modal Manifold Projection
        unified_manifold_vec, modal_coherence = self.cross_modal_projection.cross_resonate(multimodal_inputs)

        # 2. Compute Cognitive Friction & Phase Torque
        profile = sensory_profile if sensory_profile is not None else {"mechanism": "unknown"}
        phase_torque, cognitive_friction = self.prismatic_lens.compute_phase_torque_and_friction(
            sensory_profile=profile,
            observed_vector=unified_manifold_vec
        )

        # 3. Check for Cellular Bifurcation (세포 분열적 분화)
        bifurcated = False
        bifurcation_info = None
        mechanism_key = profile.get("mechanism", None)

        if cognitive_friction > self.friction_threshold and mechanism_key not in self.prismatic_lens.bifurcated_cores:
            bifurcated = True
            bifurcation_info = self.prismatic_lens.trigger_cellular_bifurcation(
                sensory_profile=profile,
                observed_vector=unified_manifold_vec,
                phase_torque=phase_torque
            )

        # 4. Meta-Plastic Prismatic Lens Refraction
        refracted_wave = self.prismatic_lens.refract(
            incoming_tensor=unified_manifold_vec,
            mechanism_hint=mechanism_key
        )

        # 5. Resonant Attractor Coupling (Conditionless attraction)
        attracted_state, active_basin_idx, peak_resonance = self.attractor_coupling.evaluate_resonance(
            input_wave=refracted_wave,
            input_frequency=input_frequency
        )

        # 6. Propagate into ELYSIA Macro Topological Phase Runtime
        macro_step_record = self.macro_phase_runtime.trigger_wave_propagation(
            external_trigger=attracted_state,
            dt=dt
        )

        # 7. Record trajectory and re-tune Meta-Plastic Lens periodically
        self.trajectory_buffer.append(attracted_state.copy())
        if len(self.trajectory_buffer) > 20:
            self.trajectory_buffer.pop(0)

        if self.step_count % 5 == 0 and len(self.trajectory_buffer) >= 5:
            traj_array = np.array(self.trajectory_buffer)
            self.prismatic_lens.adapt_lens_from_trajectory(traj_array)

        # 8. Evaluate Invariant Core Stability
        invariant_diff = float(np.linalg.norm(np.dot(self.prismatic_lens.prism_matrix, attracted_state) - attracted_state))

        step_summary = {
            "step": self.step_count,
            "mechanism": mechanism_key,
            "phase_torque_norm": float(np.linalg.norm(phase_torque)),
            "cognitive_friction": cognitive_friction,
            "bifurcated": bifurcated,
            "bifurcation_info": bifurcation_info,
            "total_bifurcated_cores": len(self.prismatic_lens.bifurcated_cores),
            "modal_coherence": modal_coherence,
            "active_basin_idx": active_basin_idx,
            "peak_resonance": peak_resonance,
            "prism_refractive_index": self.prismatic_lens.refractive_index,
            "macro_coherence_score": macro_step_record["coherence_score"],
            "macro_boundary_tension": macro_step_record["boundary_tension"],
            "invariant_core_stability": 1.0 / (1.0 + invariant_diff),
            "attracted_state": attracted_state,
        }

        self.learning_episodes.append(step_summary)
        return step_summary
