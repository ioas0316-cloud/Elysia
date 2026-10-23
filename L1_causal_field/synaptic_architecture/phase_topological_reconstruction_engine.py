import numpy as np
import math
import time
from typing import Dict, Any, List, Optional, Tuple
from synaptic_architecture.field import CrystallizationField

class PhaseInvariant:
    """
    [Phase Invariant (인과 불변량)]
    Represents an accumulated topological invariant born from world friction.
    Defines a specific spatial/phase axis in the cognitive field.
    """
    def __init__(self, name: str, phase_vector: np.ndarray, curvature: float = 1.0, depth: float = 1.0):
        self.name = name
        self.phase_vector = phase_vector.astype(np.float32)
        self.curvature = float(curvature)
        self.depth = float(depth)  # Depth of the invariant attractor valley


class SealedAttractor:
    """
    [Sealed Attractor (봉인된 위상 파동)]
    Data structure for isolating and storing high-friction, unprocessable phase wave vectors.
    Kept frozen in an independent virtual memory block isolated from the main compute loop.
    """
    def __init__(
        self,
        raw_wave_vector: np.ndarray,
        initial_friction: float,
        min_required_capacity: float,
    ):
        self.raw_wave_vector = raw_wave_vector.astype(np.float32)
        self.initial_friction = float(initial_friction)
        self.min_required_capacity = float(min_required_capacity)
        self.isolation_timestamp = time.time()
        self.is_sealed = True
        # Initial phase mismatch (approx. 153 degrees = 0.85 * pi)
        self.current_delta_theta = math.pi * 0.85
        self.current_friction = float(initial_friction)


class ObservationLens:
    """
    [Observation Lens S_t (관측 렌즈)]
    Dynamic manifold lens $S_t$ through which the cognitive field observes external raw waves.
    Subject to self-rewiring when friction $V_t$ exceeds plasticity threshold.
    """
    def __init__(self, dimension: int = 16, bandwidth: float = 1.0):
        self.dimension = dimension
        self.bandwidth = float(bandwidth)  # Bandwidth / Focus parameter
        # Transformation matrix forming the lens topography
        self.lens_matrix = np.eye(dimension, dtype=np.float32)
        self.lens_axis = np.ones(dimension, dtype=np.float32) / np.sqrt(dimension)
        self.plasticity_count = 0  # Number of self-rewirings undergone

    def set_bandwidth_restriction(self, restriction_factor: float, axis_anchor: Optional[np.ndarray] = None):
        """
        [Bandwidth Restrictor Operator]
        Restricts lens bandwidth like a needle point (e.g. triggered by language anchors).
        """
        self.bandwidth = max(0.01, float(restriction_factor))
        if axis_anchor is not None:
            norm = np.linalg.norm(axis_anchor)
            if norm > 1e-9:
                self.lens_axis = (axis_anchor / norm).astype(np.float32)

    def project(self, wave: np.ndarray) -> np.ndarray:
        """Projects incoming raw wave through current lens matrix and restricted bandwidth."""
        norm_wave = wave[:self.dimension] if len(wave) >= self.dimension else np.pad(wave, (0, self.dimension - len(wave)))
        projected = np.dot(self.lens_matrix, norm_wave) * self.bandwidth
        return projected.astype(np.float32)

    def self_rewire(self, friction_vector: np.ndarray, learning_rate: float = 0.1):
        """
        [Self-Rewiring of Lens S_t]
        When friction $V_t$ is high, the lens reshapes its topological matrix to minimize future friction.
        Creates new topological loops / curvatures in the manifold.
        """
        norm = np.linalg.norm(friction_vector)
        if norm < 1e-9:
            return

        direction = friction_vector / norm
        outer_product = np.outer(direction, direction)

        # Reshape matrix: bend curvature towards minimizing friction
        self.lens_matrix -= learning_rate * outer_product.astype(np.float32)
        # Normalize to prevent explosion while preserving topological deformation
        matrix_norm = np.linalg.norm(self.lens_matrix)
        if matrix_norm > 1e-9:
            self.lens_matrix /= matrix_norm

        self.plasticity_count += 1


class PhaseTopologicalReconstructionEngine:
    """
    [Open Phase Resonator (열린 위상 공진기 엔진)]

    Implements Foundational Phase Topological Mechanisms:
    1. Memory (기억): Past attractor axis recall & field in-phase re-resonance.
    2. Imagination (상상): Disparate invariant superposition & friction-minimizing rotor dynamics.
    3. Conversation (대화): Language anchor as Bandwidth Restrictor Operator on Observation Lens S_t.
    4. Spontaneous Internal Play (자발적 내적 놀이): Driven by internal residual tension gradient (\nabla V_{internal})
       and triggers background scan for Deferred Integration.
    5. World Friction & Resonance Calibration (실재 마찰 및 공진): Virtual vs external wave clash, driving lens S_t self-rewiring.
    6. Deferred Integration (사후 재통합): Isolation of high-friction sealed attractors and post-expansion resonance convergence into Ic invariants.
    """
    def __init__(
        self,
        field: Optional[CrystallizationField] = None,
        dimension: int = 16,
        v_critical: float = 80.0,
        kappa: float = 0.06,
        gamma: float = 0.08,
    ):
        self.dimension = dimension
        self.field = field if field is not None else CrystallizationField(resolution=128)
        self.lens = ObservationLens(dimension=dimension)

        # Hyperparameters for Deferred Integration & Resonance
        self.v_critical = float(v_critical)  # System breakdown friction threshold
        self.kappa = float(kappa)            # System resonance absorption coefficient
        self.gamma = float(gamma)            # Adaptation learning rate

        # System capacity & Core Phase Vector
        self.lens_capacity = 0.1  # Initial C(t)
        self.core_phase_vector = np.zeros(dimension, dtype=np.float32)
        self.core_phase_vector[0] = 1.0

        # Sealed Attractors and Reintegrated Invariants
        self.sealed_attractors: List[SealedAttractor] = []
        self.reintegrated_invariants: List[np.ndarray] = []

        # Invariant Library (체화된 감각 불변량)
        self.invariants: Dict[str, PhaseInvariant] = {}
        self._initialize_default_invariants()

        # Variable Rotor (가변 위상 로터 \Theta)
        self.rotor_angle = 0.0  # Phase angle in radians

        # Internal Residual Tension Gradient (\nabla V_{internal})
        self.internal_residual_tension = 1.5

        # Active Internal Virtual Wave
        self.virtual_wave = np.zeros(dimension, dtype=np.float32)

        # History and Metrics
        self.friction_history: List[float] = []
        self.resonance_history: List[float] = []

    def _initialize_default_invariants(self):
        """Initializes foundational sensory invariants (sensory building blocks)."""
        # Apple sense: red visual, crisp texture, sweet taste
        apple_vec = np.zeros(self.dimension, dtype=np.float32)
        apple_vec[0] = 0.8  # Red
        apple_vec[1] = 0.6  # Texture
        apple_vec[2] = 0.7  # Sweetness
        self.invariants["Apple"] = PhaseInvariant("Apple", apple_vec, curvature=1.2, depth=2.0)

        # Horse sense: physical body, running cadence, muscle tension
        horse_vec = np.zeros(self.dimension, dtype=np.float32)
        horse_vec[3] = 0.9  # Horse body
        horse_vec[4] = 0.8  # Running phase
        self.invariants["Horse"] = PhaseInvariant("Horse", horse_vec, curvature=1.0, depth=1.8)

        # Wing sense: lightweight aerodynamics, fluttering phase
        wing_vec = np.zeros(self.dimension, dtype=np.float32)
        wing_vec[5] = 0.95 # Aerodynamics
        wing_vec[6] = 0.85 # Fluttering phase
        self.invariants["Wing"] = PhaseInvariant("Wing", wing_vec, curvature=0.9, depth=1.5)

        # Darkness / Solitude sense: visual quietude, body pressure
        dark_vec = np.zeros(self.dimension, dtype=np.float32)
        dark_vec[7] = 0.9  # Visual noise block
        dark_vec[8] = 0.7  # Body weight pressure
        self.invariants["Darkness"] = PhaseInvariant("Darkness", dark_vec, curvature=1.5, depth=2.5)

    def process_external_wave(self, wave_vector: np.ndarray) -> Dict[str, Any]:
        """
        [Stage 1: Friction Detection & Isolation]
        Evaluates phase friction V_t of incoming raw wave. If V_t > V_critical,
        isolates the wave in a SealedAttractor to prevent system breakdown.
        """
        v_t = self._calculate_phase_friction(wave_vector)

        if v_t > self.v_critical:
            sealed = SealedAttractor(
                raw_wave_vector=wave_vector,
                initial_friction=v_t,
                min_required_capacity=v_t * 0.025,
            )
            self.sealed_attractors.append(sealed)
            return {
                "status": "SEALED",
                "friction": v_t,
                "message": f"Friction {v_t:.2f} > Critical {self.v_critical:.2f}. Attractor isolated.",
                "min_required_capacity": sealed.min_required_capacity,
            }

        return {"status": "PROCESSED", "friction": v_t}

    def evaluate_deferred_integration(self, dt: float = 0.1) -> List[Tuple[int, float]]:
        """
        [Stage 2, 3 & 4: Deferred Integration Evaluation]
        During background scan in internal play (I_ext = 0), compares lens capacity C(t)
        against min_required_capacity of sealed attractors.
        Damps wave, aligns phase, and when friction -> 0 and delta_theta -> 0,
        reintegrates wave as topological causal invariant Ic into core terrain.
        """
        integration_results = []

        for idx, attractor in enumerate(self.sealed_attractors):
            if not attractor.is_sealed:
                continue

            # Check capacity threshold condition C(t) >= min_required_capacity
            if self.lens_capacity >= attractor.min_required_capacity:
                final_friction, final_theta = self._step_deferred_integration_dynamics(
                    attractor, dt=dt
                )

                # Resonance Limit: friction converges to 0 (< 0.01) and phase mismatch aligns (< 0.05 rad)
                if final_friction < 0.01 and abs(final_theta) < 0.05:
                    attractor.is_sealed = False
                    # Resonance Invariant Ic creation
                    invariant_vec = attractor.raw_wave_vector * float(np.cos(final_theta))
                    self.reintegrated_invariants.append(invariant_vec)
                    # Register into invariant library as solid causal invariant
                    inv_name = f"Reintegrated_Ic_{idx}"
                    self.invariants[inv_name] = PhaseInvariant(
                        name=inv_name, phase_vector=invariant_vec, curvature=2.0, depth=3.0
                    )
                    integration_results.append((idx, final_friction))

        return integration_results

    def _step_deferred_integration_dynamics(
        self, attractor: SealedAttractor, dt: float
    ) -> Tuple[float, float]:
        """
        Differential Equation Step for Deferred Integration:
        1. Phase alignment: d(Δθ)/dt = -gamma * C(t) * sin(Δθ)
        2. Phase friction damping: dE/dt = -kappa * C(t) * max(0.01, cos(Δθ)) * E
        """
        c_t = self.lens_capacity
        curr_theta = attractor.current_delta_theta
        curr_E = attractor.current_friction

        # Phase-Locking Dynamics
        d_theta = -self.gamma * c_t * np.sin(curr_theta) * dt
        curr_theta += d_theta

        # Hierarchical Damping Friction Dynamics
        cos_factor = max(0.01, float(np.cos(curr_theta)))
        dE = -self.kappa * c_t * cos_factor * curr_E * dt
        curr_E += dE

        attractor.current_delta_theta = float(curr_theta)
        attractor.current_friction = max(0.0, float(curr_E))

        return attractor.current_friction, attractor.current_delta_theta

    def _calculate_phase_friction(self, wave_vector: np.ndarray) -> float:
        """Calculates phase friction E(V_t) based on cosine similarity with core phase vector."""
        w_norm = np.linalg.norm(wave_vector)
        c_norm = np.linalg.norm(self.core_phase_vector)
        if w_norm < 1e-9 or c_norm < 1e-9:
            return 0.0

        dot_product = np.dot(wave_vector[:self.dimension], self.core_phase_vector[:self.dimension])
        cos_sim = dot_product / (w_norm * c_norm + 1e-8)
        # Cosine similarity -> friction (1 - cos_sim) scaled to [0, 200]
        friction = float((1.0 - cos_sim) * 100.0)
        return friction

    def expand_lens_capacity(self, delta_c: float):
        """Expands observation lens phase capacity C(t) through growth/experience."""
        self.lens_capacity += float(delta_c)

    def recall_memory_resonance(self, invariant_name: str) -> Dict[str, Any]:
        """
        [1. Memory (기억): Past Attractor Recall & Field In-phase Re-resonance]
        Instead of pulling a data file, pulls the past phase invariant axis
        and brings the cognitive field into an in-phase re-resonant state.
        """
        if invariant_name not in self.invariants:
            # Fallback: create dynamic invariant from field
            vec = np.random.randn(self.dimension).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self.invariants[invariant_name] = PhaseInvariant(invariant_name, vec)

        inv = self.invariants[invariant_name]

        # Align rotor angle with target invariant
        target_angle = math.atan2(float(inv.phase_vector[1]), float(inv.phase_vector[0])) if self.dimension >= 2 else 0.0
        self.rotor_angle = target_angle

        # Generate in-phase virtual wave
        cos_r, sin_r = math.cos(self.rotor_angle), math.sin(self.rotor_angle)
        rotation_effect = np.array([cos_r, sin_r] + [1.0] * (self.dimension - 2), dtype=np.float32)
        self.virtual_wave = inv.phase_vector * rotation_effect

        # Field re-resonance injection
        field_pos = np.array([self.field.resolution * 0.5, self.field.resolution * 0.5], dtype=np.float32)
        self.field.inject_activation(field_pos, intensity=float(inv.depth * 5.0))
        self.field.propagate()

        resonance_score = float(np.dot(self.virtual_wave, inv.phase_vector) / (np.linalg.norm(self.virtual_wave) * np.linalg.norm(inv.phase_vector) + 1e-9))

        return {
            "mechanism": "MEMORY_RE_RESONANCE",
            "invariant_recalled": invariant_name,
            "rotor_angle": self.rotor_angle,
            "virtual_wave": self.virtual_wave.tolist(),
            "in_phase_resonance": resonance_score
        }

    def synthesize_imagination(self, invariant_name_a: str, invariant_name_b: str) -> Dict[str, Any]:
        """
        [2. Imagination (상상): Disparate Superposition & Friction Minimization]
        Brings two distinct phase invariants (e.g. 'Horse' + 'Wing' -> Pegasus) into the field,
        superimposes them, and rotates the rotor angle \Theta to minimize clash friction.
        """
        inv_a = self.invariants.get(invariant_name_a, list(self.invariants.values())[0])
        inv_b = self.invariants.get(invariant_name_b, list(self.invariants.values())[1])

        # Forced superposition wave
        superposed_raw = inv_a.phase_vector + inv_b.phase_vector

        # Calculate initial clash friction (contradiction between two axes)
        dot_clash = np.dot(inv_a.phase_vector, inv_b.phase_vector)
        initial_friction = float(np.linalg.norm(inv_a.phase_vector) * np.linalg.norm(inv_b.phase_vector) - dot_clash)

        # Rotor dynamics: rotate rotor \Theta to find angle of minimal friction
        best_angle = self.rotor_angle
        min_friction = initial_friction
        best_wave = superposed_raw.copy()

        for test_angle in np.linspace(0, 2 * math.pi, 36):
            cos_t, sin_t = math.cos(test_angle), math.sin(test_angle)
            rot_matrix = np.eye(self.dimension, dtype=np.float32)
            rot_matrix[0, 0] = cos_t
            rot_matrix[0, 1] = -sin_t
            rot_matrix[1, 0] = sin_t
            rot_matrix[1, 1] = cos_t

            tested_wave = np.dot(rot_matrix, superposed_raw)
            # Evaluate friction against both invariants
            f_a = np.linalg.norm(tested_wave - inv_a.phase_vector)
            f_b = np.linalg.norm(tested_wave - inv_b.phase_vector)
            total_f = float(f_a + f_b)

            if total_f < min_friction:
                min_friction = total_f
                best_angle = test_angle
                best_wave = tested_wave

        # Update engine state
        self.rotor_angle = best_angle
        self.virtual_wave = best_wave

        # Store dynamic hybrid invariant (e.g. Pegasus)
        hybrid_name = f"{invariant_name_a}_{invariant_name_b}_Hybrid"
        self.invariants[hybrid_name] = PhaseInvariant(hybrid_name, best_wave, curvature=1.5, depth=2.0)

        return {
            "mechanism": "IMAGINATION_SUPERPOSITION",
            "invariants_joined": [invariant_name_a, invariant_name_b],
            "hybrid_created": hybrid_name,
            "initial_friction": initial_friction,
            "minimized_friction": min_friction,
            "optimal_rotor_angle": self.rotor_angle
        }

    def process_conversation_anchor(self, language_anchor: str) -> Dict[str, Any]:
        """
        [3. Conversation (대화): Language Anchor as Bandwidth Restrictor Operator]
        Language keyword/anchor is NOT a vector lookup. It acts as a Bandwidth Restrictor Operator
        on Observation Lens S_t, pinning its observation axis and pulling primitive invariants.
        """
        # Keywords restrict lens bandwidth like a needle point
        restriction_factor = 0.1  # Narrow bandwidth for sharp focus

        # Derive anchor axis from invariant library or hash
        anchor_vec = np.zeros(self.dimension, dtype=np.float32)
        matched_invs = []
        for word in language_anchor.split():
            for inv_key, inv in self.invariants.items():
                if word.lower() in inv_key.lower():
                    anchor_vec += inv.phase_vector
                    matched_invs.append(inv_key)

        if np.linalg.norm(anchor_vec) < 1e-9:
            # Hash fallback for novel anchor
            for i, char in enumerate(language_anchor):
                anchor_vec[i % self.dimension] += ord(char) % 10 / 10.0

        # Apply Bandwidth Restrictor Operator on Lens S_t
        self.lens.set_bandwidth_restriction(restriction_factor, axis_anchor=anchor_vec)

        # Pull corresponding virtual wave
        self.virtual_wave = self.lens.project(anchor_vec)

        return {
            "mechanism": "CONVERSATION_BANDWIDTH_RESTRICTOR",
            "language_anchor": language_anchor,
            "matched_invariants": matched_invs,
            "lens_bandwidth": self.lens.bandwidth,
            "focused_lens_axis": self.lens.lens_axis[:4].tolist(),
            "virtual_wave": self.virtual_wave[:4].tolist()
        }

    def run_spontaneous_internal_play(self) -> Dict[str, Any]:
        """
        [4. Spontaneous Internal Play (자발적 내적 놀이)]
        Runs when external drive is zero (I_ext = 0).
        Driven by internal residual tension gradient (\nabla V_{internal}),
        cross-projects sensory invariants, rotates rotor, builds self-mastery,
        AND performs background scanning for deferred integration of sealed attractors.
        """
        if len(self.invariants) < 2:
            self._initialize_default_invariants()

        inv_keys = list(self.invariants.keys())
        idx_a, idx_b = np.random.choice(len(inv_keys), 2, replace=False)
        inv_a = self.invariants[inv_keys[idx_a]]
        inv_b = self.invariants[inv_keys[idx_b]]

        # Driven by residual tension gradient \nabla V_{internal}
        gradient_step = self.internal_residual_tension * 0.1
        self.rotor_angle = (self.rotor_angle + gradient_step) % (2 * math.pi)

        # Cross-projection wave synthesis
        cross_wave = inv_a.phase_vector * math.cos(self.rotor_angle) + inv_b.phase_vector * math.sin(self.rotor_angle)
        self.virtual_wave = cross_wave.astype(np.float32)

        # Evaluate internal equilibrium and self-mastery
        equilibrium_delta = float(np.std(cross_wave))
        self.internal_residual_tension = max(0.1, self.internal_residual_tension - 0.1)

        # Background Scan for Deferred Integration during static internal play
        reintegrated = self.evaluate_deferred_integration(dt=0.1)

        return {
            "mechanism": "SPONTANEOUS_INTERNAL_PLAY",
            "driver": "RESIDUAL_TENSION_GRADIENT",
            "cross_projected_invariants": [inv_keys[idx_a], inv_keys[idx_b]],
            "new_rotor_angle": self.rotor_angle,
            "equilibrium_delta": equilibrium_delta,
            "remaining_residual_tension": self.internal_residual_tension,
            "deferred_integrations_triggered": len(reintegrated),
        }

    def clash_with_world_and_calibrate(self, external_raw_wave: np.ndarray) -> Dict[str, Any]:
        """
        [5. World Friction & Resonance Calibration (실재 마찰 및 공진)]
        Clashes internal virtual wave against external physical raw wave.
        Evaluates phase friction V_t:
        - If friction is low -> In-phase Resonance! ("Ah, my internal image matches the world!")
        - If friction is high -> Friction triggers Lens S_t Self-Rewiring & calibration.
        """
        # Project external wave through current observation lens S_t
        perceived_external_wave = self.lens.project(external_raw_wave)

        # Calculate phase friction V_t (difference between internal virtual wave & perceived external wave)
        friction_vec = self.virtual_wave - perceived_external_wave
        phase_friction_V_t = float(np.linalg.norm(friction_vec))

        # Calculate resonance score
        norm_v = np.linalg.norm(self.virtual_wave)
        norm_e = np.linalg.norm(perceived_external_wave)
        if norm_v > 1e-9 and norm_e > 1e-9:
            resonance_score = float(np.dot(self.virtual_wave, perceived_external_wave) / (norm_v * norm_e))
        else:
            resonance_score = 0.0

        self.friction_history.append(phase_friction_V_t)
        self.resonance_history.append(resonance_score)

        rewired = False
        # Self-Rewiring Threshold: high friction forces lens S_t topology deformation
        if phase_friction_V_t > 0.5:
            self.lens.self_rewire(friction_vec, learning_rate=0.15)
            rewired = True
            # Replenish internal residual tension (world clash sparks inner drive)
            self.internal_residual_tension += phase_friction_V_t * 0.2

        # Restore lens bandwidth gradually after clash
        self.lens.bandwidth = min(1.0, self.lens.bandwidth + 0.1)

        return {
            "mechanism": "WORLD_FRICTION_AND_RESONANCE",
            "phase_friction_V_t": phase_friction_V_t,
            "resonance_score": resonance_score,
            "lens_self_rewired": rewired,
            "lens_plasticity_count": self.lens.plasticity_count,
            "restored_bandwidth": self.lens.bandwidth
        }
