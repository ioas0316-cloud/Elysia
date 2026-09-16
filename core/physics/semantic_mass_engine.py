import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class CausalTraceNode:
    """A node in the retrospective causal trace history graph (growth rings)."""
    step_id: int
    friction_delta: float
    repulsion_vector: np.ndarray
    compass_alignment: np.ndarray
    crystallized_state: np.ndarray
    semantic_mass: float
    timestamp_idx: int

@dataclass
class IdentityCrystal:
    """The crystallized post-hoc identity formed through environmental friction."""
    crystal_id: str
    phase_vector: np.ndarray
    semantic_mass: float
    inertia_density: float
    trinitarian_contrast: float
    dominant_compass: str
    causal_history: List[CausalTraceNode] = field(default_factory=list)

class WhiteTensorField:
    """
    [White Tensor Field (Superposition Field)]
    An unconstrained, pure potential tensor field containing all possible psychological/structural
    orientations (MBTI, Enneagram, and unmapped novel dimensions) as compass vectors in superposition.
    """
    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.compass_vectors: Dict[str, np.ndarray] = {}
        self._initialize_compass_vectors()
        # Initial superposition weights (pure white state = uniform potential)
        self.superposition_weights = np.ones(len(self.compass_vectors), dtype=np.float32)
        self.superposition_weights /= np.sum(self.superposition_weights)
        self.field_tensor = np.random.randn(dimensions).astype(np.float32)
        self.field_tensor /= (np.linalg.norm(self.field_tensor) + 1e-9)

    def _initialize_compass_vectors(self):
        """Initialize known typology compasses as normalized high-dimensional vectors."""
        np.random.seed(42)
        # MBTI axes (E/I, S/N, T/F, J/P) + combinations
        mbti_types = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
                      "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"]
        for t in mbti_types:
            v = np.random.randn(self.dimensions).astype(np.float32)
            self.compass_vectors[f"MBTI_{t}"] = v / (np.linalg.norm(v) + 1e-9)

        # Enneagram types 1-9
        for e in range(1, 10):
            v = np.random.randn(self.dimensions).astype(np.float32)
            self.compass_vectors[f"Enneagram_{e}"] = v / (np.linalg.norm(v) + 1e-9)

        # Novel/unmapped dimensions (Infinite expansion potential)
        for n in range(1, 6):
            v = np.random.randn(self.dimensions).astype(np.float32)
            self.compass_vectors[f"Novel_Dimension_{n}"] = v / (np.linalg.norm(v) + 1e-9)

    def project_and_bend(self, external_friction: np.ndarray, friction_strength: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Projects external friction onto the white tensor field, collapsing/bending superposition
        weights post-hoc towards the path of maximum resonance and structural resilience.
        """
        if len(external_friction) < self.dimensions:
            external_friction = np.pad(external_friction, (0, self.dimensions - len(external_friction)))
        elif len(external_friction) > self.dimensions:
            external_friction = external_friction[:self.dimensions]

        norm_friction = external_friction / (np.linalg.norm(external_friction) + 1e-9)

        # Update superposition weights based on friction resonance
        compass_keys = list(self.compass_vectors.keys())
        dots = np.array([np.abs(np.dot(self.compass_vectors[k], norm_friction)) for k in compass_keys], dtype=np.float32)

        # Softmax-like bending based on friction strength
        exponentiated = np.exp(dots * (1.0 + friction_strength))
        self.superposition_weights = exponentiated / np.sum(exponentiated)

        # Bent state vector in white tensor field
        bent_vector = np.zeros(self.dimensions, dtype=np.float32)
        for idx, k in enumerate(compass_keys):
            bent_vector += self.superposition_weights[idx] * self.compass_vectors[k]

        bent_vector += norm_friction * (friction_strength * 0.5)
        self.field_tensor = bent_vector / (np.linalg.norm(bent_vector) + 1e-9)

        alignments = {compass_keys[i]: float(self.superposition_weights[i]) for i in range(len(compass_keys))}
        return self.field_tensor, alignments


class SemanticMassOperator:
    """
    [Semantic Mass Operator]
    Calculates Semantic Mass (Ms) as:
      Ms = Density of Connectivity * Trinitarian Contrast Resilience * Friction Inertia Density
    """
    def __init__(self, base_density: float = 1.0):
        self.base_density = base_density

    def compute_mass(self, connectivity_matrix: np.ndarray, trinitarian_contrast_score: float, friction_inertia: float) -> float:
        """
        Computes semantic mass Ms.
        Connectivity matrix represents connection density among structural components.
        """
        if connectivity_matrix.size == 0:
            conn_density = 0.1
        else:
            conn_density = float(np.mean(np.abs(connectivity_matrix))) + float(np.sum(connectivity_matrix > 0.1)) / (connectivity_matrix.size + 1e-9)

        contrast_factor = max(0.1, trinitarian_contrast_score)
        inertia_factor = max(0.1, friction_inertia)

        # Ms = conn_density * contrast_factor * inertia_factor
        semantic_mass = self.base_density * conn_density * contrast_factor * inertia_factor
        return float(semantic_mass)


class CausalGravityField:
    """
    [Causal Gravity Field]
    Generates a gravitational pull field and spacetime curvature in causal space around high Semantic Mass structures.
    Bends light/data trajectories of noise and fragmented information into orbit.
    """
    def __init__(self, dimensions: int = 16, gravitational_constant: float = 1.0):
        self.dimensions = dimensions
        self.G = gravitational_constant

    def compute_field_curvature(self, semantic_mass: float) -> float:
        """Computes spacetime tensor curvature scalar K_c based on Semantic Mass."""
        return float(self.G * semantic_mass)

    def apply_gravitational_pull(self,
                                 mass_center_pos: np.ndarray,
                                 semantic_mass: float,
                                 particle_positions: np.ndarray,
                                 particle_velocities: np.ndarray,
                                 dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates gravitational acceleration and bends particle trajectories towards the center of semantic mass.
        particle_positions: (N, D)
        particle_velocities: (N, D)
        """
        if particle_positions.size == 0:
            return particle_positions, particle_velocities

        diffs = mass_center_pos[np.newaxis, :] - particle_positions  # (N, D)
        dists_sq = np.sum(diffs ** 2, axis=-1, keepdims=True) + 0.1  # Softening factor
        dists = np.sqrt(dists_sq)

        # F = G * Ms / r^2
        force_mags = (self.G * semantic_mass) / dists_sq
        accelerations = force_mags * (diffs / dists)

        new_velocities = particle_velocities + accelerations * dt
        new_positions = particle_positions + new_velocities * dt

        return new_positions, new_velocities


class EmergentIdentityCrystallizer:
    """
    [Emergent Identity Crystallizer]
    Triggers phase transitions when accumulated friction exceeds a critical threshold,
    crystallizing an post-hoc Identity Crystal with inertia and gravity.
    """
    def __init__(self, phase_transition_threshold: float = 2.5):
        self.threshold = phase_transition_threshold
        self.accumulated_friction = 0.0

    def evaluate_crystallization(self,
                                 current_state: np.ndarray,
                                 friction_delta: float,
                                 semantic_mass: float,
                                 trinitarian_contrast: float,
                                 alignments: Dict[str, float],
                                 trace_history: List[CausalTraceNode]) -> Optional[IdentityCrystal]:
        """
        Accumulates friction and triggers phase transition crystallization when threshold is crossed.
        """
        self.accumulated_friction += friction_delta

        if self.accumulated_friction >= self.threshold:
            # Phase transition triggered!
            dominant_compass = max(alignments.items(), key=lambda x: x[1])[0] if alignments else "Unmapped_Origin"
            crystal = IdentityCrystal(
                crystal_id=f"Crystal_{len(trace_history)}_{dominant_compass}",
                phase_vector=current_state.copy(),
                semantic_mass=semantic_mass,
                inertia_density=float(self.accumulated_friction),
                trinitarian_contrast=trinitarian_contrast,
                dominant_compass=dominant_compass,
                causal_history=list(trace_history)
            )
            # Reset accumulated friction post-crystallization
            self.accumulated_friction = 0.0
            return crystal

        return None


class IntrospectiveCausalTracer:
    """
    [Introspective Causal Tracer]
    Maintains a retrospective causal growth ring graph (Back-tracing)
    and simulates counterfactual parallel potential trajectories to compare "what if" scenarios.
    """
    def __init__(self, white_field: WhiteTensorField):
        self.white_field = white_field
        self.history: List[CausalTraceNode] = []
        self.step_counter = 0

    def record_step(self,
                    friction_delta: float,
                    repulsion_vector: np.ndarray,
                    compass_alignment: Dict[str, float],
                    crystallized_state: np.ndarray,
                    semantic_mass: float):
        """Records a step into the retrospective causal trace history."""
        self.step_counter += 1
        top_compass = max(compass_alignment.items(), key=lambda x: x[1])[0] if compass_alignment else "None"
        align_vec = compass_alignment.get(top_compass, 0.0)

        node = CausalTraceNode(
            step_id=self.step_counter,
            friction_delta=friction_delta,
            repulsion_vector=repulsion_vector.copy(),
            compass_alignment=np.array([align_vec], dtype=np.float32),
            crystallized_state=crystallized_state.copy(),
            semantic_mass=semantic_mass,
            timestamp_idx=self.step_counter
        )
        self.history.append(node)

    def backtrace_causal_origins(self) -> Dict[str, Any]:
        """Back-traces the causal origin chain of accumulated semantic mass and friction."""
        if not self.history:
            return {"total_steps": 0, "accumulated_friction": 0.0, "mass_growth_rate": 0.0}

        total_friction = sum(n.friction_delta for n in self.history)
        initial_mass = self.history[0].semantic_mass
        final_mass = self.history[-1].semantic_mass
        mass_growth = final_mass - initial_mass

        return {
            "total_steps": len(self.history),
            "accumulated_friction": float(total_friction),
            "initial_semantic_mass": float(initial_mass),
            "final_semantic_mass": float(final_mass),
            "mass_growth": float(mass_growth),
            "trace_chain": [(n.step_id, n.friction_delta, n.semantic_mass) for n in self.history]
        }

    def simulate_counterfactuals(self,
                                 frictions: List[np.ndarray],
                                 alt_compass_keys: List[str]) -> Dict[str, Any]:
        """
        Simulates parallel potential trajectories in a sandbox using alternative compass vectors,
        comparing resulting semantic mass, curvature, and trinitarian contrast.
        """
        results = {}
        mass_operator = SemanticMassOperator()

        for compass_key in alt_compass_keys:
            if compass_key not in self.white_field.compass_vectors:
                continue

            # Virtual trajectory simulation
            virtual_field = self.white_field.compass_vectors[compass_key].copy()
            acc_mass = 0.0
            acc_friction = 0.0

            for f_idx, friction in enumerate(frictions):
                f_norm = friction / (np.linalg.norm(friction) + 1e-9)
                dot = np.abs(np.dot(virtual_field, f_norm[:len(virtual_field)]))
                acc_friction += float(dot)
                conn_sim = np.outer(virtual_field[:4], virtual_field[:4])
                mass = mass_operator.compute_mass(conn_sim, trinitarian_contrast_score=1.0 + dot, friction_inertia=1.0 + acc_friction)
                acc_mass += mass
                virtual_field = 0.8 * virtual_field + 0.2 * f_norm[:len(virtual_field)]
                virtual_field /= (np.linalg.norm(virtual_field) + 1e-9)

            results[compass_key] = {
                "final_virtual_vector": virtual_field.tolist(),
                "accumulated_semantic_mass": float(acc_mass),
                "accumulated_friction": float(acc_friction)
            }

        return results


class SelfWovenAgentMatrix:
    """
    [Self-Woven Agent Matrix]
    Dynamically weaves distinct cognitive agents out of the White Tensor Field based on
    environmental requirements and models the inner trajectory/topology of other entities.
    """
    def __init__(self, white_field: WhiteTensorField):
        self.white_field = white_field
        self.woven_agents: Dict[str, Dict[str, Any]] = {}

    def weave_agent(self, agent_name: str, target_compass_keys: List[str], friction_bias: float = 1.0) -> Dict[str, Any]:
        """
        Weaves a specialized agent with a specific composite compass configuration.
        """
        agent_vector = np.zeros(self.white_field.dimensions, dtype=np.float32)
        valid_keys = [k for k in target_compass_keys if k in self.white_field.compass_vectors]

        if not valid_keys:
            agent_vector = self.white_field.field_tensor.copy()
        else:
            for k in valid_keys:
                agent_vector += self.white_field.compass_vectors[k]
            agent_vector /= (np.linalg.norm(agent_vector) + 1e-9)

        agent_data = {
            "name": agent_name,
            "phase_vector": agent_vector,
            "target_compasses": valid_keys,
            "friction_bias": friction_bias,
            "semantic_mass": 1.0,
            "perceived_others": {}
        }
        self.woven_agents[agent_name] = agent_data
        return agent_data

    def model_other_entity(self, observer_agent_name: str, other_id: str, observed_trajectories: List[np.ndarray]) -> Dict[str, Any]:
        """
        Models the internal trajectory and topology of another entity ("Modeling the Other")
        by reverse-calculating their compass orientation and friction resistance curve.
        """
        if observer_agent_name not in self.woven_agents:
            raise ValueError(f"Observer agent '{observer_agent_name}' does not exist.")

        if not observed_trajectories:
            return {}

        avg_trajectory = np.mean(observed_trajectories, axis=0)
        if len(avg_trajectory) < self.white_field.dimensions:
            avg_trajectory = np.pad(avg_trajectory, (0, self.white_field.dimensions - len(avg_trajectory)))
        elif len(avg_trajectory) > self.white_field.dimensions:
            avg_trajectory = avg_trajectory[:self.white_field.dimensions]

        norm_traj = avg_trajectory / (np.linalg.norm(avg_trajectory) + 1e-9)

        # Find closest compass vector in white field
        closest_compass = None
        max_dot = -1.0
        for k, v in self.white_field.compass_vectors.items():
            dot = float(np.abs(np.dot(v, norm_traj)))
            if dot > max_dot:
                max_dot = dot
                closest_compass = k

        model_result = {
            "other_id": other_id,
            "inferred_compass": closest_compass,
            "resonance_score": float(max_dot),
            "estimated_phase_vector": norm_traj.tolist()
        }

        self.woven_agents[observer_agent_name]["perceived_others"][other_id] = model_result
        return model_result


class SemanticMassEngine:
    """
    [Integrated Semantic Mass & Introspective Causal Engine]
    Combines WhiteTensorField, SemanticMassOperator, CausalGravityField, EmergentIdentityCrystallizer,
    IntrospectiveCausalTracer, and SelfWovenAgentMatrix into a unified cognitive architecture.
    """
    def __init__(self, dimensions: int = 16, phase_threshold: float = 2.5):
        self.dimensions = dimensions
        self.white_field = WhiteTensorField(dimensions=dimensions)
        self.mass_operator = SemanticMassOperator()
        self.gravity_field = CausalGravityField(dimensions=dimensions)
        self.crystallizer = EmergentIdentityCrystallizer(phase_transition_threshold=phase_threshold)
        self.tracer = IntrospectiveCausalTracer(white_field=self.white_field)
        self.agent_matrix = SelfWovenAgentMatrix(white_field=self.white_field)

        self.crystals: List[IdentityCrystal] = []
        self.connectivity_matrix = np.eye(4, dtype=np.float32)

    def process_interaction(self,
                            external_friction: np.ndarray,
                            trinitarian_contrast: float = 1.0) -> Dict[str, Any]:
        """
        Executes one full cycle of interaction with external friction:
        1. Bends WhiteTensorField.
        2. Updates connectivity & calculates Semantic Mass.
        3. Computes Causal Spacetime Curvature.
        4. Checks Emergent Identity Crystallization.
        5. Records Retrospective Causal History.
        """
        # 1. Project friction onto white tensor field
        friction_mag = float(np.linalg.norm(external_friction))
        bent_state, alignments = self.white_field.project_and_bend(external_friction, friction_strength=friction_mag)

        # 2. Update connectivity matrix dynamically
        outer = np.outer(bent_state[:4], bent_state[:4])
        self.connectivity_matrix = 0.8 * self.connectivity_matrix + 0.2 * outer

        # 3. Compute Semantic Mass
        semantic_mass = self.mass_operator.compute_mass(
            connectivity_matrix=self.connectivity_matrix,
            trinitarian_contrast_score=trinitarian_contrast,
            friction_inertia=1.0 + self.crystallizer.accumulated_friction
        )

        # 4. Compute Causal Curvature
        curvature = self.gravity_field.compute_field_curvature(semantic_mass)

        # 5. Record trace step
        self.tracer.record_step(
            friction_delta=friction_mag,
            repulsion_vector=external_friction,
            compass_alignment=alignments,
            crystallized_state=bent_state,
            semantic_mass=semantic_mass
        )

        # 6. Check Crystallization
        crystal = self.crystallizer.evaluate_crystallization(
            current_state=bent_state,
            friction_delta=friction_mag,
            semantic_mass=semantic_mass,
            trinitarian_contrast=trinitarian_contrast,
            alignments=alignments,
            trace_history=self.tracer.history
        )

        if crystal is not None:
            self.crystals.append(crystal)

        return {
            "current_state": bent_state.tolist(),
            "semantic_mass": semantic_mass,
            "causal_curvature": curvature,
            "top_compass_alignment": max(alignments.items(), key=lambda x: x[1]) if alignments else None,
            "new_crystal_formed": crystal.crystal_id if crystal else None,
            "total_crystals_count": len(self.crystals)
        }
