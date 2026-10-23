"""
Elysia Core Architecture: Topological Dimensional Mirror & Living Schema Engine

This module breaks free from static reductionist representations by realizing that
dimensions (1D, 2D, 3D, 4D) are continuous topological transformations of relational constraints.

Key Components:
1. DimensionalTopologicalEngine: Handles bi-directional topological mappings across
   1D (Signal/Wave), 2D (Texture/Grid), 3D (Structural Body Schema), and 4D (Causal-Temporal Field).
2. DualEnvironmentMirrorLoop: Evaluates discrepancies between internal expectations
   (Body Schema) and external projections under dual environmental conditions (e.g., Viscous Fluid vs. Standard Void).
3. LivingCognitiveSchemaEngine: Converts topological friction and structural discrepancies into
   living cognitive & sensory schemas (embodied habits, motor adjustments, metacognitive self-awareness).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EnvironmentCondition:
    """External environment physical parameters and field constraints."""
    name: str = "Standard Void"
    gravity: float = 9.81              # Gravitational acceleration (m/s^2)
    viscosity: float = 0.01             # Fluid/medium viscosity
    surface_friction: float = 0.2       # Boundary friction coefficient
    medium_density: float = 1.0         # Environmental density
    temporal_flux: float = 1.0          # Time flow rate coefficient


@dataclass
class BodySchemaState:
    """Internal structural body schema of Elysia across space and time."""
    nodes_3d: np.ndarray = field(default_factory=lambda: np.zeros((16, 3)))  # 3D keypoint coordinates
    joint_rigidity: np.ndarray = field(default_factory=lambda: np.ones(16))   # Structural rigidity per node
    chromatic_signature: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.33, 0.34]))  # Red/Blue/Yellow flux
    energy_potential: float = 1.0


@dataclass
class TopologicalDiscrepancy:
    """Measures divergence between internal expectations and external physical reality."""
    positional_drift: float = 0.0
    phase_decoherence: float = 0.0
    topological_friction: float = 0.0
    boundary_deformation: float = 0.0
    chromatic_mismatch: float = 0.0
    total_discrepancy: float = 0.0


@dataclass
class LivingSchema:
    """An organic, embodied pattern of perception and action judgment formed through environmental friction."""
    schema_id: str
    name: str
    embodied_habit_vector: np.ndarray
    metacognitive_awareness: float
    resilience_score: float
    chromatic_balance: np.ndarray


class DimensionalTopologicalEngine:
    """
    Handles bi-directional topological phase transformations between dimensions.
    Dimensions are treated not as rigid containers, but as manifold projections of relational dynamics.
    - 1D: Temporal wave / Auditory / Sequential Phase Signal
    - 2D: Causal Texture / Surface Grid / Optical Horizon
    - 3D: Structural Body Schema / Spatial Voxels
    - 4D: Spacetime Trajectory / Causal Evolution Ring
    """

    def __init__(self, num_nodes: int = 16):
        self.num_nodes = num_nodes

    def map_1d_signal_to_2d_texture(self, signal_1d: np.ndarray, grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
        """Lifts a 1D phase/wave signal into a 2D topological texture grid using phase folding."""
        rows, cols = grid_size
        flattened_len = rows * cols
        # Resample or interpolate 1D signal to match grid size
        if len(signal_1d) != flattened_len:
            interpolated = np.interp(
                np.linspace(0, 1, flattened_len),
                np.linspace(0, 1, len(signal_1d)),
                signal_1d
            )
        else:
            interpolated = signal_1d.copy()

        # Fold into 2D grid with phase interference harmonics
        grid_2d = interpolated.reshape((rows, cols))
        phase_x = np.sin(np.linspace(0, 2 * np.pi, cols))
        phase_y = np.cos(np.linspace(0, 2 * np.pi, rows))
        outer_harmonic = np.outer(phase_y, phase_x)

        return grid_2d * outer_harmonic

    def map_2d_texture_to_3d_body(self, texture_2d: np.ndarray, base_schema: BodySchemaState) -> BodySchemaState:
        """
        Projects a 2D surface texture into 3D body manifold coordinate perturbations.
        Preserves topological invariants (total topological mass & energy potential).
        """
        rows, cols = texture_2d.shape
        nodes = base_schema.nodes_3d.copy()

        # Sample texture values at node positions
        for i in range(min(self.num_nodes, len(nodes))):
            r_idx = int((i / self.num_nodes) * (rows - 1))
            c_idx = int((i / self.num_nodes) * (cols - 1))
            texture_val = texture_2d[r_idx, c_idx]

            # Apply smooth 3D displacement vector based on texture gradient
            displacement = np.array([
                math.sin(texture_val * np.pi),
                math.cos(texture_val * np.pi),
                math.tanh(texture_val)
            ]) * 0.1
            nodes[i] += displacement

        updated_schema = BodySchemaState(
            nodes_3d=nodes,
            joint_rigidity=base_schema.joint_rigidity.copy(),
            chromatic_signature=base_schema.chromatic_signature.copy(),
            energy_potential=base_schema.energy_potential
        )
        return updated_schema

    def map_3d_body_to_4d_causal_trajectory(
        self,
        schema_history: List[BodySchemaState],
        time_steps: float = 1.0
    ) -> np.ndarray:
        """
        Extends a series of 3D Body Schema states into a 4D spacetime causal tensor [T, N, 4]
        where the 4th coordinate represents temporal-causal flux (ct).
        """
        if not schema_history:
            return np.zeros((1, self.num_nodes, 4))

        num_frames = len(schema_history)
        tensor_4d = np.zeros((num_frames, self.num_nodes, 4))

        for t, state in enumerate(schema_history):
            time_coord = t * time_steps
            for n in range(min(self.num_nodes, len(state.nodes_3d))):
                xyz = state.nodes_3d[n]
                tensor_4d[t, n] = np.array([xyz[0], xyz[1], xyz[2], time_coord])

        return tensor_4d

    def fold_4d_causal_trajectory_to_1d(self, tensor_4d: np.ndarray) -> np.ndarray:
        """
        Folds a 4D causal trajectory back down to a 1D resonant topological invariant wave.
        Captures the global momentum, curvature, and phase flow of the trajectory.
        """
        num_frames, num_nodes, _ = tensor_4d.shape
        wave_1d = np.zeros(num_frames)

        for t in range(num_frames):
            frame_coords = tensor_4d[t, :, :3]
            # Calculate centroid and topological dispersion (radius of gyration)
            centroid = np.mean(frame_coords, axis=0)
            dispersion = np.sqrt(np.sum((frame_coords - centroid) ** 2) / num_nodes)
            time_val = tensor_4d[t, 0, 3]

            # Resonant wave combining dispersion and temporal progression
            wave_1d[t] = dispersion * math.sin(time_val + dispersion)

        return wave_1d


class DualEnvironmentMirrorLoop:
    """
    Simulates dual environmental conditions (e.g. Standard Void vs. High Viscosity Environment)
    and projects the system's internal Body Schema into external physical outcomes.
    Measures the exact topological discrepancies and cognitive friction between expectation and reality.
    """

    def __init__(self, topo_engine: Optional[DimensionalTopologicalEngine] = None):
        self.topo_engine = topo_engine or DimensionalTopologicalEngine()

    def simulate_external_projection(
        self,
        internal_schema: BodySchemaState,
        env: EnvironmentCondition,
        steps: int = 10
    ) -> List[BodySchemaState]:
        """
        Simulates how external physical forces in environment `env` deform and interact with
        the internal Body Schema over `steps` time frames.
        """
        projected_states = []
        current_nodes = internal_schema.nodes_3d.copy()

        for step in range(steps):
            t = step * 0.1
            deformed_nodes = current_nodes.copy()

            for i in range(len(deformed_nodes)):
                rigidity = internal_schema.joint_rigidity[i]
                # Environmental forces: Gravity drag, viscosity resistance, surface friction
                gravity_force = np.array([0.0, -env.gravity * (1.0 - rigidity * 0.5) * 0.01, 0.0])
                viscous_drag = -deformed_nodes[i] * env.viscosity * 0.05
                friction_retardation = np.array([
                    -math.sin(t) * env.surface_friction * 0.02,
                    0.0,
                    -math.cos(t) * env.surface_friction * 0.02
                ])

                # Total net force effect
                deformed_nodes[i] += (gravity_force + viscous_drag + friction_retardation) * env.temporal_flux

            state = BodySchemaState(
                nodes_3d=deformed_nodes,
                joint_rigidity=internal_schema.joint_rigidity.copy(),
                chromatic_signature=internal_schema.chromatic_signature.copy(),
                energy_potential=internal_schema.energy_potential * math.exp(-env.viscosity * t * 0.1)
            )
            projected_states.append(state)
            current_nodes = deformed_nodes

        return projected_states

    def evaluate_discrepancy(
        self,
        internal_expectation: List[BodySchemaState],
        external_reality: List[BodySchemaState]
    ) -> TopologicalDiscrepancy:
        """
        Quantifies the discrepancy between internal expected trajectories and external reality.
        Measures positional drift, phase decoherence, topological friction, boundary deformation,
        and chromatic signature mismatch.
        """
        num_steps = min(len(internal_expectation), len(external_reality))
        if num_steps == 0:
            return TopologicalDiscrepancy()

        positional_drifts = []
        phase_diffs = []
        boundary_deformations = []
        chromatic_diffs = []

        for t in range(num_steps):
            exp_state = internal_expectation[t]
            real_state = external_reality[t]

            # 1. Positional drift (mean Euclidean error between nodes)
            drift = np.mean(np.linalg.norm(exp_state.nodes_3d - real_state.nodes_3d, axis=1))
            positional_drifts.append(drift)

            # 2. Boundary deformation (dispersion mismatch)
            exp_dispersion = np.std(exp_state.nodes_3d)
            real_dispersion = np.std(real_state.nodes_3d)
            boundary_deformations.append(abs(exp_dispersion - real_dispersion))

            # 3. Phase decoherence
            exp_center = np.mean(exp_state.nodes_3d, axis=0)
            real_center = np.mean(real_state.nodes_3d, axis=0)
            exp_phase = math.atan2(exp_center[1], exp_center[0] + 1e-6)
            real_phase = math.atan2(real_center[1], real_center[0] + 1e-6)
            phase_diffs.append(abs(exp_phase - real_phase))

            # 4. Chromatic signature mismatch
            c_diff = np.linalg.norm(exp_state.chromatic_signature - real_state.chromatic_signature)
            chromatic_diffs.append(c_diff)

        mean_drift = float(np.mean(positional_drifts))
        mean_phase = float(np.mean(phase_diffs))
        mean_boundary = float(np.mean(boundary_deformations))
        mean_chromatic = float(np.mean(chromatic_diffs))

        # Topological friction = combined rate of structural distortion under force
        topological_friction = 0.5 * mean_drift + 0.3 * mean_boundary + 0.2 * mean_phase
        total_discrepancy = topological_friction + 0.1 * mean_chromatic

        return TopologicalDiscrepancy(
            positional_drift=mean_drift,
            phase_decoherence=mean_phase,
            topological_friction=topological_friction,
            boundary_deformation=mean_boundary,
            chromatic_mismatch=mean_chromatic,
            total_discrepancy=total_discrepancy
        )

    def run_reflection_cycle(
        self,
        initial_schema: BodySchemaState,
        env_a: EnvironmentCondition,
        env_b: EnvironmentCondition,
        steps: int = 10
    ) -> Dict[str, Tuple[List[BodySchemaState], TopologicalDiscrepancy]]:
        """
        Runs dual reflection cycle comparing internal Body Schema projections across Environment A and Environment B.
        Returns simulation trajectories and discrepancy metrics for both environments.
        """
        # Internal baseline expectation (vacuum/unconstrained motion)
        default_env = EnvironmentCondition(name="Ideal Internal Model", gravity=0.0, viscosity=0.0, surface_friction=0.0)
        internal_expectation = self.simulate_external_projection(initial_schema, default_env, steps=steps)

        # External reality simulations
        real_proj_a = self.simulate_external_projection(initial_schema, env_a, steps=steps)
        real_proj_b = self.simulate_external_projection(initial_schema, env_b, steps=steps)

        # Discrepancy analysis
        disc_a = self.evaluate_discrepancy(internal_expectation, real_proj_a)
        disc_b = self.evaluate_discrepancy(internal_expectation, real_proj_b)

        return {
            env_a.name: (real_proj_a, disc_a),
            env_b.name: (real_proj_b, disc_b)
        }


class LivingCognitiveSchemaEngine:
    """
    Transforms topological friction and discrepancy metrics into living cognitive & sensory schemas.
    Instead of hardcoding rules, this engine internalizes environmental friction as organic motor/sensory habits,
    enhancing metacognitive self-awareness and body schema resilience.
    """

    def __init__(self):
        self.embodied_schemas: Dict[str, LivingSchema] = {}
        self.global_metacognitive_awareness: float = 0.1

    def absorb_discrepancy_and_embodiment(
        self,
        env_name: str,
        discrepancy: TopologicalDiscrepancy,
        initial_schema: BodySchemaState
    ) -> LivingSchema:
        """
        Internalizes the discrepancy from `env_name` to weave a new LivingSchema or adapt an existing one.
        """
        schema_key = f"SCHEMA_{env_name.upper().replace(' ', '_')}"

        # Calculate habit vector (adaptation adjustments for joint rigidities / motor torque)
        # Higher friction requires higher structural rigidity and dynamic dampening
        friction = discrepancy.topological_friction
        habit_adjustment = np.clip(
            initial_schema.joint_rigidity + friction * 0.2,
            0.1, 2.0
        )

        # Chromatic balance shifts: High friction increases Order (Blue) and Flux (Red)
        red_flux = min(1.0, initial_schema.chromatic_signature[0] + friction * 0.1)
        blue_order = min(1.0, initial_schema.chromatic_signature[1] + friction * 0.15)
        yellow_entropy = max(0.0, 1.0 - (red_flux + blue_order) * 0.5)
        chromatic_balance = np.array([red_flux, blue_order, yellow_entropy])
        chromatic_balance /= np.sum(chromatic_balance)

        # Metacognitive awareness growth through self-recognition of discrepancy ("Where did I diverge?")
        awareness_gain = math.tanh(discrepancy.total_discrepancy * 2.0) * 0.15
        self.global_metacognitive_awareness = min(1.0, self.global_metacognitive_awareness + awareness_gain)

        # Resilience score = ability to maintain structural integrity under friction
        resilience = 1.0 / (1.0 + discrepancy.boundary_deformation)

        living_schema = LivingSchema(
            schema_id=schema_key,
            name=f"Embodied Habit under {env_name}",
            embodied_habit_vector=habit_adjustment,
            metacognitive_awareness=self.global_metacognitive_awareness,
            resilience_score=resilience,
            chromatic_balance=chromatic_balance
        )

        self.embodied_schemas[schema_key] = living_schema
        return living_schema

    def refine_body_schema(
        self,
        base_schema: BodySchemaState,
        active_schema: LivingSchema
    ) -> BodySchemaState:
        """
        Refines the base Body Schema using an active living cognitive schema.
        Applies learned joint rigidity adjustments and chromatic balance.
        """
        refined_nodes = base_schema.nodes_3d.copy()
        # Adapt joint rigidities based on habit vector
        refined_rigidity = active_schema.embodied_habit_vector.copy()

        return BodySchemaState(
            nodes_3d=refined_nodes,
            joint_rigidity=refined_rigidity,
            chromatic_signature=active_schema.chromatic_balance.copy(),
            energy_potential=base_schema.energy_potential * active_schema.resilience_score
        )
