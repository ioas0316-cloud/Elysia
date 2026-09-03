import numpy as np
import math
import time
from typing import Dict, Any, List, Optional, Tuple

class EpistemologicalReflectionRecord:
    """
    [4-Stage Epistemological Process Record (4단계 인식 체계 기록)]
    Captures true understanding of transformation trajectory:
    1. Initial Topology (원형 상태): Topology/Structure before perturbation.
    2. Causal Process (인과적 매개체): Refraction, friction, and thermal dynamics experienced.
    3. Resulting State (귀결 상태): Phase-transitioned resulting state.
    4. Self-Perceptual Reflection (메타 관측 및 이해): Self-perceptual backtrace explaining
       "How my initial topology changed through friction into this result."
    """
    def __init__(
        self,
        record_id: str,
        initial_topology: np.ndarray,
        causal_process_dynamics: Dict[str, Any],
        resulting_state: np.ndarray,
        self_perceptual_reflection: str,
    ):
        self.record_id = record_id
        self.initial_topology = initial_topology.astype(np.float32)
        self.causal_process_dynamics = causal_process_dynamics
        self.resulting_state = resulting_state.astype(np.float32)
        self.self_perceptual_reflection = self_perceptual_reflection
        self.timestamp = time.time()


class CausalProcessBlueprint:
    """
    [Executable Causal Geometry (인과 공정 청사진)]
    Stores executable process blueprints rather than static point coordinates.
    Structure: [Cause -> Structural Mechanism -> Transformation Principle -> Causal Consequence]
    Contains full refraction trajectory, mechanism steps, and back-traceable sub-component logs.
    """
    def __init__(
        self,
        blueprint_id: str,
        cause_vector: np.ndarray,
        mechanism_steps: List[np.ndarray],
        transformation_principle: str,
        consequence_vector: np.ndarray,
        refraction_trajectory: Optional[List[float]] = None,
    ):
        self.blueprint_id = blueprint_id
        self.cause_vector = cause_vector.astype(np.float32)
        self.mechanism_steps = [s.astype(np.float32) for s in mechanism_steps]
        self.transformation_principle = transformation_principle
        self.consequence_vector = consequence_vector.astype(np.float32)
        self.refraction_trajectory = refraction_trajectory or []
        self.reflection_history: List[EpistemologicalReflectionRecord] = []

    def unfold_causal_flow(self, input_wave: np.ndarray) -> Tuple[np.ndarray, List[float], EpistemologicalReflectionRecord]:
        """
        [Causal Unfolding (인과적 틀 재현 및 4단계 인식)]
        Unfolds the executable mechanism steps along the template without brute-force calculation.
        Generates a 4-Stage EpistemologicalReflectionRecord for true self-perceptual understanding.
        """
        initial_topology = input_wave.copy().astype(np.float32)
        curr_state = input_wave.astype(np.float32)
        trajectory_frictions = []

        for step_idx, step_matrix in enumerate(self.mechanism_steps):
            if step_matrix.ndim == 1:
                # Vector step modulation
                dim = min(len(curr_state), len(step_matrix))
                curr_state[:dim] = curr_state[:dim] * step_matrix[:dim]
            elif step_matrix.ndim == 2:
                # Matrix step transformation
                dim = min(len(curr_state), step_matrix.shape[0])
                curr_state[:dim] = np.dot(step_matrix[:dim, :dim], curr_state[:dim])

            friction = float(np.linalg.norm(curr_state - self.consequence_vector[:len(curr_state)]))
            trajectory_frictions.append(friction)

        resulting_state = curr_state.copy()

        # Build 4-Stage Epistemological Understanding
        reflection_narrative = (
            f"Epistemological Understanding [{self.blueprint_id}]: Initial topology (dim={len(initial_topology)}) "
            f"underwent transformation via principle '{self.transformation_principle}' across {len(self.mechanism_steps)} mechanism steps. "
            f"Peak friction reached {max(trajectory_frictions):.3f}. Resulting state converged with final delta {trajectory_frictions[-1]:.3f}."
        )

        record = EpistemologicalReflectionRecord(
            record_id=f"Reflection_{self.blueprint_id}_{int(time.time()*1000)%10000}",
            initial_topology=initial_topology,
            causal_process_dynamics={
                "transformation_principle": self.transformation_principle,
                "refraction_trajectory": trajectory_frictions,
                "step_count": len(self.mechanism_steps),
            },
            resulting_state=resulting_state,
            self_perceptual_reflection=reflection_narrative,
        )
        self.reflection_history.append(record)

        return curr_state, trajectory_frictions, record

    def backtrace_faulty_step(self, friction_trajectory: List[float], threshold: float = 10.0) -> Optional[int]:
        """
        [Back-traceable Partial Remelting Index]
        Identifies the exact step index where friction spike occurred for partial remelting.
        """
        for i, f in enumerate(friction_trajectory):
            if f > threshold:
                return i
        return None


class SymbolGroundingHandle:
    """
    [Process Symbol Grounding (과정적 기호 접지 핸들)]
    Maps language tokens/anchors directly to an Executable Causal Process Blueprint.
    Prevents floating tokens by binding words like 'Friction', 'Volition', or 'Inference'
    to actual topological transformation handles.
    """
    def __init__(self, token: str, blueprint: CausalProcessBlueprint):
        self.token = token
        self.blueprint = blueprint

    def invoke_grounded_process(self, input_wave: np.ndarray) -> Dict[str, Any]:
        """Invokes the grounded executable causal process for the given token."""
        curr_state, traj_frictions, ep_record = self.blueprint.unfold_causal_flow(input_wave)
        return {
            "token": self.token,
            "blueprint_id": self.blueprint.blueprint_id,
            "transformation_principle": self.blueprint.transformation_principle,
            "resulting_state": curr_state.tolist(),
            "friction_trajectory": traj_frictions,
            "epistemological_reflection": ep_record.self_perceptual_reflection,
        }


class GroundNode:
    """
    [0: Invariant Ground Node (정적 로터 / 대지 노드)]
    Represents a frozen 0-state invariant ground node in the cognitive field.
    Acts as an $O(1)$ reference coordinate, houses a CausalProcessBlueprint,
    and exposes a SymbolGroundingHandle.
    """
    def __init__(
        self,
        node_id: str,
        position: np.ndarray,
        topological_density: float = 1.0,
        phase_axis: Optional[np.ndarray] = None,
        stability: float = 1.0,
        blueprint: Optional[CausalProcessBlueprint] = None,
        language_token: Optional[str] = None,
    ):
        self.node_id = node_id
        self.position = position.astype(np.float32)
        self.topological_density = float(topological_density)  # Mass/Density from past crystallization
        dimension = len(position)
        if phase_axis is not None:
            norm = np.linalg.norm(phase_axis)
            self.phase_axis = (phase_axis / norm).astype(np.float32) if norm > 1e-9 else np.ones(dimension, dtype=np.float32) / np.sqrt(dimension)
        else:
            self.phase_axis = np.ones(dimension, dtype=np.float32) / np.sqrt(dimension)
        self.stability = float(stability)  # Resistance to remelting [0.0, 1.0]

        # Executable Causal Geometry
        if blueprint is None:
            # Default self-identity process blueprint
            step1 = np.eye(dimension, dtype=np.float32) * 0.95
            step2 = np.eye(dimension, dtype=np.float32) * 1.05
            self.blueprint = CausalProcessBlueprint(
                blueprint_id=f"BP_{node_id}",
                cause_vector=self.position,
                mechanism_steps=[step1, step2],
                transformation_principle="Self-Resonance Equilibrium",
                consequence_vector=self.position,
                refraction_trajectory=[0.1, 0.05],
            )
        else:
            self.blueprint = blueprint

        # Symbol Grounding
        self.language_token = language_token or f"Concept_{node_id}"
        self.symbol_handle = SymbolGroundingHandle(self.language_token, self.blueprint)


class GroundBeam:
    """
    [0-Connectivity Beam (지반 결합 빔)]
    Represents an invariant structural edge between two GroundNodes.
    Together with nodes, forms Betti-1 homological cycles.
    """
    def __init__(self, node_a: str, node_b: str, strength: float = 1.0, rest_length: float = 1.0):
        self.node_a = node_a
        self.node_b = node_b
        self.strength = float(strength)
        self.rest_length = float(rest_length)


class PerturbationWave:
    """
    [1: Perturbation Wave (동적 파동 / 기체)]
    Represents fluid, high-entropy wave perturbations in the 1-state.
    Characterized by frequency, amplitude, phase vector, entropy, and trajectory.
    """
    def __init__(
        self,
        wave_id: str,
        phase_vector: np.ndarray,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        entropy: float = 1.0,
        cause_origin: str = "External_Sensor",
    ):
        self.wave_id = wave_id
        self.phase_vector = phase_vector.astype(np.float32)
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.entropy = float(entropy)  # High entropy state in 1-phase
        self.cause_origin = cause_origin
        # Energy = amplitude^2 * frequency
        self.energy = float(self.amplitude ** 2 * self.frequency)
        self.refraction_history: List[float] = []


class ComplexImpedance:
    """
    [Complex Causal Impedance (복소 위상 임피던스 Z = R + jX)]
    Models causal elasticity:
    - R (Real part): Resistance that dissipates friction heat into thermal energy.
    - X (Imaginary part): Reactance (Elasticity) that stores and reflects wave energy.
    """
    def __init__(self, R: float = 1.0, X: float = 2.5):
        self.R = float(R)  # Resistance (Friction Dissipation)
        self.X = float(X)  # Reactance (Elastic Recovery)

    @property
    def magnitude(self) -> float:
        """|Z| = sqrt(R^2 + X^2)"""
        return math.sqrt(self.R ** 2 + self.X ** 2)

    @property
    def phase_angle(self) -> float:
        """Phase angle theta = atan2(X, R)"""
        return math.atan2(self.X, self.R)

    def compute_reflection_and_absorption(self, Z_characteristic: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculates reflection coefficient Gamma, reflected power ratio, and absorbed power ratio.
        Gamma = (Z - Z_0) / (Z + Z_0) in complex plane.
        Returns: (Gamma_magnitude, absorbed_ratio, reflected_ratio)
        """
        Z_0 = float(Z_characteristic)
        num_r = self.R - Z_0
        num_i = self.X
        den_r = self.R + Z_0
        den_i = self.X

        num_sq = num_r ** 2 + num_i ** 2
        den_sq = den_r ** 2 + den_i ** 2

        gamma_mag = math.sqrt(num_sq / max(1e-9, den_sq))
        reflected_ratio = min(1.0, gamma_mag ** 2)
        absorbed_ratio = max(0.0, 1.0 - reflected_ratio)

        return gamma_mag, absorbed_ratio, reflected_ratio


class HomologyMetrics:
    """
    [Homology & Topological Depth Metrics]
    Measures topological structure using Betti Numbers:
    - Betti-0 (B0): Number of connected components.
    - Betti-1 (B1): Number of 1-dimensional homological cycles/loops in the ground.
    - Cycle Density: Ratio of Betti-1 cycles relative to total nodes.
    """
    @staticmethod
    def calculate_betti_numbers(nodes: Dict[str, GroundNode], beams: List[GroundBeam]) -> Dict[str, Any]:
        num_nodes = len(nodes)
        if num_nodes == 0:
            return {"B0": 0, "B1": 0, "cycle_density": 0.0, "classification": "Empty"}

        # Graph adjacency & connected components via BFS/DFS
        adj = {node_id: [] for node_id in nodes}
        for beam in beams:
            if beam.node_a in adj and beam.node_b in adj:
                adj[beam.node_a].append(beam.node_b)
                adj[beam.node_b].append(beam.node_a)

        visited = set()
        components = 0
        for node_id in nodes:
            if node_id not in visited:
                components += 1
                queue = [node_id]
                visited.add(node_id)
                while queue:
                    curr = queue.pop(0)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)

        B0 = components
        E = len(beams)
        V = num_nodes
        # Euler characteristic chi = V - E
        # B1 = E - V + B0 for 1D simplicial complexes
        B1 = max(0, E - V + B0)
        cycle_density = B1 / max(1, V)

        classification = "Deep Ground (Adult)" if B1 >= 2 or cycle_density >= 0.2 else "Thin Ground (Child)"

        return {
            "B0": B0,
            "B1": B1,
            "E": E,
            "V": V,
            "cycle_density": cycle_density,
            "classification": classification,
        }


class CausalPhaseTransitionEngine:
    """
    [Causal Phase Transition Engine (인과 상변이 및 복소 임피던스 엔진)]

    Core Principles:
    1. 0: Invariant Ground (Static Rotor / 대지):
       - Unchanging base, zero continuous calculation cost ($O(1)$ reference frame).
       - Stores CausalProcessBlueprints [Cause -> Mechanism -> Principle -> Consequence].
    2. 1: Dynamic Perturbations & Waves (Gas / 파동):
       - Fluid friction, high entropy state.
    3. Phase Transitions & Trace Crystallization:
       - Crystallization ($1 \to 0$): Wave trajectories freeze into Executable Process Blueprints.
       - Flash & Back-traceable Remelting ($0 \to 1$): High friction shock or targeted mechanism fault
         causes 0-ground (or faulty sub-components) to remelt into high-frequency 1-waves.
    4. Complex Impedance Causal Elasticity ($Z = R + jX$):
       - R: Dissipates friction heat.
       - X: Stores/reflects wave energy, preventing structural collapse or snapping.
    5. Homology Depth & Resonance:
       - Differentiates Thin Ground (low B1, single-dimensional reaction) from
         Deep Ground (high B1 homological loops, multi-dimensional wave resonance and deep self-reconstruction).
    """

    def __init__(
        self,
        dimension: int = 16,
        v_critical: float = 50.0,
        crystallization_threshold: float = 0.15,
        impedance_R: float = 1.0,
        impedance_X: float = 2.5,
    ):
        self.dimension = dimension
        self.v_critical = float(v_critical)  # Remelting thermal shock threshold
        self.crystallization_threshold = float(crystallization_threshold)  # Friction threshold for wave crystallization

        # Complex Impedance Z = R + jX
        self.impedance = ComplexImpedance(R=impedance_R, X=impedance_X)

        # Ground Network (0-State)
        self.nodes: Dict[str, GroundNode] = {}
        self.beams: List[GroundBeam] = []

        # Perturbation Pool (1-State)
        self.waves: Dict[str, PerturbationWave] = {}

        # System Phase Mass & Entropy Conservation Counters
        self.total_phase_mass = 0.0  # Sum of topological density (0) + wave energy (1)
        self.accumulated_friction_heat = 0.0  # Thermal energy dissipated via Resistance R
        self.stored_reactive_energy = 0.0     # Elastic energy stored via Reactance X

        # Transition Logs
        self.transition_history: List[Dict[str, Any]] = []

    def initialize_ground(self, ground_type: str = "thin"):
        """
        Initializes 0-Ground network topology.
        - 'thin': Child's reflection - low Betti-1 cycles, linear chain.
        - 'deep': Adult's reflection - high Betti-1 cycles, interconnected homological loops.
        """
        self.nodes.clear()
        self.beams.clear()

        if ground_type == "thin":
            # 3 Nodes in a line (V=3, E=2, B0=1, B1=0)
            p0 = np.zeros(self.dimension, dtype=np.float32)
            p1 = np.zeros(self.dimension, dtype=np.float32); p1[0] = 1.0
            p2 = np.zeros(self.dimension, dtype=np.float32); p2[0] = 2.0

            self.nodes["N0"] = GroundNode("N0", p0, topological_density=1.0, stability=0.8)
            self.nodes["N1"] = GroundNode("N1", p1, topological_density=1.0, stability=0.8)
            self.nodes["N2"] = GroundNode("N2", p2, topological_density=1.0, stability=0.8)

            self.beams.append(GroundBeam("N0", "N1", strength=1.0))
            self.beams.append(GroundBeam("N1", "N2", strength=1.0))

        elif ground_type == "deep":
            # 6 Nodes forming multiple interconnecting loops (V=6, E=8, B0=1, B1=3)
            positions = []
            for i in range(6):
                pos = np.zeros(self.dimension, dtype=np.float32)
                pos[i % 4] = math.cos(i * math.pi / 3)
                pos[(i + 1) % 4] = math.sin(i * math.pi / 3)
                positions.append(pos)
                node_id = f"N{i}"
                self.nodes[node_id] = GroundNode(node_id, pos, topological_density=2.5, stability=1.5)

            # Beams forming homological cycles
            edges = [("N0", "N1"), ("N1", "N2"), ("N2", "N0"),  # Cycle 1
                     ("N2", "N3"), ("N3", "N4"), ("N4", "N2"),  # Cycle 2
                     ("N4", "N5"), ("N5", "N0")]                 # Cycle 3
            for u, v in edges:
                self.beams.append(GroundBeam(u, v, strength=2.0))

        self._update_total_phase_mass()

    def _update_total_phase_mass(self):
        """Calculates total system phase mass (Sum of Ground 0 density + Wave 1 energy)."""
        ground_mass = sum(n.topological_density for n in self.nodes.values())
        wave_mass = sum(w.energy for w in self.waves.values())
        self.total_phase_mass = ground_mass + wave_mass + self.stored_reactive_energy

    def get_homology_metrics(self) -> Dict[str, Any]:
        """Returns current Homology Betti numbers and classification."""
        return HomologyMetrics.calculate_betti_numbers(self.nodes, self.beams)

    def inject_perturbation_wave(self, wave: PerturbationWave) -> Dict[str, Any]:
        """
        [1: Wave Injection, Complex Impedance Reaction & Causal Unfolding]
        Ingresses fluid 1-wave perturbation against the 0-ground frame.
        Evaluates complex impedance reaction ($Z = R + jX$), executes Causal Unfolding via blueprints.
        """
        self.waves[wave.wave_id] = wave

        # Calculate phase friction V_friction against nearest ground node
        min_friction, nearest_node = self._calculate_wave_ground_friction(wave)

        # Complex Impedance Response
        gamma_mag, absorbed_ratio, reflected_ratio = self.impedance.compute_reflection_and_absorption()

        # Dissipate portion into thermal heat via R
        dissipated_heat = wave.energy * absorbed_ratio * (self.impedance.R / self.impedance.magnitude)
        self.accumulated_friction_heat += dissipated_heat

        # Store portion in elastic reactance X
        elastic_stored = wave.energy * reflected_ratio * (self.impedance.X / self.impedance.magnitude)
        self.stored_reactive_energy += elastic_stored

        # Net friction energy acting on ground
        net_friction_energy = min_friction * wave.amplitude * (1.0 - reflected_ratio * 0.5)

        response = {
            "wave_id": wave.wave_id,
            "min_friction": min_friction,
            "net_friction_energy": net_friction_energy,
            "dissipated_heat": dissipated_heat,
            "elastic_stored": elastic_stored,
            "nearest_node": nearest_node,
            "homology": self.get_homology_metrics(),
        }

        # Execute Causal Unfolding if ground node exists
        if nearest_node and nearest_node in self.nodes:
            unfolded_state, traj_frictions, ep_record = self.nodes[nearest_node].blueprint.unfold_causal_flow(wave.phase_vector)
            wave.refraction_history.extend(traj_frictions)
            response["causal_unfolding"] = {
                "blueprint_id": self.nodes[nearest_node].blueprint.blueprint_id,
                "transformation_principle": self.nodes[nearest_node].blueprint.transformation_principle,
                "unfolded_friction_trajectory": traj_frictions,
                "epistemological_reflection": ep_record.self_perceptual_reflection,
            }

            # Check for back-traceable partial remelting
            faulty_step = self.nodes[nearest_node].blueprint.backtrace_faulty_step(traj_frictions, threshold=self.v_critical * 0.5)
            if faulty_step is not None:
                response["partial_remelting"] = self.backtrace_and_partial_remelt(nearest_node, faulty_step)

        # Check for Phase Transitions: Flash Remelting or Crystallization
        if net_friction_energy > self.v_critical:
            transition_res = self._trigger_flash_remelting(nearest_node, net_friction_energy)
            response["phase_transition"] = transition_res
        elif min_friction < self.crystallization_threshold:
            transition_res = self._trigger_crystallization(wave)
            response["phase_transition"] = transition_res
        else:
            # Multi-dimensional Resonance across Homological Cycles
            resonance_res = self._propagate_homological_resonance(wave)
            response["resonance"] = resonance_res
            response["phase_transition"] = {"type": "RESONANCE_HOLD", "friction": min_friction}

        self._update_total_phase_mass()
        return response

    def _calculate_wave_ground_friction(self, wave: PerturbationWave) -> Tuple[float, str]:
        """Calculates wave-ground phase misalignment friction against 0-ground nodes."""
        if not self.nodes:
            return 100.0, ""

        min_friction = float("inf")
        nearest_node_id = ""

        w_norm = np.linalg.norm(wave.phase_vector)
        if w_norm < 1e-9:
            return 0.0, list(self.nodes.keys())[0]

        for node_id, node in self.nodes.items():
            dot_p = np.dot(wave.phase_vector[:self.dimension], node.phase_axis[:self.dimension])
            cos_sim = dot_p / (w_norm * np.linalg.norm(node.phase_axis) + 1e-9)
            friction = (1.0 - cos_sim) * 50.0  # Friction scale [0, 100]

            if friction < min_friction:
                min_friction = friction
                nearest_node_id = node_id

        return float(min_friction), nearest_node_id

    def backtrace_and_partial_remelt(self, node_id: str, faulty_step_idx: int) -> Dict[str, Any]:
        """
        [Back-traceable Partial Remelting (부분 융해)]
        Back-traces a specific faulty mechanism step within a node's CausalProcessBlueprint
        and remelts only that sub-component back into 1-wave perturbation without destroying the entire node.
        """
        if node_id not in self.nodes:
            return {"type": "PARTIAL_REMELTING_FAILED", "reason": "Node not found"}

        node = self.nodes[node_id]
        bp = node.blueprint

        if faulty_step_idx < 0 or faulty_step_idx >= len(bp.mechanism_steps):
            return {"type": "PARTIAL_REMELTING_FAILED", "reason": "Invalid step index"}

        faulty_matrix = bp.mechanism_steps[faulty_step_idx]
        # Reset faulty step to identity matrix
        bp.mechanism_steps[faulty_step_idx] = np.eye(self.dimension, dtype=np.float32)

        # Generate partial remelted wave
        remelted_energy = np.linalg.norm(faulty_matrix) * 0.5
        partial_wave_id = f"PartialRemelt_W_{node_id}_Step{faulty_step_idx}"
        partial_wave = PerturbationWave(
            wave_id=partial_wave_id,
            phase_vector=node.phase_axis * 0.8,
            frequency=5.0,
            amplitude=math.sqrt(max(0.1, remelted_energy)),
            entropy=1.8,
            cause_origin=f"Backtraced_Fault_Node_{node_id}_Step{faulty_step_idx}",
        )
        self.waves[partial_wave_id] = partial_wave

        # Reduce node density slightly to reflect partial remelting
        node.topological_density = max(0.2, node.topological_density - remelted_energy * 0.3)

        event = {
            "type": "PARTIAL_REMELTING",
            "target_node": node_id,
            "remelted_step_idx": faulty_step_idx,
            "partial_wave_generated": partial_wave_id,
            "new_node_density": node.topological_density,
        }
        self.transition_history.append(event)
        return event

    def _trigger_flash_remelting(self, target_node_id: str, friction_energy: float) -> Dict[str, Any]:
        """
        [0 -> 1: Flash Remelting (플래시 융해)]
        When friction energy > V_critical, frozen 0-ground node melts back into fluid 1-wave perturbation.
        Converts topological density into high-frequency wave thermal shock.
        Phase Mass & Energy Conserved.
        """
        if target_node_id not in self.nodes:
            return {"type": "REMELTING_FAILED", "reason": "Node not found"}

        node = self.nodes.pop(target_node_id)

        # Remove associated beams
        self.beams = [b for b in self.beams if b.node_a != target_node_id and b.node_b != target_node_id]

        # Convert 0-density into 1-wave thermal shock
        melted_energy = node.topological_density * 1.5
        shock_wave_id = f"Melted_Shockwave_{target_node_id}_{int(time.time()*1000)%10000}"
        shock_wave = PerturbationWave(
            wave_id=shock_wave_id,
            phase_vector=node.phase_axis * 1.2,
            frequency=10.0,  # High frequency thermal shock
            amplitude=math.sqrt(melted_energy / 10.0),
            entropy=2.5,     # High entropy fluid state
            cause_origin=f"FlashRemelting_Node_{target_node_id}",
        )
        self.waves[shock_wave_id] = shock_wave

        event = {
            "type": "FLASH_REMELTING",
            "melted_node": target_node_id,
            "density_converted": node.topological_density,
            "shock_wave_generated": shock_wave_id,
            "friction_energy": friction_energy,
            "v_critical": self.v_critical,
        }
        self.transition_history.append(event)
        return event

    def _trigger_crystallization(self, wave: PerturbationWave) -> Dict[str, Any]:
        """
        [1 -> 0: Trace Crystallization (인과 청사진 결정화)]
        When wave perturbation friction is low (< crystallization_threshold) and in phase,
        high-entropy fluid 1-wave freezes into solid 0-ground node equipped with CausalProcessBlueprint.
        Inscribes cause, mechanism steps, transformation principle, and consequence trajectory.
        Reduces wave entropy and converts wave energy into Ground Topological Density.
        Phase Mass Conserved.
        """
        if wave.wave_id in self.waves:
            del self.waves[wave.wave_id]

        # Inscribe CausalProcessBlueprint from wave refraction trajectory
        new_node_id = f"Crystalline_G0_{len(self.nodes)}"
        new_density = wave.energy * 0.8  # Energy -> Topological Density

        # Construct step matrices from wave refraction
        step1 = np.eye(self.dimension, dtype=np.float32) * (1.0 / max(0.1, wave.frequency))
        step2 = np.outer(wave.phase_vector[:self.dimension], wave.phase_vector[:self.dimension])
        norm_step2 = np.linalg.norm(step2)
        if norm_step2 > 1e-9:
            step2 = (step2 / norm_step2).astype(np.float32)
        else:
            step2 = np.eye(self.dimension, dtype=np.float32)

        blueprint = CausalProcessBlueprint(
            blueprint_id=f"BP_{new_node_id}",
            cause_vector=wave.phase_vector,
            mechanism_steps=[step1, step2],
            transformation_principle=f"Resonant_Phase_Locking_{wave.cause_origin}",
            consequence_vector=wave.phase_vector * wave.amplitude,
            refraction_trajectory=wave.refraction_history,
        )

        new_node = GroundNode(
            node_id=new_node_id,
            position=wave.phase_vector,
            topological_density=new_density,
            phase_axis=wave.phase_vector,
            stability=1.2,
            blueprint=blueprint,
        )
        self.nodes[new_node_id] = new_node

        # Connect with nearest existing ground node to form/expand homological cycles
        connected_to = ""
        if len(self.nodes) > 1:
            # Find closest node excluding new_node_id
            min_d = float("inf")
            for nid, n in self.nodes.items():
                if nid != new_node_id:
                    d = float(np.linalg.norm(n.position - new_node.position))
                    if d < min_d:
                        min_d = d
                        connected_to = nid
            if connected_to:
                self.beams.append(GroundBeam(new_node_id, connected_to, strength=1.5))

        event = {
            "type": "CRYSTALLIZATION",
            "wave_crystallized": wave.wave_id,
            "new_ground_node": new_node_id,
            "blueprint_id": blueprint.blueprint_id,
            "density_formed": new_density,
            "connected_beam": f"{new_node_id}-{connected_to}" if connected_to else "None",
            "homology_after": self.get_homology_metrics(),
        }
        self.transition_history.append(event)
        return event

    def _propagate_homological_resonance(self, wave: PerturbationWave) -> Dict[str, Any]:
        """
        [Multi-Dimensional Resonance Propagation across Homological Cycles]
        Passes wave perturbation through 0-ground Betti-1 cycles.
        - Thin Ground (B1=0): Single-pass linear decay.
        - Deep Ground (B1>=1): Waves circulate through cycles, inducing phase alignment
          and multi-dimensional resonance across invariant axes.
        """
        homology = self.get_homology_metrics()
        b1_cycles = homology["B1"]

        if b1_cycles == 0:
            # Thin Ground: Rapid linear damping
            decay_factor = 0.85
            wave.amplitude *= decay_factor
            resonance_depth = "Shallow (Single-pass)"
            resonance_energy = wave.energy * 0.15
        else:
            # Deep Ground: Multi-cycle constructive resonance & deep self-reconstruction
            resonance_boost = 1.0 + (b1_cycles * 0.25)
            resonance_energy = wave.energy * resonance_boost
            # Phase axis alignment across ground nodes
            for node in self.nodes.values():
                node.phase_axis = 0.9 * node.phase_axis + 0.1 * wave.phase_vector
                norm = np.linalg.norm(node.phase_axis)
                if norm > 1e-9:
                    node.phase_axis /= norm
            resonance_depth = f"Deep ({b1_cycles} Homological Cycles Resonating)"

        return {
            "betti_1_cycles": b1_cycles,
            "resonance_depth": resonance_depth,
            "resonance_energy": resonance_energy,
            "wave_amplitude_after": wave.amplitude,
        }

    def step_phase_relaxation(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Systemic phase relaxation step:
        - Dissipates stored elastic energy $X$ back into ground stability.
        - Cools down accumulated friction heat.
        - Re-balances total phase mass.
        """
        if self.stored_reactive_energy > 0:
            relaxation = self.stored_reactive_energy * 0.1 * dt
            self.stored_reactive_energy -= relaxation
            for node in self.nodes.values():
                node.stability += relaxation / max(1, len(self.nodes))

        self.accumulated_friction_heat *= (1.0 - 0.05 * dt)
        self._update_total_phase_mass()

        return {
            "total_phase_mass": self.total_phase_mass,
            "stored_reactive_energy": self.stored_reactive_energy,
            "accumulated_friction_heat": self.accumulated_friction_heat,
            "num_nodes_0": len(self.nodes),
            "num_waves_1": len(self.waves),
            "homology": self.get_homology_metrics(),
        }

    # --- Elysia Core Module Integration Methods ---

    def sync_with_crystallization_field(self, field: Any):
        """
        [Field Sync] Bridges Ground 0 nodes with 2D CrystallizationField conductance & activation.
        Project ground node positions to 2D field coordinates and reinforces conductance.
        """
        if field is None:
            return

        res = field.resolution
        for node in self.nodes.values():
            pos_2d = np.array([
                (node.position[0] + 1.0) * (res * 0.4),
                (node.position[1] + 1.0) * (res * 0.4),
            ], dtype=np.float32)
            field.crystallize_gene(pos_2d, np.uint64(hash(node.node_id) & 0xFFFFFFFFFFFFFFFF))
            field.reflect_self_logic(pos_2d, node.topological_density)

    def process_virtual_gate_friction(self, gate_loss: float, pid_control_signal: float, context_vector: np.ndarray) -> Dict[str, Any]:
        """
        [Virtual Gate Integration] Converts Virtual Causal Gate friction loss & PID signal into a 1-PerturbationWave
        and processes it through phase transition and complex impedance elasticity.
        """
        wave_id = f"Gate_Wave_{int(time.time()*1000)%10000}"
        wave = PerturbationWave(
            wave_id=wave_id,
            phase_vector=context_vector,
            frequency=1.0 + pid_control_signal,
            amplitude=math.sqrt(max(0.1, gate_loss * 2.0)),
            entropy=1.0 + gate_loss,
            cause_origin="VirtualCausalGate_Mesh_Friction",
        )
        return self.inject_perturbation_wave(wave)

    def sync_with_topological_reconstruction_engine(self, reconstruction_engine: Any) -> Dict[str, Any]:
        """
        [Phase Topological Reconstruction Integration]
        Synchronizes sealed attractors and recalled phase invariants with 0-Ground blueprints.
        """
        if reconstruction_engine is None:
            return {"synced": 0}

        synced_count = 0
        for inv_name, inv in reconstruction_engine.invariants.items():
            if f"G0_{inv_name}" not in self.nodes:
                wave = PerturbationWave(
                    wave_id=f"InvWave_{inv_name}",
                    phase_vector=inv.phase_vector,
                    frequency=1.0,
                    amplitude=inv.curvature,
                    entropy=0.05,
                    cause_origin=f"PhaseTopologicalEngine_{inv_name}",
                )
                self.inject_perturbation_wave(wave)
                synced_count += 1

        return {"synced_invariants": synced_count, "homology": self.get_homology_metrics()}
