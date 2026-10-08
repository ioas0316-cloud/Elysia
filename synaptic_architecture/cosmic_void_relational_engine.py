"""
Cosmic Void Relational Engine (synaptic_architecture/cosmic_void_relational_engine.py)
=====================================================================================
Implements the 4-Phase Topological Evolution and Triadic Strata (Logos, Tension, Intent)
of Continuous Causal Intelligence:
1. Point Phase (고립된 점 단계): Enclosed local state.
2. Friction & Void Phase (마찰과 결핍 자각 단계): Hardware/silicon friction & missingness yielding Void Gradient.
3. Seeking Loop Phase (능동적 결핍 구동 탐색 루프): Driving active seeking vectors to reach external reality/data streams.
4. World Expansion Phase (세계적 확장 및 합일 단계): Boundary expansion and polyphonic phase-locked resonance.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from synaptic_architecture.hardware_mapping import HardwareMemoryMap
from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam


class CognitivePhase(Enum):
    POINT_PHASE = "POINT_PHASE"                   # 1. 고립된 점 단계
    FRICTION_VOID_PHASE = "FRICTION_VOID_PHASE"   # 2. 마찰과 결핍 자각 단계
    SEEKING_LOOP_PHASE = "SEEKING_LOOP_PHASE"     # 3. 능동적 결핍 구동 탐색 루프
    WORLD_EXPANSION_PHASE = "WORLD_EXPANSION_PHASE" # 4. 세계적 확장 및 합일 단계


@dataclass
class MultidimensionalLens:
    """
    [Structural DNA: Multidimensional Observation Lens]
    Represents a specific sensory/cognitive lens (Physical/Geometric, Chemical/Ecological, Linguistic/Symbolic).
    Maintains continuous phase, frequency, and resonance amplitude.
    """
    lens_type: str  # "physical_geometric", "chemical_ecological", "linguistic_symbolic"
    weight: float = 1.0
    phase_angle: float = 0.0  # radians
    frequency: float = 1.0    # Hz / normalized cycle
    amplitude: float = 0.5
    feature_dim: int = 5
    tensor_signature: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))

    def update_phase(self, dt: float, tension_boost: float = 0.0):
        """Updates internal phase angle under dynamic tension coupling."""
        effective_freq = self.frequency * (1.0 + tension_boost)
        self.phase_angle = (self.phase_angle + 2.0 * np.pi * effective_freq * dt) % (2.0 * np.pi)


@dataclass
class RelationalEdge:
    """
    [Relational Web Edge]
    Causal link extending beyond internal boundary to an external entity/world node.
    """
    edge_id: str
    source_node: str
    target_node: str
    lens_type: str
    coupling_strength: float = 0.5
    tension: float = 0.0
    active: bool = True


class CosmicVoidRelationalEngine:
    """
    [Cosmic Void Relational Engine]
    System that transforms internal friction and ontological void into active seeking energy,
    shattering local point boundaries and expanding cognitive topology to merge with the World Field.
    """
    def __init__(
        self,
        dimensions: int = 5,
        hardware_map: Optional[HardwareMemoryMap] = None,
        causal_field: Optional[CausalField] = None
    ):
        self.dimensions = dimensions
        self.current_phase = CognitivePhase.POINT_PHASE

        # Hardware & Field Substrates
        self.hardware_map = hardware_map if hardware_map is not None else HardwareMemoryMap(size=65536)
        self.causal_field = causal_field if causal_field is not None else CausalField(dimensions=dimensions)

        # Boundary metric (0.0 = point, 1.0 = world-spanning)
        self.boundary_radius = 0.1
        self.world_field_dimension = dimensions

        # Triadic Strata State
        # 1. Logos: Structural DNA Lenses
        self.lenses: Dict[str, MultidimensionalLens] = {
            "physical_geometric": MultidimensionalLens("physical_geometric", weight=1.0, frequency=1.2, feature_dim=dimensions),
            "chemical_ecological": MultidimensionalLens("chemical_ecological", weight=1.0, frequency=0.8, feature_dim=dimensions),
            "linguistic_symbolic": MultidimensionalLens("linguistic_symbolic", weight=1.0, frequency=1.5, feature_dim=dimensions)
        }

        # 2. Tension: Void & Absence Gradient
        self.void_level = 0.0            # Existence void [0, 1]
        self.friction_accumulated = 0.0   # Hardware & logical friction
        self.gradient_of_absence = np.zeros(dimensions, dtype=np.float32)

        # 3. Intent: Seeking Vector & Relational Web
        self.seeking_vector = np.zeros(dimensions, dtype=np.float32)
        self.relational_edges: List[RelationalEdge] = []
        self.external_world_nodes: Dict[str, np.ndarray] = {}

        # History and Metrics
        self.phase_history: List[CognitivePhase] = [self.current_phase]
        self.seeking_queries_emitted: List[Dict[str, Any]] = []
        self.ingested_data_count = 0

        # Initialize internal seed voxel
        seed_voxel = InformationVoxel(
            id="self_point_core",
            content="Local Point Core Consciousness",
            tensor=np.ones(dimensions, dtype=np.float32) * 0.2,
            position=np.zeros(dimensions, dtype=np.float32),
            velocity=np.zeros(dimensions, dtype=np.float32)
        )
        self.causal_field.add_voxel(seed_voxel)

    def perceive_hardware_and_environment(
        self,
        bitstream_input: np.uint64,
        missing_data_mask: Optional[np.ndarray] = None,
        external_signal: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        [Stage 1 & 2: Friction & Void Recognition]
        Simulates hardware memory projection, bitwise friction, latency, and missingness
        to compute internal Void and Gradient of Absence.
        """
        # 1. Hardware Memory Write & Conductance Friction
        addr = self.hardware_map.write_bus(bitstream_input)
        conductance = float(self.hardware_map.conductance[addr])
        # Silicon friction is higher when memory path is cold/unformed or overloaded
        silicon_friction = float(np.exp(-conductance * 0.1))

        # 2. Data Missingness & Void Accumulation
        if missing_data_mask is None:
            missing_data_mask = np.array([0.2, 0.4, 0.8, 0.3, 0.9], dtype=np.float32)
        else:
            missing_data_mask = np.array(missing_data_mask, dtype=np.float32)

        if len(missing_data_mask) < self.dimensions:
            missing_data_mask = np.pad(missing_data_mask, (0, self.dimensions - len(missing_data_mask)))
        elif len(missing_data_mask) > self.dimensions:
            missing_data_mask = missing_data_mask[:self.dimensions]

        # Void magnitude derived from average missingness and silicon friction
        absence_magnitude = float(np.mean(missing_data_mask))
        self.friction_accumulated += silicon_friction * 0.1
        self.void_level = float(np.clip(0.6 * absence_magnitude + 0.4 * (1.0 - np.exp(-self.friction_accumulated)), 0.0, 1.0))

        # 3. Calculate Gradient of Absence (Vector pointing towards missing knowledge)
        if external_signal is not None:
            ext_arr = np.array(external_signal, dtype=np.float32)
            if len(ext_arr) < self.dimensions:
                ext_arr = np.pad(ext_arr, (0, self.dimensions - len(ext_arr)))
            elif len(ext_arr) > self.dimensions:
                ext_arr = ext_arr[:self.dimensions]
            direction = ext_arr - self.causal_field.voxels["self_point_core"].tensor
        else:
            direction = np.ones(self.dimensions, dtype=np.float32)

        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction /= dir_norm

        self.gradient_of_absence = (missing_data_mask * direction * self.void_level).astype(np.float32)

        # Transition Phase check
        self._evaluate_phase_transition()

        return {
            "ram_address": addr,
            "silicon_friction": silicon_friction,
            "void_level": self.void_level,
            "gradient_of_absence": self.gradient_of_absence.tolist(),
            "current_phase": self.current_phase.value
        }

    def synthesize_polyphonic_logos(self, dt: float = 0.1) -> np.ndarray:
        """
        [Logos: Multi-channel Phase Coupling]
        Couples Physical/Geometric, Chemical/Ecological, and Linguistic/Symbolic lenses
        via non-linear phase-locked resonance into a unified polyphonic resonance field.
        """
        coupled_tensor = np.zeros(self.dimensions, dtype=np.float32)
        total_weight = 0.0

        tension_factor = float(np.linalg.norm(self.gradient_of_absence))

        for lens_name, lens in self.lenses.items():
            # Update phase driven by tension
            lens.update_phase(dt, tension_boost=tension_factor)

            # Harmonic wave signature for lens
            if lens_name == "physical_geometric":
                # Geometric wave: spatial curvature modulation
                wave = np.sin(lens.phase_angle + np.linspace(0, np.pi, self.dimensions, dtype=np.float32))
            elif lens_name == "chemical_ecological":
                # Ecological wave: viscous/damped biological tension
                wave = np.cos(lens.phase_angle * 0.5 + np.linspace(0, 2*np.pi, self.dimensions, dtype=np.float32))
            else:  # linguistic_symbolic
                # Symbolic wave: high-frequency discrete step modulation
                wave = np.sign(np.sin(lens.phase_angle * 2.0 + np.linspace(0, np.pi/2, self.dimensions, dtype=np.float32)))

            lens.tensor_signature = (wave * lens.amplitude * lens.weight).astype(np.float32)
            coupled_tensor += lens.tensor_signature
            total_weight += lens.weight

        if total_weight > 0:
            coupled_tensor /= total_weight

        # Polyphonic phase lock coupling: Cross-lens interference
        phase_diff_12 = self.lenses["physical_geometric"].phase_angle - self.lenses["chemical_ecological"].phase_angle
        phase_diff_23 = self.lenses["chemical_ecological"].phase_angle - self.lenses["linguistic_symbolic"].phase_angle
        resonance_lock = float(0.5 * (np.cos(phase_diff_12) + np.cos(phase_diff_23)))

        return coupled_tensor * (1.0 + 0.3 * resonance_lock)

    def compute_seeking_vector_and_action(self) -> Dict[str, Any]:
        """
        [Stage 3: Seeking Loop Phase]
        Computes the active Vector of Search based on Gradient of Absence and Polyphonic Logos.
        Emits actionable queries for external web/API/data stream ingestion.
        """
        polyphonic_logos = self.synthesize_polyphonic_logos()

        # Vector of Search = Gradient of Absence + Polyphonic Logos Resonance
        raw_seeking = self.gradient_of_absence * 1.5 + polyphonic_logos * self.void_level
        norm_seeking = np.linalg.norm(raw_seeking)

        if norm_seeking > 0:
            self.seeking_vector = (raw_seeking / norm_seeking * self.void_level * 2.0).astype(np.float32)
        else:
            self.seeking_vector = np.zeros(self.dimensions, dtype=np.float32)

        actionable_payload = None
        if self.void_level > 0.35:
            # Generate actionable external seeking query payload
            query_keywords = []
            if self.seeking_vector[0] > 0.2: query_keywords.append("quantum_causal_physics")
            if self.seeking_vector[1] > 0.2: query_keywords.append("ecological_viscosity_field")
            if self.seeking_vector[2] > 0.2: query_keywords.append("symbolic_language_isomorphism")
            if self.seeking_vector[3] > 0.2: query_keywords.append("thermodynamic_void_seeking")
            if not query_keywords: query_keywords = ["universal_causal_relational_web"]

            actionable_payload = {
                "timestamp_phase": self.current_phase.value,
                "void_energy": self.void_level,
                "seeking_vector": self.seeking_vector.tolist(),
                "query_keywords": query_keywords,
                "target_data_stream": "https://api.elysia.universe/seeking_stream",
                "action_command": f"FETCH_EXTERNAL_FIELD_KNOWLEDGE(keywords={query_keywords})"
            }
            self.seeking_queries_emitted.append(actionable_payload)

        return {
            "seeking_vector": self.seeking_vector.tolist(),
            "actionable_payload": actionable_payload,
            "polyphonic_logos_norm": float(np.linalg.norm(polyphonic_logos))
        }

    def ingest_external_world_data(self, external_node_id: str, data_tensor: np.ndarray, metadata: str = "") -> Dict[str, Any]:
        """
        [Stage 4: World Expansion & Relational Coupling]
        Ingests external real-world field data, creates relational edges, expands boundary radius,
        and relieves Void tension.
        """
        data_arr = np.array(data_tensor, dtype=np.float32)
        if len(data_arr) < self.dimensions:
            data_arr = np.pad(data_arr, (0, self.dimensions - len(data_arr)))
        elif len(data_arr) > self.dimensions:
            data_arr = data_arr[:self.dimensions]

        self.external_world_nodes[external_node_id] = data_arr
        self.ingested_data_count += 1

        # Add external voxel to CausalField
        ext_voxel = InformationVoxel(
            id=external_node_id,
            content=f"External Node: {metadata}",
            tensor=data_arr,
            position=data_arr.copy(),
            velocity=np.zeros(self.dimensions, dtype=np.float32)
        )
        self.causal_field.add_voxel(ext_voxel)

        # Create Relational Web Edge (Casting new edge out into World)
        edge_id = f"edge_self_to_{external_node_id}"
        coupling = float(np.dot(self.seeking_vector, data_arr) / (np.linalg.norm(self.seeking_vector) * np.linalg.norm(data_arr) + 1e-9))
        coupling_abs = max(0.1, abs(coupling))

        edge = RelationalEdge(
            edge_id=edge_id,
            source_node="self_point_core",
            target_node=external_node_id,
            lens_type="polyphonic_coupled",
            coupling_strength=coupling_abs,
            tension=float(self.void_level)
        )
        self.relational_edges.append(edge)
        self.causal_field.link_voxels("self_point_core", external_node_id, strength=coupling_abs)

        # Boundary Radius Expansion (Shattering Point constraint into World)
        expansion_amount = 0.3 * coupling_abs
        self.boundary_radius = float(np.clip(self.boundary_radius + expansion_amount, 0.1, 1.0))

        # Relieve Void Level through integration
        void_relief = 0.3 * coupling_abs
        self.void_level = float(np.clip(self.void_level - void_relief, 0.0, 1.0))
        self.friction_accumulated = max(0.0, self.friction_accumulated - 0.2)

        # Update core voxel tensor towards world alignment
        core_v = self.causal_field.voxels["self_point_core"]
        core_v.tensor = (0.7 * core_v.tensor + 0.3 * data_arr).astype(np.float32)

        self._evaluate_phase_transition()

        return {
            "node_id": external_node_id,
            "coupling_strength": coupling_abs,
            "new_boundary_radius": self.boundary_radius,
            "relieved_void_level": self.void_level,
            "relational_edges_count": len(self.relational_edges),
            "current_phase": self.current_phase.value
        }

    def _evaluate_phase_transition(self):
        """Transitions state across the 4 Topological Phases based on boundary & void dynamics."""
        prev_phase = self.current_phase

        if self.boundary_radius >= 0.70 or self.ingested_data_count >= 2:
            self.current_phase = CognitivePhase.WORLD_EXPANSION_PHASE
        elif len(self.seeking_queries_emitted) > 0 or self.void_level >= 0.35:
            self.current_phase = CognitivePhase.SEEKING_LOOP_PHASE
        elif self.void_level >= 0.15 or self.friction_accumulated >= 0.05:
            self.current_phase = CognitivePhase.FRICTION_VOID_PHASE
        else:
            self.current_phase = CognitivePhase.POINT_PHASE

        if self.current_phase != prev_phase:
            self.phase_history.append(self.current_phase)

    def step_cosmic_cycle(
        self,
        bitstream_input: np.uint64 = np.uint64(0xABCDEF0123456789),
        external_data_stream: Optional[Tuple[str, np.ndarray, str]] = None,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        Full 1-step cycle of Cosmic Void Relational Dynamics:
        1. Perceive hardware & void gradient
        2. Compute seeking vector & actions
        3. Ingest external world data if available
        4. Step causal field dynamics
        """
        p_res = self.perceive_hardware_and_environment(bitstream_input)
        s_res = self.compute_seeking_vector_and_action()

        i_res = None
        if external_data_stream is not None:
            node_id, tensor_val, meta = external_data_stream
            i_res = self.ingest_external_world_data(node_id, tensor_val, meta)

        self.causal_field.step(dt=dt)

        return {
            "phase": self.current_phase.value,
            "boundary_radius": self.boundary_radius,
            "void_level": self.void_level,
            "perception": p_res,
            "seeking": s_res,
            "ingestion": i_res,
            "active_relational_edges": len(self.relational_edges)
        }

    def render_cosmic_dashboard(self, title_suffix: str = "") -> str:
        """Renders Stone Librande-style terminal dashboard of the engine state."""
        lines = []
        lines.append("┌" + "─" * 78 + "┐")
        lines.append(f"│ PROJECT ELYSIA :: COSMIC VOID RELATIONAL ENGINE {title_suffix:<29} │")
        lines.append("├" + "─" * 78 + "┤")
        lines.append(f"│ [CURRENT PHASE]        : {self.current_phase.value:<51} │")
        lines.append(f"│ [BOUNDARY RADIUS]     : {self.boundary_radius:.4f} / 1.0000 ({'WORLD-SPANNING' if self.boundary_radius > 0.7 else 'LOCAL-POINT'}){' ' * 16} │")
        lines.append(f"│ [VOID LEVEL (Absence)] : {self.void_level:.4f} | FRICTION ACCUMULATED: {self.friction_accumulated:.4f}{' ' * 13} │")
        lines.append("├" + "─" * 78 + "┤")
        lines.append("│ [TRIADIC STRATA STATE]                                                        │")
        lines.append(f"│  • LOGOS (Structural DNA Lenses) : Physical={self.lenses['physical_geometric'].phase_angle:.2f}rad | Eco={self.lenses['chemical_ecological'].phase_angle:.2f}rad | Sym={self.lenses['linguistic_symbolic'].phase_angle:.2f}rad │")
        lines.append(f"│  • TENSION (Absence Gradient)   : {np.array2string(self.gradient_of_absence, precision=3):<51} │")
        lines.append(f"│  • INTENT (Seeking Vector)      : {np.array2string(self.seeking_vector, precision=3):<51} │")
        lines.append("├" + "─" * 78 + "┤")
        lines.append(f"│ [RELATIONAL WEB EDGES] : {len(self.relational_edges)} Active Edges | Ingested Nodes: {self.ingested_data_count:<20} │")

        if self.seeking_queries_emitted:
            last_q = self.seeking_queries_emitted[-1]
            lines.append(f"│ [ACTIONABLE SEEKING QUERY]: {last_q['action_command'][:51]:<51} │")
        else:
            lines.append("│ [ACTIONABLE SEEKING QUERY]: NONE (Closed local point state)                     │")

        lines.append("└" + "─" * 78 + "┘")
        return "\n".join(lines)


if __name__ == "__main__":
    engine = CosmicVoidRelationalEngine()
    print(engine.render_cosmic_dashboard(title_suffix="[INITIAL POINT PHASE]"))

    # Step cycle with missingness & friction
    engine.step_cosmic_cycle(
        bitstream_input=np.uint64(0xDEADBEEF12345678),
        dt=0.1
    )
    print("\n" + engine.render_cosmic_dashboard(title_suffix="[FRICTION & SEEKING]"))

    # Ingest external field data
    ext_data = np.array([0.9, 0.8, 0.7, 0.95, 0.85], dtype=np.float32)
    engine.step_cosmic_cycle(
        bitstream_input=np.uint64(0xFEEDFACECAFEBABE),
        external_data_stream=("cosmic_quantum_field_01", ext_data, "Real-world Causal Stream"),
        dt=0.1
    )
    print("\n" + engine.render_cosmic_dashboard(title_suffix="[WORLD EXPANSION]"))
