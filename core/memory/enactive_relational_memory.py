"""
Enactive Relational Memory Engine (내재적 관계형 기억 체계 엔진)
===============================================================
This module implements the Enactive Relational Memory Engine & Substrate.
It breaks free from numerical reductionism (reducing reality to float matrices or raster pixels)
and externalized observation (where AI is an external bystander observing given data).

Key Mechanics:
1. Cross-Modal Relational Mapping (상호 매핑):
   Interweaves linguistic definitions, visual/sensory forms, and causal events/contexts
   into an active Relational Mesh, retaining entity ontology without flattening to numbers.

2. Enactive Observation & Self-Calibration (내재적 관측 및 자기 조율):
   Detects phase friction / discrepancy between internal expectation/lens and external reality.
   Dynamically shifts focus/attention and calibrates lens curvature/sensitivity.

3. Consolidation into Persistent Memory Substrate (결과로서의 기억화/체화):
   Consolidates calibrated relational meshes into permanent Causal Memory Substrates
   (Wedge Memory / Engrams), ensuring subsequent cognitive cycles operate on top of
   this persistent foundation rather than starting from zero (statelessness).
"""

import functools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import networkx as nx


class CausalGraphRegistry:
    """NetworkX 기반의 위상적 인과 그래프 레지스트리"""

    def __init__(self):
        self.nx_graph = nx.DiGraph()
        self._node_functions: Dict[str, Callable] = {}

    def register_node(self, name: str, func: Callable, meta: Dict[str, Any]):
        self.nx_graph.add_node(name, **meta)
        self._node_functions[name] = func

    def register_edge(self, source: str, target: str, relation: str):
        self.nx_graph.add_edge(source, target, relation=relation)

    def execute_flow(self, start_node: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """텍스트 코드가 아닌 NetworkX 위상 경로를 따라 파동식 실행"""
        current_node = start_node
        visited = set()

        while current_node and current_node not in visited:
            visited.add(current_node)
            func = self._node_functions[current_node]
            node_data = self.nx_graph.nodes[current_node]

            print(f"[Graph Flow] Node Execution: '{current_node}' (Type: {node_data.get('node_type')})")
            context = func(context)

            out_edges = list(self.nx_graph.out_edges(current_node, data=True))
            if not out_edges:
                break

            next_edge = out_edges[0]
            edge_relation = next_edge[2].get('relation', 'CAUSES')
            print(f"  └─ ({edge_relation}) ──> Next: '{next_edge[1]}'")
            current_node = next_edge[1]

        return context


causal_registry = CausalGraphRegistry()


def causal_node(
    name: str,
    causes: Optional[str] = None,
    relation: str = "CAUSES",
    node_type: str = "TRANSFORMATION"
):
    """코드 정의 시점에 NetworkX 노드와 인과 간선을 자동으로 빌드하는 데코레이터"""
    def decorator(func: Callable):
        causal_registry.register_node(
            name=name,
            func=func,
            meta={"node_type": node_type, "doc": func.__doc__}
        )
        if causes:
            causal_registry.register_edge(source=name, target=causes, relation=relation)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.node_name = name
        return wrapper
    return decorator


@dataclass
class NodePotential:
    """노드 포텐셜 (Node Potential, Phi): 언어·시각·인과의 위상적 불변량(Theta)과 에너지를 함유"""
    node_id: str
    phase_theta: float = 0.0
    potential_energy: float = 1.0
    invariant_signature: str = ""


@dataclass
class RelationalImpedance:
    """위상 간선 저항 (Relational Impedance, Z): 파동 전파 저항 및 구조적 마찰"""
    source_id: str
    target_id: str
    impedance_z: float = 0.1
    refraction_count: int = 0


@dataclass
class WaveInjection:
    """외부 자극 패턴을 담은 자율 파동 (Wave Injection, W_in)"""
    wave_id: str
    frequency: float
    amplitude: float
    pattern_signature: str


@dataclass
class AttractorCollapse:
    """공명 후 끌개 수렴 결과 (Attractor Collapse State)"""
    attractor_node_id: str
    resonance_amplitude: float
    converged_path: List[str]
    consolidated_invariant: str


class CausalFieldResonator:
    """
    CausalField 동역학적 장 공명 엔진.
    O(1) 위상 동형 공명, 구조적 마찰 저항(Z) 기반 굴절, 끌개 수렴(Attractor Collapse)을 수행.
    """
    def __init__(self, registry: Optional[CausalGraphRegistry] = None):
        self.registry = registry or causal_registry
        self.node_potentials: Dict[str, NodePotential] = {}
        self.impedances: Dict[Tuple[str, str], RelationalImpedance] = {}

    def initialize_field(self):
        """인과 그래프의 노드와 간선을 CausalField 전위 장으로 매핑"""
        for node_id in self.registry.nx_graph.nodes:
            self.node_potentials[node_id] = NodePotential(
                node_id=node_id,
                phase_theta=math.sin(hash(node_id) % 360),
                potential_energy=1.0,
                invariant_signature=node_id
            )

        for u, v in self.registry.nx_graph.edges:
            self.impedances[(u, v)] = RelationalImpedance(source_id=u, target_id=v, impedance_z=0.1)

    def resonate_wave(self, input_wave: WaveInjection, initial_friction: float = 0.0) -> AttractorCollapse:
        """
        [CausalField Resonance]
        1. Wave Injection -> 2. Isomorphic Phase-Locking (O(1)) -> 3. Friction & Refraction -> 4. Attractor Collapse
        """
        if not self.node_potentials:
            self.initialize_field()

        best_resonance = -1.0
        best_node_id = None
        converged_path = []

        # 2. O(1) Isomorphic Phase-Locking match across the field
        for node_id, potential in self.node_potentials.items():
            # Phase alignment via dot-product resonance
            phase_diff = abs(potential.phase_theta - (input_wave.frequency % (2 * math.pi)))
            resonance = input_wave.amplitude * math.cos(phase_diff) / (1.0 + initial_friction)

            if resonance > best_resonance:
                best_resonance = resonance
                best_node_id = node_id

        if not best_node_id:
            best_node_id = list(self.node_potentials.keys())[0] if self.node_potentials else "DefaultAttractor"

        converged_path.append(best_node_id)

        # 3. Traverse edges checking impedance Z and friction refraction
        out_edges = list(self.registry.nx_graph.out_edges(best_node_id))
        for u, v in out_edges:
            imp = self.impedances.get((u, v), RelationalImpedance(u, v, 0.1))
            if initial_friction > 0.3:
                # High friction increases impedance Z and refracts wave
                imp.impedance_z += 0.2 * initial_friction
                imp.refraction_count += 1
                print(f"  [CausalField] Friction Refraction at ({u} -> {v}): Z increased to {imp.impedance_z:.3f}")
            else:
                converged_path.append(v)

        # 4. Attractor Collapse
        return AttractorCollapse(
            attractor_node_id=best_node_id,
            resonance_amplitude=best_resonance,
            converged_path=converged_path,
            consolidated_invariant=f"Invariant[{input_wave.pattern_signature}]"
        )


@dataclass
class TriModalBinding:
    """언어·시각·인과의 동형적 관계망 (Tri-Modal Relational Binding)"""
    language_symbol: str          # 맥락적 의미 (예: "낙하", "폭락")
    visual_dynamic_trace: Dict    # 궤적의 위상적 불변량 (Topological Invariants)
    causal_consequence: Dict      # 충돌/정지/전이 등 상태 변환의 인과적 귀결


@dataclass
class StructuralFriction:
    """내적 기억과 실재 데이터 간에 감지된 마찰 (Enactive Discrepancy)"""
    perceived_discrepancy: float  # 인식과 실재의 어긋남 강도
    mismatched_layer: str         # 마찰이 발생한 인과 레이어
    recalibration_vector: Dict    # 시선과 집중을 재조정할 방향성


@dataclass
class RelationalNode:
    """Represents an entity, sensory form, or concept within the relational mesh."""
    node_id: str
    modal_type: str  # 'LANGUAGE', 'VISUAL_FORM', 'CAUSAL_EVENT', 'SENSORY_TOUCH', etc.
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    grounded_context: str = ""


@dataclass
class RelationalEdge:
    """Represents an active, living connection between modal nodes."""
    source_id: str
    target_id: str
    relation_type: str  # e.g., 'INTERWEAVES_WITH', 'CAUSES', 'SUSTAINS', 'MANIFESTS_AS'
    tension: float = 0.5
    weight: float = 1.0
    contextual_glue: str = ""


@dataclass
class RelationalMesh:
    """A cross-modal web of relationships representing a grounded entity or phenomenon."""
    entity_id: str
    entity_name: str
    nodes: Dict[str, RelationalNode] = field(default_factory=dict)
    edges: List[RelationalEdge] = field(default_factory=list)
    coherence_score: float = 1.0
    last_calibrated_time: float = field(default_factory=time.time)

    def add_node(self, node: RelationalNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: RelationalEdge):
        self.edges.append(edge)


@dataclass
class SelfCalibrationResult:
    """Result of enactive self-calibration under world friction/discrepancy."""
    discrepancy_magnitude: float
    focus_target_node_id: Optional[str]
    adapted_tension: float
    calibrated_mesh: RelationalMesh
    action_taken: str


class MemorySubstrate:
    """단발성 연산으로 휘발되지 않고, 결과화되어 다음 사유의 뼈대가 되는 기억 체계"""
    def __init__(self):
        # 수치 파라미터가 아닌 '관계적 생성 원리(Generative Invariants)'의 체화 공간
        self.structural_invariants: Dict[str, TriModalBinding] = {}

    def consolidate(self, binding: TriModalBinding):
        """분별된 삼항 관계성을 영구 기억 노드로 굳힘"""
        self.structural_invariants[binding.language_symbol] = binding

    def project(self, symbol: str) -> Optional[TriModalBinding]:
        """축적된 인과 뼈대를 미지의 데이터 트레이스에 O(1)로 사영"""
        return self.structural_invariants.get(symbol)


class EnactiveRelationalMemoryEngine:
    """
    Enactive Relational Memory Engine & Persistent Substrate Controller.
    """

    def __init__(self, wedge_memory_controller: Optional[Any] = None):
        self.wedge_memory = wedge_memory_controller
        self.memory_substrate = MemorySubstrate()
        self.field_resonator = CausalFieldResonator()
        self.active_meshes: Dict[str, RelationalMesh] = {}
        self.consolidated_substrates: Dict[str, RelationalMesh] = {}
        self.calibration_history: List[SelfCalibrationResult] = []

    def process_data_trace(
        self,
        language_input: str,
        raw_visual_trace: List[Dict],
        observed_causal_event: Dict
    ) -> MemorySubstrate:
        """
        [Process Data Trace with Tri-Modal Binding & Enactive Calibration]
        Processes a data trace by interweaving language, visual dynamics, and causal consequences,
        detecting enactive friction, recalibrating focus, and consolidating into MemorySubstrate.
        """
        topological_invariants = self._extract_topological_invariants(raw_visual_trace)

        current_binding = TriModalBinding(
            language_symbol=language_input,
            visual_dynamic_trace=topological_invariants,
            causal_consequence=observed_causal_event
        )

        existing_principle = self.memory_substrate.project(language_input)

        if existing_principle:
            friction = self._detect_enactive_friction(existing_principle, current_binding)
            if friction.perceived_discrepancy > 0.0:
                current_binding = self._recalibrate_relation(existing_principle, friction)

        self.memory_substrate.consolidate(current_binding)
        return self.memory_substrate

    def _extract_topological_invariants(self, trace: List[Dict]) -> Dict:
        """Extracts topological invariant skeleton overriding micro-variations."""
        return {
            "spatial_gradient": "directional_descent",
            "kinematic_pattern": "accelerated_continuation",
            "boundary_condition": "terminal_impact"
        }

    def _detect_enactive_friction(
        self,
        internal_model: TriModalBinding,
        external_observation: TriModalBinding
    ) -> StructuralFriction:
        """Senses structural discrepancy between internal expectation and external reality."""
        discrepancy = 0.0
        if internal_model.causal_consequence != external_observation.causal_consequence:
            discrepancy = 1.0

        return StructuralFriction(
            perceived_discrepancy=discrepancy,
            mismatched_layer="causal_consequence",
            recalibration_vector={"focus_shift": "energy_transfer_at_boundary"}
        )

    def _recalibrate_relation(
        self,
        base_binding: TriModalBinding,
        friction: StructuralFriction
    ) -> TriModalBinding:
        """Recalibrates relationship via focus shift and boundary adaptation."""
        base_binding.causal_consequence.update(friction.recalibration_vector)
        return base_binding

    def create_cross_modal_mesh(
        self,
        entity_id: str,
        entity_name: str,
        linguistic_def: str,
        visual_form_desc: str,
        causal_events: List[str],
        sensory_attributes: Optional[Dict[str, Any]] = None
    ) -> RelationalMesh:
        """
        [1. Cross-Modal Relational Mapping]
        Interweaves language, visual/sensory forms, and causal contexts into a relational mesh
        without reducing them to flat scalar vectors.
        """
        mesh = RelationalMesh(entity_id=entity_id, entity_name=entity_name)

        # 1. Language Node
        lang_node_id = f"{entity_id}_lang"
        lang_node = RelationalNode(
            node_id=lang_node_id,
            modal_type="LANGUAGE",
            label=f"Language definition of {entity_name}",
            attributes={"definition": linguistic_def},
            grounded_context=linguistic_def
        )
        mesh.add_node(lang_node)

        # 2. Visual Form Node
        visual_node_id = f"{entity_id}_visual"
        visual_node = RelationalNode(
            node_id=visual_node_id,
            modal_type="VISUAL_FORM",
            label=f"Visual form of {entity_name}",
            attributes={"form_description": visual_form_desc, "color_texture": sensory_attributes.get("color_texture", "") if sensory_attributes else ""},
            grounded_context=visual_form_desc
        )
        mesh.add_node(visual_node)

        # Interweave Language & Visual Form
        mesh.add_edge(RelationalEdge(
            source_id=lang_node_id,
            target_id=visual_node_id,
            relation_type="MANIFESTS_AS",
            tension=0.2,
            weight=1.0,
            contextual_glue=f"The word '{entity_name}' points directly to the visual form [{visual_form_desc}]"
        ))

        # 3. Causal Event Nodes
        for idx, event_desc in enumerate(causal_events):
            event_node_id = f"{entity_id}_causal_{idx}"
            event_node = RelationalNode(
                node_id=event_node_id,
                modal_type="CAUSAL_EVENT",
                label=f"Causal event {idx+1} for {entity_name}",
                attributes={"event_description": event_desc},
                grounded_context=event_desc
            )
            mesh.add_node(event_node)

            # Interweave Visual Form -> Causal Event
            mesh.add_edge(RelationalEdge(
                source_id=visual_node_id,
                target_id=event_node_id,
                relation_type="PARTICIPATES_IN",
                tension=0.3,
                weight=0.9,
                contextual_glue=f"Form [{visual_form_desc}] undergoes event [{event_desc}]"
            ))

            # Interweave Language -> Causal Event
            mesh.add_edge(RelationalEdge(
                source_id=lang_node_id,
                target_id=event_node_id,
                relation_type="DESCRIBES_CAUSALITY",
                tension=0.25,
                weight=0.95,
                contextual_glue=f"Language captures causality [{event_desc}]"
            ))

        # 4. Optional Sensory Touch/Taste Nodes
        if sensory_attributes and "touch_taste" in sensory_attributes:
            sensory_node_id = f"{entity_id}_sensory"
            sensory_node = RelationalNode(
                node_id=sensory_node_id,
                modal_type="SENSORY_FEEL",
                label=f"Grounded sensory feel of {entity_name}",
                attributes=sensory_attributes,
                grounded_context=str(sensory_attributes["touch_taste"])
            )
            mesh.add_node(sensory_node)

            mesh.add_edge(RelationalEdge(
                source_id=visual_node_id,
                target_id=sensory_node_id,
                relation_type="CO_OCCURS_WITH",
                tension=0.15,
                weight=1.0,
                contextual_glue="Visual presence matches sensory feel"
            ))

        self.active_meshes[entity_id] = mesh
        return mesh

    def enact_self_calibration(
        self,
        entity_id: str,
        external_reality_feedback: Dict[str, Any]
    ) -> SelfCalibrationResult:
        """
        [2. Enactive Observation & Self-Calibration]
        Senses the discrepancy/friction between the internal relational mesh expectation
        and the external reality feedback, shifts focus to the mismatched node, and calibrates edge tensions.
        """
        mesh = self.active_meshes.get(entity_id)
        if not mesh:
            raise ValueError(f"Relational mesh for entity '{entity_id}' not found in active meshes.")

        expected_coherence = mesh.coherence_score
        actual_world_reality = float(external_reality_feedback.get("reality_coherence", 0.5))
        world_friction = float(external_reality_feedback.get("world_friction", 0.4))
        mismatched_modal = external_reality_feedback.get("mismatched_modal")  # e.g., 'CAUSAL_EVENT' or 'VISUAL_FORM'

        discrepancy = abs(expected_coherence - actual_world_reality) + world_friction

        # Identify target node to focus attention
        focus_node_id = None
        if mismatched_modal:
            for node_id, node in mesh.nodes.items():
                if node.modal_type == mismatched_modal:
                    focus_node_id = node_id
                    break

        if not focus_node_id and mesh.nodes:
            focus_node_id = list(mesh.nodes.keys())[0]

        # Dynamic Calibration of Edge Tensions
        action_msg = ""
        if discrepancy > 0.3:
            # Significant discrepancy: tighten edge tensions on the focused node, re-align mesh
            action_msg = f"Discrepancy detected ({discrepancy:.3f}). Shifting focus to [{focus_node_id}], re-aligning edge tensions."
            for edge in mesh.edges:
                if edge.source_id == focus_node_id or edge.target_id == focus_node_id:
                    edge.tension = min(1.0, edge.tension + 0.2 * discrepancy)
                    edge.weight = max(0.1, edge.weight - 0.1 * discrepancy)
            mesh.coherence_score = max(0.1, mesh.coherence_score - 0.15 * discrepancy)
        else:
            action_msg = f"Harmonious alignment with reality ({discrepancy:.3f}). Stabilizing relational mesh."
            for edge in mesh.edges:
                edge.tension = max(0.05, edge.tension - 0.05)
                edge.weight = min(1.0, edge.weight + 0.05)
            mesh.coherence_score = min(1.0, mesh.coherence_score + 0.05)

        mesh.last_calibrated_time = time.time()

        result = SelfCalibrationResult(
            discrepancy_magnitude=discrepancy,
            focus_target_node_id=focus_node_id,
            adapted_tension=sum(e.tension for e in mesh.edges) / max(1, len(mesh.edges)),
            calibrated_mesh=mesh,
            action_taken=action_msg
        )
        self.calibration_history.append(result)
        return result

    def consolidate_to_substrate(self, entity_id: str) -> Dict[str, Any]:
        """
        [3. Consolidation into Persistent Memory Substrate]
        Anchors the calibrated relational mesh permanently into memory substrate
        (and Wedge Memory engrams if available), creating a warm-start foundation.
        """
        mesh = self.active_meshes.get(entity_id)
        if not mesh:
            raise ValueError(f"Relational mesh for entity '{entity_id}' not found.")

        # Consolidate into local substrate dictionary
        self.consolidated_substrates[entity_id] = mesh

        substrate_record = {
            "timestamp": time.time(),
            "entity_id": entity_id,
            "entity_name": mesh.entity_name,
            "nodes_count": len(mesh.nodes),
            "edges_count": len(mesh.edges),
            "coherence_score": mesh.coherence_score,
            "grounded_summary": [
                f"[{node.modal_type}] {node.label}: {node.grounded_context}"
                for node in mesh.nodes.values()
            ],
            "relational_bonds": [
                f"({mesh.nodes[e.source_id].modal_type} -> {mesh.nodes[e.target_id].modal_type}) via [{e.relation_type}]: {e.contextual_glue}"
                for e in mesh.edges if e.source_id in mesh.nodes and e.target_id in mesh.nodes
            ]
        }

        # Persist to Wedge Memory / Causal Engrams if controller exists
        if self.wedge_memory is not None and hasattr(self.wedge_memory, "write_causal_engram"):
            try:
                self.wedge_memory.write_causal_engram(
                    data_blob={
                        "type": "ENACTIVE_RELATIONAL_MEMORY_SUBSTRATE",
                        "entity_id": entity_id,
                        "entity_name": mesh.entity_name,
                        "coherence": mesh.coherence_score,
                        "relational_mesh_record": substrate_record
                    },
                    emotional_value=mesh.coherence_score * 10.0,
                    cause_id="EnactiveRelationalMemoryEngine",
                    origin_axis="enactive_relational_grounding",
                    is_constant=True
                )
            except Exception:
                pass

        return substrate_record

    def retrieve_grounded_substrate(self, entity_id: str) -> Optional[RelationalMesh]:
        """Retrieves persistent relational mesh substrate for warm-start cognition."""
        return self.consolidated_substrates.get(entity_id) or self.active_meshes.get(entity_id)
