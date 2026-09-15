"""
Elysia Causal Topology Foundation: Exclusivity & Coexistence Topology Engine
============================================================================
충돌이나 모순을 '에러'나 '틀린 답'으로 치부하지 않고,
어떤 조건과 환경에서 두 원리가 함께 존재할 수 없는가(상호 배타성의 경계선: Exclusivity Boundary)를
인과적 이정표로 삼아 위상적 분기(Mitosis / Layer Separation) 및 공존 지도(Coexistence Topology)를 구축하는 엔진입니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from core.topology.causal_structure import InformationTopology, TopologyLink, CausalSymbol, CausalNumber


@dataclass
class ExclusivityBoundary:
    """
    공존 불가능성의 경계선 (Exclusivity Boundary)
    - 두 위상 노드/원리(A, B)가 특정 환경 조건(Condition Vector) 하에서
      함께 존재할 수 없을 때 발생하는 파열음과 그 제약 조건 정보.
    """
    node_a_id: str
    node_b_id: str
    condition_vector: np.ndarray      # 충돌을 유발한 환경/맥락 위상 벡터
    friction_magnitude: float         # 마찰 크기
    is_mutually_exclusive: bool       # 상호 배타적 불가능성 여부
    layer_a_id: str                   # 분기된 A의 위상 층위 ID
    layer_b_id: str                   # 분기된 B의 위상 층위 ID
    description: str = ""


@dataclass
class CoexistenceLayer:
    """
    공존 위상 층위 (Coexistence Topology Layer)
    - 상호 조화롭게 융합되어 공존할 수 있는 원리와 노드들의 위상 다양체.
    """
    layer_id: str
    layer_name: str
    topology: InformationTopology
    parent_layer_id: Optional[str] = None
    depth: int = 0
    active_conditions: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


@dataclass
class CoexistenceTopologyMap:
    """
    인과적 공존 지도 (Coexistence Topology Map)
    - 무엇이 함께 존재할 수 있고(Coexistence), 무엇이 분기되어야 하는가(Exclusivity)를
      다층적 위상 레이어들(Layered Topology)의 네트워크로 보존하는 지형도.
    """
    primary_layer_id: str
    layers: Dict[str, CoexistenceLayer] = field(default_factory=dict)
    boundaries: List[ExclusivityBoundary] = field(default_factory=list)

    def get_layer(self, layer_id: str) -> Optional[CoexistenceLayer]:
        return self.layers.get(layer_id)


class ExclusivityCoexistenceTopologyEngine:
    """
    공존-배타 위상 엔진 (Exclusivity & Coexistence Topology Engine)
    - 자극 유입 시 내부 원리와의 마찰을 감지.
    - 마찰을 에러로 버리지 않고 공존 불가능성의 경계선(Exclusivity Boundary)을 추적.
    - 상호 배타적인 요소는 위상적 세포분열(Mitosis / Bifurcation)을 일으켜 별도 층위로 격리 및 수용.
    - 공존 가능한 요소는 위상적 융합(Fusion / Coexistence)을 통해 공존 지도를 조화롭게 확장.
    """
    def __init__(
        self,
        primary_topology: Optional[InformationTopology] = None,
        exclusivity_threshold: float = 0.65,
        fusion_threshold: float = 0.30
    ):
        base_topo = primary_topology or InformationTopology("RootCoexistenceTopology")
        self.exclusivity_threshold = exclusivity_threshold
        self.fusion_threshold = fusion_threshold

        root_layer = CoexistenceLayer(
            layer_id="layer_root",
            layer_name="Root_Layer",
            topology=base_topo,
            depth=0
        )
        self.coexistence_map = CoexistenceTopologyMap(
            primary_layer_id="layer_root",
            layers={"layer_root": root_layer}
        )
        self._layer_counter = 1

    def evaluate_node_friction(
        self,
        node_a_vector: np.ndarray,
        node_b_vector: np.ndarray,
        context_vector: np.ndarray
    ) -> float:
        """
        두 위상 노드/원리 간의 주어진 맥락(Context) 하에서의 마찰력(Friction) 산출.
        단순 벡터 거리가 아닌 맥락과의 직교성 및 위상 가열도 반영.
        """
        diff = node_a_vector - node_b_vector
        norm_diff = float(np.linalg.norm(diff))

        if len(context_vector) == len(diff) and np.linalg.norm(context_vector) > 1e-6:
            context_projection = abs(float(np.dot(diff, context_vector) / (np.linalg.norm(diff) * np.linalg.norm(context_vector) + 1e-6)))
            friction = norm_diff * (1.0 + 1.5 * context_projection)
        else:
            friction = norm_diff

        return float(np.clip(friction, 0.0, 2.0))

    def detect_exclusivity_boundary(
        self,
        node_a_id: str,
        node_b_id: str,
        node_a_vector: np.ndarray,
        node_b_vector: np.ndarray,
        context_vector: np.ndarray
    ) -> ExclusivityBoundary:
        """
        두 노드 간 공존 불가능성의 경계선(Exclusivity Boundary) 도출
        """
        friction = self.evaluate_node_friction(node_a_vector, node_b_vector, context_vector)
        is_exclusive = friction >= self.exclusivity_threshold

        layer_a_id = self.coexistence_map.primary_layer_id
        layer_b_id = layer_a_id

        if is_exclusive:
            layer_b_id = f"layer_mitosis_{self._layer_counter}"
            self._layer_counter += 1

        boundary = ExclusivityBoundary(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            condition_vector=context_vector.copy(),
            friction_magnitude=friction,
            is_mutually_exclusive=is_exclusive,
            layer_a_id=layer_a_id,
            layer_b_id=layer_b_id,
            description=f"Friction {friction:.3f} under context vector {context_vector.tolist()}"
        )

        self.coexistence_map.boundaries.append(boundary)
        return boundary

    def perform_topological_mitosis(
        self,
        source_layer_id: str,
        exclusive_node_id: str,
        boundary: ExclusivityBoundary
    ) -> CoexistenceLayer:
        """
        위상적 세포분열 (Topological Mitosis / Layer Bifurcation):
        공존할 수 없는 노드를 기존 층위에서 별도의 새로운 위상 층위로 분기하여 수용.
        """
        source_layer = self.coexistence_map.get_layer(source_layer_id)
        if not source_layer:
            raise ValueError(f"Source layer {source_layer_id} does not exist.")

        new_layer_id = boundary.layer_b_id
        new_layer_name = f"MitoticLayer_from_{source_layer_id}_{exclusive_node_id}"

        new_topo = source_layer.topology.clone()
        new_topo.name = new_layer_name

        new_layer = CoexistenceLayer(
            layer_id=new_layer_id,
            layer_name=new_layer_name,
            topology=new_topo,
            parent_layer_id=source_layer_id,
            depth=source_layer.depth + 1,
            active_conditions=boundary.condition_vector.copy()
        )

        self.coexistence_map.layers[new_layer_id] = new_layer
        return new_layer

    def perform_topological_fusion(
        self,
        layer_a_id: str,
        layer_b_id: str,
        coexistent_nodes: List[Tuple[str, str]]
    ) -> CoexistenceLayer:
        """
        위상적 융합 (Topological Fusion):
        마찰이 낮고 공존 가능한 요소들을 하나의 조화로운 층위로 융합.
        """
        layer_a = self.coexistence_map.get_layer(layer_a_id)
        layer_b = self.coexistence_map.get_layer(layer_b_id)

        if not layer_a or not layer_b:
            raise ValueError("Invalid layer IDs provided for fusion.")

        fused_layer_id = f"fused_{layer_a_id}_{layer_b_id}"
        fused_topo = layer_a.topology.clone()
        fused_topo.name = f"FusedTopology_{layer_a_id}_{layer_b_id}"

        for num_id, num_obj in layer_b.topology.numbers.items():
            if num_id not in fused_topo.numbers:
                fused_topo.add_number(num_obj)

        for sym_id, sym_obj in layer_b.topology.symbols.items():
            if sym_id not in fused_topo.symbols:
                fused_topo.add_symbol(sym_obj)

        for n_a, n_b in coexistent_nodes:
            fused_topo.add_link(TopologyLink(
                source_id=n_a,
                target_id=n_b,
                relation_type="coexistent_fusion",
                strength=0.9,
                tension=0.05
            ))

        fused_layer = CoexistenceLayer(
            layer_id=fused_layer_id,
            layer_name=f"FusedLayer({layer_a.layer_name}, {layer_b.layer_name})",
            topology=fused_topo,
            depth=max(layer_a.depth, layer_b.depth)
        )

        self.coexistence_map.layers[fused_layer_id] = fused_layer
        return fused_layer

    def process_stimulus(
        self,
        incoming_topology: InformationTopology,
        context_vector: np.ndarray
    ) -> Dict[str, object]:
        """
        유입되는 외부 세계 자극 위상체와 내부 공존 지도의 대조,
        공존 불가능 경계선 추출 및 위상적 분기/융합 자율 수행.
        """
        primary_layer = self.coexistence_map.layers[self.coexistence_map.primary_layer_id]
        self_topo = primary_layer.topology

        mitotic_layers_created = []
        fused_layers_created = []
        boundaries_detected = []

        for inc_id, inc_sym in incoming_topology.symbols.items():
            inc_vec = inc_sym.get_cross_dimensional_signature()

            for self_id, self_sym in self_topo.symbols.items():
                self_vec = self_sym.get_cross_dimensional_signature()

                boundary = self.detect_exclusivity_boundary(
                    node_a_id=self_id,
                    node_b_id=inc_id,
                    node_a_vector=self_vec,
                    node_b_vector=inc_vec,
                    context_vector=context_vector
                )
                boundaries_detected.append(boundary)

                if boundary.is_mutually_exclusive:
                    new_layer = self.perform_topological_mitosis(
                        source_layer_id=self.coexistence_map.primary_layer_id,
                        exclusive_node_id=inc_id,
                        boundary=boundary
                    )
                    mitotic_layers_created.append(new_layer)
                elif boundary.friction_magnitude <= self.fusion_threshold:
                    fusion_layer = self.perform_topological_fusion(
                        layer_a_id=self.coexistence_map.primary_layer_id,
                        layer_b_id=self.coexistence_map.primary_layer_id,
                        coexistent_nodes=[(self_id, inc_id)]
                    )
                    fused_layers_created.append(fusion_layer)

        return {
            "boundaries_detected_count": len(boundaries_detected),
            "mitotic_layers_count": len(mitotic_layers_created),
            "fused_layers_count": len(fused_layers_created),
            "boundaries": boundaries_detected,
            "total_coexistence_layers": len(self.coexistence_map.layers)
        }
