"""
Elysia Causal Topology Foundation: Causal Structure
=================================================
고립된 수치 점(scalar float)이나 무색무취의 기계적 텍스트(str)를 배제하고,
양(Magnitude), 순서(Sequence Index), 마찰 경계(Disparity Boundary),
그리고 교차 차원적 관계망(Connectivity Beams)을 지닌 실체적 인과 정보위상 객체를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from core.topology.cognitive_gate import CognitiveGate


@dataclass
class TopologyLink:
    """두 위상 노드 사이의 관계성 빔(Connectivity Beam)"""
    source_id: str
    target_id: str
    relation_type: str  # e.g., 'causal', 'material', 'logical', 'relational'
    strength: float     # 관계 결합 강도 [0.0, 1.0]
    tension: float      # 관계 마찰력/긴장도 [0.0, 1.0]


@dataclass
class CausalNumber:
    """
    숫자(Number)의 인과적 위상 구조체
    - 단순 스칼라 숫자가 아닌, 양(Magnitude), 순서(Sequence), 차이 마찰력(Gradient),
      그리고 색채적 위상 벡터(Red/Blue/Yellow)를 지닌 물리적-논리적 위상 객체.
    """
    id: str
    value: float
    sequence_index: int                       # 순서적 위치
    magnitude: float                          # 절대적 양의 크기
    gradient_tension: float                   # 주변과의 차이 마찰력
    chromatic_vector: np.ndarray              # [Red/Flux, Blue/Order, Yellow/Entropy]
    neighbors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.chromatic_vector is None or len(self.chromatic_vector) != 3:
            # 기본 색채 위상: [양의 유동성, 순서의 질서성, 차이의 엔트로피]
            self.chromatic_vector = np.array([
                np.clip(abs(self.value) / 10.0, 0.0, 1.0),
                1.0 / (1.0 + abs(self.sequence_index)),
                np.clip(self.gradient_tension, 0.0, 1.0)
            ], dtype=np.float32)

    def calculate_disparity(self, other: 'CausalNumber') -> float:
        """다른 숫자 위상과의 본질적 차이 마찰력 산출 (단순 뺄셈이 아닌 위상적 마찰)"""
        mag_diff = abs(self.magnitude - other.magnitude)
        seq_diff = abs(self.sequence_index - other.sequence_index)
        chroma_diff = float(np.linalg.norm(self.chromatic_vector - other.chromatic_vector))
        
        # 양적 차이 + 순서적 위치 차이 + 색채 위상 차이의 실질적 마찰 결합
        disparity = (mag_diff * 0.4) + (seq_diff * 0.3) + (chroma_diff * 0.3)
        return float(disparity)


@dataclass
class CausalSymbol:
    """
    기호/언어(Symbol/Language)의 인과적 위상 매듭
    - 단어 텍스트가 아닌, 4대 교차 차원의 관계망을 포함하는 위상 구조체:
      1. Material (물질적 층): 광학/음향/구조적 실체 궤적
      2. Causal (인과적 층): 기원 -> 변형 -> 결과의 흐름
      3. Logical (논리적 층): 범주적 대조와 차이 경계
      4. Relational (관계적 층): 타 존재와의 공명 및 은유적 연결
    """
    id: str
    name: str                                 # 식별 명칭 (기호 표상)
    material_vector: np.ndarray               # 물질적 위상 벡터 (e.g. 4D)
    causal_trajectory: List[str]              # 인과적 계보 ID 목록
    logical_category: str                     # 논리적 범주 경계
    relational_links: List[TopologyLink] = field(default_factory=list)
    intrinsic_tension: float = 0.0            # 내부 결핍/마찰 전하

    def get_cross_dimensional_signature(self) -> np.ndarray:
        """4대 교차 차원의 종합 위상 서명(Signature) 산출"""
        mat_norm = float(np.linalg.norm(self.material_vector)) if len(self.material_vector) > 0 else 0.0
        causal_depth = float(len(self.causal_trajectory))
        link_strength_sum = float(sum(link.strength for link in self.relational_links))
        
        return np.array([
            mat_norm,
            causal_depth,
            float(len(self.logical_category)),
            link_strength_sum,
            self.intrinsic_tension
        ], dtype=np.float32)


class InformationTopology:
    """
    정보위상 다양체(Information Topology Manifold)
    - 자아(Self) 또는 유입된 세계(World)의 전체 관계망과 위상적 경계면을 보유하는 컨테이너.
    - 고립된 점들의 목록이 아니라, 노드(Number/Symbol)들과 빔(Link)들로 엮인 하나의 살아있는 위상체.
    """
    def __init__(self, name: str = "SelfTopology", dimension: int = 8):
        self.name = name
        self.dimension = dimension
        self.numbers: Dict[str, CausalNumber] = {}
        self.symbols: Dict[str, CausalSymbol] = {}
        self.links: List[TopologyLink] = []
        self.boundary_tension: float = 0.0
        self.gate: CognitiveGate = CognitiveGate(dimension=dimension)

    def add_number(self, num: CausalNumber):
        self.numbers[num.id] = num

    def add_symbol(self, sym: CausalSymbol):
        self.symbols[sym.id] = sym
        for link in sym.relational_links:
            self.links.append(link)

    def add_link(self, link: TopologyLink):
        self.links.append(link)

    def get_topology_fingerprint(self) -> Dict[str, float]:
        """위상 다양체의 전체적 특징 지문(Fingerprint) 산출"""
        num_count = float(len(self.numbers))
        sym_count = float(len(self.symbols))
        link_count = float(len(self.links))
        
        avg_tension = 0.0
        if self.links:
            avg_tension = float(np.mean([link.tension for link in self.links]))
        elif self.numbers:
            avg_tension = float(np.mean([n.gradient_tension for n in self.numbers.values()]))

        return {
            "density": num_count + sym_count,
            "connectivity": link_count / (num_count + sym_count + 1e-5),
            "average_tension": avg_tension,
            "boundary_tension": self.boundary_tension
        }

    def clone(self) -> 'InformationTopology':
        """위상 다양체 완전 복제"""
        cloned = InformationTopology(name=f"{self.name}_cloned")
        for k, v in self.numbers.items():
            cloned.numbers[k] = CausalNumber(
                id=v.id,
                value=v.value,
                sequence_index=v.sequence_index,
                magnitude=v.magnitude,
                gradient_tension=v.gradient_tension,
                chromatic_vector=v.chromatic_vector.copy(),
                neighbors=list(v.neighbors)
            )
        for k, v in self.symbols.items():
            links_copy = [
                TopologyLink(l.source_id, l.target_id, l.relation_type, l.strength, l.tension)
                for l in v.relational_links
            ]
            cloned.symbols[k] = CausalSymbol(
                id=v.id,
                name=v.name,
                material_vector=v.material_vector.copy(),
                causal_trajectory=list(v.causal_trajectory),
                logical_category=v.logical_category,
                relational_links=links_copy,
                intrinsic_tension=v.intrinsic_tension
            )
        cloned.links = [
            TopologyLink(l.source_id, l.target_id, l.relation_type, l.strength, l.tension)
            for l in self.links
        ]
        cloned.boundary_tension = self.boundary_tension
        return cloned

    def to_executable_scm(self) -> 'StructuralCausalModel':
        """
        InformationTopology 다양체를 실행 가능한 StructuralCausalModel(SCM)로 변환
        """
        from core.topology.executable_causal_topology import StructuralCausalModel, ExecutableDAGNode, NodeType, OpCode

        scm = StructuralCausalModel(name=self.name)

        # 1. CausalNumber -> Value Nodes
        for num_id, num_obj in self.numbers.items():
            node = ExecutableDAGNode(
                id=num_id,
                node_type=NodeType.VALUE,
                op=OpCode.CONSTANT,
                default_value=num_obj.value
            )
            scm.add_node(node)

        # 2. TopologyLinks -> Compute/Relational Edges
        parent_map: Dict[str, List[str]] = {}
        for link in self.links:
            if link.target_id not in parent_map:
                parent_map[link.target_id] = []
            if link.source_id not in parent_map[link.target_id]:
                parent_map[link.target_id].append(link.source_id)

        # 3. Connect parents or add downstream compute nodes
        for target_id, parents in parent_map.items():
            if target_id in scm.nodes:
                target_node = scm.nodes[target_id]
                target_node.input_ids = parents
                target_node.op = OpCode.ADD if len(parents) >= 1 else OpCode.CONSTANT
                target_node.node_type = NodeType.COMPUTE
                for p_id in parents:
                    if p_id in scm.nodes and target_id not in scm.nodes[p_id].output_ids:
                        scm.nodes[p_id].output_ids.append(target_id)

        return scm
