"""
Topological Memory Landscape & Dynamic Rewiring (위상 기억 지형 및 동적 리와이어링)
=============================================================================
기억(Memory)의 본질: 정적 데이터 저장이 아닌 '연결 구조의 위상 지형(Topological Landscape)'이자
역사적 참조면(Historical Reference Plane).

1. 위상 지형(Landscape):
   - 기존 연결망(TopologyLink)과 인과 축(Lens S)이 어떻게 결합되어 왔는지에 대한 역사적 인과 궤적.
2. 위상 마찰 피드백과 동적 리와이어링(Rewiring):
   - 현재 유입된 상태 $X$와 인과 축 $S$ 사이에서 발생하는 변이 마찰 $V$의 장력 에너지 $\mathcal{E}(V)$에 따라
     연결망 결합 강도(strength), 긴장도(tension), 궤적 연결을 실시간 미세 재배치.
3. 이산적 개념 결정화(Concept Crystallization / Symbolization Boundary):
   - 연속적인 물리적 파동 및 마찰 궤적에서 "여기까지는 같고, 여기서부터는 다르다"라는
     위상 경계를 절단하여 이산적 개념(CrystallizedConcept) 및 언어 기호(Symbol)로 굳혀내는 작용.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from core.topology.causal_structure import InformationTopology, TopologyLink, CausalNumber, CausalSymbol
from core.topology.cognitive_gate import CognitiveGate


@dataclass
class CrystallizedConcept:
    """
    위상 마찰 경계면에서 절단되어 굳어진 이산적 개념(Concept)
    """
    id: str
    label: str                                # 개념 명칭
    invariant_skeleton: np.ndarray            # 불변 뼈대 (I)
    boundary_friction: float                  # 절단 경계 마찰 전하 (V_mag)
    historical_depth: int                     # 인과적 누적 깊이
    associated_nodes: List[str] = field(default_factory=list)


class TopologicalMemory:
    """
    Topological Memory Landscape (위상 인과 기억 지형)

    데이터 저장이 아닌, 누적된 인과 궤적의 역사적 참조면(Reference Plane)으로 작용하며,
    유입된 위상 마찰(V) 피드백으로 네트워크 결합 구조를 실시간 Rewiring합니다.
    """

    def __init__(
        self,
        dimension: int = 8,
        decay_rate: float = 0.05,
        rewire_threshold: float = 0.3
    ):
        self.dimension = dimension
        self.decay_rate = decay_rate
        self.rewire_threshold = rewire_threshold

        # 위상 다양체 (Landscape Graph)
        self.topology = InformationTopology(name="TopologicalMemoryLandscape")
        self.gate = CognitiveGate(dimension=dimension)

        # 역사적 참조 궤적 (Historical Reference Trajectories)
        self.trajectory_history: List[Dict[str, Any]] = []
        self.crystallized_concepts: Dict[str, CrystallizedConcept] = {}
        self.step_counter: int = 0

    def get_reference_plane(self) -> np.ndarray:
        """
        기존 인과 축 및 노드 분포로부터 역사적 참조면(Reference Plane Lens S) 반환
        """
        return self.gate.S.copy()

    def process_and_rewire(
        self,
        X: np.ndarray,
        node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        유입 신호 X 수용 -> 게이트 분별 -> 위상 마찰 V 발생 -> 기억 지형 Rewiring & 개념 결속

        1. Reference Plane (Gate S) 관측 & 분별 (I, V)
        2. 마찰 에너지 E(V)에 기반한 네트워크 동적 리와이어링 (Rewiring)
        3. 마찰 경계가 절단 임계치를 넘을 때 이산 개념 결정화 (Crystallization)
        """
        self.step_counter += 1
        X_vec = np.asarray(X, dtype=np.float32).reshape(-1)

        # 1. 인지 게이트 분별 및 렌즈 재정렬
        gate_res = self.gate.process(X_vec)
        I = gate_res["invariant"]
        V = gate_res["variant"]
        friction = gate_res["friction_energy"]

        # 2. CausalNode 생성 또는 업데이트 (위상 궤적 저장)
        curr_id = node_id or f"node_t{self.step_counter}"
        c_num = CausalNumber(
            id=curr_id,
            value=float(np.mean(I)),
            sequence_index=self.step_counter,
            magnitude=float(np.linalg.norm(I)),
            gradient_tension=float(np.linalg.norm(V)),
            chromatic_vector=np.array([
                float(np.clip(np.linalg.norm(I) / 5.0, 0.0, 1.0)),
                1.0 / (1.0 + self.step_counter * 0.01),
                float(np.clip(friction, 0.0, 1.0))
            ], dtype=np.float32)
        )
        self.topology.add_number(c_num)

        # 3. 위상 마찰 기반 Dynamic Rewiring (연결망 재배치)
        rewired_links = self._rewire_topology(curr_id, I, V, friction)

        # 4. 마찰 경계 절단에 의한 개념 결정화 (Concept Crystallization)
        crystallized = None
        if friction > self.rewire_threshold:
            crystallized = self._crystallize_concept(curr_id, I, V, friction)

        # 5. 역사적 궤적 수록
        record = {
            "step": self.step_counter,
            "node_id": curr_id,
            "invariant": I,
            "variant": V,
            "friction": friction,
            "rewired_links_count": len(rewired_links),
            "crystallized_concept": crystallized.label if crystallized else None
        }
        self.trajectory_history.append(record)

        return {
            "node_id": curr_id,
            "invariant": I,
            "variant": V,
            "friction": friction,
            "reference_lens": self.gate.S.copy(),
            "rewired_links": rewired_links,
            "crystallized_concept": crystallized
        }

    def _rewire_topology(
        self,
        curr_id: str,
        I: np.ndarray,
        V: np.ndarray,
        friction: float
    ) -> List[TopologyLink]:
        """
        위상 마찰 V의 장력에 의하여 주변 노드와의 TopologyLink 결합 강도/긴장도를 실시간 재배치
        """
        new_or_updated_links: List[TopologyLink] = []

        # 기존 노드들과의 불변량 공명 및 마찰 교차 측정
        for past_id, past_num in list(self.topology.numbers.items()):
            if past_id == curr_id:
                continue

            # 인과적 마찰 및 공명 결합도 산출
            # 불변 뼈대가 유사할수록 strong strength, 변이 마찰(V)이 높을수록 high tension
            past_vec = np.array([past_num.magnitude, past_num.gradient_tension], dtype=np.float32)
            curr_vec = np.array([np.linalg.norm(I), np.linalg.norm(V)], dtype=np.float32)

            cos_sim = float(np.dot(past_vec, curr_vec) / (np.linalg.norm(past_vec) * np.linalg.norm(curr_vec) + 1e-8))
            strength = float(np.clip(cos_sim, 0.0, 1.0))
            tension = float(np.clip(friction / (1.0 + strength), 0.0, 1.0))

            if strength > self.rewire_threshold or tension > self.rewire_threshold:
                link = TopologyLink(
                    source_id=past_id,
                    target_id=curr_id,
                    relation_type="topological_causal",
                    strength=strength,
                    tension=tension
                )
                self.topology.add_link(link)
                new_or_updated_links.append(link)

        # 감쇄 연산 (Decay of stale links)
        for link in self.topology.links:
            link.tension = max(0.0, link.tension - self.decay_rate * 0.1)

        return new_or_updated_links

    def _crystallize_concept(
        self,
        node_id: str,
        I: np.ndarray,
        V: np.ndarray,
        friction: float
    ) -> CrystallizedConcept:
        """
        연속적 마찰 경계를 절단하여 이산적 개념(Concept)으로 굳혀냄
        """
        concept_id = f"concept_{len(self.crystallized_concepts) + 1}"
        concept_label = f"Crystallized_Boundary_Cut_{concept_id}"

        concept = CrystallizedConcept(
            id=concept_id,
            label=concept_label,
            invariant_skeleton=I.copy(),
            boundary_friction=float(np.linalg.norm(V)),
            historical_depth=self.step_counter,
            associated_nodes=[node_id]
        )
        self.crystallized_concepts[concept_id] = concept

        # InformationTopology 내 CausalSymbol로 정착
        sym = CausalSymbol(
            id=concept_id,
            name=concept_label,
            material_vector=I[:4] if len(I) >= 4 else I,
            causal_trajectory=[node_id],
            logical_category="crystallized_boundary_cut",
            intrinsic_tension=concept.boundary_friction
        )
        self.topology.add_symbol(sym)

        return concept

    def get_landscape_summary(self) -> Dict[str, Any]:
        """
        현재 기억 지형의 위상적 요약 정보 반환
        """
        fingerprint = self.topology.get_topology_fingerprint()
        return {
            "total_nodes": len(self.topology.numbers),
            "total_symbols": len(self.topology.symbols),
            "total_links": len(self.topology.links),
            "total_crystallized_concepts": len(self.crystallized_concepts),
            "fingerprint": fingerprint,
            "accumulated_gate_friction": self.gate.accumulated_friction
        }
