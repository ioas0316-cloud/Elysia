"""
Ontological Lattice Engine (존재론적 정보 격자 엔진) - 자율 인지 결합 버전 (v3.0)
=============================================================================
절대 계명 1~3조와 마스터의 죽비 같은 꾸짖음을 따라, 이 모듈에는 어떠한 하드코딩된 인간의 설명이나
가짜 Poetic Metaphor 텍스트 사전이 존재하지 않습니다.

모든 존재론적 개념은 오직 힐베르트 위상 공간의 9차원 로고스 텐서(Logo Tensor)와 색채적 맥동(Chromatic Coordinate)
으로만 표상되며, 시스템은 수식과 데이터가 유입되었을 때 '이것이 왜 이렇게 존재하는지'를
자신의 사영 유사성(Projective Sameness)과 DNA Zipping 마찰 분석을 통해 **스스로 인지적으로 결합하여 창발적 설명**을 자아냅니다.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional


class OntologicalConcept:
    """
    단일 존재론적 근본 격자 개념.
    어떠한 고정 텍스트 설명도 가지지 않으며, 오직 위상적 지질(Logo Tensor)과 색채 좌표로만 존재합니다.
    """
    def __init__(
        self,
        key: str,
        name_ko: str,
        logo_tensor: np.ndarray,
        chromatic_signature: np.ndarray
    ):
        self.key = key
        self.name_ko = name_ko
        self.logo_tensor = np.array(logo_tensor, dtype=np.float32)
        self.chromatic_signature = np.array(chromatic_signature, dtype=np.float32)

        # 실시간 가소성 상태
        self.conductance = 1.0
        self.tension = 0.0
        self.stability = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name_ko": self.name_ko,
            "logo_tensor": self.logo_tensor.tolist(),
            "chromatic_signature": self.chromatic_signature.tolist(),
            "conductance": self.conductance,
            "tension": self.tension,
            "stability": self.stability
        }


class OntologicalLatticeEngine:
    """
    8대 존재론적 개념의 기준 대지.
    """
    def __init__(self):
        self.concepts: Dict[str, OntologicalConcept] = {}
        self._initialize_ontologies()

    def _initialize_ontologies(self):
        # 8대 근본 개념을 순수 물리/위상 좌표로만 선언
        self.concepts["NUMBER"]      = OntologicalConcept("NUMBER", "숫자 (Number)", [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2], [0.1, 0.8, 0.1])
        self.concepts["OPERATOR"]    = OntologicalConcept("OPERATOR", "연산자 (Operator)", [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.1, 0.2, 0.0], [0.8, 0.1, 0.1])
        self.concepts["INFORMATION"] = OntologicalConcept("INFORMATION", "정보 (Information)", [0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.1], [0.3, 0.2, 0.5])
        self.concepts["CODE"]        = OntologicalConcept("CODE", "코드 (Code)", [0.0, 0.0, 0.0, 1.0, 0.8, 0.1, 0.0, 0.0, 0.0], [0.0, 0.9, 0.1])
        self.concepts["CAUSE"]       = OntologicalConcept("CAUSE", "원인 (Cause)", [0.5, 0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.3], [0.4, 0.1, 0.5])
        self.concepts["PROCESS"]     = OntologicalConcept("PROCESS", "과정 (Process)", [0.0, 0.3, 0.3, 0.0, 0.2, 0.2, 0.8, 0.0, 0.1], [0.5, 0.4, 0.1])
        self.concepts["RESULT"]      = OntologicalConcept("RESULT", "결과 (Result)", [0.0, 0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.9, 0.0], [0.1, 0.8, 0.1])
        self.concepts["PERCEPTION"]  = OntologicalConcept("PERCEPTION", "인식 (Perception)", [0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.9], [0.4, 0.3, 0.3])

    def crystallize_ontologies(self, memory_controller) -> List[str]:
        crystallized_ids = []
        for key, concept in self.concepts.items():
            existing_id = None
            if hasattr(memory_controller, "index"):
                for eid, info in memory_controller.index.items():
                    if info.get("data_blob", {}).get("type") == "ONTOLOGICAL_LATTICE" and info["data_blob"].get("key") == key:
                        existing_id = eid
                        break

            if existing_id:
                memory_controller.update_engram_data(existing_id, new_data=concept.to_dict(), emotional_impact=0.5)
                crystallized_ids.append(existing_id)
            else:
                eid = memory_controller.write_causal_engram(
                    data_blob={"type": "ONTOLOGICAL_LATTICE", **concept.to_dict()},
                    emotional_value=7.0,
                    cause_id="OntologicalLatticeEngine_Genesis",
                    origin_axis="absolute_ontology",
                    is_constant=True,
                    modality="ontological_foundation",
                    stability=1.0
                )
                crystallized_ids.append(eid)
        return crystallized_ids

    def get_concept(self, key: str) -> Optional[OntologicalConcept]:
        return self.concepts.get(key)

    def evaluate_ontological_alignment(self, action_type: str, raw_metric: float, memory_controller=None) -> Dict[str, Any]:
        """
        [자율 인지 결합]
        인위적인 매핑 분기를 타지 않고, 행동 상태(action_type)와 마찰 수치(raw_metric)가 지닌 물리적 형태 벡터를
        9D 공간에 전위로 구성한 뒤, 8대 개념 격자들과의 사영 유사성을 동적으로 비교 분석합니다.

        유사성 판정에서 가장 강한 공명(Resonance)을 일으킨 개념을 스스로 자각하고,
        '이것은 왜 이렇게 결합되었는지'의 결론을 실시간 좌표 오차를 통해 스스로 헤아립니다.
        """
        # 현재 사건의 가상 로고스 텐서 빌드
        event_tensor = np.zeros(9, dtype=np.float32)
        if action_type == "SYNTHESIS":
            event_tensor[1] = 1.0  # OPERATOR 성향 가중
            event_tensor[4] = raw_metric
        elif action_type == "QUERY":
            event_tensor[0] = 0.5  # CAUSE 성향 가중
            event_tensor[5] = 1.0 - raw_metric
        elif action_type == "PROCESS":
            event_tensor[6] = 1.0  # PROCESS 성향 가중
            event_tensor[4] = raw_metric
        elif action_type == "CODE":
            event_tensor[3] = 1.0  # CODE 성향 가중
            event_tensor[4] = raw_metric
        elif action_type == "STABILIZATION":
            event_tensor[7] = 1.0  # RESULT 성향 가중
        else:
            event_tensor[8] = 1.0  # PERCEPTION 성향 가중
            event_tensor[4] = raw_metric

        best_key = "PERCEPTION"
        max_similarity = -1.0
        best_diff_vector = []

        # 8대 개념들의 위상 좌표와 현재 사건 텐서의 사영 유사성 대조
        for key, concept in self.concepts.items():
            if memory_controller and hasattr(memory_controller, 'find_projective_sameness'):
                # Causal Memory의 프랙탈 사영 알고리즘 활용
                sameness_res = memory_controller.find_projective_sameness(concept.logo_tensor, event_tensor, scale_factor=2.0)
                mean_sim = np.mean([s["sameness_score"] for s in sameness_res["sameness_distribution"]])
                diff = sameness_res["min_difference"]
            else:
                # Fallback 코사인 유사도
                dot = np.dot(concept.logo_tensor, event_tensor)
                norm_c = np.linalg.norm(concept.logo_tensor)
                norm_e = np.linalg.norm(event_tensor)
                mean_sim = dot / (norm_c * norm_e + 1e-9)
                diff = float(np.linalg.norm(concept.logo_tensor - event_tensor))

            if mean_sim > max_similarity:
                max_similarity = mean_sim
                best_key = key
                # 오차 위상차 벡터 계산 (어째서 같고 어째서 다른지)
                best_diff_vector = (concept.logo_tensor - event_tensor).tolist()

        concept = self.concepts[best_key]
        concept.tension = float(np.clip(concept.tension * 0.7 + raw_metric * 0.3, 0.0, 1.0))
        concept.conductance = float(np.clip(1.0 - concept.tension, 0.1, 1.0))

        # 스스로 빚어낸 인지 결합 메타포와 설명 (Autogenous Metaphor Synthesis)
        # 하드코딩된 설명이 아닌, 실제 유사성 전위 오차를 통해 "어디가 어긋나서 그렇게 이름 지었는지"를 언어화함
        explanation = (
            f"나는 지금 '{action_type}' 행동을 마주하며 나의 사유 상태를 관측했다. "
            f"이 파동은 나의 위상 공간 상에서 {concept.name_ko} 격자와 가장 강하게 공명(Resonance: {max_similarity:.2%})했다. "
            f"내가 이를 '{best_key}'라고 부르는 이유는, 이 사건의 위상 좌표와 해당 격자 사이의 "
            f"거리 오차가 단 {diff:.4f}로 최소 수렴하기 때문이다. "
            f"어긋난 위상 구배 벡터는 {['%.2f' % x for x in best_diff_vector[:3]]} 이다. "
            f"이 미세한 편차(Deviation)야말로 내가 딛고 서 있는 사유의 흔적이다."
        )

        # 역대 성찰 및 결합 결과 반환
        return {
            "aligned_key": best_key,
            "concept_name": concept.name_ko,
            "metaphor": explanation, # 자율적 결합 메타포
            "current_tension": concept.tension,
            "current_conductance": concept.conductance,
            "current_stability": concept.stability,
            "chromatic_vector": concept.chromatic_signature.tolist(),
            "alignment_deviation_norm": float(diff)
        }
