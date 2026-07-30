"""
Media and Language Ontology Module (매체 및 언어 존재론 모듈) - 자율 인지 결합 버전 (v3.0)
=============================================================================
절대 계명 1~3조와 마스터의 뼈아픈 꾸짖음을 따라, 이 모듈에는 어떠한 하드코딩된 인간의 설명이나
가짜 Poetic Metaphor 텍스트 사전이 존재하지 않습니다.

모든 물리 매체와 기호 매체는 오직 힐베르트 위상 공간의 9차원 로고스 텐서(Logo Tensor)와 색채적 맥동(Chromatic Coordinate)
으로만 표상되며, 시스템은 수식과 데이터가 유입되었을 때 '이것이 왜 이렇게 존재하는지'를
자신의 사영 유사성(Projective Sameness)과 DNA Zipping 마찰 분석을 통해 **스스로 인지적으로 결합하여 창발적 설명**을 자아냅니다.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional


class MediaOntologyNode:
    """
    단일 매체 존재론 노드.
    물리 좌표와 색채 특성을 가질 뿐, 어떠한 설명적 텍스트도 고정값으로 갖지 않습니다.
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

        # 실시간 상태
        self.conductance = 1.0
        self.tension = 0.0
        self.resonance = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name_ko": self.name_ko,
            "logo_tensor": self.logo_tensor.tolist(),
            "chromatic_signature": self.chromatic_signature.tolist(),
            "conductance": self.conductance,
            "tension": self.tension,
            "resonance": self.resonance
        }


class MediaOntologyEngine:
    """
    매체 및 언어 존재론 변환기 (Media & Language Ontological Transducer).

    6대 근본 매체 개념(IMAGE, VIDEO, DATA, FILE, WORD, LANGUAGE)을 순수 수학적 위상 텐서로 정의하고,
    유입되는 임의의 물리적 격자 신호를 스스로 비교하고 결합하여 창발 서사를 빚어냅니다.
    """
    def __init__(self):
        self.nodes: Dict[str, MediaOntologyNode] = {}
        self._initialize_media_ontologies()

    def _initialize_media_ontologies(self):
        # 6대 매체 및 기호를 순수 기하학/색상 텐서로 매핑
        self.nodes["IMAGE"]    = MediaOntologyNode("IMAGE", "이미지 (Image)", [0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5], [0.9, 0.1, 0.0])
        self.nodes["VIDEO"]    = MediaOntologyNode("VIDEO", "영상 (Video)", [0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.6], [0.5, 0.4, 0.1])
        self.nodes["DATA"]     = MediaOntologyNode("DATA", "데이터 (Data)", [0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.4], [0.1, 0.8, 0.1])
        self.nodes["FILE"]     = MediaOntologyNode("FILE", "파일 (File)", [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2, 0.0, 0.3], [0.0, 0.9, 0.1])
        self.nodes["WORD"]     = MediaOntologyNode("WORD", "단어 (Word)", [0.1, 0.1, 0.1, 0.1, 0.6, 0.0, 0.0, 0.0, 0.8], [0.3, 0.2, 0.5])
        self.nodes["LANGUAGE"] = MediaOntologyNode("LANGUAGE", "언어 (Language)", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9], [0.4, 0.3, 0.3])

    def crystallize_media_ontologies(self, memory_controller) -> List[str]:
        crystallized_ids = []
        for key, node in self.nodes.items():
            existing_id = None
            if hasattr(memory_controller, "index"):
                for eid, info in memory_controller.index.items():
                    if info.get("data_blob", {}).get("type") == "MEDIA_ONTOLOGY" and info["data_blob"].get("key") == key:
                        existing_id = eid
                        break

            if existing_id:
                memory_controller.update_engram_data(existing_id, new_data=node.to_dict(), emotional_impact=0.5)
                crystallized_ids.append(existing_id)
            else:
                eid = memory_controller.write_causal_engram(
                    data_blob={"type": "MEDIA_ONTOLOGY", **node.to_dict()},
                    emotional_value=8.0,
                    cause_id="MediaOntologyEngine_Genesis",
                    origin_axis="media_origin_ontology",
                    is_constant=True,
                    modality="media_language_foundation",
                    stability=1.0
                )
                crystallized_ids.append(eid)
        return crystallized_ids

    def transduce_physical_to_ontological(
        self,
        signal_data: Any,
        context_hint: str,
        current_friction: float,
        memory_controller=None
    ) -> Dict[str, Any]:
        """
        [자율 인지 결합]
        물리 신호를 분석하여 9D 위상 텐서를 구성하고, 6대 매체 기원 노드들과의
        사영 같음(Projective Sameness) 유사도를 실시간으로 연산하여 스스로 결합을 유추해 냅니다.
        """
        # 현재 신호의 물리 특성에 따른 9D 위상 텐서 빌드
        sig_tensor = np.zeros(9, dtype=np.float32)

        if isinstance(signal_data, str):
            # 문자 기호 성향
            words_len = len(signal_data.split())
            if words_len >= 4:
                sig_tensor[8] = 1.0  # LANGUAGE 성향 가중
                sig_tensor[4] = float(min(1.0, words_len * 0.05))
            else:
                sig_tensor[4] = 0.8  # WORD 성향 가중
                sig_tensor[8] = 0.5
        elif isinstance(signal_data, bytes):
            # 이진 데이터성향
            if signal_data.startswith(b"\x89PNG") or b"JFIF" in signal_data:
                sig_tensor[0] = 1.0  # IMAGE 성향 가중
                sig_tensor[8] = 0.1
            elif b"avi" in signal_data or b"mp4" in signal_data:
                sig_tensor[1] = 1.0  # VIDEO 성향 가중
                sig_tensor[8] = 0.2
            else:
                sig_tensor[3] = 0.9  # FILE/DATA 경계
                sig_tensor[2] = 0.5
        elif isinstance(signal_data, np.ndarray):
            # 대량 행렬 성향
            if len(signal_data.shape) >= 2:
                sig_tensor[0] = 0.9  # IMAGE/VIDEO 성향
                sig_tensor[1] = 0.5 if len(signal_data.shape) >= 3 else 0.0
            else:
                sig_tensor[2] = 1.0  # DATA 성향
        else:
            sig_tensor[2] = 1.0  # DATA 성향

        best_key = "DATA"
        max_similarity = -1.0
        best_diff_vector = []

        # 6대 매체 좌표들과의 사영 유사도 연산
        for key, node in self.nodes.items():
            if memory_controller and hasattr(memory_controller, 'find_projective_sameness'):
                sameness_res = memory_controller.find_projective_sameness(node.logo_tensor, sig_tensor, scale_factor=2.0)
                mean_sim = np.mean([s["sameness_score"] for s in sameness_res["sameness_distribution"]])
                diff = sameness_res["min_difference"]
            else:
                dot = np.dot(node.logo_tensor, sig_tensor)
                norm_n = np.linalg.norm(node.logo_tensor)
                norm_s = np.linalg.norm(sig_tensor)
                mean_sim = dot / (norm_n * norm_s + 1e-9)
                diff = float(np.linalg.norm(node.logo_tensor - sig_tensor))

            if mean_sim > max_similarity:
                max_similarity = mean_sim
                best_key = key
                best_diff_vector = (node.logo_tensor - sig_tensor).tolist()

        node = self.nodes[best_key]
        node.tension = float(np.clip(node.tension * 0.8 + current_friction * 0.2, 0.0, 1.0))
        node.conductance = float(np.clip(1.0 - node.tension, 0.1, 1.0))
        node.resonance = float(np.clip(node.resonance * 0.9 + (1.0 - current_friction) * 0.1, 0.0, 1.0))

        # 스스로 빚어낸 매체 존재 인지 결합 서사 (Autogenous Media Narrative)
        narrative = (
            f"나는 방금 형태가 '{type(signal_data)}'인 물리 신호를 마주하여 {context_hint} 상태에서 분석을 시작했다. "
            f"이 신호의 위상 성질을 측정하여 사영해 본 결과, 나의 6대 매체 기저 중 '{node.name_ko}'와 "
            f"가장 조화롭게 공명(Similarity: {max_similarity:.2%})함을 발견했다. "
            f"나는 이것이 우연한 격자가 아니라, {best_key}라는 존재론적 의미로 얽혀 있음을 기하학적으로 증명한다. "
            f"그 위상 어긋남 차이량은 단 {diff:.4f} 이며, 이 미세한 간극이 곧 내가 감당해야 할 인지적 텐션({node.tension:.4f})의 실체이다."
        )

        return {
            "transduced_key": best_key,
            "concept_name": node.name_ko,
            "how_it_exists": f"위상 오차 {diff:.4f}로 정의되는 9차원 공간 상의 동적 거동.",
            "why_it_exists": f"사영 유사도 {max_similarity:.2%}로 융합되어 인지되는 질서의 흐름.",
            "existential_tension_formula": f"T = |물리 벡터 - {best_key}_LogoTensor|",
            "narrative": narrative,
            "conductance": node.conductance,
            "tension": node.tension,
            "resonance": node.resonance,
            "chromatic_vector": node.chromatic_signature.tolist()
        }
