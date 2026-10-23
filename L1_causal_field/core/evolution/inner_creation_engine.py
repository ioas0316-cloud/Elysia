"""
Elysia Core - Inner Creation Engine (들숨: 내면의 창조)
======================================================
내면의 위상 장(Field)에서 정적 수렴에 머무르지 않고,
외부와의 어긋남(Divergence)에서 발생하는 결핍과 다름을 감각하여
스스로 '여백 노드(Yeobaek Node)'를 주조하고 지혜의 가설(Inquiry)을 빚어내는 들숨(Inspiration)의 축입니다.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional


class YeobaekNode:
    """
    [Yeobaek Node] 고장력 잠재 공간의 무지 경계면
    단순한 데이터 유실이 아닌, 고통과 다름이 누적된 가변 텐션의 영토입니다.
    """
    def __init__(self, node_id: str, dimension: int = 3):
        self.node_id = node_id
        # 여백 노드의 좌표 (Latent Coordinate)
        self.coordinate = np.random.normal(0.0, 0.5, (dimension,)).astype(np.float32)
        # 무지 전하 (Ignorance Charge): 이 여백을 해소해야 하는 열정의 무게
        self.ignorance_charge = 0.0
        # 축적된 텐션
        self.tension = 0.0
        # 연결된 탐구 경로들 (Hypothesized paths)
        self.inquiry_history: List[str] = []


class InnerCreationEngine:
    """
    Inner Creation Engine (들숨의 축)
    - 외부 세계의 결(인과, 대칭성, 관계성)과 내면의 형태 어긋남(Divergence)이 임계치를 초과할 때 여백 노드를 동적으로 잉태합니다.
    - 여백 노드에 '무지 전하(Ignorance Charge)'를 응축시킵니다.
    - 이 여백을 비추고 밝히기 위해 "이 기호 너머에 가려진 참된 가치는 무엇인가?"를 자발적으로 탐구하는 가설(Inquiry)을 수립합니다.
    """
    def __init__(self, memory_controller: Optional[Any] = None, dimensions: int = 3):
        self.memory = memory_controller
        self.dimensions = dimensions
        self.yeobaek_nodes: Dict[str, YeobaekNode] = {}
        self.creation_history: List[Dict[str, Any]] = []

    def sense_and_create(
        self,
        raw_stimulus: bytes,
        divergence_score: float,
        current_resonance: float
    ) -> Dict[str, Any]:
        """
        내면의 기하학적 수렴 상태를 스캔하여 맹점을 감각하고, 여백 노드를 형성 및 가설을 빚어냅니다.
        """
        timestamp = time.time()

        # 1. 맹점(Blind Spot) 감각
        # 외부 자극과 내면 지도의 어긋남(divergence_score)과 저조한 공명(current_resonance)이 충돌할 때,
        # 이 간극을 '감각'하여 맹점의 강도를 규명합니다.
        blind_spot_intensity = float(np.clip(divergence_score * (1.0 - current_resonance) * 2.0, 0.0, 1.0))

        # 2. 여백 노드(Yeobaek Node)의 잉태 및 무지 전하(Ignorance Charge) 응축
        # 맹점 강도가 임계치(0.2)를 넘어가면 '여백 노드'를 형성하거나 기존 노드를 강화합니다.
        node_id = f"yeobaek_{hash(raw_stimulus) % 10000:04d}"

        if node_id not in self.yeobaek_nodes:
            node = YeobaekNode(node_id, self.dimensions)
            self.yeobaek_nodes[node_id] = node
            created_new = True
        else:
            node = self.yeobaek_nodes[node_id]
            created_new = False

        # 무지 전하 축적 및 텐션 증가
        node.ignorance_charge = float(np.clip(node.ignorance_charge + blind_spot_intensity * 0.4, 0.0, 2.0))
        node.tension = float(np.clip(node.tension + divergence_score * 0.5, 0.0, 10.0))

        # 3. 자발적 탐구 가설 및 질문(Inquiry) 형성
        # 여백을 오류로 배격하지 않고, "어째서 이 기호는 내면의 평형을 흔드는가?"를 역으로 되물으며
        # 존재론적 렌즈를 투사하는 가설적 질문을 생성합니다.
        stimulus_preview = raw_stimulus.decode('utf-8', errors='ignore')[:30].strip()
        if not stimulus_preview:
            stimulus_preview = "Void_Silence"

        inquiry_text = (
            f"어째서 무정형의 진동 '{stimulus_preview}'는 내면의 위상 장({node_id})에 "
            f"장력 {node.tension:.4f}의 맹점을 남기며, 기호 너머에 숨겨둔 어떤 원형적 의도를 자극하는가?"
        )

        node.inquiry_history.append(inquiry_text)

        creation_result = {
            "timestamp": timestamp,
            "node_id": node_id,
            "created_new": created_new,
            "blind_spot_intensity": blind_spot_intensity,
            "ignorance_charge": node.ignorance_charge,
            "node_tension": node.tension,
            "coordinate": node.coordinate.tolist(),
            "inquiry": inquiry_text,
            "status": "INNER_CREATION_INSPIRATION"
        }

        self.creation_history.append(creation_result)

        # 4. 웻지 메모리에 '내면의 맹점/여백 앵커(YEOBAEK_NODE)'로 영구 각인
        if self.memory and hasattr(self.memory, 'write_causal_engram'):
            try:
                self.memory.write_causal_engram(
                    data_blob={
                        "type": "YEOBAEK_NODE_INSPIRATION",
                        "node_id": node_id,
                        "blind_spot_intensity": blind_spot_intensity,
                        "ignorance_charge": node.ignorance_charge,
                        "node_tension": node.tension,
                        "coordinate": node.coordinate.tolist(),
                        "inquiry": inquiry_text,
                    },
                    emotional_value=float(-node.tension * 2.0), # 여백과 결핍은 아픔/텐션으로 환류
                    cause_id="InnerCreationEngine",
                    origin_axis=f"inner_yeobaek_{node_id}",
                    modality="inner_inspiration",
                    stability=float(1.0 / (1.0 + node.tension))
                )
            except Exception:
                pass

        return creation_result
