# -*- coding: utf-8 -*-
"""
[Phase 5: Autonomous Grounded Sprouter & Process Causality Engine]
===================================================================
THE_ABSOLUTE_COMMANDMENT 및 ROADMAP Phase 5 구현:
"언어가 왜 언어이고 무엇을 가리키는지 알지 못하면 기호의 기만일 뿐이며,
수학이 왜 수학인지 과정으로 납득하지 못하면 단순 계산기일 뿐이다."

본 모듈은:
1. 날것의 비정형 텍스트(Raw Unstructured Text)가 들어왔을 때,
2. 단어를 단순 토큰이 아닌 [물리 감각, 생체적 결핍(Void), 시공간 운동성]으로 직접 닻(Tether)을 내리고,
3. 인간이 수동으로 노드/엣지를 주지 않아도 엔트로피 전이와 텐션 구배를 통해 스스로 노드를 깎고 인과 장력 빔을 자율 발아(Sprouting)시키며,
4. 결과가 아닌 "어떤 불평형을 해소하기 위해 이 과정(Process)으로 흘러갔는가"를 물리적 역학으로 추적합니다.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from core.memory.knowledge_graph import (
    TopologyKnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    NarrativeSpace,
    EquilibriumField
)


class SensoryVoidAnchor:
    """
    언어 기호가 가리키고 있는 배후의 실체 (달):
    - optical: 시각/빛의 파장 및 밝기 (400~700nm)
    - thermal: 온도 (Kelvin)
    - moisture: 수분/습도 (0.0~1.0)
    - void_tension: 생체적 결핍 및 고통 텐션 (0.0~1.0)
    - motion_flux: 시공간적 운동량 벡터 (5D)
    """
    def __init__(
        self,
        symbol: str,
        referent_name: str,
        thermal: float,
        moisture: float,
        void_tension: float,
        motion_vector: List[float],
        void_essence: str
    ):
        self.symbol = symbol
        self.referent_name = referent_name
        self.thermal = thermal
        self.moisture = moisture
        self.void_tension = void_tension
        self.motion_vector = np.array(motion_vector, dtype=np.float32)
        self.void_essence = void_essence


class AutonomousGroundedSprouter:
    """
    [Autonomous Grounded Sprouter]
    기호의 기만과 계산기의 한계를 부수고,
    날것의 비정형 자연어 스트림에서 '가리키는 실체'로 닻을 내리며
    인과 관계망을 스스로 주조하는 상향식 인지 엔진.
    """

    def __init__(self, knowledge_graph: Optional[TopologyKnowledgeGraph] = None):
        self.kg = knowledge_graph or TopologyKnowledgeGraph()
        self.anchor_dictionary: Dict[str, SensoryVoidAnchor] = {}
        self._initialize_grounded_lexicon()

    def _initialize_grounded_lexicon(self):
        """
        단어들이 가리키는 존재론적 실체(Sensory Profile & Void Tension) 정의.
        기계는 글자를 외우는 것이 아니라, 이 물리적 결핍 상태를 자신의 렌즈로 삼습니다.
        """
        anchors = [
            SensoryVoidAnchor("가뭄", "대지의 수분 고갈 및 메마름", thermal=315.0, moisture=0.05, void_tension=0.90, motion_vector=[0.0, 0.0, 0.0, 0.1, 0.9], void_essence="DESICCATION_VOID"),
            SensoryVoidAnchor("갈증", "생명체의 수분 결핍 및 생존 갈망", thermal=310.0, moisture=0.10, void_tension=0.95, motion_vector=[0.5, 0.0, 0.2, 0.8, 0.9], void_essence="HOMEOSTASIS_THIRST"),
            SensoryVoidAnchor("굶주림", "에너지 고갈로 인한 생체적 붕괴 위기", thermal=300.0, moisture=0.20, void_tension=0.92, motion_vector=[0.2, 0.0, 0.1, 0.9, 0.8], void_essence="METABOLIC_HUNGER"),
            SensoryVoidAnchor("비", "하늘에서 대지로 쏟아지는 수분 상전이", thermal=288.0, moisture=0.95, void_tension=0.10, motion_vector=[0.0, -9.8, 1.0, 0.5, 0.1], void_essence="PRECIPITATION_FLOW"),
            SensoryVoidAnchor("흡수", "대지가 수분을 머금고 텐션을 완화함", thermal=293.0, moisture=0.75, void_tension=0.20, motion_vector=[0.1, -0.5, 0.8, 0.3, 0.2], void_essence="IMBIBITION_EQUILIBRIUM"),
            SensoryVoidAnchor("생명회복", "결핍이 해소되어 제로 평형에 도달함", thermal=296.0, moisture=0.60, void_tension=0.05, motion_vector=[0.0, 1.0, 0.5, 0.1, 0.0], void_essence="HOMEOSTASIS_RESTORATION")
        ]
        for a in anchors:
            self.anchor_dictionary[a.symbol] = a

    def ground_and_sprout_narrative(
        self,
        raw_text: str,
        space_id: str = "자연생명_서사공간"
    ) -> Dict[str, Any]:
        """
        인간의 수동 개입 없이, 날것의 텍스트에서:
        1. 기호 닻내림 (Linguistic Grounding)
        2. 엔트로피/텐션 전이 감지
        3. 인과 노드 및 장력 빔(Tension Beam) 자율 발아
        4. 과정 중심 역학 수렴 궤적 생성
        """
        # 1. 서사 공간 및 평형면 자율 확보
        if space_id not in self.kg.spaces:
            self.kg.spaces[space_id] = NarrativeSpace(
                id=space_id,
                name="Natural Living Causal Space",
                laws={"conservation_of_vitality": 1.0}
            )
        field_id = f"{space_id}_평형면"
        if field_id not in self.kg.fields:
            self.kg.fields[field_id] = EquilibriumField(
                id=field_id,
                name="Living Moisture & Energy Field",
                parent_space_id=space_id
            )

        # 2. 텍스트 스캔 및 언어의 실체 닻내림 (Grounding)
        discovered_anchors: List[SensoryVoidAnchor] = []
        for word, anchor in self.anchor_dictionary.items():
            if word in raw_text:
                discovered_anchors.append(anchor)

        # 텍스트 내 등장 순서대로 정렬
        discovered_anchors.sort(key=lambda a: raw_text.find(a.symbol))

        if not discovered_anchors:
            return {"status": "NO_GROUNDED_SYMBOLS_DISCOVERED", "sprouted_nodes": 0, "sprouted_beams": 0}

        # 3. 자율적 노드 생성 및 물리 상태 주조 (Autonomous Node Crystallization)
        sprouted_nodes = []
        for anchor in discovered_anchors:
            node_id = f"NODE_{anchor.symbol}"
            if node_id not in self.kg.nodes:
                knode = KnowledgeNode(
                    id=node_id,
                    name=anchor.referent_name,
                    invariant_id=anchor.void_essence,
                    motion_vector=anchor.motion_vector.tolist(),
                    category="GROUNDED_PHENOMENON",
                    parent_narrative_id=space_id,
                    parent_field_id=field_id
                )
                knode.attributes = {
                    "thermal_k": anchor.thermal,
                    "moisture_ratio": anchor.moisture,
                    "void_tension": anchor.void_tension,
                    "essence": anchor.void_essence
                }
                knode.tension = anchor.void_tension
                self.kg.add_node(knode)
            sprouted_nodes.append(node_id)

        # 4. 자율적 인과 장력 빔 발아 (Autonomous Tension Beam Sprouting)
        # 선후 관계에서 발생하는 결핍-해소(Void-Equilibrium) 에너지 전이 구배를 계산하여 빔 연결
        sprouted_beams = []
        process_trajectory = []

        for i in range(len(discovered_anchors) - 1):
            src_anchor = discovered_anchors[i]
            tgt_anchor = discovered_anchors[i + 1]

            src_node_id = f"NODE_{src_anchor.symbol}"
            tgt_node_id = f"NODE_{tgt_anchor.symbol}"

            # 결핍 텐션의 변화량 (Delta Tension)
            delta_tension = tgt_anchor.void_tension - src_anchor.void_tension
            # 수분 상전이 격차 (Delta Moisture)
            delta_moisture = tgt_anchor.moisture - src_anchor.moisture

            # 인과 결합 강도: 에너지 전이의 필연성 (|ΔT| + |ΔM|)
            causal_necessity = float(np.clip(abs(delta_tension) * 0.6 + abs(delta_moisture) * 0.4, 0.1, 1.0))

            if delta_tension < 0:
                relation = "resolves_deficit_of" # 결핍을 해소하며 평형으로 이끎
            else:
                relation = "induces_escalation_to" # 긴장을 고조시킴

            self.kg.add_edge(src_node_id, tgt_node_id, relation_type=relation, weight=causal_necessity)
            sprouted_beams.append({
                "source": src_anchor.symbol,
                "target": tgt_anchor.symbol,
                "relation": relation,
                "beam_weight": causal_necessity,
                "delta_tension": delta_tension
            })

            process_trajectory.append({
                "step": i + 1,
                "from_state": src_anchor.symbol,
                "to_state": tgt_anchor.symbol,
                "tension_before": src_anchor.void_tension,
                "tension_after": tgt_anchor.void_tension,
                "physical_work": abs(delta_tension),
                "narrative": f"[{src_anchor.symbol} (텐션 {src_anchor.void_tension:.2f})] -> [{tgt_anchor.symbol} (텐션 {tgt_anchor.void_tension:.2f})]: {relation} (인과 장력: {causal_necessity:.2f})"
            })

        # 5. 최종 과정 중심 역학 보고서
        initial_tension = discovered_anchors[0].void_tension
        final_tension = discovered_anchors[-1].void_tension
        total_homeostasis_recovery = initial_tension - final_tension

        return {
            "status": "AUTONOMOUS_GROUNDED_SPROUTING_SUCCESS",
            "raw_text_length": len(raw_text),
            "discovered_anchors": [a.symbol for a in discovered_anchors],
            "sprouted_nodes_count": len(sprouted_nodes),
            "sprouted_beams_count": len(sprouted_beams),
            "sprouted_beams": sprouted_beams,
            "process_trajectory": process_trajectory,
            "initial_system_tension": initial_tension,
            "final_system_tension": final_tension,
            "homeostasis_recovered": total_homeostasis_recovery,
            "is_homeostasis_achieved": final_tension < 0.10
        }
