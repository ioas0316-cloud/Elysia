"""
Organismic Relational Continuum (유기적 관계적 연속체)
======================================================
단일 세계(Unified Relational World) 안에서 부분과 전체가 서로의 존재 조건, 정체성,
그리고 가용 결합 가능성(Relational Affordances)을 동시 변형시키는 최소 원리 모델입니다.

핵심 설계 원리:
1. 최상위 통제 모듈 배격: 단일 관제탑이 아닌, 상호 의존적인 관계적 노드(Relational Entities)들이
   동일한 인과 위상장(Causal Relational Field)을 공유하는 연속체.
2. 다층적 좋음(Multi-layered Goodness)의 복합적 상충과 개화:
   - 세포적 좋음 (Cellular: 항상성 / 생존)
   - 호르몬/오감적 좋음 (Sensory/Somatic: 자극 / 공명 / 유쾌)
   - 인지적 좋음 (Cognitive: 구조적 질서 / 미분 분별)
   - 정신/마음적 좋음 (Mental/Volitional: 의미 / 내어줌 / 자발적 사랑 $G_{\text{Love}}$)
   단순한 보상 스칼라나 지속가능성으로 환원하지 않고, 층위 간 지지/충돌을 통해 판단 근거가 재구성됨.
3. 비파이프라인 상호 변형 동역학 (Non-pipeline Co-dependent Dynamics):
   순차적 execution step 대신, A와 B의 결합이 A와 B 각각의 내적 상태, 정체성 텐서,
   미래 결합 가능성 매트릭스($\mathbf{A}_{\text{afford}}$)를 동시 굴절시키는 상태 전이 수식.
4. 상위 스케일 창발 및 하향식 조건 재설정 (Scale Co-emergence & Top-down Recalibration):
   국소적 관계 결합이 상위 스케일 필터/방향성을 창발시키고, 창발된 방향성이 다시 국소 결합 조건을 재정의.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class RelationalEntity:
    """
    관계적 연속체 내부에서 존재하는 관계적 단위 (Relational Unit).
    육체적, 감각적, 인지적, 정신적 층위의 좋음을 고유한 가치 프로필로 보유하며,
    타자와의 결합에 따라 자신의 정체성 및 가용 가능성이 유연하게 변화합니다.
    """

    def __init__(
        self,
        entity_id: str,
        layer_type: str, # "cellular", "sensory", "cognitive", "mental"
        dimension: int = 16,
        initial_goodness_profile: Optional[Dict[str, float]] = None,
        initial_identity_vector: Optional[np.ndarray] = None,
    ):
        self.entity_id = entity_id
        self.layer_type = layer_type
        self.dimension = dimension

        # 다층적 좋음 (Multi-layered Goodness Profile)
        # 세포적, 오감적, 인지적, 정신적 좋음의 가치 지형
        if initial_goodness_profile is None:
            self.goodness_profile = {
                "cellular": 0.5,
                "sensory": 0.5,
                "cognitive": 0.5,
                "mental": 0.5,
            }
        else:
            self.goodness_profile = dict(initial_goodness_profile)

        # 내적 정체성 및 위상 벡터 (Identity & Phase Vector)
        if initial_identity_vector is None:
            rng = np.random.default_rng(hash(entity_id) % (2**32))
            vec = rng.standard_normal(self.dimension)
            self.identity_vector = vec / (np.linalg.norm(vec) + 1e-9)
        else:
            norm = np.linalg.norm(initial_identity_vector)
            self.identity_vector = initial_identity_vector / (norm + 1e-9)

        # 내적 위상 각도 (Rotor Phase Angle)
        self.phase_angle = float(np.arctan2(self.identity_vector[1], self.identity_vector[0]))

        # 가용 결합 가능성 매트릭스 (Relational Affordance Vector - 다른 층위/개체와의 결합 가용성)
        self.affordance_capacity = np.ones(self.dimension, dtype=np.float64) * 0.5

        # 결합 이력 및 마찰 흉터 (Scar Tensor)
        self.scar_history = np.zeros(self.dimension, dtype=np.float64)

    def evaluate_goodness_conflict(self, external_stimulus: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        외부 자극 또는 타개체와의 접촉 시 4대 층위의 좋음 간 상충과 조화를 계측합니다.
        단순 0~1 평형이 아닌, 충돌 마찰(Conflict Friction)을 통해 판단 근거의 긴장을 형성합니다.
        """
        stim_norm = external_stimulus / (np.linalg.norm(external_stimulus) + 1e-9)
        cos_sim = float(np.dot(self.identity_vector, stim_norm))

        # 층위별 반응성
        c_good = self.goodness_profile["cellular"] * (1.0 + 0.2 * cos_sim)
        s_good = self.goodness_profile["sensory"] * (1.0 + 0.4 * (1.0 - abs(cos_sim))) # 신선한 감각적 자극
        cog_good = self.goodness_profile["cognitive"] * (0.5 + 0.5 * cos_sim) # 질서성
        m_good = self.goodness_profile["mental"] * (1.0 + 0.5 * sin_val if (sin_val := float(np.sin(cos_sim))) else 0.5)

        updated_profile = {
            "cellular": float(np.clip(c_good, 0.0, 1.0)),
            "sensory": float(np.clip(s_good, 0.0, 1.0)),
            "cognitive": float(np.clip(cog_good, 0.0, 1.0)),
            "mental": float(np.clip(m_good, 0.0, 1.0)),
        }

        # 층위 간 분산/마찰 (Goodness Variance = Friction)
        vals = list(updated_profile.values())
        goodness_conflict_friction = float(np.std(vals))

        return goodness_conflict_friction, updated_profile


class OrganismicRelationalContinuum:
    """
    [Organismic Relational Continuum: 단일 유기적 관계 세계]
    거대 통제 모듈이 아니며, 각 관계적 엔티티들이 서로의 존재 조건을 형성하고
    결합에 따라 정체성, 가용성, 상위 필터가 비선형적으로 동시 변화하는 관계의 장.
    """

    def __init__(self, dimension: int = 16):
        self.dimension = dimension

        # 단일 세계 안의 유기적 엔티티 집합
        self.entities: Dict[str, RelationalEntity] = {
            "cellular_soma": RelationalEntity("cellular_soma", "cellular", dimension, {"cellular": 0.9, "sensory": 0.3, "cognitive": 0.2, "mental": 0.1}),
            "sensory_organ": RelationalEntity("sensory_organ", "sensory", dimension, {"cellular": 0.4, "sensory": 0.9, "cognitive": 0.4, "mental": 0.2}),
            "cognitive_mind": RelationalEntity("cognitive_mind", "cognitive", dimension, {"cellular": 0.2, "sensory": 0.5, "cognitive": 0.9, "mental": 0.4}),
            "mental_spirit": RelationalEntity("mental_spirit", "mental", dimension, {"cellular": 0.1, "sensory": 0.3, "cognitive": 0.5, "mental": 0.9}),
        }

        # 상위 스케일 가치/방향성 필드 (Emergent Macro Directional Field)
        self.macro_orientation_vector = np.zeros(self.dimension, dtype=np.float64)
        self.macro_orientation_vector[0] = 1.0 # 초기 임의 스케일 축

        # 관계적 결합 역사적 매트릭스 (Coupling Matrix)
        self.coupling_matrix = np.eye(len(self.entities), dtype=np.float64) * 0.2

    def couple_entities_mutually(
        self,
        entity_id_a: str,
        entity_id_b: str,
        coupling_intensity: float = 0.5
    ) -> Dict[str, Any]:
        """
        [핵심 원리 1 & 3: 상호 변형 동역학 (Mutual Identity & Affordance Transformation)]
        A와 B가 결합할 때, 단순한 통로 연결이 아닌 A와 B 각자의:
        1. 내적 정체성 벡터 ($\mathbf{v}_A, \mathbf{v}_B$)가 서로를 향해 굴절.
        2. 가용 결합 가능성 ($\mathbf{A}_{\text{afford}}$)이 새롭게 열리거나 제한됨.
        3. 다층적 좋음의 프로필이 지지/충돌을 거쳐 재구성됨.
        """
        if entity_id_a not in self.entities or entity_id_b not in self.entities:
            raise ValueError(f"Entities {entity_id_a} or {entity_id_b} do not exist in continuum.")

        ea = self.entities[entity_id_a]
        eb = self.entities[entity_id_b]

        # 결합 전 상태 기록
        prev_a_vec = np.copy(ea.identity_vector)
        prev_b_vec = np.copy(eb.identity_vector)
        prev_a_aff = np.copy(ea.affordance_capacity)
        prev_b_aff = np.copy(eb.affordance_capacity)

        # 1. 내적 정체성 벡터의 상호 굴절 (Mutual Refraction)
        # A는 B의 정체성을 받아들이고, B는 A의 정체성을 받아들여 둘 다 변형됨
        coupling_alpha = float(np.clip(coupling_intensity * 0.3, 0.05, 0.45))

        new_a_vec = (1.0 - coupling_alpha) * ea.identity_vector + coupling_alpha * eb.identity_vector
        new_b_vec = (1.0 - coupling_alpha) * eb.identity_vector + coupling_alpha * ea.identity_vector

        ea.identity_vector = new_a_vec / (np.linalg.norm(new_a_vec) + 1e-9)
        eb.identity_vector = new_b_vec / (np.linalg.norm(new_b_vec) + 1e-9)

        # 2. 가용 결합 가능성 매트릭스의 재구조화 (Affordance Transformation)
        # 결합으로 인해 이전과 다른 새로운 형태의 위상적 결합 가능성이 창발
        shared_coupling_resonance = float(np.dot(ea.identity_vector, eb.identity_vector))
        novel_affordance_wave = np.sin(ea.identity_vector * eb.identity_vector * math.pi)

        ea.affordance_capacity = np.clip(ea.affordance_capacity + 0.2 * novel_affordance_wave * shared_coupling_resonance, 0.01, 1.0)
        eb.affordance_capacity = np.clip(eb.affordance_capacity + 0.2 * novel_affordance_wave * shared_coupling_resonance, 0.01, 1.0)

        # 3. 다층적 좋음의 상호 전이 및 충돌 (Multi-layered Goodness Mutual Influence)
        friction_a, new_prof_a = ea.evaluate_goodness_conflict(eb.identity_vector)
        friction_b, new_prof_b = eb.evaluate_goodness_conflict(ea.identity_vector)

        ea.goodness_profile = new_prof_a
        eb.goodness_profile = new_prof_b

        # 4. 상위 스케일 방향성 필드 창발 (Emergence of Macro Orientation Field)
        emergent_delta = (ea.identity_vector + eb.identity_vector) * (1.0 - abs(friction_a - friction_b))
        self.macro_orientation_vector = 0.8 * self.macro_orientation_vector + 0.2 * emergent_delta
        self.macro_orientation_vector /= (np.linalg.norm(self.macro_orientation_vector) + 1e-9)

        # 5. 상위 필드의 하향식 조건 재설정 (Top-Down Recalibration)
        # 창발된 상위 스케일 방향성이 다시 전체 엔티티들의 위상 각도와 반응성 조율
        for ent in self.entities.values():
            macro_alignment = float(np.dot(ent.identity_vector, self.macro_orientation_vector))
            ent.phase_angle += 0.1 * macro_alignment

        return {
            "entity_a": entity_id_a,
            "entity_b": entity_id_b,
            "shared_coupling_resonance": shared_coupling_resonance,
            "a_identity_shift_norm": float(np.linalg.norm(ea.identity_vector - prev_a_vec)),
            "b_identity_shift_norm": float(np.linalg.norm(eb.identity_vector - prev_b_vec)),
            "a_affordance_change_norm": float(np.linalg.norm(ea.affordance_capacity - prev_a_aff)),
            "b_affordance_change_norm": float(np.linalg.norm(eb.affordance_capacity - prev_b_aff)),
            "goodness_conflict_a": friction_a,
            "goodness_conflict_b": friction_b,
            "macro_orientation_updated": self.macro_orientation_vector.tolist(),
        }

    def interact_with_world_environment(
        self,
        environmental_stimulus: np.ndarray,
        intensity: float = 0.6
    ) -> Dict[str, Any]:
        r"""
        세계와의 상호작용: 외부 자극이 들어올 때, 특정 단일 모듈만 반응하는 것이 아니라
        단일 세계 전체의 무늬(Macro Orientation)와 엔티티들의 가치 판단 근구가 동시 굴절함.
        """
        stim_norm = environmental_stimulus / (np.linalg.norm(environmental_stimulus) + 1e-9)

        interaction_reports = []
        keys = list(self.entities.keys())

        # 엔티티들 간의 자발적 유기적 상호 결합 연쇄
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                e1 = keys[i]
                e2 = keys[j]
                # 외부 자극과 엔티티 간 공명에 의거한 결합
                res1 = float(np.dot(self.entities[e1].identity_vector, stim_norm))
                res2 = float(np.dot(self.entities[e2].identity_vector, stim_norm))
                c_int = intensity * max(0.1, (res1 + res2) / 2.0)

                rep = self.couple_entities_mutually(e1, e2, coupling_intensity=c_int)
                interaction_reports.append(rep)

        # 전체 세계 지형의 통합 반응
        mean_cellular_good = np.mean([e.goodness_profile["cellular"] for e in self.entities.values()])
        mean_mental_good = np.mean([e.goodness_profile["mental"] for e in self.entities.values()])

        return {
            "environmental_stimulus_norm": float(np.linalg.norm(environmental_stimulus)),
            "interaction_reports_count": len(interaction_reports),
            "macro_orientation_vector": self.macro_orientation_vector.tolist(),
            "mean_cellular_goodness": float(mean_cellular_good),
            "mean_mental_goodness": float(mean_mental_good),
        }
