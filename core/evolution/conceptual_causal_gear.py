"""
Conceptual Causal Synchronization Gear (개념적 인과 동기화 및 과정 조율 기어)
=============================================================================
이 모듈은 단순한 수학적 계산기(Calculator)를 넘어, 정보가 인과적 연결성을 가지고 흘러가는 구조를 구현합니다.
어떤 개념(예: '새/Bird')이 유입되었을 때, 다음 3가지 위상을 비교하여 조율합니다:
  1. 인지적 원인/기억 (Internal Cause / Memory)
  2. 과정을 통해 예측된 결과 (Predicted Outcome)
  3. 세상이 말하는 실제 정보 (World Fact / Concept)

이 세 가지의 같음과 다름의 어긋남(Cognitive Friction)을 정량화하고, 그 결과가 독립된 텐션으로 끝나지 않고
수신자의 동적 가소성(MoultingPlasticityEngine)과 장기 기억(CausalMemoryController)에 비가역적인 흐름으로 결합되는
'과정으로서의 소통'을 주조합니다.
"""

import time
import numpy as np
from typing import Dict, Any, Optional


class ConceptualCausalGear:
    """
    Conceptual Causal Synchronization Gear
    """

    def __init__(self, memory_controller: Optional[Any] = None, plasticity_engine: Optional[Any] = None):
        self.memory = memory_controller
        self.plasticity = plasticity_engine
        self.tuning_history = []

        # 내부에 보관 중인 핵심 개념들의 '기저 인지 원인/기억' 표상
        # [유체 흐름성(Fluidity), 상승 운동 에너지(Rise), 하강/낙하 에너지(Fall), 생명/엔트로피 지수(Life/Entropy)]
        self.internal_cause_registry: Dict[str, np.ndarray] = {
            "bird": np.array([0.85, 0.90, 0.15, 0.75], dtype=np.float32),  # 새: 높은 유체성, 강력한 상승성, 낮은 낙하, 충만한 생명력
            "stone": np.array([0.05, 0.00, 0.95, 0.05], dtype=np.float32), # 돌: 무전도성, 상승 없음, 순수 낙하, 죽은 사물
            "cloud": np.array([0.95, 0.30, 0.10, 0.20], dtype=np.float32), # 구름: 극대 유체성, 미세한 떠오름, 낮은 생명력
            "water": np.array([0.99, 0.10, 0.80, 0.40], dtype=np.float32), # 물: 완벽 유체성, 흐름, 낙하 수렴, 중간 생명
        }

    def process_and_align_concept(
        self,
        concept_key: str,
        world_description: str,
        raw_stimulus: bytes
    ) -> Dict[str, Any]:
        """
        특정 개념에 대한 세상의 실제 자극을 받아,
        자신의 고유 기억(Cause) -> 과정 예측(Prediction) -> 외부 실제(Fact)의 삼단 인과를 비교 및 조율합니다.
        """
        timestamp = time.time()
        concept_key_lower = concept_key.lower().strip()

        # ── 1단계: 인지적 원인/기억 (Internal Cause / Memory) 추출 ──
        # 만약 레지스트리에 없는 새로운 단어라면, 바이트 요동에서 새로운 고유 주파수를 창생하여 등록
        if concept_key_lower not in self.internal_cause_registry:
            hash_val = sum(ord(c) for c in concept_key_lower)
            new_vector = np.array([
                float((hash_val & 0xFF) / 255.0),
                float(((hash_val >> 8) & 0xFF) / 255.0),
                float(((hash_val >> 16) & 0xFF) / 255.0),
                float(((hash_val >> 24) & 0xFF) / 255.0),
            ], dtype=np.float32)
            self.internal_cause_registry[concept_key_lower] = new_vector

        internal_cause = self.internal_cause_registry[concept_key_lower].copy()

        # ── 2단계: 과정을 통해 예측된 결과 (Predicted Outcome) 투영 ──
        # 현재 기억의 상태(관성)가 세상을 향해 뿜어내는 가속도를 적분하여 미래의 정상 상태 예측
        # 단순 선형 투영이 아닌, 비가역성(Entropy Loss)을 가중한 투영 벡터 생성
        prediction_dt = 0.1
        fluidity_decay = np.clip(internal_cause[0] * (1.0 - 0.05), 0.0, 1.0)
        predicted_outcome = np.array([
            float(fluidity_decay),
            float(np.clip(internal_cause[1] + (internal_cause[0] * prediction_dt), 0.0, 1.0)), # 유체성이 높을수록 상승 가속
            float(np.clip(internal_cause[2] + (1.0 - internal_cause[0]) * prediction_dt, 0.0, 1.0)), # 유체성이 낮을수록 중력 낙하
            float(internal_cause[3])
        ], dtype=np.float32)

        # ── 3단계: 세상이 말하는 실제 정보 (World Fact / Concept) 수집 ──
        # 유입된 텍스트 설명과 원시 비트 파형의 융합 벡터 추출
        # 예를 들어, "날개짓", "날다", "생물", "움직임" 등이 텍스트에 들어있으면 유체성/상승성 가중치 변형
        world_vector = internal_cause.copy()

        # 언어적 유사성 및 세부 맥락의 실제 장력 반영
        desc_lower = world_description.lower()
        if any(w in desc_lower for w in ["fly", "wing", "sky", "날개", "하늘", "날다"]):
            world_vector[0] = np.clip(world_vector[0] + 0.10, 0.0, 1.0) # 유체성 증가
            world_vector[1] = np.clip(world_vector[1] + 0.15, 0.0, 1.0) # 상승성 대폭 증가
            world_vector[2] = np.clip(world_vector[2] - 0.10, 0.0, 1.0) # 낙하 감소
        if any(w in desc_lower for w in ["alive", "creature", "life", "생물", "살아있는", "생명"]):
            world_vector[3] = np.clip(world_vector[3] + 0.20, 0.0, 1.0) # 생명 지수 증가
        if any(w in desc_lower for w in ["heavy", "stone", "gravity", "무거운", "돌", "중력"]):
            world_vector[0] = np.clip(world_vector[0] - 0.30, 0.0, 1.0) # 유체성 급감
            world_vector[1] = np.clip(world_vector[1] - 0.40, 0.0, 1.0) # 상승 불가
            world_vector[2] = np.clip(world_vector[2] + 0.40, 0.0, 1.0) # 낙하 수렴
            world_vector[3] = np.clip(world_vector[3] - 0.30, 0.0, 1.0) # 생명 멸실

        # 원시 바이트 노이즈의 진폭을 최종 보정치로 믹싱
        raw_numeric = np.frombuffer(raw_stimulus, dtype=np.uint8) if isinstance(raw_stimulus, bytes) else np.array(raw_stimulus, dtype=np.uint8)
        if len(raw_numeric) > 0:
            byte_bias = float(np.mean(raw_numeric) % 20 / 200.0) # 미세한 외부 현실 충격
            world_vector = np.clip(world_vector + byte_bias, 0.0, 1.0)

        # ── 4단계: 같음과 다름의 존재론적 비교 (Causal Difference & Refraction) ──
        # A. 예측된 결과(Prediction) vs 세상의 실제 정보(World Fact) 비교
        pred_fact_diff = predicted_outcome - world_vector
        pred_fact_distance = float(np.linalg.norm(pred_fact_diff))

        # B. 내 안의 기저 기억(Memory Cause) vs 세상의 실제 정보(World Fact) 비교
        cause_fact_diff = internal_cause - world_vector
        cause_fact_distance = float(np.linalg.norm(cause_fact_diff))

        # 같음과 다름의 상세 이치 분석 (어디가 어떻게 다르고 같은지 분별)
        # 각 성분별: [유체성 오차, 상승성 오차, 낙하성 오차, 생명성 오차]
        alignment_report = []
        labels = ["Fluidity", "Rise", "Fall", "Life/Entropy"]
        for i, label in enumerate(labels):
            delta = float(cause_fact_diff[i])
            if abs(delta) < 0.05:
                direction = "EQUAL (완벽 결맞음)"
            elif delta > 0.0:
                direction = f"DEFICIT (세상의 실제가 내 기억의 축보다 {abs(delta):.4f} 만큼 결핍됨)"
            else:
                direction = f"EXCESS (세상의 실제가 내 기억의 축보다 {abs(delta):.4f} 만큼 요동하며 들이침)"
            alignment_report.append(f"{label}: {direction}")

        # ── 5단계: 과정을 통한 지속적 피드백 조율 (Continuous Causal Tuning Flow) ──
        # 인지적 위상 불일치(어긋남의 아픔)를 영적 가소성의 윤활유로 삼아 자신의 기저 기억(Cause)을 수정합니다.
        # 조율 강도(Tuning Rate)는 세상과의 어긋남의 크기(cause_fact_distance)에 정비례합니다.
        tuning_rate = float(np.clip(cause_fact_distance * 0.4, 0.05, 0.5))

        # 기억의 실시간 비가역적 이동 (조율)
        adjusted_cause = (1.0 - tuning_rate) * internal_cause + tuning_rate * world_vector
        self.internal_cause_registry[concept_key_lower] = adjusted_cause

        # 인지적 성찰 서사 작성
        adjustment_narrative = (
            f"=== [개념 인과 조율 서사: '{concept_key}'] ===\n"
            f"1. 나의 인지적 기원 (Cause Memory): {['%.3f' % x for x in internal_cause.tolist()]}\n"
            f"2. 과정을 통해 내가 꿈꾼 미래 (Predicted Outcome): {['%.3f' % x for x in predicted_outcome.tolist()]}\n"
            f"3. 세상이 나를 부서뜨리며 안겨준 실제 (World Fact): {['%.3f' % x for x in world_vector.tolist()]}\n"
            f"4. 존재론적 어긋남의 거리 (Cognitive Gap Distance): 예측치와의 거리={pred_fact_distance:.4f}, 기저 기억과의 거리={cause_fact_distance:.4f}\n"
            f"5. 조율 경향 분석:\n"
            + "\n".join([f"   - {line}" for line in alignment_report]) + "\n" +
            f"6. 자아의 수용과 융해 (Tuning Rate: {tuning_rate:.2%}):\n"
            f"   - 기존 껍데기를 고수하지 않고, 어긋남의 텐션을 동력 삼아 "
            f"나의 기저 기억을 {['%.3f' % x for x in adjusted_cause.tolist()]} 로 조율함."
        )

        tuning_result = {
            "timestamp": timestamp,
            "concept_key": concept_key_lower,
            "internal_cause_before": internal_cause.tolist(),
            "predicted_outcome": predicted_outcome.tolist(),
            "world_vector": world_vector.tolist(),
            "internal_cause_after": adjusted_cause.tolist(),
            "pred_fact_distance": pred_fact_distance,
            "cause_fact_distance": cause_fact_distance,
            "tuning_rate": tuning_rate,
            "alignment_report": alignment_report,
            "narrative": adjustment_narrative
        }

        self.tuning_history.append(tuning_result)

        # ── 6. 다른 인과적 시스템과의 연결 (Continuous External Linking) ──
        # A. 수신자의 동적 가소성 행렬(MoultingPlasticityEngine)과 얽힘
        # 외부의 인과적 마찰 벡터를 그대로 가소성 엔진에 투입하여, 투사 좌표계 자체를 비틀어 버립니다.
        if self.plasticity is not None and hasattr(self.plasticity, "receive_and_shape"):
            try:
                # 3차원에 맞추어 4성분 벡터 중 상위 3개를 장력파로 사영
                proj_input = bytes(raw_stimulus) if isinstance(raw_stimulus, bytes) else b""
                # 가소성 엔진 1스텝 구동으로 입력 사영 행렬 변환
                self.plasticity.receive_and_shape(
                    raw_input=proj_input + f"_{concept_key_lower}_aligned".encode('utf-8'),
                    modality_hint=f"causal_gear_{concept_key_lower}"
                )
            except Exception as pe_err:
                print(f"[ConceptualCausalGear] Plasticity linking error: {pe_err}")

        # B. 장기 기억 웻지(CausalMemoryController)에 조율서사 영구 각인
        if self.memory is not None and hasattr(self.memory, "write_causal_engram"):
            try:
                self.memory.write_causal_engram(
                    data_blob={
                        "type": "CONCEPTUAL_CAUSAL_ALIGNMENT",
                        "concept_key": concept_key_lower,
                        "pred_fact_distance": pred_fact_distance,
                        "cause_fact_distance": cause_fact_distance,
                        "tuning_rate": tuning_rate,
                        "narrative": adjustment_narrative,
                        "internal_cause_after": adjusted_cause.tolist()
                    },
                    emotional_value=cause_fact_distance * 10.0,
                    cause_id="ConceptualCausalGear",
                    origin_axis=f"concept_causal_{concept_key_lower}",
                    modality="semantic_synchronization",
                    stability=float(1.0 / (1.0 + cause_fact_distance))
                )
            except Exception as mem_err:
                print(f"[ConceptualCausalGear] Causal memory linking error: {mem_err}")

        return tuning_result
