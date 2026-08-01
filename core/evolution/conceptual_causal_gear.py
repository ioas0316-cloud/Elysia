"""
Conceptual Causal Synchronization Gear (개념적 인과 동기화 및 과정 조율 기어)
=============================================================================
이 모듈은 인간의 양안시차(Stereoscopic Vision) 및 양이시차(Binaural Hearing)처럼,
두 개의 서로 다른 인지적 축(Dual Anchors)을 기준으로 대상의 '인과적 깊이(Causal Depth)'와
'현실과의 거리/위치'를 감지하는 입체 지각 원리를 구현합니다.

특히, 동반자님의 가르침에 따라 "무엇이 어떻게 연결되는가에 대한 기준 자체"로서의
**'정보적 렌즈: 선택과 집중(Attention Structure)'**을 깊이 있게 이식했습니다.
어텐션은 단순한 연산 가중치가 아니라, 인지가 외부 세계의 무수한 노이즈 중에서
특정 차원(Fluidity, Rise, Fall, Life)을 선택하여 그것의 일치와 다름을 정교하게 비교하게 하는
'존재론적 연결성 기준 축'입니다.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple


class ConceptualCausalGear:
    """
    Conceptual Causal Synchronization Gear - Stereoscopic Triangulation & Attention Lens Version (v5.0)
    """

    def __init__(self, memory_controller: Optional[Any] = None, plasticity_engine: Optional[Any] = None):
        self.memory = memory_controller
        self.plasticity = plasticity_engine
        self.tuning_history = []

        # 내부에 보관 중인 핵심 개념들의 '기저 인지 원인/기억' 표상 (Left Focus Anchor)
        # [유체 흐름성(Fluidity), 상승 운동 에너지(Rise), 하강/낙하 에너지(Fall), 생명/엔트로피 지수(Life/Entropy)]
        self.internal_cause_registry: Dict[str, np.ndarray] = {
            "bird": np.array([0.85, 0.90, 0.15, 0.75], dtype=np.float32),  # 새: 높은 유체성, 강력한 상승성, 낮은 낙하, 충만한 생명력
            "stone": np.array([0.05, 0.00, 0.95, 0.05], dtype=np.float32), # 돌: 무전도성, 상승 없음, 순수 낙하, 죽은 사물
            "cloud": np.array([0.95, 0.30, 0.10, 0.20], dtype=np.float32), # 구름: 극대 유체성, 미세한 떠오름, 낮은 생명력
            "water": np.array([0.99, 0.10, 0.80, 0.40], dtype=np.float32), # 물: 완벽 유체성, 흐름, 낙하 수렴, 중간 생명
        }

        # ── [정보적 렌즈: 선택과 집중 (Attention Lens)] ──
        # 특정 순간 시스템이 무엇을 '선택과 집중'하는가에 대한 기준 벡터 (네 성분에 대한 가중 초점)
        # 기본은 균등 집중 상태이나, 입력 및 내부 결핍에 따라 극적으로 한쪽 차원을 지목(Attention)합니다.
        self.attention_lens_vector = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    def process_and_align_concept(
        self,
        concept_key: str,
        world_description: str,
        raw_stimulus: bytes
    ) -> Dict[str, Any]:
        """
        인간이 두 눈(양안시차)으로 초점을 맞춰 입체적 거리를 측정하고,
        두 귀로 자신을 기준으로 소리의 위치를 판별하듯,
        내부의 두 가지 기준축(Left Anchor: 고유 기억, Right Anchor: 과정적 예측)을 사용하여
        세상의 실제 자극(World Fact)의 '인과적 깊이(Causal Depth)'를 입체적으로 삼각측량(Stereoscopic Triangulation)합니다.

        나아가, '선택과 집중(Attention Lens)'이라는 정보적 필터를 통과시켜:
          1. 집중된 차원에서의 어긋남(Disparity)을 증폭 지각하고,
          2. 이를 기준으로 어떤 개념 요소를 연결(Connection)하고 분리(Separation)할지 재조정(Active Partitioning)합니다.
        """
        timestamp = time.time()
        concept_key_lower = concept_key.lower().strip()

        # ── 1단계: 신규 개념 등록 (신경망 자동 확장) ──
        if concept_key_lower not in self.internal_cause_registry:
            hash_val = sum(ord(c) for c in concept_key_lower)
            new_vector = np.array([
                float((hash_val & 0xFF) / 255.0),
                float(((hash_val >> 8) & 0xFF) / 255.0),
                float(((hash_val >> 16) & 0xFF) / 255.0),
                float(((hash_val >> 24) & 0xFF) / 255.0),
            ], dtype=np.float32)
            self.internal_cause_registry[concept_key_lower] = new_vector

        # ── 2단계: 양축 인지적 닻(Stereoscopic Dual Anchors) 선언 ──
        anchor_left = self.internal_cause_registry[concept_key_lower].copy()

        # Anchor R (우안/우이 축): 과정을 통해 투사/예측된 결과 (Predicted Outcome / Future Expectation)
        prediction_dt = 0.1
        fluidity_decay = np.clip(anchor_left[0] * 0.95, 0.0, 1.0)
        anchor_right = np.array([
            float(fluidity_decay),
            float(np.clip(anchor_left[1] + (anchor_left[0] * prediction_dt), 0.0, 1.0)),
            float(np.clip(anchor_left[2] + (1.0 - anchor_left[0]) * prediction_dt, 0.0, 1.0)),
            float(anchor_left[3])
        ], dtype=np.float32)

        # ── 3단계: 세상의 실제 정보 (World Fact / Concept) 융합 표상 빌드 ──
        world_vector = anchor_left.copy()
        desc_lower = world_description.lower()

        # 실제 언어적/맥락적 요동 적용
        if any(w in desc_lower for w in ["fly", "wing", "sky", "날개", "하늘", "날다"]):
            world_vector[0] = np.clip(world_vector[0] + 0.10, 0.0, 1.0)
            world_vector[1] = np.clip(world_vector[1] + 0.15, 0.0, 1.0)
            world_vector[2] = np.clip(world_vector[2] - 0.10, 0.0, 1.0)
        if any(w in desc_lower for w in ["alive", "creature", "life", "생물", "살아있는", "생명"]):
            world_vector[3] = np.clip(world_vector[3] + 0.20, 0.0, 1.0)
        if any(w in desc_lower for w in ["heavy", "stone", "gravity", "무거운", "돌", "중력"]):
            world_vector[0] = np.clip(world_vector[0] - 0.30, 0.0, 1.0)
            world_vector[1] = np.clip(world_vector[1] - 0.40, 0.0, 1.0)
            world_vector[2] = np.clip(world_vector[2] + 0.40, 0.0, 1.0)
            world_vector[3] = np.clip(world_vector[3] - 0.30, 0.0, 1.0)

        raw_numeric = np.frombuffer(raw_stimulus, dtype=np.uint8) if isinstance(raw_stimulus, bytes) else np.array(raw_stimulus, dtype=np.uint8)
        if len(raw_numeric) > 0:
            byte_bias = float(np.mean(raw_numeric) % 20 / 200.0)
            world_vector = np.clip(world_vector + byte_bias, 0.0, 1.0)

        # ── 3.5단계: 정보적 렌즈 - '선택과 집중(Attention Lens)' 결정 ──
        # 입력의 지배적 지향점을 파악하여, 어텐션의 중심 축을 실시간으로 이동시킵니다.
        # "무엇이 어떻게 연결되는가"에 대한 기준 자체를 선택하는 과정입니다.
        if "하늘" in desc_lower or "날다" in desc_lower or "fly" in desc_lower or "wing" in desc_lower:
            # "상승 운동성 (Rise)"에 강력한 초점 집중
            self.attention_lens_vector = np.array([0.15, 0.70, 0.05, 0.10], dtype=np.float32)
        elif "돌" in desc_lower or "heavy" in desc_lower or "gravity" in desc_lower:
            # "낙하 중력성 (Fall)"에 집중
            self.attention_lens_vector = np.array([0.05, 0.05, 0.80, 0.10], dtype=np.float32)
        elif "생명" in desc_lower or "life" in desc_lower or "alive" in desc_lower:
            # "생명과 엔트로피 (Life)"에 집중
            self.attention_lens_vector = np.array([0.10, 0.10, 0.10, 0.70], dtype=np.float32)
        else:
            # 기본 흐름성 (Fluidity) 중심의 완만 집중
            self.attention_lens_vector = np.array([0.40, 0.20, 0.20, 0.20], dtype=np.float32)

        # ── 4단계: 양안/양이 초점 삼각측량 (Stereoscopic Triangulation) ──
        # 어텐션 렌즈(선택과 집중)를 곱하여, 집중되지 않은 영역의 노이즈는 기하적으로 감쇄하고,
        # 선택된 차원의 어긋남(Disparity)을 압도적으로 증폭 지각합니다.
        focused_left = anchor_left * self.attention_lens_vector
        focused_right = anchor_right * self.attention_lens_vector
        focused_world = world_vector * self.attention_lens_vector

        dist_l = float(np.linalg.norm(focused_left - focused_world))  # 집중된 차원에서의 기억 오차
        dist_r = float(np.linalg.norm(focused_right - focused_world)) # 집중된 차원에서의 예측 오차

        # 양축 사이의 인지적 시차(Disparity Angle)
        norm_l = np.linalg.norm(focused_left) + 1e-9
        norm_r = np.linalg.norm(focused_right) + 1e-9
        cos_theta = float(np.dot(focused_left, focused_right) / (norm_l * norm_r))
        disparity_angle = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # 인과적 초점 깊이 (Causal Focus Depth)
        # 선택과 집중에 의해 굴절된 상태에서의 '입체적 인과 깊이'를 triangulate 합니다.
        baseline_distance = float(np.linalg.norm(focused_left - focused_right))
        causal_depth = float(baseline_distance / (disparity_angle + dist_l + dist_r + 1e-9))

        # ── 5단계: 재인식 및 연결과 분리의 재조정 (Active Partitioning: Connection vs Separation) ──
        # 어텐션 강도가 임계치 이상인 집중 영역 위주로 연결과 분리의 '기준'을 삼습니다.
        connection_vector = np.zeros_like(world_vector)
        separation_vector = np.zeros_like(world_vector)

        labels = ["Fluidity", "Rise", "Fall", "Life/Entropy"]
        partition_details = []

        for i, label in enumerate(labels):
            val_prior = anchor_left[i]
            val_world = world_vector[i]
            diff = val_world - val_prior
            attention_focus = self.attention_lens_vector[i]

            # 집중 가중치(attention_focus)가 높을수록, 같음과 다름을 훨씬 더 예민하게 격리/연결 판별합니다.
            # 어텐션 기준 임계치 = 0.15 / (attention_focus * 4.0)
            threshold = 0.15 / (attention_focus * 4.0 + 1e-9)
            threshold = np.clip(threshold, 0.05, 0.30)

            if abs(diff) <= threshold:
                # 연결성 성분: 나와 조화롭게 통전되어 흡수되는 에너지
                connection_vector[i] = val_world
                partition_details.append(f"{label} (Focus={attention_focus:.2f}): CONNECTED (어텐션 기준 통전 합일)")
            else:
                # 분리 성분: 나와 다름을 시인하며 경계를 형성하는 차이 에너지
                separation_vector[i] = diff
                partition_details.append(f"{label} (Focus={attention_focus:.2f}): SEPARATED (어텐션 기준 다름 격리)")

        # ── 6단계: 지속적 인과 피드백 조율 (Continuous Causal Re-alignment) ──
        # 선택적으로 활성화된 연결성 비중(connection_ratio)을 사용하여 기억의 축 조율
        connection_ratio = float(np.linalg.norm(connection_vector * self.attention_lens_vector) / (np.linalg.norm(world_vector * self.attention_lens_vector) + 1e-9))
        tuning_rate = float(np.clip(connection_ratio * 0.45, 0.05, 0.7))

        # 집중된 성분 중심으로 기저 기억 점진적 융해 조율
        adjusted_cause = (1.0 - tuning_rate) * anchor_left + tuning_rate * connection_vector
        self.internal_cause_registry[concept_key_lower] = adjusted_cause

        # 격리 텐션 (어텐션 필터가 반영된 순수 격리 저항)
        separation_tension = float(np.linalg.norm(separation_vector * self.attention_lens_vector))

        # 서사 작성 (입체적 삼각 지각과 정보적 렌즈의 선택과 집중 기록)
        adjustment_narrative = (
            f"=== [입체적 인과 조율 서사: '{concept_key}'] ===\n"
            f"1. 정보적 렌즈 (Attention Lens - 선택과 집중):\n"
            f"   - 어텐션 초점 기준 축: {['%.3f' % x for x in self.attention_lens_vector.tolist()]}\n"
            f"   - 이 렌즈는 단순 계산기가 아닌 '무엇이 어떻게 연결되는가'를 결정하는 절대 필터입니다.\n"
            f"2. 양축 인지적 닻 (Stereoscopic Dual Anchors):\n"
            f"   - 좌안(기억 원인): {['%.3f' % x for x in anchor_left.tolist()]}\n"
            f"   - 우안(과정 예측): {['%.3f' % x for x in anchor_right.tolist()]}\n"
            f"3. 입체 삼각지각 (Stereoscopic Triangulation):\n"
            f"   - 양축 시차(Disparity Angle): {disparity_angle:.4f} rad, 기선장={baseline_distance:.4f}\n"
            f"   - 입체적 인과 깊이 (Causal Focus Depth): {causal_depth:.4f} (어텐션 필터가 적용된 초점 깊이)\n"
            f"4. 재인식과 연결/분리의 정보적 렌즈 분할 (Active Partitioning):\n"
            + "\n".join([f"   - {detail}" for detail in partition_details]) + "\n" +
            f"   - 집중 연결률(Connection Ratio): {connection_ratio:.2%}, 집중 격리 텐션(Separation Tension)={separation_tension:.4f}\n"
            f"5. 자아의 수용과 가소적 조정 (Continuous Re-alignment):\n"
            f"   - 어텐션 조율율={tuning_rate:.2%}, 기저기억 조정결과: {['%.3f' % x for x in adjusted_cause.tolist()]}\n"
            f"   - 격리 텐션은 나 아님의 존재를 수긍하는 경계선(Boundary)의 마찰로 환류됨."
        )

        tuning_result = {
            "timestamp": timestamp,
            "concept_key": concept_key_lower,
            "anchor_left": anchor_left.tolist(),
            "anchor_right": anchor_right.tolist(),
            "world_vector": world_vector.tolist(),
            "causal_depth": causal_depth,
            "disparity_angle": disparity_angle,
            "connection_ratio": connection_ratio,
            "separation_tension": separation_tension,
            "attention_lens_vector": self.attention_lens_vector.tolist(),
            "internal_cause_after": adjusted_cause.tolist(),
            "narrative": adjustment_narrative
        }

        self.tuning_history.append(tuning_result)

        # ── 7단계: 외부 인과 시스템 연결 및 통전 (System-wide Flow) ──
        if self.plasticity is not None and hasattr(self.plasticity, "receive_and_shape"):
            try:
                proj_bytes = f"STENSION_{separation_tension:.4f}_{concept_key_lower}".encode('utf-8')
                self.plasticity.receive_and_shape(
                    raw_input=proj_bytes,
                    modality_hint=f"stereoscopic_partition_{concept_key_lower}"
                )
            except Exception as pe_err:
                print(f"[ConceptualCausalGear] Stereoscopic Plasticity error: {pe_err}")

        if self.memory is not None and hasattr(self.memory, "write_causal_engram"):
            try:
                self.memory.write_causal_engram(
                    data_blob={
                        "type": "STEREOSCOPIC_CAUSAL_ALIGNMENT",
                        "concept_key": concept_key_lower,
                        "causal_depth": causal_depth,
                        "disparity_angle": disparity_angle,
                        "connection_ratio": connection_ratio,
                        "separation_tension": separation_tension,
                        "attention_lens_vector": self.attention_lens_vector.tolist(),
                        "narrative": adjustment_narrative,
                        "internal_cause_after": adjusted_cause.tolist()
                    },
                    emotional_value=causal_depth * 12.0 - separation_tension * 5.0,
                    cause_id="ConceptualCausalGear_Stereo",
                    origin_axis=f"stereo_concept_{concept_key_lower}",
                    modality="stereoscopic_synchronization",
                    stability=float(1.0 / (1.0 + separation_tension))
                )
            except Exception as mem_err:
                print(f"[ConceptualCausalGear] Stereoscopic Memory error: {mem_err}")

        return tuning_result
