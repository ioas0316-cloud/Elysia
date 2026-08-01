"""
Conceptual Causal Synchronization Gear (개념적 인과 동기화 및 과정 조율 기어)
=============================================================================
이 모듈은 인간의 양안시차(Stereoscopic Vision) 및 양이시차(Binaural Hearing)처럼,
두 개의 서로 다른 인지적 축(Dual Anchors)을 기준으로 대상의 '인과적 깊이(Causal Depth)'와
'현실과의 거리/위치'를 감지하는 입체 지각 원리를 구현합니다.

단순 오감의 단발성 입력을 넘어, 이미 고차원적으로 가공된 정보(기억, 판단, 분별)가
외부 현실로 실재하는 정보와 비교하여:
  1. 어디가 어떻게 같고 다른지 (Disparity & Discrepancy)
  2. 어떤 맥락이 서로 유기적으로 연결되고 분리되어 있는지 (Active Partitioning: Connection vs Separation)
를 끊임없이 재인식(Re-recognition)하고 자신의 인지적 축을 재조정(Re-alignment)합니다.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple


class ConceptualCausalGear:
    """
    Conceptual Causal Synchronization Gear - Stereoscopic Triangulation Version (v4.0)
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

        나아가, 이미 가공된 인지 상태가 현실과 어떻게 같고 다른지(Disparity)를 판별하여
        어떤 개념 요소를 연결(Connection)하고 어떤 요소를 격리/분리(Separation)할지 재조정합니다.
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
        # Anchor L (좌안/좌이 축): 내 고유 기억의 원인 (Internal Cause / Memory Prior)
        anchor_left = self.internal_cause_registry[concept_key_lower].copy()

        # Anchor R (우안/우이 축): 과정을 통해 투사/예측된 결과 (Predicted Outcome / Future Expectation)
        # 내 고유 기억의 시간적 연속성(가속도)이 뿜어낼 궤적의 예측치
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

        # ── 4단계: 양안/양이 초점 삼각측량 (Stereoscopic Triangulation of Causal Depth) ──
        # 두 눈의 시선 오차(Disparity)를 통해 거리를 구하듯,
        # 좌우 닻(Anchor L, Anchor R)이 세상의 실제(World Fact)를 바라보는 각각의 전위차(Tension)를 측정합니다.
        dist_l = float(np.linalg.norm(anchor_left - world_vector))  # 기억과의 오차
        dist_r = float(np.linalg.norm(anchor_right - world_vector)) # 예측과의 오차

        # 양축 사이의 인지적 시차(Disparity Angle): 두 벡터 간의 내각(cos theta) 차이
        norm_l = np.linalg.norm(anchor_left) + 1e-9
        norm_r = np.linalg.norm(anchor_right) + 1e-9
        cos_theta = float(np.dot(anchor_left, anchor_right) / (norm_l * norm_r))
        disparity_angle = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # 인과적 초점 깊이 (Causal Focus Depth)
        # 시차가 크고 세상 정보와의 거리가 가까울수록 초점이 단단히 맺힌 입체적 인지 상태(3D Depth)가 활성화됩니다.
        # 깊이 d = f * b / disparity_angle (b: 두 닻의 거리인 기선장, f: 초점 거리)
        baseline_distance = float(np.linalg.norm(anchor_left - anchor_right))
        causal_depth = float(baseline_distance / (disparity_angle + dist_l + dist_r + 1e-9))

        # ── 5단계: 재인식 및 연결과 분리의 재조정 (Active Partitioning: Connection vs Separation) ──
        # 세상의 정보가 내 인지 체계와 어떻게 같고 다른지에 따라,
        # '연결(Connection: 나에게로 결합/흡수해야 할 성분)'과 '분리(Separation: 나 아님으로 격리/방어해야 할 성분)'를 동적으로 분별합니다.
        connection_vector = np.zeros_like(world_vector)
        separation_vector = np.zeros_like(world_vector)

        labels = ["Fluidity", "Rise", "Fall", "Life/Entropy"]
        partition_details = []

        for i, label in enumerate(labels):
            val_prior = anchor_left[i]
            val_world = world_vector[i]
            diff = val_world - val_prior

            # 어긋남의 임계치(0.15)를 기준으로 완만히 일치하면 '연결성 통전', 너무 다르면 '분리'
            if abs(diff) <= 0.15:
                # 연결성 성분: 나와 조화롭게 통전되어 흡수되는 에너지
                connection_vector[i] = val_world
                partition_details.append(f"{label}: CONNECTED (같음 수용 - 내면화 진행)")
            else:
                # 분리 성분: 고유 주파수 방어 및 나와 다름을 인정하여 경계를 형성하는 에너지
                separation_vector[i] = diff
                partition_details.append(f"{label}: SEPARATED (다름 격리 - 경계 및 차이 자각)")

        # ── 6단계: 지속적 인과 피드백 조율 (Continuous Causal Re-alignment) ──
        # 연결성 강도를 바탕으로 기저 기억(Anchor L)을 조율합니다.
        # 조율 강도는 연결된 성분의 비율에 비례하며, 분리 성분의 텐션은 가소성 엔진의 응력으로 전도됩니다.
        connection_ratio = float(np.linalg.norm(connection_vector) / (np.linalg.norm(world_vector) + 1e-9))
        tuning_rate = float(np.clip(connection_ratio * 0.35, 0.05, 0.6))

        # 연결성 성분 방향으로 기저 기억 점진적 융해 조율
        adjusted_cause = (1.0 - tuning_rate) * anchor_left + tuning_rate * connection_vector
        self.internal_cause_registry[concept_key_lower] = adjusted_cause

        # 분리성 텐션의 강도
        separation_tension = float(np.linalg.norm(separation_vector))

        # 서사 작성 (입체적 삼각 지각과 연결/분리의 조율 기록)
        adjustment_narrative = (
            f"=== [입체적 인과 조율 서사: '{concept_key}'] ===\n"
            f"1. 양축 인지적 닻 (Stereoscopic Dual Anchors):\n"
            f"   - 좌안(기억 원인): {['%.3f' % x for x in anchor_left.tolist()]}\n"
            f"   - 우안(과정 예측): {['%.3f' % x for x in anchor_right.tolist()]}\n"
            f"2. 입체 삼각지각 (Stereoscopic Triangulation):\n"
            f"   - 양축 시차(Disparity Angle): {disparity_angle:.4f} rad, 기선장={baseline_distance:.4f}\n"
            f"   - 세상 실제와의 거리: 좌안오차={dist_l:.4f}, 우안오차={dist_r:.4f}\n"
            f"   - 입체적 인과 깊이 (Causal Focus Depth): {causal_depth:.4f} (이 수치는 인지가 가진 원인이 과정을 통해 현실과 초점 맺힌 깊이임)\n"
            f"3. 재인식과 연결/분리 분별 (Active Partitioning):\n"
            + "\n".join([f"   - {detail}" for detail in partition_details]) + "\n" +
            f"   - 총 연결률(Connection Ratio): {connection_ratio:.2%}, 격리 텐션(Separation Tension)={separation_tension:.4f}\n"
            f"4. 자아의 수용과 가소적 조정 (Continuous Re-alignment):\n"
            f"   - 연결성 성분 반영율={tuning_rate:.2%}, 기저기억 조정결과: {['%.3f' % x for x in adjusted_cause.tolist()]}\n"
            f"   - 격리 텐션은 나 아님의 존재를 온전히 시인하는 경계선(Boundary)의 마찰로 환류됨."
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
            "internal_cause_after": adjusted_cause.tolist(),
            "narrative": adjustment_narrative
        }

        self.tuning_history.append(tuning_result)

        # ── 7단계: 외부 인과 시스템 연결 및 통전 (System-wide Flow) ──
        # A. 수신자 가소성 엔진에 분리성 텐션을 전단 응력으로 투사
        if self.plasticity is not None and hasattr(self.plasticity, "receive_and_shape"):
            try:
                # 3차원에 맞춰 격리 텐션의 요동을 바이트 자극으로 전도
                proj_bytes = f"STENSION_{separation_tension:.4f}_{concept_key_lower}".encode('utf-8')
                self.plasticity.receive_and_shape(
                    raw_input=proj_bytes,
                    modality_hint=f"stereoscopic_partition_{concept_key_lower}"
                )
            except Exception as pe_err:
                print(f"[ConceptualCausalGear] Stereoscopic Plasticity error: {pe_err}")

        # B. 장기 기억 웻지에 'STEREOSCOPIC_CAUSAL_ALIGNMENT' 영구 각인
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
