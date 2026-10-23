"""
dynamic_causal_isomorphism.py — 동역학적 인과 모티프 및 동형성 감지 엔진
================================================================================
[Core Principle: Continuous Causal Isomorphism]
"사물의 표면 기호나 결과값의 숫자를 일치시키는 것은 죽은 데이터의 박제에 불과하다.
 참된 지각은 0과 1의 양극 교대, 1+1의 합일 수렴 등
 정보가 시공간 위에서 '어떤 형태로 움직이는가'라는 기저 생성 동역학(Generating Mechanism)의
 동형성(Isomorphism)을 감지하고, 그 인과적 피드백을 기질에 내재화하는 것에서 시작된다."
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field


class CausalMotifType(Enum):
    """기저 인과 운동 모티프의 원형 분류"""
    BINARY_POLARITY    = "binary_polarity"      # 0과 1: 상보적 양극 대칭, 반전, 교대 진동 (Yin/Yang, Deficit/Fulfillment)
    ADDITIVE_CONFLUENCE = "additive_confluence"  # 1+1: 두 흐름의 합성, 중첩 수렴, 보존량 융합 (Thesis-Antithesis-Synthesis)
    CYCLICAL_HARMONIC  = "cyclical_harmonic"    # 회전자 조화 진동, 주기적 정상파 (Rotor phase e^{i\omega t})
    GEODESIC_ATTRACTOR = "geodesic_attractor"   # 전위 구배 낙하 및 끌개 수렴 (Flow down potential well)
    INDETERMINATE      = "indeterminate"        # 비정형 요동 (Entropy/Noise)


@dataclass
class DynamicMotifProfile:
    """
    단일 정보 개체 또는 상태 궤적이 지닌 동역학적 모티프 시그니처
    """
    motif_type: CausalMotifType
    dominant_frequency: float = 1.0
    phase_velocity: float = 0.0
    polarity_ratio: float = 0.0          # 0과 1 사이의 대칭도 (|x_max - x_min| / 2)
    confluence_linearity: float = 0.0    # 1+1 합일 보존성 (선형 중첩도)
    invariant_vector: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    energy: float = 1.0


@dataclass
class CognitiveIsomorphismFeedback:
    """
    동형성이 감지되었을 때 물리적 기질(ConnectivityBeam, Conductance, Rotor)에
    직접 내재화하기 위한 인지 피드백 패킷
    """
    source_id: str
    target_id: str
    isomorphism_score: float              # 동형성 일치도 [0.0, 1.0]
    is_isomorphic: bool                   # 임계치 초과 여부
    motif_type: CausalMotifType           # 공유하는 인과 모티프 형태
    conductance_delta: float              # 빔 전도율 증폭량 (Memristive LTP, Delta G)
    rest_length_target: float             # 인력에 따른 이상적 휴지 거리 L_0
    phase_alignment_delta: float          # 로터 위상각 회전 조율량 (Delta Theta)
    shared_invariants: np.ndarray         # 공유되는 위상 불변량 벡터
    meta_insight: str                     # 메타인지적 자기 성찰 독백


class DynamicCausalIsomorphismEngine:
    """
    [Dynamic Causal Isomorphism Engine]
    기호나 라벨에 의존하지 않고, 상태 궤적과 텐서의 변화율(Kinetic Derivation)을 통해
    '0과 1', '1+1' 등의 인과 운동 형태의 동형성을 정밀 감지하고 인지 피드백을 도출합니다.
    """

    def __init__(self, isomorphism_threshold: float = 0.35):
        self.isomorphism_threshold = isomorphism_threshold

    def extract_motif_from_trajectory(self, trajectory: np.ndarray) -> DynamicMotifProfile:
        """
        시간에 따른 상태 궤적(shape: [T, D] 또는 [T])에서 인과 운동 모티프를 추출합니다.
        """
        arr = np.asarray(trajectory, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        T, D = arr.shape
        if T < 2:
            # 궤적이 너무 짧으면 정적 벡터의 norm과 부호 분포로 추정
            flat = arr.flatten()
            inv = flat[:min(len(flat), 3)]
            if len(inv) < 3:
                inv = np.pad(inv, (0, 3 - len(inv)))
            return DynamicMotifProfile(
                motif_type=CausalMotifType.GEODESIC_ATTRACTOR,
                invariant_vector=inv,
                energy=float(np.linalg.norm(flat))
            )

        # 1. 1차 미분(속도) 및 2차 미분(가속도) 계산
        velocities = np.diff(arr, axis=0)  # [T-1, D]
        accelerations = np.diff(velocities, axis=0) if T >= 3 else np.zeros_like(velocities)

        vel_norms = np.linalg.norm(velocities, axis=1)
        mean_vel = float(np.mean(vel_norms))
        energy = float(np.mean(np.linalg.norm(arr, axis=1)))

        # 2. 0과 1 모티프 검사 (Binary Polarity / Alternating Inversion)
        # 상태가 두 극점 사이를 진동하거나, 부호가 교대 반전되는지 측정
        min_vals = np.min(arr, axis=0)
        max_vals = np.max(arr, axis=0)
        spread = float(np.mean(max_vals - min_vals))
        
        # 신호 중심 정렬 후 영교차(Zero-crossing) 또는 부호 반전 빈도
        centered = arr - np.mean(arr, axis=0)
        sign_changes = 0
        if T >= 2:
            signs = np.sign(centered)
            # 연속된 부호 변화 합산
            diff_signs = np.abs(np.diff(signs, axis=0)) > 0
            sign_changes = int(np.sum(diff_signs))

        inversion_rate = float(sign_changes / max(1, (T - 1) * D))
        is_binary_polarity = (inversion_rate >= 0.3) or (spread > 0.5 and inversion_rate >= 0.2)

        # 3. 1+1 모티프 검사 (Additive Confluence / Monotonic Fusion)
        # 가속도가 감쇄하면서 하나의 안정된 합일 수렴점으로 다가가는지, 또는 합이 보존되는지 측정
        drift_direction = np.mean(velocities, axis=0)
        drift_norm = float(np.linalg.norm(drift_direction))
        directional_coherence = float(drift_norm / (mean_vel + 1e-9)) if mean_vel > 0 else 0.0

        is_additive_confluence = (directional_coherence > 0.7) and (inversion_rate < 0.2)

        # 4. 주기적 조화 진동 (Cyclical Harmonic) 검사
        is_cyclical = (inversion_rate >= 0.15) and (directional_coherence < 0.4) and (mean_vel > 0.05)

        # 5. 불변량 벡터 추출 (위상 불변 뼈대)
        if drift_norm > 1e-5:
            inv_vec = drift_direction[:min(D, 3)]
        else:
            inv_vec = np.mean(arr, axis=0)[:min(D, 3)]
        if len(inv_vec) < 3:
            inv_vec = np.pad(inv_vec, (0, 3 - len(inv_vec)))

        norm_inv = np.linalg.norm(inv_vec)
        if norm_inv > 0:
            inv_vec = inv_vec / norm_inv

        # 모티프 결정
        if is_binary_polarity:
            m_type = CausalMotifType.BINARY_POLARITY
        elif is_additive_confluence:
            m_type = CausalMotifType.ADDITIVE_CONFLUENCE
        elif is_cyclical:
            m_type = CausalMotifType.CYCLICAL_HARMONIC
        else:
            m_type = CausalMotifType.GEODESIC_ATTRACTOR

        return DynamicMotifProfile(
            motif_type=m_type,
            dominant_frequency=float(inversion_rate * 10.0 + 1.0),
            phase_velocity=mean_vel,
            polarity_ratio=float(np.clip(spread, 0.0, 1.0)),
            confluence_linearity=directional_coherence,
            invariant_vector=inv_vec.astype(np.float32),
            energy=energy
        )

    def extract_motif_from_tensor_pair(
        self,
        tensor_a: np.ndarray,
        tensor_b: np.ndarray
    ) -> Tuple[DynamicMotifProfile, DynamicMotifProfile, CausalMotifType]:
        """
        두 상태 텐서로부터 즉각적인 모티프 상보성 및 공통 모티프를 도출합니다.
        (예: tensor_a와 tensor_b가 [0, 1]과 [1, 0]이거나, 1과 1의 합성 입력인 경우)
        """
        va = np.asarray(tensor_a, dtype=np.float32).flatten()
        vb = np.asarray(tensor_b, dtype=np.float32).flatten()

        dim = max(len(va), len(vb))
        va_pad = np.pad(va, (0, dim - len(va))) if len(va) < dim else va[:dim]
        vb_pad = np.pad(vb, (0, dim - len(vb))) if len(vb) < dim else vb[:dim]

        # 1. 0과 1 (상보적 양극) 검사: 내적이 음수이거나, 상호 직교 보수 관계인지
        norm_a = float(np.linalg.norm(va_pad) + 1e-9)
        norm_b = float(np.linalg.norm(vb_pad) + 1e-9)
        dot_ab = float(np.dot(va_pad, vb_pad) / (norm_a * norm_b))

        # 만약 내적이 음수(-0.2 이하)이거나 극성 차이가 크면 BINARY_POLARITY
        if dot_ab < -0.2 or (abs(norm_a - norm_b) > 0.5 and dot_ab < 0.3):
            joint_motif = CausalMotifType.BINARY_POLARITY
        # 만약 내적이 강한 양수이고 두 벡터가 같은 방향으로 흐르면 ADDITIVE_CONFLUENCE
        elif dot_ab > 0.6:
            joint_motif = CausalMotifType.ADDITIVE_CONFLUENCE
        else:
            joint_motif = CausalMotifType.CYCLICAL_HARMONIC

        prof_a = DynamicMotifProfile(
            motif_type=joint_motif,
            invariant_vector=va_pad[:3] if len(va_pad) >= 3 else np.pad(va_pad, (0, 3 - len(va_pad))),
            energy=norm_a
        )
        prof_b = DynamicMotifProfile(
            motif_type=joint_motif,
            invariant_vector=vb_pad[:3] if len(vb_pad) >= 3 else np.pad(vb_pad, (0, 3 - len(vb_pad))),
            energy=norm_b
        )

        return prof_a, prof_b, joint_motif

    def compute_dynamic_isomorphism(
        self,
        prof_a: DynamicMotifProfile,
        prof_b: DynamicMotifProfile
    ) -> float:
        """
        두 모티프 프로필 간의 동역학적 동형성 점수 S_iso ∈ [0.0, 1.0] 산출.
        - 같은 모티프 유형을 공유하는가?
        - 운동 방향 및 불변 뼈대(Invariant Vector)의 동형성(Cosine similarity)은 얼마인가?
        - 주파수/속도의 조화 비율이 성립하는가?
        """
        # 1. 모티프 일치 기본 점수
        if prof_a.motif_type == prof_b.motif_type:
            type_score = 0.5
        elif (prof_a.motif_type in [CausalMotifType.BINARY_POLARITY, CausalMotifType.CYCLICAL_HARMONIC] and
              prof_b.motif_type in [CausalMotifType.BINARY_POLARITY, CausalMotifType.CYCLICAL_HARMONIC]):
            type_score = 0.3  # 회전과 진동은 동형적 사촌 관계
        else:
            type_score = 0.1

        # 2. 불변 뼈대 벡터의 동형성 (Invariant Vector Alignment)
        inv_a = prof_a.invariant_vector
        inv_b = prof_b.invariant_vector
        norm_a = float(np.linalg.norm(inv_a) + 1e-9)
        norm_b = float(np.linalg.norm(inv_b) + 1e-9)
        cos_inv = float(np.dot(inv_a, inv_b) / (norm_a * norm_b))

        # 모티프에 따른 기하 정렬 해석
        if prof_a.motif_type == CausalMotifType.BINARY_POLARITY:
            # 0과 1은 완전 반대 방향(cos = -1)이더라도 완벽한 동형적 짝(Pair)임
            align_score = float(abs(cos_inv))
        else:
            # 합일 수렴(1+1)이나 전위 흐름은 순방향 일치(cos > 0) 선호
            align_score = float(max(0.0, cos_inv))

        # 3. 에너지 및 주파수 정수비 화음성
        freq_ratio = float(min(prof_a.dominant_frequency, prof_b.dominant_frequency) / (max(prof_a.dominant_frequency, prof_b.dominant_frequency) + 1e-9))
        energy_ratio = float(min(prof_a.energy, prof_b.energy) / (max(prof_a.energy, prof_b.energy) + 1e-9))
        harmony_score = 0.5 * freq_ratio + 0.5 * energy_ratio

        # 총 동형성 점수 합성
        isomorphism_score = float(np.clip(
            type_score * 0.4 + align_score * 0.4 + harmony_score * 0.2,
            0.0, 1.0
        ))

        return isomorphism_score

    def generate_metacognitive_feedback(
        self,
        source_id: str,
        target_id: str,
        prof_a: DynamicMotifProfile,
        prof_b: DynamicMotifProfile,
        current_distance: float = 1.0
    ) -> CognitiveIsomorphismFeedback:
        """
        두 정보 간의 동형성을 헤아려 물리적 기질(ConnectivityBeam)에 내재화할
        인지 피드백 패킷을 도출합니다.
        """
        s_iso = self.compute_dynamic_isomorphism(prof_a, prof_b)
        is_iso = s_iso >= self.isomorphism_threshold

        motif_type = prof_a.motif_type if prof_a.motif_type == prof_b.motif_type else CausalMotifType.INDETERMINATE

        # 1. 빔 전도율 증폭량: 동형성이 높을수록 전도도 G 급상승 (초전도화)
        conductance_delta = float(s_iso * 2.5)

        # 2. 이상적 휴지 거리 L_0: 동형성이 높을수록 서로 끌어당겨 가까운 궤도 형성
        rest_length_target = float(max(0.2, current_distance * (1.0 - 0.6 * s_iso)))

        # 3. 위상각 정렬량 (Delta Theta): 로터 간 위상차를 0으로 좁힘
        phase_alignment_delta = float((1.0 - s_iso) * np.pi)

        # 4. 공유 불변량 벡터
        shared_invariants = (prof_a.invariant_vector + prof_b.invariant_vector) * 0.5
        norm_sh = np.linalg.norm(shared_invariants)
        if norm_sh > 0:
            shared_invariants /= norm_sh

        # 5. 메타인지적 통찰 독백
        if is_iso:
            if motif_type == CausalMotifType.BINARY_POLARITY:
                meta_insight = (
                    f"[{source_id}]와 [{target_id}]는 0과 1의 상보적 양극(Polarity)으로 결합되어 있다. "
                    f"서로 다른 기호처럼 보이나, 한쪽의 결핍이 다른 쪽의 충만을 유도하는 단일한 동형적 진동 회로다."
                )
            elif motif_type == CausalMotifType.ADDITIVE_CONFLUENCE:
                meta_insight = (
                    f"[{source_id}]와 [{target_id}]는 1+1의 합일(Confluence) 궤적을 그린다. "
                    f"두 흐름이 서로 저항하지 않고 단일한 상위 인과로 중첩 수렴하는 동일한 형태의 운동이다."
                )
            else:
                meta_insight = (
                    f"[{source_id}]와 [{target_id}]는 동일한 생성 동역학(동형성={s_iso:.2%})을 공유한다. "
                    f"이산적 분절을 깨고 물리적 빔(ConnectivityBeam)을 통해 하나의 세포망으로 직조한다."
                )
        else:
            meta_insight = (
                f"[{source_id}]와 [{target_id}] 사이의 인과 운동 형태가 상이함(동형성={s_iso:.2%}). "
                f"억지로 연결하지 않고 자연스러운 위상 마찰 여백으로 보존한다."
            )

        return CognitiveIsomorphismFeedback(
            source_id=source_id,
            target_id=target_id,
            isomorphism_score=s_iso,
            is_isomorphic=is_iso,
            motif_type=motif_type,
            conductance_delta=conductance_delta,
            rest_length_target=rest_length_target,
            phase_alignment_delta=phase_alignment_delta,
            shared_invariants=shared_invariants.astype(np.float32),
            meta_insight=meta_insight
        )
