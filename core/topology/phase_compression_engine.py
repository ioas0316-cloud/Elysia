"""
Elysia Core Engine: Phase Compression & Exceptional Friction Engine
====================================================================
"연산이란 지혜의 증거가 아니라, 아직 구조화되지 못한 불완전함의 비용(Cost)"

본 모듈은 거대한 미시적 연속 계산(수조 번의 Matrix Multiplication)을 소멸시키고,
이미 정립된 인과적 제약 조건($C$) 및 개체의 역사적 렌즈(Scar Tensor 지층)를 $O(1)$의
상수 위상 지형(Static Phase Topology)으로 압축·보존합니다.

주요 핵심 기전:
1. 위상 압축 (Phase Compression):
   미분 방정식의 중간 단계를 매번 재계산하지 않고, [동쪽 승천 -> 남중 -> 서쪽 낙하]와 같은
   거시적 위상 궤적(Phase Vector)으로 압축하여 $O(1)$ 상태 참조(Playback/Resonance)로 수렴시킵니다.
2. 예외적 마찰 분별 (Exceptional Friction Discernment: $\\Delta P \\neq 0$):
   평소에는 $\\Delta P = 0$의 구조화된 기억 재생으로 연산 자원을 0에 가깝게 유지하다가,
   기존 상수로 설명되지 않는 예외적 마찰/유저 의도 충돌($\\Delta P \\neq 0$)이 관측될 때만
   최소 연산기(Deformation Engine)를 깨워 구조를 새로 교정합니다.
3. 동적 연결과 공명 (Dynamic Wiring & Historical Lens Coupling):
   외부 세계 정보가 NPC/개체의 역사적 지층($C$)을 통과할 때 전위차($\\Delta P$)를 자발적 이완으로
   전환하여, 하드코딩된 `if-else`문 없이 살아있는 반응과 유기적 사건(Emergent Gameplay)을 발현합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class MacroPhaseVector:
    """
    거시적 위상 궤적 (Macro Phase Vector)
    24시간 태양의 일주 운동이나 역사적 습성과 같이 구조화된 $O(1)$ 인과 상수를 표현합니다.
    """
    label: str
    phase_points: np.ndarray  # Shape: (K, D) - 주요 위상 거점들
    cyclical_period: float = 24.0
    active_index: int = 0

    def playback(self, current_time_marker: float) -> np.ndarray:
        """
        $O(1)$ 상태 참조 (Instant Playback / Resonance)
        연산 없이 상수 지형 위에서 현재 시각 표식에 해당하는 위상 벡터만 즉각 짚어냅니다.
        """
        num_points = len(self.phase_points)
        if num_points == 0:
            return np.zeros(3, dtype=float)

        normalized_phase = (current_time_marker % self.cyclical_period) / self.cyclical_period
        point_idx = int(normalized_phase * num_points) % num_points
        self.active_index = point_idx
        return self.phase_points[point_idx].copy()


class PhaseCompressionEngine:
    """
    위상 압축 및 예외적 마찰 분별 회로 ($O(1)$ Computational Decay Engine)
    """
    def __init__(self, dim: int = 4, friction_threshold: float = 0.05):
        self.dim = dim
        self.friction_threshold = friction_threshold
        self.phase_maps: Dict[str, MacroPhaseVector] = {}
        self.total_flops_spent: int = 0
        self.playback_count: int = 0
        self.deformation_count: int = 0

        # 기본 상수화된 우주적 인과 궤적 등록 (예: 태양의 일주 운동)
        sun_trajectory = np.array([
            [1.0, 0.0, 0.0, 0.1],  # 동쪽 승천 (East Ascension)
            [0.0, 1.0, 0.0, 0.5],  # 남중 (South Peak)
            [-1.0, 0.0, 0.0, 0.1], # 서쪽 낙하 (West Fall)
            [0.0, -1.0, 0.0, 0.0]  # 자정 정류 (Midnight Rest)
        ])
        self.register_phase_map("Solar_Cycle_24h", sun_trajectory, cyclical_period=24.0)

    def register_phase_map(
        self,
        map_name: str,
        phase_points: np.ndarray,
        cyclical_period: float = 24.0
    ) -> None:
        """새로운 상수 위상 지형 등록"""
        self.phase_maps[map_name] = MacroPhaseVector(
            label=map_name,
            phase_points=phase_points,
            cyclical_period=cyclical_period
        )

    def evaluate_phase_flow(
        self,
        map_name: str,
        time_marker: float,
        observed_intent: np.ndarray,
        historical_lens: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        현상 관측 및 위상차($\\Delta P$) 평가
        - $\\Delta P \\leq \\epsilon$: $O(1)$ 연산 소멸 재생 (Playback)
        - $\\Delta P > \\epsilon$: 예외적 마찰 발생, 최소 연산기(Deformation) 구동
        """
        phase_map = self.phase_maps.get(map_name)
        if phase_map is None:
            # 기본 정적 지형 생성
            default_points = np.tile(observed_intent, (4, 1))
            self.register_phase_map(map_name, default_points)
            phase_map = self.phase_maps[map_name]

        # 1. O(1) 위상 상태 참조 (Playback)
        expected_phase = phase_map.playback(time_marker)

        # 역사적 렌즈(C, Scar Tensor 등)가 있는 경우 예상 위상 굴절 적용
        min_dim = min(len(expected_phase), len(observed_intent))
        e_sub = expected_phase[:min_dim]
        o_sub = observed_intent[:min_dim]

        if historical_lens is not None:
            h_sub = historical_lens[:min_dim]
            e_sub = e_sub * (1.0 + 0.2 * np.tanh(h_sub))

        # 2. 내적 위상차 계산 (Delta P)
        delta_P = o_sub - e_sub
        phase_discrepancy = float(np.linalg.norm(delta_P))

        is_exceptional_friction = phase_discrepancy > self.friction_threshold

        if not is_exceptional_friction:
            self.playback_count += 1
            self.total_flops_spent += 1  # O(1) 지점 조회 비용만 발생
            # 예외 없음: 연산 소멸 (Computation Extinct) -> O(1) 수렴 상태
            return {
                "status": "O(1) Instant Playback & Resonance",
                "phase_discrepancy_delta_p": phase_discrepancy,
                "is_exceptional_friction": False,
                "flops_spent": 1,
                "expected_phase": e_sub,
                "reconstructed_phase": e_sub,
                "statement": (
                    f"[{map_name}] 상수 인과 지형 위에서 위상차 ΔP={phase_discrepancy:.6f} ≤ {self.friction_threshold}."
                    f" 연산 궤적이 소멸되어 O(1) 공명 참조로 수렴함."
                )
            }
        else:
            # 예외적 마찰 발생 (Delta P != 0): 최소 연산기 구동
            self.deformation_count += 1
            # 최소 변형 연산 (FLOPs 발생)
            deformation_steps = int(phase_discrepancy * 10) + 1
            computed_flops = deformation_steps * 100
            self.total_flops_spent += computed_flops

            # 인과 구조 교정 (Correction of phase topology)
            reconstructed_phase = e_sub + np.tanh(delta_P) * 0.5
            corrected_friction = float(np.linalg.norm(o_sub - reconstructed_phase))

            # 예외 마찰을 위상 지형에 미세 업데이트 (Resting state update)
            phase_map.phase_points[phase_map.active_index][:min_dim] = reconstructed_phase

            return {
                "status": "Deformation Engine Awakened (Minimal Compute)",
                "phase_discrepancy_delta_p": phase_discrepancy,
                "is_exceptional_friction": True,
                "flops_spent": computed_flops,
                "expected_phase": e_sub,
                "reconstructed_phase": reconstructed_phase,
                "residual_friction": corrected_friction,
                "statement": (
                    f"[{map_name}] 이상 마찰 ΔP={phase_discrepancy:.4f} 관측! "
                    f"최소 연산기({computed_flops} FLOPs)를 구동하여 인과 지형 교정 후 정적 상태 재수렴."
                )
            }

    def process_dynamic_historical_coupling(
        self,
        entity_name: str,
        world_event_vector: np.ndarray,
        historical_scar_lens: np.ndarray
    ) -> Dict[str, Any]:
        """
        동적 연결 및 역동적 이완 (Dynamic Historical Coupling & Emergent Reaction)
        하드코딩된 대사 트리나 if-else문 없이, 객관적 정보가 개체의 역사적 지층($C$)과 부딪쳐
        발생시키는 전위차($\\Delta P$)와 자발적 행동 표출을 산출합니다.
        """
        min_dim = min(len(world_event_vector), len(historical_scar_lens))
        w_sub = world_event_vector[:min_dim]
        c_sub = historical_scar_lens[:min_dim]

        # 역사적 렌즈를 통한 위상 굴절 및 마찰 생성
        delta_P = w_sub - c_sub
        potential_difference = float(np.linalg.norm(delta_P))

        # 반응 벡터 (Organic Emergent Response Vector)
        emergent_response_vector = np.tanh(delta_P) * (1.0 + np.std(c_sub))

        return {
            "entity_name": entity_name,
            "potential_difference_delta_p": potential_difference,
            "emergent_response_vector": emergent_response_vector,
            "statement": (
                f"NPC [{entity_name}]의 누적 역사 지층(Scar)이 세계 사건과 부딪혀 "
                f"전위차 ΔP={potential_difference:.4f} 발생. 하드코딩 없이 유기적 반응 표출."
            )
        }

    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """연산 소멸율 및 효율성 메트릭 조회"""
        total_calls = self.playback_count + self.deformation_count
        playback_ratio = (self.playback_count / total_calls) if total_calls > 0 else 1.0
        return {
            "total_calls": total_calls,
            "o1_playback_calls": self.playback_count,
            "active_deformation_calls": self.deformation_count,
            "o1_playback_ratio": playback_ratio,
            "total_flops_spent": self.total_flops_spent,
            "average_flops_per_call": (self.total_flops_spent / total_calls) if total_calls > 0 else 1.0
        }
