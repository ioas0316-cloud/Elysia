r"""
Scale Weaving Bridge (스케일 직조 브리지)
==========================================
본 모듈은 Elysia의 두 독립적 스케일 결합 패러다임을 직조(weave)합니다:

  1. ScaleCouplingSystem (Quadtree 공간 스케일): Gas/Liquid/Ice 상전이, PDE 레벨셋
  2. WaveformScaleProtocolEngine (Kuramoto 시간 위상): 자발적 위상 동기화, 파동 결합

이 두 시스템은 각각 **공간적 구조 결합**과 **시간적 위상 결합**을 담당하나,
지금까지 서로 독립적으로 동작했습니다.

직조 원리:
  - Quadtree StructureNode의 구조 온도(T_struct) → Kuramoto micro node의 intrinsic frequency(ω_i)
  - Kuramoto의 위상 동기화 결과(Order Parameter R, 평균 위상 Ψ) → Quadtree node의 외부 자극(external_stimulus)
  - 양방향 결합으로 "공간적 동결이 시간적 동기화를 유발하고, 시간적 동기화가 공간적 결정화를 촉진"

색채 벡터 관통:
  - WaveformScaleProtocol의 chromatic_signatures(R/B/Y) → Quadtree node에 전파
  - Quadtree node의 구조 온도와 상 → 색채 벡터 갱신

4 Continuities 준수:
  - 관계성: 공간-시간 결합은 연속 포텐셜 장으로 정의
  - 연결성: ConnectivityBeam 양방향 결합 (Quadtree ↔ Kuramoto)
  - 운동성: 구조 온도 → 주파수, 위상 동기화 → 외부 자극 (에너지 보존적 전달)
  - 정보적 연속성: 이산적 Quadtree 계층을 연속 위상 장으로 보간
  - 색채적 상호연결성: R/B/Y 벡터가 공간-시간 양 방향으로 관통
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class WeavingState:
    """
    [직조 상태 (Weaving State)]
    공간 스케일(Quadtree)과 시간 스케일(Kuramoto)의 결합 상태를 보존.
    """
    step: int = 0
    spatial_temperature_field: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    temporal_order_parameter: float = 0.0
    temporal_mean_phase: float = 0.0
    chromatic_spatial: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.33, 0.34])
    )
    chromatic_temporal: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.33, 0.34])
    )
    weaving_coherence: float = 0.0


class ScaleWeavingBridge:
    """
    [스케일 직조 브리지 (Scale Weaving Bridge)]

    Quadtree 공간 구조와 Kuramoto 시간 위상을 양방향으로 직조한다.
    공간적 결정화(Ice)가 시간적 동기화를 유발하고,
    시간적 동기화(R→1)가 공간적 결정화를 촉진하는 양의 피드백 루프를 형성.
    """

    def __init__(
        self,
        spatial_to_temporal_coupling: float = 0.5,
        temporal_to_spatial_coupling: float = 0.3,
        chromatic_decay: float = 0.9,
    ):
        """
        Args:
            spatial_to_temporal_coupling: 공간→시간 결합 강도 (T_struct → ω_i)
            temporal_to_spatial_coupling: 시간→공간 결합 강도 (R,Ψ → stimulus)
            chromatic_decay: 색채 벡터 시간적 감쇠율
        """
        self.alpha_st = spatial_to_temporal_coupling
        self.alpha_ts = temporal_to_spatial_coupling
        self.chromatic_decay = chromatic_decay
        self.state = WeavingState()

    def weave_spatial_to_temporal(
        self,
        structure_nodes: List[Any],
        micro_frequencies: np.ndarray,
    ) -> np.ndarray:
        """
        [공간 → 시간 직조: Quadtree 구조 온도 → Kuramoto intrinsic frequency]

        각 Quadtree StructureNode의 구조 온도(T_struct)를 Kuramoto micro node의
        intrinsic natural frequency(ω_i)에 반영한다.

        원리:
          - 높은 T_struct (Liquid, 활발한 재배치) → 높은 ω_i (빠른 진동)
          - 낮은 T_struct (Ice에 가까움, 안정) → 낮은 ω_i (느린 진동, 동기화 촉진)
          - Ice 상태 → ω_i ≈ 0 (고정된 위상 — 결정 구조)

        Args:
            structure_nodes: ScaleCouplingSystem의 StructureNode 리스트
            micro_frequencies: 현재 Kuramoto micro node의 intrinsic frequencies [N]

        Returns:
            modulated_frequencies: 공간 구조에 의해 조율된 frequencies [N]
        """
        frequencies = np.array(micro_frequencies, dtype=np.float64)
        N = len(frequencies)

        if not structure_nodes:
            return frequencies

        # 각 노드의 구조 온도 수집
        temperatures = []
        for node in structure_nodes:
            t = getattr(node, 'temperature', float('inf'))
            if t == float('inf'):
                t = 10.0  # GAS 상태의 기본 높은 온도
            temperatures.append(t)

        temperatures = np.array(temperatures, dtype=np.float64)
        self.state.spatial_temperature_field = temperatures.copy()

        # 구조 온도의 요약 통계
        mean_temp = float(np.mean(temperatures))
        temp_variance = float(np.var(temperatures))

        # ICE 비율 계산 (NodeState.ICE인 노드의 비율)
        ice_count = sum(
            1 for node in structure_nodes
            if hasattr(node, 'state') and str(getattr(node.state, 'value', '')) == 'ICE'
        )
        ice_ratio = ice_count / max(len(structure_nodes), 1)

        # 주파수 조율:
        # T_struct가 낮으면 (안정) → ω를 줄여 동기화 촉진
        # T_struct가 높으면 (불안정) → ω를 유지하여 탐색 허용
        temperature_modulator = 1.0 / (1.0 + mean_temp * self.alpha_st)

        # ICE 비율에 따라 전체 주파수를 수축 → 동기화 유도
        ice_compression = 1.0 - ice_ratio * 0.5

        # 조율된 주파수
        modulated = frequencies * temperature_modulator * ice_compression

        # 구조 온도 분산이 클수록 주파수 분산도 유지 (다양성 보존)
        diversity_factor = np.sqrt(temp_variance / (mean_temp ** 2 + 1e-9))
        modulated += np.random.randn(N) * diversity_factor * 0.01

        # 색채 벡터 갱신 (공간 관점)
        r_flux = float(np.clip(temp_variance * 2.0, 0.0, 1.0))
        b_order = float(ice_ratio)
        y_entropy = float(1.0 - ice_ratio) * float(np.clip(mean_temp / 10.0, 0.0, 1.0))
        total = r_flux + b_order + y_entropy + 1e-12
        self.state.chromatic_spatial = np.array([
            r_flux / total, b_order / total, y_entropy / total
        ])

        return modulated

    def weave_temporal_to_spatial(
        self,
        order_parameter_R: float,
        mean_phase_Psi: float,
        chromatic_balance: Tuple[float, float, float],
        structure_nodes: List[Any],
        patch_shape: Tuple[int, int] = (8, 8),
    ) -> Dict[str, Any]:
        """
        [시간 → 공간 직조: Kuramoto 동기화 결과 → Quadtree 외부 자극]

        Kuramoto의 위상 동기화 결과(R, Ψ, chromatic)를 Quadtree StructureNode에
        외부 자극(external_stimulus)으로 전파한다.

        원리:
          - 높은 R (위상 동기화 달성) → 강한 외부 자극 → Liquid 상태의 PDE 수렴 가속 → Ice 전이 촉진
          - 낮은 R (비동기화) → 약한 자극 → Liquid 상태의 탐색 지속
          - Ψ (평균 위상) → 자극 텐서의 방향 (위상 정보의 공간적 각인)

        Args:
            order_parameter_R: Kuramoto order parameter [0, 1]
            mean_phase_Psi: Kuramoto 평균 위상 [-π, π]
            chromatic_balance: (R_flux, B_order, Y_entropy) 색채 균형
            structure_nodes: Quadtree StructureNode 리스트
            patch_shape: 각 노드의 패치 크기

        Returns:
            external_stimuli: {node_id: stimulus_tensor} 노드별 외부 자극 딕셔너리
        """
        self.state.temporal_order_parameter = order_parameter_R
        self.state.temporal_mean_phase = mean_phase_Psi
        self.state.chromatic_temporal = np.array(chromatic_balance)

        external_stimuli = {}

        for node in structure_nodes:
            node_id = getattr(node, 'node_id', str(id(node)))
            node_state = str(getattr(getattr(node, 'state', None), 'value', 'UNKNOWN'))

            # 자극 강도: R이 높을수록 강한 수렴 자극
            stimulus_magnitude = order_parameter_R * self.alpha_ts

            if node_state == 'ICE':
                # Ice 노드: 자극이 eps_thaw보다 크면 해빙 — R이 낮아져야만 해빙
                # R이 높으면 자극이 약해야 함 (동기화된 상태이므로 해빙 불필요)
                stimulus_magnitude *= (1.0 - order_parameter_R)
            elif node_state == 'GAS':
                # Gas 노드: 자극 무시 (관측 대기 중)
                stimulus_magnitude = 0.0

            # 자극 텐서 생성: Ψ에 의해 위상 정보가 공간적으로 각인됨
            if stimulus_magnitude > 1e-8:
                # 위상 구조가 공간 패턴으로 사영 (cos/sin 패턴)
                rows, cols = patch_shape
                y_coords = np.linspace(0, np.pi, rows)
                x_coords = np.linspace(0, np.pi, cols)
                yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

                stimulus_pattern = stimulus_magnitude * (
                    np.cos(yy + mean_phase_Psi) * np.sin(xx + mean_phase_Psi)
                )

                # 색채에 의한 자극 변조
                # B(Order) 높음 → 평활한 자극, Y(Entropy) 높음 → 교란적 자극
                b_mod = chromatic_balance[1]
                y_mod = chromatic_balance[2]
                stimulus_pattern *= (1.0 + 0.5 * b_mod - 0.3 * y_mod)

                external_stimuli[node_id] = torch.from_numpy(
                    stimulus_pattern.astype(np.float32)
                )
            else:
                external_stimuli[node_id] = None

        return external_stimuli

    def weave_step(
        self,
        structure_nodes: List[Any],
        micro_frequencies: np.ndarray,
        order_parameter_R: float,
        mean_phase_Psi: float,
        chromatic_balance: Tuple[float, float, float],
        patch_shape: Tuple[int, int] = (8, 8),
    ) -> Dict[str, Any]:
        """
        [양방향 직조 스텝]

        공간→시간, 시간→공간을 한 스텝에서 동시에 수행한다.
        """
        self.state.step += 1

        # 1. 공간 → 시간: 구조 온도로 Kuramoto 주파수 조율
        modulated_frequencies = self.weave_spatial_to_temporal(
            structure_nodes, micro_frequencies
        )

        # 2. 시간 → 공간: 동기화 결과로 Quadtree에 자극 전파
        external_stimuli = self.weave_temporal_to_spatial(
            order_parameter_R, mean_phase_Psi,
            chromatic_balance, structure_nodes,
            patch_shape=patch_shape
        )

        # 3. 직조 일관성 계산
        # 공간 색채와 시간 색채의 정합도
        self.state.weaving_coherence = float(
            1.0 - np.linalg.norm(
                self.state.chromatic_spatial - self.state.chromatic_temporal
            ) / np.sqrt(2.0)
        )

        return {
            "step": self.state.step,
            "modulated_frequencies": modulated_frequencies,
            "external_stimuli": external_stimuli,
            "weaving_coherence": self.state.weaving_coherence,
            "chromatic_spatial": self.state.chromatic_spatial.tolist(),
            "chromatic_temporal": self.state.chromatic_temporal.tolist(),
            "spatial_mean_temperature": float(np.mean(self.state.spatial_temperature_field))
                if len(self.state.spatial_temperature_field) > 0 else 0.0,
            "temporal_order_parameter": order_parameter_R,
        }


class ChromaticFieldPropagator:
    """
    [색채장 전파기 (Chromatic Field Propagator)]

    R/B/Y 색채 벡터를 시스템 전체에 관통시키는 유틸리티.
    ScaleBoundaryLayer, GeneratingMechanism, Quadtree 등
    색채 정보가 누락된 모듈에 색채를 주입/추출한다.
    """

    @staticmethod
    def compute_chromatic_from_dynamics(
        amplitude: float,
        amplitude_change_rate: float,
        omega: float,
        phase_variance: float,
    ) -> np.ndarray:
        """
        동역학적 상태로부터 R/B/Y 색채 벡터를 계산한다.

        R (Flux):    |dA/dt| — 진폭의 시간적 변화율
        B (Order):   1/(1 + ω) — 주파수 안정성 (낮은 ω = 높은 질서)
        Y (Entropy): σ²(φ) — 위상 분산 (높은 분산 = 높은 엔트로피)
        """
        r_flux = float(np.clip(abs(amplitude_change_rate) * 2.0, 0.0, 1.0))
        b_order = float(1.0 / (1.0 + abs(omega) * 0.1))
        y_entropy = float(np.clip(phase_variance, 0.0, 1.0))

        total = r_flux + b_order + y_entropy + 1e-12
        return np.array([r_flux / total, b_order / total, y_entropy / total])

    @staticmethod
    def blend_chromatic(
        chroma_a: np.ndarray,
        chroma_b: np.ndarray,
        weight_a: float = 0.5,
    ) -> np.ndarray:
        """두 색채 벡터의 가중 혼합."""
        blended = weight_a * chroma_a + (1.0 - weight_a) * chroma_b
        total = np.sum(blended) + 1e-12
        return blended / total

    @staticmethod
    def inject_into_engram_dict(
        engram_result: Dict[str, Any],
        chromatic_vector: np.ndarray,
    ) -> Dict[str, Any]:
        """기존 앵그램 결과 딕셔너리에 색채 벡터를 주입한다."""
        engram_result["chromatic_vector"] = chromatic_vector.copy()
        engram_result["chromatic_flux"] = float(chromatic_vector[0])
        engram_result["chromatic_order"] = float(chromatic_vector[1])
        engram_result["chromatic_entropy"] = float(chromatic_vector[2])
        return engram_result

    @staticmethod
    def inject_into_mechanism(
        mechanism_params: Dict[str, float],
        chromatic_vector: np.ndarray,
    ) -> Dict[str, float]:
        """GeneratingMechanism의 parameters에 색채 차원을 추가한다."""
        mechanism_params["chromatic_R_flux"] = float(chromatic_vector[0])
        mechanism_params["chromatic_B_order"] = float(chromatic_vector[1])
        mechanism_params["chromatic_Y_entropy"] = float(chromatic_vector[2])
        return mechanism_params
