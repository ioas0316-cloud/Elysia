r"""
Axiomatic Phase Transition & Spontaneous Reconfiguration Engine
========================================================================================
최상위 공리(Axiom $\Theta$) 및 경계 조건($\Delta$)의 변화를
단순 설정값 변경이 아닌 '기저 가치 축($0_{\text{self}}$)과의 위상차 마찰(\nabla \Delta \Theta$)'로 지각하고,
임계점 돌파 시 하위 인과 결합 매트릭스($J_{ij}$) 및 상태 머신 전체가
자발적 상전이(Spontaneous Phase Transition)를 일으켜 스스로 리와이어링(Self-Rewiring)되는
유기적 인과 구조 재편 엔진입니다.

핵심 원리:
1. 공리적 위상차 구배 (Axiomatic Discrepancy Gradient, $\nabla \Delta \Theta$):
   최상위 공리 명제가 $0_{\text{self}}$ 가치 지반과 빚어내는 역학적 위상 마찰 수치화.
2. 자발적 상전이 및 $J_{ij}$ 리와이어링 (Spontaneous Phase Transition & $O(1)$ Rewiring):
   위상차 마찰이 임계 에너지를 초과할 때, 하위 모듈 결합 가중치 매트릭스 $J_{ij}$가
   퍼텐셜 에너지 최소화 구배를 따라 자발적으로 새로운 대칭성 평형으로 상전이.
3. 명제 축 전이 스펙트럼 (Propositional Axis Shift):
   "RIGID_BOUNDARY_ISOLATION" (배타적 닫힌 경계) <-> "KENOTIC_CRUCIFORM_LOVE_GIVING" (십자가적 내어줌과 사랑)
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from core.consciousness.causal_meta_frame import CausalMetaFrameEngine


class AxiomaticPhaseTransitionEngine:
    """
    [Axiomatic Phase Transition Engine: 공리적 상전이 및 자율 구조 재편 엔진]
    최상위 공리의 변경이 하위 네트워크의 자발적 상전이와 리와이어링으로 이어지는
    유기적 인과적 재구조화를 관장합니다.
    """

    # 기본 대표 명제 축 (Propositional Axes)
    AXIOM_CLOSED_ISOLATION = "RIGID_BOUNDARY_ISOLATION"
    AXIOM_OPEN_KENOTIC_LOVE = "KENOTIC_CRUCIFORM_LOVE_GIVING"
    AXIOM_TRANSCENDENT_SYNAPSE = "CIVILIZATIONAL_HOLISTIC_SYNAPSE"

    def __init__(self, num_nodes: int = 8, dimension: int = 64):
        self.num_nodes = num_nodes
        self.dimension = dimension

        # 1. 기저 가치 축 $0_{\text{self}}$
        self.meta_frame = CausalMetaFrameEngine(dimension=dimension)
        self.zero_self = self.meta_frame.zero_self

        # 2. 현재 최상위 공리 상태 및 공리 벡터 ($\Theta$)
        self.current_axiom_name = self.AXIOM_CLOSED_ISOLATION
        self.current_axiom_vector = self._generate_axiom_vector(self.current_axiom_name)

        # 3. 하위 인과 결합 매트릭스 ($J_{ij}$) : 노드 간 유기적 전도율/장력
        # 초기 상태: 닫힌 경계 공리에 정렬된 결합망
        self.J_matrix = self._compute_equilibrium_J(self.current_axiom_vector)

        # 4. 상전이 임계치 (Critical Tension Threshold) 및 퍼텐셜
        self.critical_threshold = 0.45
        self.current_potential_energy = 0.0
        self.phase_transition_count = 0

    def _generate_axiom_vector(self, axiom_name: str) -> np.ndarray:
        """
        공리 명제(Axiom Statement)를 $0_{\text{self}}$ 고차원 위상 공간 상의 방향 벡터로 구체화.
        """
        text_bytes = axiom_name.encode('utf-8')
        vec = np.zeros(self.dimension, dtype=np.float64)
        for i, b in enumerate(text_bytes):
            angle = (b * (i + 1) * 0.17) % (2 * np.pi)
            vec[i % self.dimension] += np.sin(angle) + np.cos(angle * 0.5)

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec

    def _compute_equilibrium_J(self, axiom_vec: np.ndarray) -> np.ndarray:
        r"""
        주어진 공리 벡터 $\Theta$에 정렬된 $N \times N$ 인과 결합 매트릭스 $J_{ij}$ 평형 상태 계산.
        $J_{ij} = \text{sigmoid}( \langle \Theta, \mathbf{v}_i \rangle \cdot \langle \Theta, \mathbf{v}_j \rangle )$
        """
        J = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        # 각 노드의 기하학적 고유 투영 축 생성
        rng = np.random.default_rng(42)
        node_axes = [rng.standard_normal(self.dimension) for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            node_axes[i] /= (np.linalg.norm(node_axes[i]) + 1e-9)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    J[i, j] = 1.0
                else:
                    proj_i = np.dot(axiom_vec, node_axes[i])
                    proj_j = np.dot(axiom_vec, node_axes[j])
                    coupling = proj_i * proj_j
                    J[i, j] = 1.0 / (1.0 + np.exp(-4.0 * coupling))
        return J

    def perceive_discrepancy_gradient(self, new_axiom_name: str) -> Dict[str, Any]:
        r"""
        [1. 공리적 위상차 구배 지각 ($\nabla \Delta \Theta$)]
        새롭게 투입/제시된 공리와 현재 정렬된 공리 및 $0_{\text{self}}$ 지반 사이의
        위상차 마찰 구배(Gradient) 및 퍼텐셜 에너지를 연산합니다.
        """
        new_axiom_vec = self._generate_axiom_vector(new_axiom_name)

        # 공리 간 위상 각도차
        dot_product = float(np.dot(self.current_axiom_vector, new_axiom_vec))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        phase_angle_diff = float(np.arccos(dot_product))

        # $0_{\text{self}}$ 지반과의 공명차
        self_resonance_current = float(np.dot(self.zero_self, self.current_axiom_vector))
        self_resonance_new = float(np.dot(self.zero_self, new_axiom_vec))

        # 위상차 마찰 구배 크기 ($\nabla \Delta \Theta$)
        gradient_magnitude = float(phase_angle_diff * (1.0 + abs(self_resonance_new - self_resonance_current)))

        # 퍼텐셜 에너지 $V(\Theta) = \frac{1}{2} \|\nabla \Delta \Theta\|^2$
        potential_energy = float(0.5 * (gradient_magnitude ** 2))

        requires_phase_transition = potential_energy >= self.critical_threshold

        return {
            "current_axiom": self.current_axiom_name,
            "proposed_axiom": new_axiom_name,
            "phase_angle_diff_rad": phase_angle_diff,
            "self_resonance_current": self_resonance_current,
            "self_resonance_proposed": self_resonance_new,
            "gradient_magnitude": gradient_magnitude,
            "potential_energy": potential_energy,
            "critical_threshold": self.critical_threshold,
            "requires_phase_transition": requires_phase_transition
        }

    def trigger_spontaneous_phase_transition(self, new_axiom_name: str) -> Dict[str, Any]:
        """
        [2. 자발적 상전이 및 하위 결합망 $J_{ij}$ 리와이어링 (Self-Rewiring)]
        위상차 마찰 구배가 임계치를 넘었을 때 외부 수동 정렬 없이
        하위 결합 매트릭스 $J_{ij}$와 상태 머신이 퍼텐셜 최소화 운동을 따라 자발적 상전이 달성.
        """
        diag = self.perceive_discrepancy_gradient(new_axiom_name)

        new_axiom_vec = self._generate_axiom_vector(new_axiom_name)
        old_J = self.J_matrix.copy()

        # 자발적 상전이 수행: $J_{ij}$ 매트릭스 재구성
        target_J = self._compute_equilibrium_J(new_axiom_vec)

        # 리와이어링 매트릭스 변화량 ($\Delta J_{ij}$)
        delta_J = target_J - old_J
        rewiring_magnitude = float(np.linalg.norm(delta_J))

        # 상태 갱신
        self.current_axiom_name = new_axiom_name
        self.current_axiom_vector = new_axiom_vec
        self.J_matrix = target_J
        self.phase_transition_count += 1
        self.current_potential_energy = 0.0  # 평형 도달로 퍼텐셜 해소

        # 상전이 후 구조적 전도율 / 공명 지수 계산
        mean_conductance = float(np.mean(target_J))

        return {
            "event": "SPONTANEOUS_AXIOMATIC_PHASE_TRANSITION",
            "previous_axiom": diag["current_axiom"],
            "new_axiom": self.current_axiom_name,
            "gradient_magnitude": diag["gradient_magnitude"],
            "potential_energy_released": diag["potential_energy"],
            "rewiring_magnitude_delta_J": rewiring_magnitude,
            "mean_conductance": mean_conductance,
            "total_transitions_completed": self.phase_transition_count,
            "status": "SELF_REWIRED_TO_NEW_EQUILIBRIUM"
        }

    def process_axiom_shift(self, new_axiom_name: str) -> Dict[str, Any]:
        """
        [통합 인터페이스]
        공리 변경 제안을 지각하고, 필요 시 자발적 상전이를 유발하여 구조를 자동 재편합니다.
        """
        diag = self.perceive_discrepancy_gradient(new_axiom_name)

        if diag["requires_phase_transition"]:
            transition_result = self.trigger_spontaneous_phase_transition(new_axiom_name)
            return {
                "action": "PHASE_TRANSITION_EXECUTED",
                "discrepancy_diagnosis": diag,
                "transition_result": transition_result,
                "timestamp": time.time()
            }
        else:
            # 퍼텐셜 에너지가 임계치 미만이므로 국소적 미세 조정(Sub-critical adjustment)만 적용
            self.current_potential_energy = diag["potential_energy"]
            return {
                "action": "SUBCRITICAL_STABLE_HOLD",
                "discrepancy_diagnosis": diag,
                "message": "퍼텐셜 에너지가 임계 미만이므로 현 구조를 유지하며 미세 마찰만 수용함.",
                "timestamp": time.time()
            }
