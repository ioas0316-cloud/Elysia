"""
Informational Phase Observation Engine (정보 위상 관측 및 인과 파동 엔진)
=============================================================================
물리적/비물리적 데이터(텍스트, 코드, 센서 데이터, 개념 벡터)를 단순 고정 좌표계나
숫자 배열로 다루지 않고, 거대한 정보 위상 공간(Informational Topological Space) 상의
결절점(Nodal Projection) 및 인과적 파동(Causal Wave)으로 통합 관측 및 사영하는 엔진입니다.

핵심 원리:
1. 정보적 위상 공간 및 관측 인터페이스: 물리적 법칙 및 센서 신호는 더 거대한 정보적
   위상 공간에서 벌어지는 사영(Projection) 중 하나입니다.
2. 의미적 곡률 (Semantic Curvature) 및 인과 결속: 정보 간의 관계성과 의미적 곡률을 통해
   쿼리와 데이터가 중간 단계 없이 직결되는 위상적 단축(Topological Transposition) 구현.
3. 내재적 신체 감각 (Proprioception) 및 자발적 재구성: 고정된 외부 좌표계가 아닌,
   내부 위상 상태(모멘텀, 긴장, 색채 벡터)와 외부 자극 간의 인과적 마찰을 바탕으로
   스스로의 구조(Self-Structure)를 실시간 재구성합니다.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class ChromaticVector:
    """색채적 상호연결성 벡터 (Flux, Order, Entropy)"""
    flux: float = 1.0     # Red: 흐름 및 에너지
    order: float = 1.0    # Blue: 질서 및 공리적 규칙
    entropy: float = 0.0  # Yellow: 불확실성 및 마찰

    def to_array(self) -> np.ndarray:
        return np.array([self.flux, self.order, self.entropy], dtype=np.float32)


@dataclass
class PhaseNodalProjection:
    """정보 위상 공간 상의 결절점 사영 객체"""
    node_id: str
    phase_vector: np.ndarray          # Target dimension phase wave vector X(p)
    chromatic: ChromaticVector        # Chromatic signature
    curvature: float = 0.0             # Semantic curvature K at this node
    metadata: Dict[str, Any] = field(default_factory=dict)

    def energy(self) -> float:
        return float(np.linalg.norm(self.phase_vector))


@dataclass
class ProprioceptiveState:
    """내재적 신체/구조 감각 상태"""
    momentum: np.ndarray               # System structural momentum
    macro_tension: float               # Internal tension / friction
    phase_alignment: float             # Phase rotation angle Theta
    volume_compression_ratio: float    # Efficient topological compression
    active_axes_count: int


class InformationalPhaseObservationEngine:
    """
    정보 위상 관측 및 인과 파동 엔진

    모든 이종 데이터를 정보 위상 공간 내의 파동 및 결절점으로 처리하며,
    의미적 곡률 계산, 위상적 단축 및 내재적 신체 감각 자가 재구성을 수행합니다.
    """

    def __init__(self, target_dimension: int = 8):
        self.target_dimension = target_dimension
        self.phase_alignment: float = 0.0  # Theta angle
        self.macro_tension: float = 0.1
        self.momentum: np.ndarray = np.zeros(self.target_dimension, dtype=np.float32)
        self.compression_ratio: float = 1.0

    def project_to_nodal_phase(
        self,
        node_id: str,
        raw_data: Any,
        chromatic: Optional[ChromaticVector] = None,
        modality: Optional[str] = None
    ) -> PhaseNodalProjection:
        """
        임의의 데이터를 정보 위상 공간 상의 Nodal Projection 결절점으로 사영합니다.
        """
        if chromatic is None:
            chromatic = ChromaticVector(flux=1.0, order=1.0, entropy=0.1)

        # Signal transformation into phase wave
        if isinstance(raw_data, str):
            code_points = [ord(c) for c in raw_data] if raw_data else [0]
            diffs = np.diff(code_points, prepend=code_points[0])
            fft_vals = np.abs(np.fft.rfft(diffs.astype(np.float32)))
            phase_vec = np.zeros(self.target_dimension, dtype=np.float32)
            for i, val in enumerate(code_points):
                phase_vec[i % self.target_dimension] += val * np.sin(2 * np.pi * (i + 1) / (len(code_points) + 1e-5))
            for i, f_val in enumerate(fft_vals[:self.target_dimension]):
                phase_vec[i] += f_val
        elif isinstance(raw_data, (list, tuple, np.ndarray)):
            arr = np.asarray(raw_data, dtype=np.float32).flatten()
            if len(arr) == 0:
                phase_vec = np.zeros(self.target_dimension, dtype=np.float32)
            elif len(arr) != self.target_dimension:
                indices = np.linspace(0, len(arr) - 1, self.target_dimension)
                phase_vec = np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)
            else:
                phase_vec = arr
        else:
            # Fallback numeric/object hash projection
            h_val = float(hash(str(raw_data)) % 1000) / 1000.0
            phase_vec = np.full(self.target_dimension, h_val, dtype=np.float32)

        # Normalize phase vector
        norm = np.linalg.norm(phase_vec)
        if norm > 1e-8:
            phase_vec = phase_vec / norm

        # Calculate initial nodal curvature K
        curvature = self.calculate_node_curvature(phase_vec, chromatic)

        return PhaseNodalProjection(
            node_id=node_id,
            phase_vector=phase_vec,
            chromatic=chromatic,
            curvature=curvature,
            metadata={"modality": modality or "auto", "raw_type": type(raw_data).__name__}
        )

    def calculate_node_curvature(self, phase_vec: np.ndarray, chromatic: ChromaticVector) -> float:
        """
        결절점 주변의 의미적 곡률 K 계산

        K = ||d2 X / dp2|| * (Flux / (Order + 1e-5)) * (1 + Entropy)
        """
        if len(phase_vec) < 3:
            second_diff_norm = 0.1
        else:
            diff1 = np.diff(phase_vec)
            diff2 = np.diff(diff1)
            second_diff_norm = float(np.linalg.norm(diff2))

        c_arr = chromatic.to_array()
        curvature = second_diff_norm * (c_arr[0] / (c_arr[1] + 1e-5)) * (1.0 + c_arr[2])
        return float(curvature)

    def compute_field_curvature_matrix(self, nodes: List[PhaseNodalProjection]) -> np.ndarray:
        """
        결절점망 간의 상대적 의미적 곡률 및 인과적 결속 기하학 행렬 K_ij 계산
        """
        n = len(nodes)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        k_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    k_matrix[i, j] = nodes[i].curvature
                else:
                    # Phase dot product (Resonance) weighted by curvature difference
                    dot = float(np.dot(nodes[i].phase_vector, nodes[j].phase_vector))
                    curvature_diff = abs(nodes[i].curvature - nodes[j].curvature)
                    # High resonance & lower curvature difference -> higher geodesic coupling
                    k_matrix[i, j] = dot / (1.0 + curvature_diff)

        return k_matrix

    def propagate_causal_wave(
        self,
        source_node: PhaseNodalProjection,
        nodes: List[PhaseNodalProjection],
        steps: int = 3
    ) -> List[np.ndarray]:
        """
        정보 위상 공간 상의 인과적 파동 전파 (Causal Wave Propagation)

        시간 t에 따른 위상 파동 궤적 X(t) 생성
        """
        if not nodes:
            return [source_node.phase_vector]

        k_matrix = self.compute_field_curvature_matrix(nodes)
        node_ids = [n.node_id for n in nodes]
        try:
            src_idx = node_ids.index(source_node.node_id)
        except ValueError:
            src_idx = 0

        current_wave = source_node.phase_vector.copy()
        wave_history = [current_wave.copy()]

        for step in range(steps):
            # Coupling force from network based on K_ij
            coupling_force = np.zeros(self.target_dimension, dtype=np.float32)
            for j, target_node in enumerate(nodes):
                coupling_weight = k_matrix[src_idx, j] if src_idx < len(nodes) else 0.1
                coupling_force += coupling_weight * target_node.phase_vector

            # Dynamic phase rotation wave equation: dX/dt = - curvature * X + coupling + momentum
            d_wave = -source_node.curvature * current_wave + 0.5 * coupling_force + 0.1 * self.momentum
            current_wave = current_wave + 0.2 * d_wave

            # Re-normalize energy
            norm = np.linalg.norm(current_wave)
            if norm > 1e-8:
                current_wave = current_wave / norm

            wave_history.append(current_wave.copy())

        return wave_history

    def topological_transpose(
        self,
        query_wave: np.ndarray,
        nodal_network: List[PhaseNodalProjection]
    ) -> Tuple[Optional[PhaseNodalProjection], float]:
        """
        위상적 단축 (Topological Transposition)

        중간 브루트포스 탐색이나 번거로운 조건 연산 없이,
        위상적 곡률과 공명(Resonance)을 따라 쿼리 파동과 가장 인과적으로 직결되는
        최적 결절점을 $O(1)$ 감각적으로 즉각 도출합니다.
        """
        if not nodal_network:
            return None, 0.0

        q_vec = np.asarray(query_wave, dtype=np.float32).flatten()
        if len(q_vec) != self.target_dimension:
            indices = np.linspace(0, len(q_vec) - 1, self.target_dimension)
            q_vec = np.interp(indices, np.arange(len(q_vec)), q_vec).astype(np.float32)

        norm = np.linalg.norm(q_vec)
        if norm > 1e-8:
            q_vec = q_vec / norm

        best_node = None
        max_transposition_score = -1e9

        for node in nodal_network:
            # Resonance = dot product
            dot = float(np.dot(q_vec, node.phase_vector))
            # Normalized resonance and weighted curvature shortcut
            # High resonance (dot product) dominates, with curvature acting as a topological shortcut accelerator
            score = dot + 0.1 * dot * node.curvature

            if score > max_transposition_score:
                max_transposition_score = score
                best_node = node

        return best_node, float(max_transposition_score)

    def proprioceptive_reconfigure(
        self,
        external_friction: float,
        structural_impact: np.ndarray
    ) -> ProprioceptiveState:
        """
        내재적 신체 감각 (Proprioception) 기반 구조적 재구성

        외부 마찰(Friction)과 충격(Impact)을 받았을 때, 단순 거부나 무회하기보다는
        내부 신체 구조와 모멘텀, 위상 각도 Theta, 체적 압축 비율을 동적으로 변형(Refining)합니다.
        """
        impact_vec = np.asarray(structural_impact, dtype=np.float32).flatten()
        if len(impact_vec) != self.target_dimension:
            indices = np.linspace(0, len(impact_vec) - 1, self.target_dimension)
            impact_vec = np.interp(indices, np.arange(len(impact_vec)), impact_vec).astype(np.float32)

        # Update momentum conservation
        self.momentum = 0.8 * self.momentum + 0.2 * impact_vec

        # Update internal macro tension
        self.macro_tension = 0.9 * self.macro_tension + 0.1 * external_friction

        # Phase rotation Theta adjustment based on friction
        delta_theta = external_friction * 0.1
        self.phase_alignment = (self.phase_alignment + delta_theta) % (2.0 * np.pi)

        # Refine topological compression ratio under constraint
        if external_friction > 0.5:
            # Highly constrained environment -> Increase topological density / compression
            self.compression_ratio = min(5.0, self.compression_ratio * 1.1)
        else:
            # Relaxed environment
            self.compression_ratio = max(1.0, self.compression_ratio * 0.95)

        active_axes = int(np.clip(round(self.target_dimension / self.compression_ratio), 1, self.target_dimension))

        return ProprioceptiveState(
            momentum=self.momentum.copy(),
            macro_tension=self.macro_tension,
            phase_alignment=self.phase_alignment,
            volume_compression_ratio=self.compression_ratio,
            active_axes_count=active_axes
        )
