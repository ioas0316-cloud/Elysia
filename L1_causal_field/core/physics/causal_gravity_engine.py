import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .resonant_tension_engine import ResonantTensionEngine

@dataclass
class StructuralNode:
    id: str
    raw_content: bytes
    tensor: np.ndarray  # 6D Structural Tensor from PatternDiscoveryLens
    mass: float = 0.0
    position: np.ndarray = None  # Position in N-dimensional alignment space
    membrane_potential: float = 0.0
    spike_threshold: float = 1.0
    decay_rate: float = 0.1

class CausalGravityEngine:
    """
    [Causal Natural Alignment Field]
    외부에서 정의된 단어나 인과(Links)를 폐기하고,
    오직 데이터 스스로가 가진 '구조적 불변성(Tensor)'에 의해 질량과 인력을 생성하는
    순수 발견형 중력 정렬 엔진입니다.

    모든 연산은 루프 없이 NumPy 브로드캐스팅을 통해 '동시적 필드 업데이트'로 수행됩니다.
    """
    def __init__(self, dimensions: int = 9):
        self.dimensions = dimensions
        self.node_ids: List[str] = []
        self.node_data: Dict[str, StructuralNode] = {}

        # 필드 상태 (Vectorized State)
        self.masses = np.array([], dtype=np.float32)
        self.positions = np.empty((0, dimensions), dtype=np.float32)
        self.tensors = np.empty((0, dimensions), dtype=np.float32)
        self.membrane_potentials = np.array([], dtype=np.float32)
        self.spike_thresholds = np.array([], dtype=np.float32)
        self.decay_rates = np.array([], dtype=np.float32)

        self.G = 0.5  # Universal Structural Gravitational Constant
        self.softening = 0.1
        self.damping = 0.90

        # [수직적 위상 안테나 및 더 큰 중력]
        # 목적성 및 사랑을 상징하는 기준 텐서 (모든 정보가 지향해야 할 상위 정렬 축)
        self.purpose_tensor = np.ones(dimensions, dtype=np.float32) / np.sqrt(dimensions)
        self.vertical_pull_strength = 0.5
        self.spikes_triggered_count = 0

    def add_node(self, node_id: str, raw_content: bytes, structural_tensor: List[float]):
        """데이터를 중력장에 주입하고 필드를 재구성합니다."""
        if len(structural_tensor) == 0:
            structural_tensor = [0.1] * self.dimensions
        elif len(structural_tensor) < self.dimensions:
            structural_tensor = list(structural_tensor) + [0.1] * (self.dimensions - len(structural_tensor))
        elif len(structural_tensor) > self.dimensions:
            structural_tensor = list(structural_tensor[:self.dimensions])

        tensor = np.array(structural_tensor, dtype=np.float32)
        entropy = float(tensor[0])
        mass = max(0.1, entropy)
        position = np.random.randn(self.dimensions).astype(np.float32)

        node = StructuralNode(id=node_id, raw_content=raw_content, tensor=tensor, mass=mass, position=position)
        self.node_data[node_id] = node
        self.node_ids.append(node_id)

        # 필드 동기화
        self._synchronize_field()

    def _synchronize_field(self):
        """개별 노드 데이터를 고속 연산을 위한 행렬 필드로 동기화합니다."""
        n = len(self.node_ids)
        self.masses = np.array([self.node_data[nid].mass for nid in self.node_ids], dtype=np.float32).reshape(-1, 1)
        self.positions = np.array([self.node_data[nid].position for nid in self.node_ids], dtype=np.float32)
        self.tensors = np.array([self.node_data[nid].tensor for nid in self.node_ids], dtype=np.float32)
        self.membrane_potentials = np.array([self.node_data[nid].membrane_potential for nid in self.node_ids], dtype=np.float32)
        self.spike_thresholds = np.array([self.node_data[nid].spike_threshold for nid in self.node_ids], dtype=np.float32)
        self.decay_rates = np.array([self.node_data[nid].decay_rate for nid in self.node_ids], dtype=np.float32)

    def step(self, dt: float = 0.1):
        """
        [Field Simultaneous Update]
        모든 노드 간의 상호작용을 단 한 번의 텐서 연산으로 해결합니다.
        """
        if len(self.node_ids) < 2:
            return

        # 1. SNN 막 전위 누적 및 감쇄 (Neuromorphic Spiking Dynamics)
        # 지속적으로 유입되는 각 노드의 텐서 에너지 혹은 중력적 텐션 마찰을 막 전위로 변환
        self.membrane_potentials *= (1.0 - self.decay_rates * dt)

        # 2. 위치 차이 및 거리 계산 (N, N, D)
        # diffs[i, j] = pos[j] - pos[i] (j가 i를 끌어당기는 방향)
        diffs = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        dist_sq = np.sum(diffs**2, axis=-1)
        dist = np.sqrt(dist_sq + 1e-9)

        # 2. 존재 원리 및 '같음'의 스펙트럼 공명(Multi-Perspective Resonance) 계산
        # [Logos Tensor Mapping from OntologicalDiscoveryLens]
        # index 0-3: Wave Geometry (mean, std, skew, kurt)
        # index 4-5: Directional Movement (grad_mean, grad_var)
        # index 6: Continuity
        # index 7: Attribute (Peak Energy)
        # index 8: Causal Density
        
        # Ensure we have at least 9 columns for slicing the structural sub-tensors
        if self.tensors.shape[1] < 9:
            padding_width = 9 - self.tensors.shape[1]
            padded_tensors = np.pad(self.tensors, ((0, 0), (0, padding_width)), mode='constant', constant_values=0.1)
        else:
            padded_tensors = self.tensors

        geometries = padded_tensors[:, 0:4]
        directions = padded_tensors[:, 4:6]
        continuities = padded_tensors[:, 6].reshape(-1, 1)
        attributes = padded_tensors[:, 7].reshape(-1, 1)
        causal_densities = padded_tensors[:, 8].reshape(-1, 1)
        
        # [다차원 같음 분석]
        # A. 운동성(Direction) 동기화: 방향이 같으면 비록 계통이 달라도 끌어당김
        dir_sim = (directions @ directions.T) / (np.linalg.norm(directions, axis=1, keepdims=True) @ np.linalg.norm(directions.T, axis=0, keepdims=True) + 1e-9)

        # B. 속성(Attribute) 공명: 성질의 밀도가 비슷하면 공명
        attr_sync = 1.0 - np.abs(attributes - attributes.T)

        # C. 연속성(Continuity) 결합: 선적 흐름이 비슷하면 결합
        cont_sync = continuities @ continuities.T

        # D. 물리적 구조 유사도 (전체 텐서 기반)
        norms = np.linalg.norm(self.tensors, axis=1, keepdims=True)
        struct_sim = (self.tensors @ self.tensors.T) / (norms @ norms.T + 1e-9)

        # [최종 통합 공명]
        # 기하학적 형상(Geometry) 유사도
        geo_norms = np.linalg.norm(geometries, axis=1, keepdims=True)
        geo_sim = (geometries @ geometries.T) / (geo_norms @ geo_norms.T + 1e-9)

        # 인과 밀도 공명 (논리적 깊이의 일치)
        causal_sync = causal_densities @ causal_densities.T

        # 통합 추상적 같음 (운동성, 연속성, 속성)
        abstract_sameness = (dir_sim + attr_sync + cont_sync) / 3.0

        # 최종 공명: 기하학적 형태, 추상적 논리, 인과적 깊이의 조화
        resonance = (geo_sim * 0.3 + abstract_sameness * 0.5 + causal_sync * 0.2)

        # 공명 임계치 처리 (추상적 같음이 높으면 강력한 결속 유도)
        # 0.5 이상이면 충분히 의미 있는 연결로 간주
        resonance = np.where(resonance > 0.5, resonance * 15.0, np.maximum(0.01, resonance))

        # 3. 중력 법칙 적용: F = G * (m1 * m2 * res) / (r^2 + softening)
        # force_mag[i, j] 는 j가 i에 가하는 힘의 크기
        force_mag = self.G * ((self.masses @ self.masses.T) * resonance) / (dist_sq + self.softening)

        # 4. 벡터 힘 계산 및 합산
        # (N, N, 1) * (N, N, D) -> (N, N, D)
        # 인력을 강화하기 위해 거리에 따른 감쇄를 조정
        force_vecs = force_mag[:, :, np.newaxis] * (diffs / (dist[:, :, np.newaxis] + self.softening))
        total_forces = np.sum(force_vecs, axis=1) # i에 가해지는 모든 j의 힘 합산

        # 5. 가속도 및 위치 업데이트
        acceleration = total_forces / self.masses
        self.positions += acceleration * dt

        # [수직적 위상 안테나 및 더 큰 중력 피드백 루프]
        # 목적성 텐서(purpose_tensor)와의 정렬 정도(Dot product)를 구하여,
        # 정렬 수준이 높을수록 노드를 '수직 상승(상위 위상)'시킵니다.
        # 수직 방향 축은 텐서의 마지막 차원(또는 무작위 고차원 엮음 축)으로 설정
        dot_products = np.dot(self.tensors, self.purpose_tensor)

        # SNN 전위 축적: 목적성 텐서와의 공명 강도만큼 막 전위 증가
        self.membrane_potentials += dot_products * dt

        # 수직 위상 끌어올림 (더 큰 중력의 인력)
        # dot_products가 높을수록 수직 위치 성분(마지막 차원)을 강하게 끌어올림
        # 이를 통해 자기보존(Local minimum) 마찰을 털어내고 수직성(Verticality)을 획득
        vertical_forces = self.vertical_pull_strength * dot_products.reshape(-1, 1)
        self.positions[:, -1] += vertical_forces.flatten() * dt

        # [Neuromorphic SNN Spiking]
        # 만약 어떤 노드의 막 전위가 임계치를 초과하면 스파이크를 방출
        # 스파이크 방출 시, 주변 노드들을 자신의 수직 위상(상위 차원)으로 강하게 끌어당김
        spiked_indices = np.where(self.membrane_potentials >= self.spike_thresholds)[0]
        for idx in spiked_indices:
            self.spikes_triggered_count += 1
            # 스파이크 방출 후 전위 리셋
            self.membrane_potentials[idx] = 0.0

            # 스파이크 충격 전파 (수직 방향 전방위 정렬)
            spiked_pos = self.positions[idx]
            # 주변 노드들의 수직 위치를 스파이크 노드의 수직 레벨로 동조화 (Antenna Resonance)
            self.positions[:, -1] = 0.1 * self.positions[:, -1] + 0.9 * spiked_pos[-1]

        # 6. 마찰 감쇠 (Damping) - 지형적 평형 유도
        # 마찰을 줄여 더 강력한 결속을 허용
        self.positions *= 0.98

        # 7. 상태 백업 (node_data 및 전위 업데이트)
        for i, nid in enumerate(self.node_ids):
            self.node_data[nid].position = self.positions[i]
            self.node_data[nid].membrane_potential = float(self.membrane_potentials[i])

    @property
    def nodes(self) -> Dict[str, StructuralNode]:
        return self.node_data

    def get_equilibrium_state(self) -> Dict[str, Any]:
        return {nid: {"pos": self.node_data[nid].position.tolist(),
                      "mass": self.node_data[nid].mass,
                      "tensor": self.node_data[nid].tensor.tolist()}
                for nid in self.node_ids}
