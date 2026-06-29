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

        self.G = 0.5  # Universal Structural Gravitational Constant
        self.softening = 0.1
        self.damping = 0.90

    def add_node(self, node_id: str, raw_content: bytes, structural_tensor: List[float]):
        """데이터를 중력장에 주입하고 필드를 재구성합니다."""
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

    def step(self, dt: float = 0.1):
        """
        [Field Simultaneous Update]
        모든 노드 간의 상호작용을 단 한 번의 텐서 연산으로 해결합니다.
        """
        if len(self.node_ids) < 2:
            return

        # 1. 위치 차이 및 거리 계산 (N, N, D)
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
        
        geometries = self.tensors[:, 0:4]
        directions = self.tensors[:, 4:6]
        continuities = self.tensors[:, 6].reshape(-1, 1)
        attributes = self.tensors[:, 7].reshape(-1, 1)
        causal_densities = self.tensors[:, 8].reshape(-1, 1)
        
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

        # 6. 마찰 감쇠 (Damping) - 지형적 평형 유도
        # 마찰을 줄여 더 강력한 결속을 허용
        self.positions *= 0.98

        # 7. 상태 백업 (node_data 업데이트)
        for i, nid in enumerate(self.node_ids):
            self.node_data[nid].position = self.positions[i]

    def get_equilibrium_state(self) -> Dict[str, Any]:
        return {nid: {"pos": self.node_data[nid].position.tolist(),
                      "mass": self.node_data[nid].mass,
                      "tensor": self.node_data[nid].tensor.tolist()}
                for nid in self.node_ids}
