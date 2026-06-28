import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
    """
    def __init__(self, dimensions: int = 6):
        self.dimensions = dimensions
        self.nodes: Dict[str, StructuralNode] = {}
        self.G = 0.5  # Universal Structural Gravitational Constant

    def add_node(self, node_id: str, raw_content: bytes, structural_tensor: List[float]):
        """
        데이터를 우주(중력장)에 던져 넣습니다.
        구조적 텐서가 이미 이 데이터의 질량과 인력을 결정할 모든 정보를 갖고 있습니다.
        """
        tensor = np.array(structural_tensor, dtype=np.float32)
        
        # 질량(Mass)의 자율적 발견: 데이터의 엔트로피 밀도 (tensor[0])
        entropy = float(tensor[0])
        mass = max(0.1, entropy) # 무질서도가 높을수록/정보가 많을수록 질량이 큼

        # 초기 위치는 무작위로 할당되나, 이후 자기장(구조적 공명)에 의해 정렬됩니다.
        position = np.random.randn(self.dimensions).astype(np.float32)

        node = StructuralNode(id=node_id, raw_content=raw_content, tensor=tensor, mass=mass, position=position)
        self.nodes[node_id] = node

    def calculate_attraction(self, node_a_id: str, node_b_id: str) -> np.ndarray:
        """
        두 노드 간의 중력 벡터를 계산합니다.
        F = G * (m1 * m2 * Resonance) / r^2
        """
        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]

        direction = node_b.position - node_a.position
        distance = np.linalg.norm(direction)

        softening = 0.1
        if distance < 0.0001:
            return np.zeros(self.dimensions)

        # ── 핵심: 구조적 공명 (Structural Resonance) 발견 ──
        # 두 데이터가 얼마나 비슷한 주파수(Frequency)와 위상 곡률(Gradient)을 가졌는가?
        # tensor[1:4] = Frequencies, tensor[4:6] = Gradients
        vec_a = node_a.tensor[1:]
        vec_b = node_b.tensor[1:]
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            resonance = 0.1
        else:
            # 코사인 유사도를 공명도(0 ~ 1)로 사용
            cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            resonance = max(0.01, float(cos_sim))
            
            # 구조가 완벽히 같으면(공명) 폭발적인 인력 발생
            if resonance > 0.95:
                resonance *= 10.0

        force_magnitude = self.G * (node_a.mass * node_b.mass * resonance) / (distance**2 + softening)
        return (direction / distance) * force_magnitude

    def step(self, dt: float = 0.1):
        """중력장 시뮬레이션 한 스텝 진행 (자연 정렬)"""
        forces = {nid: np.zeros(self.dimensions) for nid in self.nodes}

        ids = list(self.nodes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                f = self.calculate_attraction(ids[i], ids[j])
                forces[ids[i]] += f
                forces[ids[j]] -= f

        # Update positions
        for nid, node in self.nodes.items():
            acceleration = forces[nid] / node.mass
            node.position += acceleration * dt
            
            # 마찰 감쇠(Damping)를 통해 군집(Constellation)을 형성하며 평형에 도달하게 함
            node.position *= 0.90

    def get_equilibrium_state(self) -> Dict[str, Any]:
        return {nid: {"pos": node.position.tolist(), "mass": node.mass, "tensor": node.tensor.tolist()}
                for nid, node in self.nodes.items()}
