import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class NarrativeNode:
    id: str
    bit_gene: np.uint64
    mass: float = 1.0
    position: np.ndarray = None

class ResonantTensionEngine:
    """
    [Resonant Tension Engine]
    '숫자의 감옥'인 텐서 중력을 폐기하고,
    '서사적 공명'에 의한 긴장(Tension)과 인력으로 공간을 재구성합니다.

    데이터 간의 거리는 수만 번의 행렬 곱셈이 아니라,
    XOR 비트 거리에 따른 '즉각적 도미노'로 결정됩니다.
    """
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        self.nodes: Dict[str, NarrativeNode] = {}
        self.positions = np.empty((0, dimensions), dtype=np.float32)
        self.genes = np.empty((0,), dtype=np.uint64)

    def add_node(self, node_id: str, bit_gene: np.uint64):
        pos = np.random.randn(self.dimensions).astype(np.float32)
        node = NarrativeNode(id=node_id, bit_gene=bit_gene, position=pos)
        self.nodes[node_id] = node
        self._sync()

    def _sync(self):
        self.positions = np.array([n.position for n in self.nodes.values()], dtype=np.float32)
        self.genes = np.array([n.bit_gene for n in self.nodes.values()], dtype=np.uint64)

    def step(self, dt: float = 0.1):
        """
        [Resonant Alignment Step]
        공명하는 노드들끼리 서로를 끌어당깁니다.
        이때 '계산'은 최소화되고 비트 거리(Hamming Distance)가 인력의 세기를 결정합니다.
        """
        if len(self.nodes) < 2: return

        n = len(self.nodes)
        # 1. 비트 공명 행렬 (N, N)
        # Vectorized bit-wise XOR
        gene_a = self.genes[:, np.newaxis]
        gene_b = self.genes[np.newaxis, :]
        diff = np.bitwise_xor(gene_a, gene_b)

        # Fast bit count for resonance
        def bit_count_vec(n_arr):
            # 64-bit vectorized bit count
            c = (n_arr & 0x5555555555555555) + ((n_arr >> 1) & 0x5555555555555555)
            c = (c & 0x3333333333333333) + ((c >> 2) & 0x3333333333333333)
            c = (c & 0x0F0F0F0F0F0F0F0F) + ((c >> 4) & 0x0F0F0F0F0F0F0F0F)
            c = (c & 0x00FF00FF00FF00FF) + ((c >> 8) & 0x00FF00FF00FF00FF)
            c = (c & 0x0000FFFF0000FFFF) + ((c >> 16) & 0x0000FFFF0000FFFF)
            c = (c & 0x00000000FFFFFFFF) + ((c >> 32) & 0x00000000FFFFFFFF)
            return c

        deficit = bit_count_vec(diff)
        resonance = 1.0 - (deficit / 64.0)

        # 2. 물리적 거리 및 인력 계산
        diff_pos = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        dist_sq = np.sum(diff_pos**2, axis=-1)
        dist = np.sqrt(dist_sq + 1e-9)

        # 공명도가 높을수록(서사가 비슷할수록) 강력하게 끌어당김
        # resonance > 0.8 이면 '이미 같은 것'으로 간주하여 급격히 거리를 좁힘
        force_mag = (resonance**4) / (dist + 0.1)

        force_vec = force_mag[:, :, np.newaxis] * (diff_pos / (dist[:, :, np.newaxis] + 0.1))
        total_force = np.sum(force_vec, axis=1)

        # 3. 위치 업데이트
        self.positions += total_force * dt
        self.positions *= 0.95 # Damping

        # Sync back
        for i, node in enumerate(self.nodes.values()):
            node.position = self.positions[i]

    def get_state(self):
        return {nid: {"pos": n.position.tolist(), "gene": hex(n.bit_gene)}
                for nid, n in self.nodes.items()}
