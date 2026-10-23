import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.utils.math_utils import popcount_vectorized

@dataclass
class HierarchicalNode:
    id: str
    genes: Dict[str, np.uint64] # micro, meso, macro
    scale: str
    position: np.ndarray = None
    mass: float = 1.0

class ResonantTensionEngine:
    """
    [Hierarchical Resonant Engine]
    '비트' 하나에만 매몰되지 않고, 바이트, KB, MB 단위의 거시적 서사들이
    동시에 공명하며 서로를 정렬하는 다층적 중력장입니다.

    상위 스케일(MACRO)의 공명은 하위 스케일(MESO, MICRO)에 '서사적 장력'을 부여하여
    하위의 미세한 불일치가 있더라도 거시적 흐름 안으로 끌어당깁니다.
    """
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        self.nodes: Dict[str, HierarchicalNode] = {}

    def add_node(self, node_id: str, genes: Dict[str, np.uint64], scale: str):
        pos = np.random.randn(self.dimensions).astype(np.float32)
        # 스케일에 따른 질량 부여 (거시적일수록 무거움)
        mass_map = {"MICRO": 1.0, "MESO": 10.0, "MACRO": 100.0}
        mass = mass_map.get(scale, 1.0)

        node = HierarchicalNode(id=node_id, genes=genes, scale=scale, position=pos, mass=mass)
        self.nodes[node_id] = node

    def step(self, dt: float = 0.1):
        if len(self.nodes) < 2: return

        node_list = list(self.nodes.values())
        n = len(node_list)
        positions = np.array([node.position for node in node_list], dtype=np.float32)
        masses = np.array([node.mass for node in node_list], dtype=np.float32).reshape(-1, 1)

        # 1. 다층적 공명 계산 (Multi-Scale Resonance)
        def get_res_matrix(scale_key: str):
            genes = np.array([node.genes[scale_key] for node in node_list], dtype=np.uint64)
            diff = np.bitwise_xor(genes[:, np.newaxis], genes[np.newaxis, :])
            deficit = popcount_vectorized(diff)
            return 1.0 - (deficit / 64.0)

        res_micro = get_res_matrix("micro")
        res_meso = get_res_matrix("meso")
        res_macro = get_res_matrix("macro")

        # [거시적 인도의 법칙]
        # 상위 차원의 공명이 높으면, 하위 차원의 차이는 무시될 수 있음
        total_resonance = (res_macro * 0.6 + res_meso * 0.3 + res_micro * 0.1)

        # 2. 물리적 변위 계산
        diff_pos = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dist_sq = np.sum(diff_pos**2, axis=-1)
        dist = np.sqrt(dist_sq + 1e-9)

        # 질량과 공명에 따른 인력
        force_mag = (masses @ masses.T) * (total_resonance**2) / (dist + 0.1)

        force_vec = force_mag[:, :, np.newaxis] * (diff_pos / (dist[:, :, np.newaxis] + 0.1))
        total_force = np.sum(force_vec, axis=1)

        # 3. 위치 업데이트
        new_positions = positions + (total_force / masses) * dt
        new_positions *= 0.9 # 강력한 댐핑으로 평형 유도

        for i, node in enumerate(node_list):
            node.position = new_positions[i]

    def get_state(self):
        return {nid: {"pos": n.position.tolist(), "scale": n.scale} for nid, n in self.nodes.items()}

# HierarchicalResonantEngine alias for compatibility
HierarchicalResonantEngine = ResonantTensionEngine
