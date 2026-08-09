"""
Elysia Meta-Algorithm Foundation: Algorithm Tensor Representation
================================================================
추적 알고리즘(실행 파이프라인)의 각 연산 단계와 데이터 흐름을
행렬(Adjacency Matrix) 및 노드 속성 텐서(Node Feature Tensor) 자료구조로 인코딩하는 메타 표현 모듈입니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np


@dataclass
class PipelineNode:
    """파이프라인 내 개별 연산 단계 노드"""
    id: str
    op_type: str                  # 연산 유형 (e.g., 'input', 'transform', 'filter', 'output')
    param_vector: np.ndarray      # 연산 파라미터 벡터


class AlgorithmTensor:
    """
    알고리즘 텐서 (Algorithm Tensor)
    - 파이프라인 노드 간의 데이터 흐름 관계를 인접 행렬(Adjacency Matrix)로 관리하고,
      각 연산 노드의 속성을 텐서 행렬로 인코딩한 메타 데이터 구조.
    """
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, PipelineNode] = {}
        self.node_order: List[str] = []
        self.adjacency_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.feature_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add_node(self, node_id: str, op_type: str, param_vector: np.ndarray):
        """파이프라인 연산 노드 추가"""
        self.nodes[node_id] = PipelineNode(
            id=node_id,
            op_type=op_type,
            param_vector=np.array(param_vector, dtype=np.float32)
        )
        self.node_order.append(node_id)
        self._rebuild_matrices()

    def add_connection(self, source_id: str, target_id: str, weight: float = 1.0):
        """노드 간 데이터 흐름(연결 관계) 설정"""
        if source_id in self.nodes and target_id in self.nodes:
            src_idx = self.node_order.index(source_id)
            tgt_idx = self.node_order.index(target_id)
            self.adjacency_matrix[src_idx, tgt_idx] = weight

    def _rebuild_matrices(self):
        """노드 추가에 따른 인접 행렬 및 노드 속성 텐서 재구성"""
        n = len(self.node_order)
        new_adj = np.zeros((n, n), dtype=np.float32)
        if self.adjacency_matrix.size > 0:
            old_n = self.adjacency_matrix.shape[0]
            new_adj[:old_n, :old_n] = self.adjacency_matrix
        self.adjacency_matrix = new_adj

        # 노드 속성 텐서 재구성
        if n > 0:
            max_param_len = max(len(node.param_vector) for node in self.nodes.values())
            feature_mat = np.zeros((n, max_param_len + 1), dtype=np.float32)
            for idx, node_id in enumerate(self.node_order):
                node = self.nodes[node_id]
                # 연산 유형의 해시값을 첫 번째 열로 기록
                op_hash = float(abs(hash(node.op_type)) % 1000) / 1000.0
                feature_mat[idx, 0] = op_hash
                feature_mat[idx, 1:1 + len(node.param_vector)] = node.param_vector
            self.feature_matrix = feature_mat
