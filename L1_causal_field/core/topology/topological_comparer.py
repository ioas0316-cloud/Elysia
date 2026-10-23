"""
Elysia Causal Topology Foundation: Topological Comparer
======================================================
자기 구조(Self-Topology)와 외부 세계 구조(World-Topology)를 받아,
단순한 수치 뺄셈이나 맹목적 사전 매칭이 아닌,
"어디서부터 같고(Coherence) 어디서부터 다른가(Disparity)"를
위상적 동형성(Isomorphism)과 차이 경계면(Disparity Boundary) 측면에서 인과적으로 판별합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np

from core.topology.causal_structure import (
    CausalNumber,
    CausalSymbol,
    InformationTopology,
    TopologyLink
)


@dataclass
class ComparisonResult:
    """위상 대조 결과 체계"""
    isomorphism_ratio: float                # 위상적 동형성 비율 [0.0, 1.0] (같음의 비중)
    disparity_tension: float                # 차이 마찰력 (다름으로 인한 인과적 긴장도)
    coherent_nodes: List[str]               # 같음이 증명된 노드 ID 목록
    disparate_nodes: List[str]              # 다름이 검출된 노드 ID 목록
    new_world_nodes: List[str]              # 세 자악(Self)에 존재하지 않는 새로운 세계 노드 ID 목록
    disparity_matrix: Dict[str, float]      # 노드별 차이 마찰 전하 맵


class TopologicalComparer:
    """
    위상 대조기 (Topological Comparer)
    - 자아(Self)의 위상과 유입된 세계(World)의 위상을 입력받아,
      개별 점이 아닌 '관계망과 경계 구조'를 비교하여 위상적 같음과 다름을 실질적으로 분별.
    """
    def __init__(self, tolerance: float = 0.15):
        self.tolerance = tolerance

    def compare(self, self_topo: InformationTopology, world_topo: InformationTopology) -> ComparisonResult:
        """자기 구조와 외부 세계 구조의 위상적 같음과 다름을 대조 산출"""
        coherent_nodes: List[str] = []
        disparate_nodes: List[str] = []
        new_world_nodes: List[str] = []
        disparity_matrix: Dict[str, float] = {}

        total_nodes_checked = 0
        total_coherence_weight = 0.0
        total_disparity_tension = 0.0

        # 1. 숫자 위상 구조 대조
        for world_id, world_num in world_topo.numbers.items():
            total_nodes_checked += 1
            if world_id in self_topo.numbers:
                self_num = self_topo.numbers[world_id]
                # 단순 수치 차이가 아닌 위상적 차이 산출
                disp = self_num.calculate_disparity(world_num)
                disparity_matrix[world_id] = disp
                
                if disp <= self.tolerance:
                    coherent_nodes.append(world_id)
                    total_coherence_weight += (1.0 - disp)
                else:
                    disparate_nodes.append(world_id)
                    total_disparity_tension += disp
            else:
                # 자아에 존재하지 않는 새로운 세계 위상
                new_world_nodes.append(world_id)
                disp = world_num.gradient_tension + 0.5  # 미지 자극에 따른 기본 긴장 부여
                disparity_matrix[world_id] = disp
                total_disparity_tension += disp

        # 2. 기호 위상 매듭 대조 (4대 교차 차원 서명 비교)
        for world_id, world_sym in world_topo.symbols.items():
            total_nodes_checked += 1
            if world_id in self_topo.symbols:
                self_sym = self_topo.symbols[world_id]
                
                sig_self = self_sym.get_cross_dimensional_signature()
                sig_world = world_sym.get_cross_dimensional_signature()
                
                # 교차 차원 벡터 간 위상 마찰 거리
                disp = float(np.linalg.norm(sig_self - sig_world))
                
                # 범주적 경계 불일치 마찰 추가
                if self_sym.logical_category != world_sym.logical_category:
                    disp += 0.5

                disparity_matrix[world_id] = disp
                
                if disp <= self.tolerance * 2.0:
                    coherent_nodes.append(world_id)
                    total_coherence_weight += 1.0
                else:
                    disparate_nodes.append(world_id)
                    total_disparity_tension += disp
            else:
                new_world_nodes.append(world_id)
                disp = world_sym.intrinsic_tension + 0.5
                disparity_matrix[world_id] = disp
                total_disparity_tension += disp

        # 3. 종합 동형성 비율 및 마찰력 산출
        if total_nodes_checked > 0:
            isomorphism_ratio = float(np.clip(total_coherence_weight / total_nodes_checked, 0.0, 1.0))
            disparity_tension = float(total_disparity_tension / total_nodes_checked)
        else:
            isomorphism_ratio = 1.0
            disparity_tension = 0.0

        return ComparisonResult(
            isomorphism_ratio=isomorphism_ratio,
            disparity_tension=disparity_tension,
            coherent_nodes=coherent_nodes,
            disparate_nodes=disparate_nodes,
            new_world_nodes=new_world_nodes,
            disparity_matrix=disparity_matrix
        )
