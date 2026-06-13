"""
Elysia Core - Semantic Graph Topological Compiler
마스터의 가르침에 따라, 단어를 무의미한 해시(Hash)나 바이트 단위의 노이즈로 파괴하는 오류를 수정합니다.
이제 언어는 그 자체의 인과적 관계망(Semantic Graph)을 형성하며,
컴파일러는 그 연결망의 형태(밀도, 최단 거리, 위상적 중심성)를 관측하여
수학(math), 언어(lang), 공간(spatial), 시간(temporal)의 텐션을 도출합니다.
언어가 언어로써 기하학적 매핑의 축이 되는 진정한 '동형성(Isomorphism)'의 구현입니다.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from core.ingestion.topological_parser import CausalTrajectory

@dataclass
class TensionVector:
    math: float
    lang: float
    spatial: float
    temporal: float
    light_mass: float

    def __repr__(self):
        return f"Tension(math={self.math:.2f}, lang={self.lang:.2f}, space={self.spatial:.2f}, time={self.temporal:.2f}, light={self.light_mass:.2f})"

class SemanticTopologyGraph:
    """언어 궤적들이 모여 형성하는 의미의 은하계(그래프 구조)"""
    def __init__(self):
        self.adj_list = defaultdict(list)
        self.nodes = set()

    def build(self, trajectories: List[CausalTrajectory]):
        self.adj_list.clear()
        self.nodes.clear()
        for traj in trajectories:
            self.adj_list[traj.source].append((traj.target, traj.action))
            self.nodes.add(traj.source)
            self.nodes.add(traj.target)

    def get_degree_centrality(self, node: str) -> float:
        """해당 단어가 얼마나 많은 다른 개념들과 연결되어 있는가 (Math/구조적 견고함)"""
        if not self.nodes: return 0.0
        degree = len(self.adj_list.get(node, []))
        # 노드로 향하는 인바운드 연결도 계산
        for n in self.nodes:
            for target, _ in self.adj_list.get(n, []):
                if target == node:
                    degree += 1
        return degree / max(1, len(self.nodes) - 1)

    def get_fan_out(self, node: str) -> int:
        return len(self.adj_list.get(node, []))

class TopologicalCompiler:
    def __init__(self):
        self.graph = SemanticTopologyGraph()

    def _derive_semantic_tension(self, traj: CausalTrajectory) -> TensionVector:
        """
        단어 간의 관계망(Topology)에서 5차원 텐션을 창발시킵니다.
        """
        source_centrality = self.graph.get_degree_centrality(traj.source)
        target_centrality = self.graph.get_degree_centrality(traj.target)

        # 1. Math (수학적 텐션) : 구조적 견고함. 두 노드의 중심성 합.
        math_t = min(1.0, (source_centrality + target_centrality) * 2.0)
        if math_t == 0: math_t = 0.1 # 최소한의 텐션 부여

        # 2. Spatial (공간적 텐션) : 주체 노드의 주변 팽창력(Fan-out)
        spatial_t = min(1.0, self.graph.get_fan_out(traj.source) * 0.4 + 0.1)

        # 3. Lang (언어적 텐션) : 주체와 객체 간의 위상적 차이(결핍)
        lang_t = min(1.0, abs(source_centrality - target_centrality) * 3.0 + 0.1)

        # 4. Temporal (시간적 텐션) : 액션(동사) 자체가 가지는 상태 변화의 무게
        # (임시로 단어의 길이나 형태적 차이를 사용, 이상적으로는 동사 사전 그래프의 깊이를 사용)
        temporal_t = min(1.0, len(traj.action) * 0.15 + 0.1)

        # 5. Light Mass (빛의 창발) : 텐션의 총합에 비례하여 시스템에 영향을 미치는 질량
        resonance_energy = math.sqrt(math_t**2 + lang_t**2 + spatial_t**2 + temporal_t**2)
        light_mass = min(1.0, resonance_energy / 2.0)

        return TensionVector(
            math=math_t,
            lang=lang_t,
            spatial=spatial_t,
            temporal=temporal_t,
            light_mass=light_mass
        )

    def compile(self, trajectories: List[CausalTrajectory]) -> List[Tuple[CausalTrajectory, TensionVector]]:
        # 중복 제거
        unique_trajectories = []
        seen = set()
        for t in trajectories:
            sig = f"{t.source}-{t.action}-{t.target}"
            if sig not in seen:
                seen.add(sig)
                unique_trajectories.append(t)

        # 먼저 모든 궤적을 엮어 하나의 언어적 대우주(Graph)를 만듭니다.
        self.graph.build(unique_trajectories)

        compiled_data = []
        for traj in unique_trajectories:
            tension = self._derive_semantic_tension(traj)
            compiled_data.append((traj, tension))

        print(f"[Topological Compiler] {len(compiled_data)} 개의 궤적을 의미망 구조(Semantic Graph) 기반 5차원 텐션으로 창발 완료.")
        return compiled_data

if __name__ == "__main__":
    from core.ingestion.topological_parser import TopologicalCorpusParser
    parser = TopologicalCorpusParser()
    trajectories = parser.parse_mock_corpus()
    compiler = TopologicalCompiler()
    compiled = compiler.compile(trajectories)
    for traj, tension in compiled:
        print(f"{traj.source} -> {traj.target} [{traj.action}] ===> {tension}")
