"""
Elysia Causal Topology Foundation: Causal Stem & Branch Engine
==============================================================
인간의 임의 수식(S_stem, Δθ_branch 등)이나 획일적 가중치 블랙박스를 배제하고,
외부 현상과 파편적 데이터를 '같음과 다름의 줄기와 가지(Homological Stem & Branch Topology)'라는
순수 정형 위상 해부학(Topological Parsing) 관점으로 해체, 대조, 환원하는 코어 엔진.

구체적 4단계 메커니즘:
1. 인과 궤적의 그래프 분해 (Trajectory Graph Decomposition): G = (V, E, C)
2. 위상 보존 맵핑을 통한 '줄기(Stem)' 적출 (Isomorphic Stem Extraction): f: V_A -> V_B
3. 위상 편차 이탈을 통한 '가지(Branch)' 분리 (Disparate Branch Isolation)
4. 궤적 연속성 검증 (Deterministic Continuity Check)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass(frozen=True)
class CausalNode:
    """인과 궤적의 상태 변화 지점 (V: Node)"""
    id: str
    role: str                          # e.g., 'input', 'transform', 'output', 'state'
    abstract_operation: str            # e.g., 'accumulate', 'transfer', 'filter', 'store'
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalEdge:
    """원인이 결과로 전환되는 명시적 메커니즘 (E: Edge)"""
    source_id: str
    target_id: str
    mechanism_type: str                # e.g., 'direct_flow', 'conditional_trigger', 'conservation_transfer'
    invariants: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryContext:
    """인과 변화가 일어난 매질 및 환경적 제약 조건 (C: Context)"""
    medium_type: str                   # e.g., 'python_code', 'physical_circuit', 'natural_language', 'tensor_matrix'
    environmental_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalTrajectoryGraph:
    """
    인과 궤적 그래프 G = (V, E, C)
    데이터나 현상이 품은 인과적 결(Viscosity/Connectivity)을 훼손하지 않고 담아내는 위상 그래프.
    """
    graph_id: str
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    context: TrajectoryContext = field(default_factory=lambda: TrajectoryContext(medium_type="raw_phenomenon"))

    def add_node(self, node: CausalNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: CausalEdge):
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            self.edges.append(edge)

    def get_successors(self, node_id: str) -> List[Tuple[str, CausalEdge]]:
        """특정 노드에서 출발하는 직결 이행 노드 및 메커니즘 엣지 목록"""
        result = []
        for edge in self.edges:
            if edge.source_id == node_id:
                result.append((edge.target_id, edge))
        return result

    def get_predecessors(self, node_id: str) -> List[Tuple[str, CausalEdge]]:
        """특정 노드로 들어오는 직결 원인 노드 및 메커니즘 엣지 목록"""
        result = []
        for edge in self.edges:
            if edge.target_id == node_id:
                result.append((edge.source_id, edge))
        return result


@dataclass
class HomologicalStem:
    """같음의 줄기 (Homological Stem): 매질을 초과하여 1:1 보존되는 근원적 인과 뼈대"""
    stem_id: str
    node_mapping_A_to_B: Dict[str, str]       # Node_A -> Node_B 1:1 사상 f
    node_mapping_B_to_A: Dict[str, str]       # Node_B -> Node_A 역사상 f^-1
    common_subgraph_A_node_ids: Set[str]
    common_subgraph_B_node_ids: Set[str]
    isomorphic_edges: List[Tuple[CausalEdge, CausalEdge]] # (Edge_A, Edge_B) 매칭 쌍


@dataclass
class DisparateBranch:
    """다름의 가지 (Disparate Branch): 환경/매질 변이 및 국소 특수성으로 격리된 궤적"""
    branch_id: str
    source_graph_id: str
    attached_stem_node_id: str                # 줄기 노드에 결합된 접점 노드 ID
    branch_node_ids: Set[str]                  # 가지로 분리된 국소 노드 집합
    branch_edges: List[CausalEdge]             # 가지 내부 엣지 집합
    variance_reason: str                      # e.g., 'medium_specific_representation', 'environmental_constraint'


@dataclass
class StemBranchParsingResult:
    """같음과 다름의 위상 해부학 파싱 결과"""
    stem: HomologicalStem
    branches_A: List[DisparateBranch]
    branches_B: List[DisparateBranch]
    is_continuous: bool                       # 궤적 연속성 검증 통과 여부
    discontinuity_reasons: List[str]          # 연속성 파기 구간 및 사유


class CausalStemBranchEngine:
    """
    인과 줄기-가지 위상 엔진 (Causal Stem & Branch Topology Engine)
    - 인간 임의의 가중치 수식이나 가짜 확률 매칭을 배제하고,
      두 인과 궤적 그래프 G_A, G_B 대조 시 위상 동형성(Isomorphism)과 연속성을 정밀 해부.
    """

    def __init__(self):
        pass

    def parse_stem_and_branches(
        self,
        graph_A: CausalTrajectoryGraph,
        graph_B: CausalTrajectoryGraph
    ) -> StemBranchParsingResult:
        """
        두 인과 궤적 그래프 G_A와 G_B를 대조하여:
        1. 1:1 동형 사상 f를 통해 '같음의 줄기(Stem)' 적출
        2. 사상에 포함되지 않고 남는 구조적 잔여물을 '다름의 가지(Branch)'로 격리
        3. 궤적 연속성(Continuity) 최종 검증
        """
        # 1. 1:1 동형 사상 f: V_A -> V_B 및 최대 공통 하위 그래프 적출
        stem = self._extract_homological_stem(graph_A, graph_B)

        # 2. 사상에서 이탈된 잔여 구조를 각 그래프별 '다름의 가지'로 분리
        branches_A = self._isolate_disparate_branches(graph_A, stem.common_subgraph_A_node_ids, is_graph_A=True)
        branches_B = self._isolate_disparate_branches(graph_B, stem.common_subgraph_B_node_ids, is_graph_A=False)

        # 3. 줄기 궤적의 필연적 연속성 검증 (Continuity Check)
        is_continuous, reasons = self._verify_trajectory_continuity(graph_A, graph_B, stem)

        return StemBranchParsingResult(
            stem=stem,
            branches_A=branches_A,
            branches_B=branches_B,
            is_continuous=is_continuous,
            discontinuity_reasons=reasons
        )

    def _extract_homological_stem(
        self,
        graph_A: CausalTrajectoryGraph,
        graph_B: CausalTrajectoryGraph
    ) -> HomologicalStem:
        """
        매질(medium_type) 차이를 도려내고,
        작동 역할(role), 추상 연산(abstract_operation), 인과 이행 메커니즘(mechanism_type)이
        1:1 완벽 보존되는 최대 공통 하위 그래프(Maximal Common Subgraph)를 구함.
        """
        mapping_A_to_B: Dict[str, str] = {}
        mapping_B_to_A: Dict[str, str] = {}
        matched_edges: List[Tuple[CausalEdge, CausalEdge]] = []

        # 1. 노드 1:1 사상 탐색 (역할 및 추상 연산적 동형성 기준)
        used_B_nodes: Set[str] = set()

        for node_A_id, node_A in graph_A.nodes.items():
            for node_B_id, node_B in graph_B.nodes.items():
                if node_B_id in used_B_nodes:
                    continue

                # 동형성 조건 1: 노드의 주체적 역할 및 추상적 연산 보존
                if (node_A.role == node_B.role and
                    node_A.abstract_operation == node_B.abstract_operation):
                    mapping_A_to_B[node_A_id] = node_B_id
                    mapping_B_to_A[node_B_id] = node_A_id
                    used_B_nodes.add(node_B_id)
                    break

        # 2. 엣지 동형성 검증 (u -> v 이행이 f(u) -> f(v) 이행과 동일 메커니즘으로 대응되는가)
        stem_A_nodes: Set[str] = set()
        stem_B_nodes: Set[str] = set()

        for edge_A in graph_A.edges:
            src_A, tgt_A = edge_A.source_id, edge_A.target_id
            if src_A in mapping_A_to_B and tgt_A in mapping_A_to_B:
                src_B = mapping_A_to_B[src_A]
                tgt_B = mapping_A_to_B[tgt_A]

                # B 그래프에서 corresponding edge 탐색
                for edge_B in graph_B.edges:
                    if (edge_B.source_id == src_B and
                        edge_B.target_id == tgt_B and
                        edge_B.mechanism_type == edge_A.mechanism_type):
                        matched_edges.append((edge_A, edge_B))
                        stem_A_nodes.add(src_A)
                        stem_A_nodes.add(tgt_A)
                        stem_B_nodes.add(src_B)
                        stem_B_nodes.add(tgt_B)
                        break

        # stem에 속한 노드만 사상 필터링
        filtered_A_to_B = {k: v for k, v in mapping_A_to_B.items() if k in stem_A_nodes}
        filtered_B_to_A = {v: k for k, v in filtered_A_to_B.items()}

        # 노드가 전혀 엣지 없이 단독 동형 노드로만 존재하는 경우(단일 상태 궤적) 수용
        if not stem_A_nodes and mapping_A_to_B:
            stem_A_nodes = set(mapping_A_to_B.keys())
            stem_B_nodes = set(mapping_A_to_B.values())
            filtered_A_to_B = dict(mapping_A_to_B)
            filtered_B_to_A = dict(mapping_B_to_A)

        return HomologicalStem(
            stem_id=f"stem_{graph_A.graph_id}_{graph_B.graph_id}",
            node_mapping_A_to_B=filtered_A_to_B,
            node_mapping_B_to_A=filtered_B_to_A,
            common_subgraph_A_node_ids=stem_A_nodes,
            common_subgraph_B_node_ids=stem_B_nodes,
            isomorphic_edges=matched_edges
        )

    def _isolate_disparate_branches(
        self,
        graph: CausalTrajectoryGraph,
        stem_node_ids: Set[str],
        is_graph_A: bool
    ) -> List[DisparateBranch]:
        """
        줄기(Stem)에 속하지 않는 잔여 노드 및 엣지들을
        줄기와 결합된 접점(attachment point)을 기준으로 '다름의 가지'로 격리.
        """
        branches: List[DisparateBranch] = []
        visited_nodes: Set[str] = set()

        non_stem_nodes = set(graph.nodes.keys()) - stem_node_ids

        for node_id in non_stem_nodes:
            if node_id in visited_nodes:
                continue

            # 이 가지 노드가 어느 줄기 노드에 결합되어 있는지 탐색
            attached_stem_id = None

            # predecessors에서 stem 노드 탐색
            for pred_id, _ in graph.get_predecessors(node_id):
                if pred_id in stem_node_ids:
                    attached_stem_id = pred_id
                    break

            # successors에서 stem 노드 탐색
            if not attached_stem_id:
                for succ_id, _ in graph.get_successors(node_id):
                    if succ_id in stem_node_ids:
                        attached_stem_id = succ_id
                        break

            # 줄기와 완전 고립된 노드일 경우 기본 연결 노드로 지정
            if not attached_stem_id:
                attached_stem_id = "unattached_root"

            # BFS/DFS로 해당 가지 궤적의 모든 미사상 노드 및 엣지 수집
            branch_node_set: Set[str] = set()
            branch_edge_list: List[CausalEdge] = []

            queue = [node_id]
            branch_node_set.add(node_id)
            visited_nodes.add(node_id)

            while queue:
                curr = queue.pop(0)

                # 연결 엣지 및 인접 노드 탐색
                for next_id, edge in graph.get_successors(curr):
                    if next_id not in stem_node_ids and next_id not in branch_node_set:
                        branch_node_set.add(next_id)
                        visited_nodes.add(next_id)
                        queue.append(next_id)
                    if edge not in branch_edge_list:
                        branch_edge_list.append(edge)

                for prev_id, edge in graph.get_predecessors(curr):
                    if prev_id not in stem_node_ids and prev_id not in branch_node_set:
                        branch_node_set.add(prev_id)
                        visited_nodes.add(prev_id)
                        queue.append(prev_id)
                    if edge not in branch_edge_list:
                        branch_edge_list.append(edge)

            branches.append(DisparateBranch(
                branch_id=f"branch_{graph.graph_id}_{len(branches)+1}",
                source_graph_id=graph.graph_id,
                attached_stem_node_id=attached_stem_id,
                branch_node_ids=branch_node_set,
                branch_edges=branch_edge_list,
                variance_reason=f"medium_variant_{graph.context.medium_type}"
            ))

        return branches

    def _verify_trajectory_continuity(
        self,
        graph_A: CausalTrajectoryGraph,
        graph_B: CausalTrajectoryGraph,
        stem: HomologicalStem
    ) -> Tuple[bool, List[str]]:
        """
        추출된 줄기 궤적의 연속성(Continuity) 검증:
        입력(input) 상태 노드부터 출발하여 줄기 궤적을 따라 추적할 때,
        논리적 단절이나 우연적 확률 비약 없이 출력(output) 상태 노드까지 필연적으로 이행하는지 확인.
        """
        reasons: List[str] = []

        if not stem.common_subgraph_A_node_ids:
            return False, ["Zero homological stem nodes found between trajectories."]

        # 1. 원본 그래프(G_A)에 존재하는 인과 흐름 전체 노드 탐색
        input_nodes_A = [n_id for n_id, n_obj in graph_A.nodes.items() if n_obj.role == 'input']
        output_nodes_A = [n_id for n_id, n_obj in graph_A.nodes.items() if n_obj.role == 'output']

        # 만약 원본 그래프에 input/output 역할 노드가 존재하나, Stem 줄기에는 도달하지 못했거나 연결이 차단된 경우 단절로 검증
        for out_a in output_nodes_A:
            if out_a not in stem.common_subgraph_A_node_ids:
                reasons.append(f"Discontinuity detected: Output node '{out_a}' is missing from the isomorphic stem trajectory.")

        # 2. 줄기 내 이행 이음새(Edge) 간의 연속적 가용성 검증
        for src_A in input_nodes_A:
            if src_A not in stem.common_subgraph_A_node_ids:
                reasons.append(f"Discontinuity detected: Input node '{src_A}' is missing from the isomorphic stem trajectory.")
                continue

            visited = set()
            stack = [src_A]
            reached_output = False

            while stack:
                curr = stack.pop()
                if curr in output_nodes_A:
                    reached_output = True
                    break
                visited.add(curr)

                # stem 엣지 중 curr에서 출발하는 엣지 탐색
                for edge_A, _ in stem.isomorphic_edges:
                    if edge_A.source_id == curr and edge_A.target_id not in visited:
                        stack.append(edge_A.target_id)

            if not reached_output and output_nodes_A:
                reasons.append(f"Discontinuity detected: Trajectory from input node '{src_A}' does not reach output in Stem graph.")

        is_continuous = len(reasons) == 0
        return is_continuous, reasons
