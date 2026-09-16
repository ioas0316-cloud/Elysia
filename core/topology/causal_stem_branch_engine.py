"""
Elysia Causal Topology Foundation: Causal Stem & Branch Topology Engine
=======================================================================
인간의 임의 수식(S_stem, Δθ_branch 등)이나 획일적 가중치 블랙박스를 배제하고,
외부 현상, 코드, 수식, 하드웨어 데이터를 '같음과 다름의 줄기와 가지(Homological Stem & Branch Topology)'라는
순수 정형 위상 해부학(Topological Parsing) 관점으로 해체, 대조, 환원하는 코어 엔진.

표준 인과 위상 그래프 규격:
G = (V, E, C)
- V: CausalNode (불변 인과 서명 Invariant Signature, 노드 타입 STEM/BRANCH/CONTEXT)
- E: CausalEdge (선행조건 Precondition, 필수성 Is Necessary)
- C: Context Constraints (매질/구문/공차 격리 제약)
"""

from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union


class NodeType(Enum):
    STEM = "STEM"          # 보존되는 불변의 핵심 인과 줄기 (Invariant Core)
    BRANCH = "BRANCH"      # 매질/환경 변이로 이탈된 국소 가지 (Disparate Branch)
    CONTEXT = "CONTEXT"    # 구문적/환경적 제약 조건 (Medium Context)


@dataclass
class CausalNode:
    """인과 궤적의 상태 및 전이 노드 (V: Node)"""
    node_id: str = ""
    node_type: NodeType = NodeType.STEM
    invariant_signature: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    # 하위 호환성 (Legacy compatibility) 필드
    id: Optional[str] = None
    role: Optional[str] = None
    abstract_operation: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id and self.id:
            self.node_id = self.id
        if not self.id and self.node_id:
            self.id = self.node_id
        if not self.invariant_signature and self.abstract_operation:
            self.invariant_signature = self.abstract_operation
        elif not self.abstract_operation and self.invariant_signature:
            self.abstract_operation = self.invariant_signature
        if not self.role:
            self.role = self.node_type.value.lower()


@dataclass
class CausalEdge:
    """원인이 결과로 전환되는 명시적 인과 메커니즘 (E: Edge)"""
    source_id: str = ""
    target_id: str = ""
    precondition: str = "seq_flow"
    is_necessary: bool = True

    # 하위 호환성 필드
    mechanism_type: Optional[str] = None
    invariants: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mechanism_type and self.precondition == "seq_flow":
            self.precondition = self.mechanism_type
        elif not self.mechanism_type:
            self.mechanism_type = self.precondition


@dataclass
class TrajectoryContext:
    """인과 변화가 일어난 매질 및 환경적 제약 조건 (C: Context)"""
    medium_type: str = "raw_phenomenon"
    environmental_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalGraph:
    """
    인과 위상 그래프 G = (V, E, C)
    데이터나 현상이 품은 인과적 결을 훼손하지 않고 담아내는 표준 위상 그래프.
    """
    graph_id: str = "causal_graph"
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    context_constraints: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    context: TrajectoryContext = field(default_factory=TrajectoryContext)

    def add_node(self, node: CausalNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self.context_constraints:
            self.context_constraints[node.node_id] = set()

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

    def clone(self) -> 'CausalGraph':
        cloned = CausalGraph(
            graph_id=self.graph_id,
            context=TrajectoryContext(
                medium_type=self.context.medium_type,
                environmental_constraints=dict(self.context.environmental_constraints)
            )
        )
        for n_id, node in self.nodes.items():
            cloned.nodes[n_id] = CausalNode(
                node_id=node.node_id,
                node_type=node.node_type,
                invariant_signature=node.invariant_signature,
                payload=dict(node.payload),
                id=node.id,
                role=node.role,
                abstract_operation=node.abstract_operation,
                properties=dict(node.properties)
            )
        for edge in self.edges:
            cloned.edges.append(CausalEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                precondition=edge.precondition,
                is_necessary=edge.is_necessary,
                mechanism_type=edge.mechanism_type,
                invariants=dict(edge.invariants)
            ))
        for n_id, c_set in self.context_constraints.items():
            cloned.context_constraints[n_id] = set(c_set)
        return cloned


# 레거시 명칭 별칭
CausalTrajectoryGraph = CausalGraph


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
    variance_reason: str                      # e.g., 'medium_specific_representation'


@dataclass
class StemBranchParsingResult:
    """같음과 다름의 위상 해부학 파싱 결과"""
    stem: HomologicalStem
    branches_A: List[DisparateBranch]
    branches_B: List[DisparateBranch]
    is_continuous: bool                       # 궤적 연속성 검증 통과 여부
    discontinuity_reasons: List[str]          # 연속성 파기 구간 및 사유


class CausalStemBranchTopologyEngine:
    """
    인과 줄기-가지 위상 엔진 (Causal Stem & Branch Topology Engine)
    - 인간 임의의 가중치 수식이나 가짜 확률 매칭을 배제하고,
      두 인과 궤적 그래프 G_A, G_B 대조 시 위상 동형성(Isomorphism)과 연속성을 정밀 해부.
    """

    def extract_stem_and_branch(
        self,
        reference_graph: CausalGraph,
        target_graph: CausalGraph
    ) -> Tuple[CausalGraph, Dict[str, str]]:
        """
        두 이종 매질 그래프 간의 1:1 동형성 사상을 대조하여,
        target_graph의 각 노드를 STEM / BRANCH로 라벨링하고,
        target_id -> reference_id 1:1 사상 맵(iso_map)을 반환.
        """
        analyzed_target = target_graph.clone()
        iso_map: Dict[str, str] = {}
        used_ref_ids: Set[str] = set()

        # 1. 시그니처 1:1 매칭
        for tgt_id, tgt_node in analyzed_target.nodes.items():
            tgt_sig = tgt_node.invariant_signature
            for ref_id, ref_node in reference_graph.nodes.items():
                if ref_id in used_ref_ids:
                    continue
                ref_sig = ref_node.invariant_signature
                if tgt_sig == ref_sig and tgt_sig != "":
                    iso_map[tgt_id] = ref_id
                    used_ref_ids.add(ref_id)
                    tgt_node.node_type = NodeType.STEM
                    break

        # 2. 사상되지 않은 노드는 BRANCH로 설정
        for tgt_id, tgt_node in analyzed_target.nodes.items():
            if tgt_id not in iso_map:
                tgt_node.node_type = NodeType.BRANCH

        return analyzed_target, iso_map

    def parse_stem_and_branches(
        self,
        graph_A: CausalGraph,
        graph_B: CausalGraph
    ) -> StemBranchParsingResult:
        """
        두 인과 궤적 그래프 G_A와 G_B를 대조하여:
        1. 1:1 동형 사상 f를 통해 '같음의 줄기(Stem)' 적출
        2. 사상에 포함되지 않고 남는 구조적 잔여물을 '다름의 가지(Branch)'로 격리
        3. 궤적 연속성(Continuity) 최종 검증
        """
        stem = self._extract_homological_stem(graph_A, graph_B)
        branches_A = self._isolate_disparate_branches(graph_A, stem.common_subgraph_A_node_ids, is_graph_A=True)
        branches_B = self._isolate_disparate_branches(graph_B, stem.common_subgraph_B_node_ids, is_graph_A=False)
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
        graph_A: CausalGraph,
        graph_B: CausalGraph
    ) -> HomologicalStem:
        mapping_A_to_B: Dict[str, str] = {}
        mapping_B_to_A: Dict[str, str] = {}
        matched_edges: List[Tuple[CausalEdge, CausalEdge]] = []

        used_B_nodes: Set[str] = set()

        for node_A_id, node_A in graph_A.nodes.items():
            for node_B_id, node_B in graph_B.nodes.items():
                if node_B_id in used_B_nodes:
                    continue

                sig_match = (
                    bool(node_A.invariant_signature) and 
                    node_A.invariant_signature == node_B.invariant_signature
                )
                legacy_match = (
                    bool(node_A.role) and bool(node_B.role) and
                    node_A.role == node_B.role and
                    node_A.abstract_operation == node_B.abstract_operation
                )

                if sig_match or legacy_match:
                    mapping_A_to_B[node_A_id] = node_B_id
                    mapping_B_to_A[node_B_id] = node_A_id
                    used_B_nodes.add(node_B_id)
                    break

        stem_A_nodes: Set[str] = set()
        stem_B_nodes: Set[str] = set()

        for edge_A in graph_A.edges:
            src_A, tgt_A = edge_A.source_id, edge_A.target_id
            if src_A in mapping_A_to_B and tgt_A in mapping_A_to_B:
                src_B = mapping_A_to_B[src_A]
                tgt_B = mapping_A_to_B[tgt_A]

                for edge_B in graph_B.edges:
                    if (edge_B.source_id == src_B and
                        edge_B.target_id == tgt_B and
                        (edge_B.precondition == edge_A.precondition or 
                         edge_B.mechanism_type == edge_A.mechanism_type)):
                        matched_edges.append((edge_A, edge_B))
                        stem_A_nodes.add(src_A)
                        stem_A_nodes.add(tgt_A)
                        stem_B_nodes.add(src_B)
                        stem_B_nodes.add(tgt_B)
                        break

        filtered_A_to_B = {k: v for k, v in mapping_A_to_B.items() if k in stem_A_nodes}
        filtered_B_to_A = {v: k for k, v in filtered_A_to_B.items()}

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
        graph: CausalGraph,
        stem_node_ids: Set[str],
        is_graph_A: bool
    ) -> List[DisparateBranch]:
        branches: List[DisparateBranch] = []
        visited_nodes: Set[str] = set()
        non_stem_nodes = set(graph.nodes.keys()) - stem_node_ids

        for node_id in non_stem_nodes:
            if node_id in visited_nodes:
                continue

            attached_stem_id = None
            for pred_id, _ in graph.get_predecessors(node_id):
                if pred_id in stem_node_ids:
                    attached_stem_id = pred_id
                    break

            if not attached_stem_id:
                for succ_id, _ in graph.get_successors(node_id):
                    if succ_id in stem_node_ids:
                        attached_stem_id = succ_id
                        break

            if not attached_stem_id:
                attached_stem_id = "unattached_root"

            branch_node_set: Set[str] = set()
            branch_edge_list: List[CausalEdge] = []
            queue = [node_id]
            branch_node_set.add(node_id)
            visited_nodes.add(node_id)

            while queue:
                curr = queue.pop(0)
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
        graph_A: CausalGraph,
        graph_B: CausalGraph,
        stem: HomologicalStem
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        if not stem.common_subgraph_A_node_ids:
            return False, ["Zero homological stem nodes found between trajectories."]

        input_nodes_A = [n_id for n_id, n_obj in graph_A.nodes.items() if n_obj.role in ('input', 'stem')]
        output_nodes_A = [n_id for n_id, n_obj in graph_A.nodes.items() if n_obj.role in ('output', 'stem')]

        for out_a in output_nodes_A:
            if out_a not in stem.common_subgraph_A_node_ids:
                reasons.append(f"Discontinuity detected: Output node '{out_a}' is missing from the isomorphic stem trajectory.")

        for src_A in input_nodes_A:
            if src_A not in stem.common_subgraph_A_node_ids:
                reasons.append(f"Discontinuity detected: Input node '{src_A}' is missing from the isomorphic stem trajectory.")
                continue

            visited = set()
            stack = [src_A]
            reached_output = False

            while stack:
                curr = stack.pop()
                if curr in output_nodes_A and curr != src_A:
                    reached_output = True
                    break
                visited.add(curr)
                for edge_A, _ in stem.isomorphic_edges:
                    if edge_A.source_id == curr and edge_A.target_id not in visited:
                        stack.append(edge_A.target_id)

            if not reached_output and output_nodes_A and len(stem.isomorphic_edges) > 0:
                reasons.append(f"Discontinuity detected: Trajectory from input node '{src_A}' does not reach output in Stem graph.")

        is_continuous = len(reasons) == 0
        return is_continuous, reasons


# 레거시 클래스명 호환 별칭
CausalStemBranchEngine = CausalStemBranchTopologyEngine
