"""
Elysia CAD Kinematic Topological Parser (Mechanical Assembly Medium)
====================================================================
3D CAD 및 기구학 조립체(Part, Mates, Joint Constraints)의 설계 구속 조건을
죽은 폴리곤 메쉬나 부동소수점 물리 연산으로 퇴행시키지 않고,
실행 가능한 인과 격자 G = (V, E, C)로 변환하는 기구학 위상 파서.
"""

from typing import Dict, List, Set, Any, Optional, Union
from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    BaseTopologicalParser,
    INVARIANT_RIGID_BODY,
    INVARIANT_JOINT_REVOLUTE,
    INVARIANT_JOINT_PRISMATIC,
    INVARIANT_SURFACE_MATE,
    INVARIANT_GEAR_COUPLING
)


class CADKinematicParser(BaseTopologicalParser):
    """
    CAD 어셈블리 및 기구학적 구속 조건 모델을 인과 궤적 그래프 G = (V, E, C)로 변환하는 파서.
    """

    def __init__(self):
        super().__init__(medium_type="cad_assembly_kinematics")

    def parse(self, assembly_data: Dict[str, Any]) -> CausalGraph:
        return self.parse_assembly(assembly_data)

    def parse_assembly(self, assembly_data: Dict[str, Any]) -> CausalGraph:
        self.reset_counter()
        graph = CausalGraph(
            graph_id=assembly_data.get("assembly_name", "cad_assembly"),
            context=TrajectoryContext(
                medium_type=self.medium_type,
                environmental_constraints={"tolerance": assembly_data.get("global_tolerance", "0.01mm")}
            )
        )

        parts = assembly_data.get("parts", [])
        mates = assembly_data.get("mates", [])

        part_id_to_node_id: Dict[str, str] = {}

        # 1. 기구 부품(Rigid Body) 노드 등록
        for part in parts:
            p_id = part.get("id", f"part_{self._node_idx}")
            node_id = self._generate_node_id("N_PART")
            part_id_to_node_id[p_id] = node_id

            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=INVARIANT_RIGID_BODY,
                payload={
                    "part_name": part.get("name", p_id),
                    "mass": part.get("mass", 1.0),
                    "base_dof": 6
                }
            )
            graph.add_node(node)
            graph.context_constraints[node_id].add(f"cad_material:{part.get('material', 'generic')}")

        # 2. 기구학적 메이트(Mate) 및 구속조건을 인과 전이 엣지(ConnectivityBeam)로 사상
        for mate in mates:
            parent_p = mate.get("parent_part_id")
            child_p = mate.get("child_part_id")

            if parent_p not in part_id_to_node_id or child_p not in part_id_to_node_id:
                continue

            src_node_id = part_id_to_node_id[parent_p]
            tgt_node_id = part_id_to_node_id[child_p]

            mate_type = mate.get("type", "revolute").lower()
            if mate_type in ("revolute", "hinge", "pivot"):
                dof_sig = INVARIANT_JOINT_REVOLUTE
                dof_desc = "DOF_TRANSFER:1_ROTATION"
            elif mate_type in ("prismatic", "slider", "linear_rail"):
                dof_sig = INVARIANT_JOINT_PRISMATIC
                dof_desc = "DOF_TRANSFER:1_TRANSLATION"
            elif mate_type in ("gear", "coupling"):
                dof_sig = INVARIANT_GEAR_COUPLING
                ratio = mate.get("ratio", 1.0)
                dof_desc = f"DOF_TRANSFER:GEAR_RATIO_{ratio}"
            else:
                dof_sig = INVARIANT_SURFACE_MATE
                dof_desc = "DOF_TRANSFER:SURFACE_LOCK"

            # 구속 엣지 연결
            graph.add_edge(CausalEdge(
                source_id=src_node_id,
                target_id=tgt_node_id,
                precondition=dof_desc,
                is_necessary=True
            ))

            # 가동 범위 및 공차 한계는 맥락 제약(C)으로 엄정 격리
            if "range_limit" in mate:
                r_min, r_max = mate["range_limit"]
                graph.context_constraints[tgt_node_id].add(f"limit_range:[{r_min},{r_max}]")
            if "tolerance" in mate:
                graph.context_constraints[tgt_node_id].add(f"clearance_tolerance:{mate['tolerance']}")

        return graph
