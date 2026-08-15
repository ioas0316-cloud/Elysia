import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple
from core.memory.state_dag import StateDAGManager, StateNode, PhysicalStateSlabPool

class CounterfactualBranchingEngine:
    """
    [Counterfactual Branching Engine via do(X)]
    """
    def __init__(self, dag_manager: StateDAGManager):
        self.dag_manager = dag_manager

    def apply_do_operator(
        self,
        variable: str,
        value: Any,
        affected_dimensions: Optional[List[int]] = None
    ) -> StateNode:
        """
        직교적 개입 do(X = x') 적용 및 가상 관측 자식 노드 생성.
        비트마스크를 먼저 계산하여 do_intervention 시 슬래브 할당에 바로 전달합니다.
        """
        bitmask = None
        if affected_dimensions is not None:
            mask = 0
            for dim in affected_dimensions:
                if 0 <= dim < 64:
                    mask |= (1 << dim)
            bitmask = mask if mask != 0 else None

        intervened_node = self.dag_manager.do_intervention(
            variable=variable,
            value=value,
            custom_bitmask=bitmask
        )
        return intervened_node

    def compute_causal_cone_delta(
        self,
        baseline_node_id: str,
        intervened_node_id: str
    ) -> Dict[str, Any]:
        if baseline_node_id not in self.dag_manager.nodes or intervened_node_id not in self.dag_manager.nodes:
            raise ValueError("Invalid node IDs provided for causal cone computation.")

        base_node = self.dag_manager.nodes[baseline_node_id]
        intervened_node = self.dag_manager.nodes[intervened_node_id]

        base_state = base_node.get_state_chain()
        intervened_state = intervened_node.get_state_chain()

        cone_delta = {}
        all_keys = set(base_state.keys()).union(set(intervened_state.keys()))

        for key in all_keys:
            val_base = base_state.get(key)
            val_inter = intervened_state.get(key)

            if val_base != val_inter:
                cone_delta[key] = {
                    "baseline": val_base,
                    "intervened": val_inter
                }

        vec_base = self.dag_manager.slab_pool.get_slab_state(base_node.slab_offset)
        vec_inter = self.dag_manager.slab_pool.get_slab_state(intervened_node.slab_offset)
        spatial_divergence = float(np.linalg.norm(vec_inter - vec_base))

        return {
            "cone_delta_dict": cone_delta,
            "spatial_divergence": spatial_divergence,
            "baseline_slab_offset": base_node.slab_offset,
            "intervened_slab_offset": intervened_node.slab_offset
        }

    def simulate_parallel_trajectories(
        self,
        start_node_id: str,
        interventions: List[Tuple[str, Any]]
    ) -> List[StateNode]:
        self.dag_manager.rewind_to(start_node_id)
        derived_nodes = []

        for var, val in interventions:
            self.dag_manager.rewind_to(start_node_id)
            node = self.apply_do_operator(var, val)
            derived_nodes.append(node)

        return derived_nodes
