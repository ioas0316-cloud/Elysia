import math
import numpy as np
from typing import Dict, Any, Optional, Set, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from core.memory.state_dag import StateDAGManager, StateNode

class CausalAwareGC:
    """
    [Causal-Aware Garbage Collection (CGC) & Job System Dispatcher]
    self-guided analog GC 및 Delta Superposition 링 버퍼 회전에 따른
    자연 사멸(Self-Pruning / Entropy Expiry) 분기 회수를 지원합니다.
    """
    def __init__(
        self,
        dag_manager: StateDAGManager,
        lambda_decay: float = 0.2,
        threshold_gc: float = 1.0,
        num_workers: int = 4
    ):
        self.dag_manager = dag_manager
        self.lambda_decay = lambda_decay
        self.threshold_gc = threshold_gc
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def calculate_kl_divergence(self, p_vec: np.ndarray, q_vec: np.ndarray) -> float:
        p = np.clip(np.abs(p_vec), 1e-7, None)
        q = np.clip(np.abs(q_vec), 1e-7, None)

        p_prob = np.exp(p - np.max(p))
        p_prob /= np.sum(p_prob)

        q_prob = np.exp(q - np.max(q))
        q_prob /= np.sum(q_prob)

        kl_div = float(np.sum(p_prob * np.log(p_prob / q_prob)))
        return max(0.0, kl_div)

    def calculate_node_information_value(self, node: StateNode) -> float:
        if not node.parent:
            return 0.0

        v_curr = self.dag_manager.slab_pool.get_slab_state(node.slab_offset)
        v_parent = self.dag_manager.slab_pool.get_slab_state(node.parent.slab_offset)

        kl_val = self.calculate_kl_divergence(v_curr, v_parent)
        node_div = node.compute_node_divergence()

        return kl_val + node_div

    def calculate_branch_viability(self, node: StateNode) -> float:
        score = 0.0
        curr = node
        depth = 0

        while curr and curr.parent:
            v_i = self.calculate_node_information_value(curr)
            decay = math.exp(-self.lambda_decay * depth)
            score += v_i * decay
            curr = curr.parent
            depth += 1

        return score

    def prune_expired_superposition_nodes(self) -> int:
        """
        [Self-Guided Ring Buffer Expiry GC]
        링 버퍼 회전(Wrap-Around)에 의해 델타 파형이 Overwrite/소멸된 분기 노드를
        별도의 비트마스크 순회 비용 없이 자연스럽게 정리합니다.
        """
        pruned_count = 0
        ring_buffer = self.dag_manager.superposition_engine.ring_buffer

        with self.dag_manager._lock:
            protected_node_ids = set()
            curr = self.dag_manager.current_node
            while curr:
                protected_node_ids.add(curr.id)
                curr = curr.parent

            nodes_list = list(self.dag_manager.nodes.values())
            for node in nodes_list:
                if node.id in protected_node_ids or node.id == self.dag_manager.root.id:
                    continue

                if node.observer_view is not None and node.observer_view.active_indices:
                    # Check if all active wave indices of this node's view have expired in the ring buffer
                    active_indices = node.observer_view.active_indices
                    has_valid = False
                    for idx in active_indices:
                        if ring_buffer.get_kv_delta(idx) is not None or ring_buffer.get_vector_delta(idx) is not None:
                            has_valid = True
                            break

                    # If none of its active delta waves remain valid in the ring buffer, prune node
                    if not has_valid and not node.children:
                        if node.parent and node in node.parent.children:
                            node.parent.children.remove(node)
                        if node.id in self.dag_manager.nodes:
                            del self.dag_manager.nodes[node.id]
                        pruned_count += 1

        return pruned_count

    def collapse_isomorphic_nodes(self) -> int:
        """
        [DAG Collapsing / Deduplication]
        현재 활성 관측 조상 경로(root -> current_node)상의 노드는 절대 삭제/병합하지 않습니다.
        """
        collapsed_count = 0
        with self.dag_manager._lock:
            # Protect all nodes along current observation path
            protected_ids = set()
            curr = self.dag_manager.current_node
            while curr:
                protected_ids.add(curr.id)
                curr = curr.parent

            state_map: Dict[bytes, StateNode] = {}
            nodes_list = list(self.dag_manager.nodes.values())

            for node in nodes_list:
                if node.id in protected_ids:
                    continue

                vec = self.dag_manager.slab_pool.get_slab_state(node.slab_offset)
                vec_bytes = np.round(vec, decimals=4).tobytes()

                if vec_bytes in state_map:
                    target_node = state_map[vec_bytes]
                    for child in list(node.children):
                        child.parent = target_node
                        target_node.children.add(child)

                    if node.parent and node in node.parent.children:
                        node.parent.children.remove(node)

                    if node.id in self.dag_manager.nodes:
                        del self.dag_manager.nodes[node.id]

                    collapsed_count += 1
                else:
                    state_map[vec_bytes] = node

        return collapsed_count

    def run_cgc(self, custom_threshold: Optional[float] = None) -> int:
        """
        [Causal-Aware Garbage Collection]
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold_gc

        with self.dag_manager._lock:
            protected_node_ids = set()
            curr = self.dag_manager.current_node
            while curr:
                protected_node_ids.add(curr.id)
                curr = curr.parent

            expired_pruned = self.prune_expired_superposition_nodes()
            self.collapse_isomorphic_nodes()

            pruned_count = expired_pruned

            def prune_subtree(node: StateNode):
                nonlocal pruned_count
                for child in list(node.children):
                    prune_subtree(child)

                if not node.children and node.id not in protected_node_ids:
                    viability = self.calculate_branch_viability(node)
                    self.dag_manager.slab_pool.viability_scores[node.slab_offset] = viability

                    if viability < threshold:
                        if node.parent and node in node.parent.children:
                            node.parent.children.remove(node)
                        if node.id in self.dag_manager.nodes:
                            del self.dag_manager.nodes[node.id]
                        pruned_count += 1

            prune_subtree(self.dag_manager.root)
            return pruned_count

    def parallel_viability_eval(self, node_list: List[StateNode]) -> List[Tuple[str, float]]:
        def eval_chunk(chunk: List[StateNode]) -> List[Tuple[str, float]]:
            res = []
            for n in chunk:
                v = self.calculate_branch_viability(n)
                res.append((n.id, v))
            return res

        chunk_size = max(1, len(node_list) // self.num_workers)
        chunks = [node_list[i:i + chunk_size] for i in range(0, len(node_list), chunk_size)]

        futures = [self.executor.submit(eval_chunk, c) for c in chunks]
        results = []
        for f in futures:
            results.extend(f.result())

        return results

    def close(self):
        self.executor.shutdown(wait=False)
