"""
Localized Rollback Engine for Causal DAG Subtree Resimulation.
Instead of rolling back the entire ECS world, only contaminated subtrees
belonging to delayed Signal Cartridges are tracked and resimulated in topological order.
"""

from dataclasses import dataclass, field
from typing import List, Set, Callable, Optional, Dict, Any
from collections import deque

@dataclass
class CausalNode:
    node_id: int
    state_data: bytearray = field(default_factory=lambda: bytearray(64))  # 64-byte hot data
    last_updated_frame: int = 0
    active_signal_id: int = 0
    children: List[int] = field(default_factory=list)  # Dependent child node indices
    parents: List[int] = field(default_factory=list)   # Parent node indices

@dataclass
class SignalCartridge:
    signal_id: int
    frame: int
    target_node_id: int
    payload: Dict[str, Any] = field(default_factory=dict)

    def target_node_idx(self) -> int:
        return self.target_node_id

class LocalizedRollbackEngine:
    def __init__(self, nodes: Optional[List[CausalNode]] = None):
        self.nodes: List[CausalNode] = nodes if nodes is not None else []
        self.dirty_mask: Set[int] = set()  # BitSet simulation using a set of dirty node indices

    def add_node(self, node: CausalNode):
        self.nodes.append(node)

    def add_edge(self, parent_idx: int, child_idx: int):
        if child_idx not in self.nodes[parent_idx].children:
            self.nodes[parent_idx].children.append(child_idx)
        if parent_idx not in self.nodes[child_idx].parents:
            self.nodes[child_idx].parents.append(parent_idx)

    def mark_dirty_subtree(self, root_node_idx: int):
        """
        Traverses child subtree starting at root_node_idx using BFS
        and marks nodes as dirty in dirty_mask. Prevents re-visiting nodes.
        """
        queue = deque([root_node_idx])
        while queue:
            idx = queue.popleft()
            if idx not in self.dirty_mask:
                self.dirty_mask.add(idx)
                for child_idx in self.nodes[idx].children:
                    queue.append(child_idx)

    def get_topological_dirty_order(self, root_node_idx: int) -> List[int]:
        """
        Computes topological ordering of dirty subtree nodes using Kahn's algorithm or post-order DFS.
        Guarantees parent causal updates occur before child evaluations.
        """
        dirty_set = set(self.dirty_mask)
        in_degree = {idx: 0 for idx in dirty_set}

        for idx in dirty_set:
            for child_idx in self.nodes[idx].children:
                if child_idx in in_degree:
                    in_degree[child_idx] += 1

        queue = deque([idx for idx, deg in in_degree.items() if deg == 0])
        topo_order = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for child_idx in self.nodes[curr].children:
                if child_idx in in_degree:
                    in_degree[child_idx] -= 1
                    if in_degree[child_idx] == 0:
                        queue.append(child_idx)

        # Fallback if cycles exist or disconnected components remain
        if len(topo_order) < len(dirty_set):
            remaining = sorted(list(dirty_set - set(topo_order)))
            topo_order.extend(remaining)

        return topo_order

    def resimulate_dirty_subtrees(
        self,
        missed_signal: SignalCartridge,
        current_frame: int,
        protocol_fn: Optional[Callable[[CausalNode, SignalCartridge], None]] = None
    ) -> List[int]:
        """
        1. Inject delayed Signal Cartridge into root target node.
        2. Mark dirty subtree.
        3. Re-evaluate causal protocol ONLY on marked dirty nodes in topological order.
        4. Return list of resimulated node indices.
        """
        root_idx = missed_signal.target_node_idx()
        self.mark_dirty_subtree(root_idx)

        # Retrieve topological execution order for subtree
        resimulated_indices = self.get_topological_dirty_order(root_idx)

        for node_idx in resimulated_indices:
            node = self.nodes[node_idx]
            if protocol_fn:
                protocol_fn(node, missed_signal)
            else:
                self._default_execute_causal_protocol(node, missed_signal)

            node.last_updated_frame = current_frame
            node.active_signal_id = missed_signal.signal_id

        # Clear dirty mask for next frame
        self.dirty_mask.clear()
        return resimulated_indices

    def _default_execute_causal_protocol(self, node: CausalNode, signal: SignalCartridge):
        """
        Default causal protocol update on node state_data.
        """
        val = signal.payload.get("value", 1)
        if len(node.state_data) >= 4:
            curr_val = int.from_bytes(node.state_data[:4], byteorder='little')
            new_val = (curr_val + val) % 256
            node.state_data[:4] = new_val.to_bytes(4, byteorder='little')
