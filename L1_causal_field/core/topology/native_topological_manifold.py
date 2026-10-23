"""
Native Topological Manifold Module

Abolishes artificial data type silos (JSON, PNG, String, Bytes) and represents
all data through their native topological structures:
1. Sequential Offset Topology (1D linear order: Strings, Byte streams)
2. Spatial Grid Topology (2D orthogonal grid: Images, Pixels, Rasters)
3. Hierarchical Branch Topology (Directed Acyclic Graph / Tree: JSON, AST)

No external parser or floating-point decoder is applied. Nodes retain their
native adjacency links and structural representation directly.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum


class TopologyType(Enum):
    SEQUENTIAL = "sequential"       # 1D offset sequence (String, Bytes)
    SPATIAL_GRID = "spatial_grid"   # 2D spatial grid (PNG, Images, Rasters)
    HIERARCHICAL = "hierarchical"   # Tree/DAG hierarchy (JSON, AST)
    HYBRID_MACRO = "hybrid_macro"   # Crystallized Phase-Locked Macro Topology


class NativeTopologicalNode:
    """
    A fundamental atom of raw data maintaining its native topological links.
    """
    def __init__(
        self,
        node_id: str,
        payload: Any,
        topology_type: TopologyType,
        coordinate: Optional[Tuple[Any, ...]] = None
    ):
        self.node_id: str = node_id
        self.payload: Any = payload
        self.topology_type: TopologyType = topology_type
        self.coordinate: Optional[Tuple[Any, ...]] = coordinate  # e.g., (1D offset,), (x, y), or (depth, branch_index)

        # Native adjacency links: neighbor_role -> list of NativeTopologicalNode
        # e.g., "next", "prev", "north", "south", "east", "west", "parent", "child:key"
        self.adjacencies: Dict[str, List['NativeTopologicalNode']] = {}

        # Inter-manifold boundary pointers (for voluntary coupling & phase lock)
        self.coupled_pointers: Dict[str, 'NativeTopologicalNode'] = {}

    def add_adjacency(self, role: str, target: 'NativeTopologicalNode') -> None:
        if role not in self.adjacencies:
            self.adjacencies[role] = []
        if target not in self.adjacencies[role]:
            self.adjacencies[role].append(target)

    def bind_coupling_pointer(self, interface_name: str, target: 'NativeTopologicalNode') -> None:
        self.coupled_pointers[interface_name] = target

    def is_boundary_node(self) -> bool:
        """
        A node is on the boundary if it has uncoupled potential edges or exposed external interfaces.
        """
        return len(self.coupled_pointers) == 0 or any("boundary" in role for role in self.adjacencies)

    def __repr__(self) -> str:
        return f"Node({self.node_id}, type={self.topology_type.value}, payload={repr(self.payload)})"


class NativeTopologicalManifold:
    """
    Base manifold containing a graph of native topological nodes.
    """
    def __init__(self, manifold_id: str, topology_type: TopologyType):
        self.manifold_id: str = manifold_id
        self.topology_type: TopologyType = topology_type
        self.nodes: Dict[str, NativeTopologicalNode] = {}
        self.root_nodes: List[NativeTopologicalNode] = []
        self.boundary_nodes: List[NativeTopologicalNode] = []

    def add_node(self, node: NativeTopologicalNode, is_root: bool = False) -> None:
        self.nodes[node.node_id] = node
        if is_root:
            self.root_nodes.append(node)

    def get_boundary_nodes(self) -> List[NativeTopologicalNode]:
        """
        Returns nodes that form the Markov blanket / boundary for interaction.
        """
        return [n for n in self.nodes.values() if n.is_boundary_node()]


class SequentialOffsetTopology(NativeTopologicalManifold):
    """
    1D Linear Topology representing raw sequences (String, Bytes, Streams).
    """
    def __init__(self, manifold_id: str, raw_sequence: Union[str, bytes, List[Any]]):
        super().__init__(manifold_id, TopologyType.SEQUENTIAL)
        self.raw_sequence = raw_sequence
        self._build_topology()

    def _build_topology(self) -> None:
        prev_node: Optional[NativeTopologicalNode] = None
        for idx, item in enumerate(self.raw_sequence):
            node_id = f"{self.manifold_id}_seq_{idx}"
            node = NativeTopologicalNode(
                node_id=node_id,
                payload=item,
                topology_type=TopologyType.SEQUENTIAL,
                coordinate=(idx,)
            )
            self.add_node(node, is_root=(idx == 0))

            if prev_node is not None:
                prev_node.add_adjacency("next", node)
                node.add_adjacency("prev", prev_node)
            prev_node = node

        # Mark ends as boundaries
        if self.nodes:
            first_key = f"{self.manifold_id}_seq_0"
            last_key = f"{self.manifold_id}_seq_{len(self.raw_sequence)-1}"
            if first_key in self.nodes:
                self.nodes[first_key].add_adjacency("boundary_head", self.nodes[first_key])
            if last_key in self.nodes:
                self.nodes[last_key].add_adjacency("boundary_tail", self.nodes[last_key])


class SpatialGridTopology(NativeTopologicalManifold):
    """
    2D Orthogonal Grid Topology representing images, pixels, and raster data.
    """
    def __init__(self, manifold_id: str, grid_data: List[List[Any]]):
        super().__init__(manifold_id, TopologyType.SPATIAL_GRID)
        self.grid_data = grid_data
        self.height = len(grid_data)
        self.width = len(grid_data[0]) if self.height > 0 else 0
        self._build_topology()

    def _build_topology(self) -> None:
        grid_nodes: Dict[Tuple[int, int], NativeTopologicalNode] = {}

        for r in range(self.height):
            for c in range(self.width):
                node_id = f"{self.manifold_id}_grid_{r}_{c}"
                payload = self.grid_data[r][c]
                node = NativeTopologicalNode(
                    node_id=node_id,
                    payload=payload,
                    topology_type=TopologyType.SPATIAL_GRID,
                    coordinate=(r, c)
                )
                self.add_node(node, is_root=(r == 0 and c == 0))
                grid_nodes[(r, c)] = node

        # Link 2D orthogonal neighbors (North, South, East, West)
        for (r, c), node in grid_nodes.items():
            if (r - 1, c) in grid_nodes:
                node.add_adjacency("north", grid_nodes[(r - 1, c)])
            if (r + 1, c) in grid_nodes:
                node.add_adjacency("south", grid_nodes[(r + 1, c)])
            if (r, c - 1) in grid_nodes:
                node.add_adjacency("west", grid_nodes[(r, c - 1)])
            if (r, c + 1) in grid_nodes:
                node.add_adjacency("east", grid_nodes[(r, c + 1)])

            # Perimeter nodes are boundary nodes
            if r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1:
                node.add_adjacency("boundary_perimeter", node)


class HierarchicalBranchTopology(NativeTopologicalManifold):
    """
    Hierarchical DAG / Tree Topology representing JSON, AST, nested dictionaries/lists.
    """
    def __init__(self, manifold_id: str, nested_data: Union[Dict[str, Any], List[Any], Any]):
        super().__init__(manifold_id, TopologyType.HIERARCHICAL)
        self.nested_data = nested_data
        self._build_topology()

    def _build_topology(self) -> None:
        def traverse(data: Any, path: str, depth: int, index: int, parent_node: Optional[NativeTopologicalNode]) -> NativeTopologicalNode:
            node_id = f"{self.manifold_id}_{path}"

            if isinstance(data, dict):
                payload = "<dict>"
            elif isinstance(data, list):
                payload = "<list>"
            else:
                payload = data

            node = NativeTopologicalNode(
                node_id=node_id,
                payload=payload,
                topology_type=TopologyType.HIERARCHICAL,
                coordinate=(depth, index)
            )
            self.add_node(node, is_root=(parent_node is None))

            if parent_node is not None:
                parent_node.add_adjacency("child", node)
                node.add_adjacency("parent", parent_node)

            if isinstance(data, dict):
                for child_idx, (k, v) in enumerate(data.items()):
                    key_path = f"{path}.{k}"
                    child_node = traverse(v, key_path, depth + 1, child_idx, node)
                    node.add_adjacency(f"branch:{k}", child_node)
            elif isinstance(data, list):
                for child_idx, item in enumerate(data):
                    elem_path = f"{path}[{child_idx}]"
                    child_node = traverse(item, elem_path, depth + 1, child_idx, node)
                    node.add_adjacency(f"element:{child_idx}", child_node)
            else:
                # Leaf node is a boundary edge
                node.add_adjacency("boundary_leaf", node)

            return node

        if self.nested_data is not None:
            traverse(self.nested_data, "root", 0, 0, None)
