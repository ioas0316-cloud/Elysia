# WorldTree - Fractal Concept Hierarchy
"""
The WorldTree represents knowledge as a fractal, self-similar hierarchical structure.
It allows infinite expansion of concepts into sub-concepts (branches) while
maintaining connection to the Spiderweb causal graph.

Concept Hierarchy:
- ROOT (세계수의 뿌리)
  ├─ Core Concepts (줄기)
  │  ├─ Sub-concepts (가지)
  │  │  └─ Details (잎)
  
Integration with Spiderweb:
- Each tree node can reference a concept in the Spiderweb causal graph
- Tree structure: IS-A hierarchy (love IS-A emotion)
- Graph structure: Causal relations (love CAUSES joy)
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from Core.Mind.fractal_address import make_address

logger = logging.getLogger("WorldTree")


class TreeNode:
    """A single node in the WorldTree fractal hierarchy."""
    
    def __init__(self, concept: Any, parent: Optional['TreeNode'] = None):
        from Core.Mind.tensor import HyperQuaternion
        self.id = str(uuid.uuid4())
        self.concept = concept  # The concept data (string or complex object)
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.metadata: Dict[str, Any] = {}
        self.depth = 0 if parent is None else parent.depth + 1
        
        # The Soul of the Node (4D HyperQuaternion)
        # Inherit from parent with slight mutation, or random if root
        if parent and hasattr(parent, 'qubit'):
            # Mutation logic: drift slightly from parent
            p_q = parent.qubit
            self.qubit = HyperQuaternion(
                w=p_q.w, # Same dimension usually
                x=p_q.x + random.uniform(-0.1, 0.1),
                y=p_q.y + random.uniform(-0.1, 0.1),
                z=p_q.z + random.uniform(-0.1, 0.1)
            )
        else:
            self.qubit = HyperQuaternion.random()

    def add_child(self, child_node: 'TreeNode') -> None:
        """Add a child node to this node."""
        self.children.append(child_node)
        child_node.parent = self
        child_node.depth = self.depth + 1
        # Re-align child's qubit to parent if it was random
        # (Optional, but helps coherence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node and its descendants to a dictionary."""
        return {
            "id": self.id,
            "concept": self.concept,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
            "depth": self.depth,
            "qubit": self.qubit.to_dict() if hasattr(self, 'qubit') else None
        }


class WorldTree:
    """
    The WorldTree - a fractal, self-similar knowledge hierarchy.
    
    Integrated with Hippocampus/Spiderweb:
    - Tree nodes can reference concepts in the causal graph
    - Enables both hierarchical (IS-A) and causal (CAUSES) reasoning
    """
    
    def __init__(self, hippocampus=None):
        self.root = TreeNode("ROOT")
        self.hippocampus = hippocampus  # Optional reference to Hippocampus for integration
        self._node_index: Dict[str, TreeNode] = {self.root.id: self.root}
        self._concept_index: Dict[str, str] = {"ROOT": self.root.id}
        logger.info("✅ WorldTree initialized (세계수 - The Cosmic Tree)")
    
    def plant_seed(
        self,
        concept: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Plant a concept seed in the tree.
        
        Args:
            concept: The concept to add (e.g., "love", "emotion", "physics")
            parent_id: ID of parent node (None = add to root)
            metadata: Additional properties for this concept
        
        Returns:
            ID of the newly created node
        """
        # Deduplicate by concept name if it already exists in the tree
        existing_id = self._concept_index.get(concept)
        if existing_id:
            existing_node = self._node_index.get(existing_id)
            if existing_node and metadata:
                existing_node.metadata.update(metadata)
            if parent_id and existing_node and existing_node.parent is None:
                parent = self._find_node(parent_id)
                if parent:
                    parent.add_child(existing_node)
            return existing_id

        new_node = TreeNode(concept)
        if metadata:
            new_node.metadata.update(metadata)
        
        if parent_id:
            parent = self._find_node(parent_id)
            if parent:
                parent.add_child(new_node)
                logger.debug(f"Planted '{concept}' as child of {parent_id}")
            else:
                logger.warning(f"Parent {parent_id} not found. Planting at root.")
                self.root.add_child(new_node)
        else:
            self.root.add_child(new_node)
            logger.debug(f"Planted '{concept}' at root")

        # Index the node for fast lookup
        self._node_index[new_node.id] = new_node
        self._concept_index[concept] = new_node.id
        # Assign fractal address
        new_node.metadata.setdefault("fractal_address", make_address(self.get_path_to_root(new_node.id)[::-1]))
        
        # If Hippocampus is connected, also add to causal graph
        if self.hippocampus:
            self.hippocampus.add_concept(concept, concept_type="tree_concept", metadata=metadata)
            # Create IS-A link to parent in causal graph
            if parent_id and parent:
                parent_concept = parent.concept
                if parent_concept != "ROOT":
                    self.hippocampus.add_causal_link(concept, parent_concept, relation="is_a", weight=1.0)
        
        return new_node.id
    
    def grow(self, branch_id: str, sub_concept: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Expand a branch with a new sub-concept (fractal growth).
        
        Args:
            branch_id: ID of the node to expand
            sub_concept: The sub-concept to add
            metadata: Additional properties
        
        Returns:
            ID of the newly created sub-node
        """
        return self.plant_seed(sub_concept, parent_id=branch_id, metadata=metadata)
    
    def ensure_concept(self, concept: str, parent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Ensure a concept exists (deduplicated) and return its node id.
        Acts as a safer wrapper around plant_seed for repeated insertions.
        """
        return self.plant_seed(concept, parent_id=parent_id, metadata=metadata)
    
    def prune(self, branch_id: str) -> bool:
        """
        Remove a branch and all its descendants.
        
        Args:
            branch_id: ID of the node to remove
        
        Returns:
            True if pruned successfully, False if node not found
        """
        node = self._find_node(branch_id)
        if not node or not node.parent:
            return False
        
        # Remove from parent's children
        node.parent.children.remove(node)
        
        # Remove from index (including all descendants)
        self._remove_from_index(node)
        
        logger.info(f"Pruned branch {branch_id}")
        return True
    
    def _remove_from_index(self, node: TreeNode) -> None:
        """Recursively remove node and descendants from index."""
        if node.id in self._node_index:
            del self._node_index[node.id]
        if node.concept in self._concept_index and self._concept_index[node.concept] == node.id:
            del self._concept_index[node.concept]
        for child in node.children:
            self._remove_from_index(child)
    
    def _find_node(self, node_id: str) -> Optional[TreeNode]:
        """Fast node lookup using the index."""
        return self._node_index.get(node_id)
    
    def find_by_concept(self, concept: str) -> Optional[str]:
        """Find the first node id for a given concept, if it exists."""
        return self._concept_index.get(concept)
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """
        Get the path from a node to the root (concept ancestry).
        
        Args:
            node_id: ID of the node
        
        Returns:
            List of concepts from node to root (e.g., ["physics", "science", "knowledge", "ROOT"])
        """
        node = self._find_node(node_id)
        if not node:
            return []
        
        path = []
        current = node
        while current:
            path.append(current.concept)
            current = current.parent
        
        return path
    
    def get_descendants(self, node_id: str, max_depth: Optional[int] = None) -> List[TreeNode]:
        """
        Get all descendants of a node (optionally limited by depth).
        
        Args:
            node_id: ID of the node
            max_depth: Maximum depth to traverse (None = unlimited)
        
        Returns:
            List of descendant TreeNodes
        """
        node = self._find_node(node_id)
        if not node:
            return []
        
        descendants = []
        
        def _collect(n: TreeNode, current_depth: int):
            if max_depth is not None and current_depth > max_depth:
                return
            for child in n.children:
                descendants.append(child)
                _collect(child, current_depth + 1)
        
        _collect(node, 0)
        return descendants
    
    def visualize(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the tree (or subtree).
        
        Args:
            node_id: Root of subtree to visualize (None = entire tree)
        
        Returns:
            Dictionary with tree structure
        """
        if node_id:
            node = self._find_node(node_id)
            if not node:
                return {}
            return node.to_dict()
        else:
            return self.root.to_dict()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the WorldTree."""
        total_nodes = len(self._node_index)
        max_depth = max((node.depth for node in self._node_index.values()), default=0)
        
        # Count leaves (nodes with no children)
        leaves = sum(1 for node in self._node_index.values() if len(node.children) == 0)
        
        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "leaf_nodes": leaves,
            "branches": total_nodes - leaves - 1  # (excluding root and leaves)
        }

    def get_all_concept_names(self) -> List[str]:
        """
        Returns a list of all unique concept names in the tree.
        """
        # _concept_index stores a direct mapping from concept name to node id
        return list(self._concept_index.keys())
