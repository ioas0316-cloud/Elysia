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

logger = logging.getLogger("WorldTree")


class TreeNode:
    """A single node in the WorldTree fractal hierarchy."""
    
    def __init__(self, concept: Any, parent: Optional['TreeNode'] = None):
        self.id = str(uuid.uuid4())
        self.concept = concept  # The concept data (string or complex object)
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.metadata: Dict[str, Any] = {}
        self.depth = 0 if parent is None else parent.depth + 1
    
    def add_child(self, child_node: 'TreeNode') -> None:
        """Add a child node to this node."""
        self.children.append(child_node)
        child_node.parent = self
        child_node.depth = self.depth + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node and its descendants to a dictionary."""
        return {
            "id": self.id,
            "concept": self.concept,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
            "depth": self.depth
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
        new_node = TreeNode(concept)
        if metadata:
            new_node.metadata = metadata
        
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
        for child in node.children:
            self._remove_from_index(child)
    
    def _find_node(self, node_id: str) -> Optional[TreeNode]:
        """Fast node lookup using the index."""
        return self._node_index.get(node_id)
    
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
