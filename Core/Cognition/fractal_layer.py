"""
Fractal Memory Layer: The Node & System
=======================================
Core.Cognition.fractal_layer

"As Above, So Below."

This module implements the recursive node structure that allows
memory to be viewed at different scales (Zoom In/Out).
"""

import uuid
import time
import logging
import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field

from Core.Cognition.strata import MemoryStratum, StratumPhysics
# We will use lazy imports or dependency injection for backend storage
# to avoid circular dependencies with Hypersphere/Sediment.

logger = logging.getLogger("FractalMemory")

@dataclass
class FractalMemoryNode:
    """
    A single unit of memory that can exist in any Stratum.
    It links 'Up' to abstract concepts and 'Down' to concrete details.
    """
    content: Any  # The actual data (Text, Image path, Vector, etc.)
    stratum: MemoryStratum

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    # Physics State
    energy: float = 1.0       # Current vividness/importance
    mass: float = 1.0         # Resistance to movement/change (e.g. repetition count)

    # Fractal Topology
    parent_id: Optional[str] = None  # The "Gist" or "Category" this belongs to
    child_ids: List[str] = field(default_factory=list) # The "Details" or "Moments"

    # Semantic Coordinates (simplified for this layer, mapped to Hypersphere)
    vector: List[float] = field(default_factory=lambda: [0.0]*7)

    # Metadata (Tags, Source, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    def zoom_in(self) -> List[str]:
        """Returns IDs of children (Details)."""
        return self.child_ids

    def zoom_out(self) -> Optional[str]:
        """Returns ID of parent (Context)."""
        return self.parent_id

    def decay(self, amount: float):
        self.energy = max(0.0, self.energy - amount)

    def reinforce(self, amount: float):
        self.energy = min(1.0, self.energy + amount)
        self.mass += amount * 0.1 # Mass grows slower than energy

    def to_dict(self):
        return {
            "id": self.id,
            "content": str(self.content),
            "stratum": self.stratum.name,
            "energy": self.energy,
            "mass": self.mass,
            "parent": self.parent_id,
            "children": len(self.child_ids)
        }

class FractalMemorySystem:
    """
    The Manager that handles the CRUD of Fractal Nodes
    and interfaces with the storage backends (Hypersphere, Sediment).
    """
    def __init__(self, hypersphere_backend=None, sediment_backend=None):
        self.nodes: Dict[str, FractalMemoryNode] = {}

        # In a real system, these would be the actual persistent stores.
        # For this prototype, we keep an in-memory index `self.nodes`
        # but conceptually map them to the backends.
        self.hypersphere = hypersphere_backend
        self.sediment = sediment_backend

        # Root nodes for quick access (Top-level Crystals)
        self.roots: List[str] = []

    def add_memory(self, content: Any, stratum: MemoryStratum,
                   parent_id: Optional[str] = None,
                   vector: List[float] = None) -> FractalMemoryNode:
        """
        Creates a new memory node and links it topologically.
        """
        if vector is None:
            vector = [0.0]*7

        node = FractalMemoryNode(
            content=content,
            stratum=stratum,
            parent_id=parent_id,
            vector=vector
        )

        self.nodes[node.id] = node

        # Link to parent
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].child_ids.append(node.id)
            # Energy flows up: Parent gets reinforced by new child
            self.nodes[parent_id].reinforce(0.1)
        elif stratum == MemoryStratum.CRYSTAL:
            # Crystals without parents are Roots
            self.roots.append(node.id)

        # Log to appropriate backend (Conceptual)
        if stratum == MemoryStratum.SEDIMENT and self.sediment:
            # In a full impl, we'd serialize and write to binary here
            pass
        elif stratum in [MemoryStratum.CRYSTAL, MemoryStratum.GARDEN] and self.hypersphere:
            # Sync with Hypersphere
            pass

        return node

    def get_node(self, node_id: str) -> Optional[FractalMemoryNode]:
        return self.nodes.get(node_id)

    def get_layer_view(self, stratum: MemoryStratum) -> List[FractalMemoryNode]:
        """Returns all nodes in a specific layer."""
        return [n for n in self.nodes.values() if n.stratum == stratum]

    def move_node(self, node_id: str, target_stratum: MemoryStratum):
        """Moves a node to a different geological layer."""
        if node_id not in self.nodes: return
        node = self.nodes[node_id]

        old_stratum = node.stratum
        node.stratum = target_stratum
        logger.info(f"  Memory Moved: [{node.content}] {old_stratum.name} -> {target_stratum.name}")

    def merge_nodes(self, node_ids: List[str], new_content: Any) -> FractalMemoryNode:
        """
        [Crystallization] Merges multiple nodes into a new higher-abstraction node.
        The old nodes become children of the new node.
        """
        # Calculate average vector
        vectors = [self.nodes[nid].vector for nid in node_ids if nid in self.nodes]
        avg_vector = [sum(col)/len(col) for col in zip(*vectors)] if vectors else [0.0]*7

        # Create the Crystal (Wisdom)
        crystal = self.add_memory(
            content=new_content,
            stratum=MemoryStratum.CRYSTAL,
            vector=avg_vector
        )

        # Re-parent the children
        for nid in node_ids:
            if nid in self.nodes:
                node = self.nodes[nid]
                node.parent_id = crystal.id
                crystal.child_ids.append(nid)
                # Usually, when crystallizing, the details sink to Sediment or Garden
                if node.stratum == MemoryStratum.STREAM:
                    self.move_node(nid, MemoryStratum.GARDEN)

        logger.info(f"  Crystallization Complete: '{new_content}' formed from {len(node_ids)} fragments.")
        return crystal

    def status_report(self) -> str:
        counts = {s.name: 0 for s in MemoryStratum}
        for n in self.nodes.values():
            counts[n.stratum.name] += 1
        return f"Memory Status: {json.dumps(counts, indent=2)}"
