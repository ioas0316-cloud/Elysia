# Hippocampus - Long-Term Memory & Causal Reasoning Graph
"""
Elysia's memory system that combines:
1. Context graph (original Hippocampus) - keyword-based conversation retrieval
2. Spiderweb (Legacy integration) - causal reasoning graph with directed relationships

This unified system enables both simple context recall ("what did we talk about love?")
and complex causal reasoning ("if X causes Y, and Y enables Z, what leads to Z?").
"""

import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple, FrozenSet

logger = logging.getLogger("Hippocampus")


class Hippocampus:
    """
    Unified memory and reasoning system.
    - Stores conversation turns indexed by keywords (lightweight context)
    - Maintains causal graph of concepts with typed relationships
    """
    
    def __init__(self) -> None:
        # === Context Graph (Original Hippocampus) ===
        # {frozenset(keywords): [(user_text, response), ...]}
        self.context_graph: Dict[FrozenSet[str], List[Tuple[str, str]]] = {}
        
        # === Spiderweb - Causal Reasoning Graph ===
        # NetworkX directed graph: nodes=concepts, edges=causal links
        self.causal_graph = nx.DiGraph()
        
        logger.info("✅ Hippocampus initialized with causal reasoning (Spiderweb)")
    
    @property
    def graph(self) -> nx.DiGraph:
        """
        Compatibility alias for legacy Spiderweb API.
        Returns the underlying causal graph so legacy modules can treat
        Hippocampus as a drop-in replacement for Spiderweb.graph.
        """
        return self.causal_graph
    
    # ========== Context Graph Methods (Original) ==========
    
    def _keywords(self, text: str) -> FrozenSet[str]:
        """Extract keywords from text (words longer than 1 char)."""
        words = [w for w in text.lower().split() if len(w) > 1]
        return frozenset(words)
    
    def add_turn(self, user_text: str, response: str) -> None:
        """Store a conversation turn indexed by keywords."""
        key = self._keywords(user_text)
        if key not in self.context_graph:
            self.context_graph[key] = []
        self.context_graph[key].append((user_text, response))
        
        # Also add concepts to causal graph as nodes
        for word in key:
            self.add_concept(word, concept_type="keyword")
        
        logger.debug(f"Stored turn under {len(key)} keywords")
    
    def retrieve(self, current_text: str) -> List[Dict[str, str]]:
        """
        Retrieve stored turns that share keywords with current_text.
        Returns list of dicts with 'user_text' and 'response' keys.
        """
        cur_key = self._keywords(current_text)
        matches = []
        for stored_key, turns in self.context_graph.items():
            overlap = len(cur_key & stored_key)
            if overlap > 0:
                matches.append((overlap, turns))
        matches.sort(key=lambda x: x[0], reverse=True)
        
        # Flatten top 3 matches, return as dict list
        result: List[Dict[str, str]] = []
        for _, turns in matches[:3]:
            for user_t, resp_t in turns[-3:]:
                result.append({"user_text": user_t, "response": resp_t})
        
        logger.debug(f"Retrieved {len(result)} relevant turns")
        return result
    
    # ========== Spiderweb - Causal Graph Methods ==========
    
    def add_concept(
        self,
        concept_id: str,
        concept_type: str = "concept",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a concept node to the causal graph.
        
        Args:
            concept_id: Unique identifier (e.g., "love", "fire", "burn_event_123")
            concept_type: Type of node (e.g., "concept", "event", "emotion", "keyword")
            metadata: Additional properties
        """
        if not self.causal_graph.has_node(concept_id):
            self.causal_graph.add_node(
                concept_id,
                type=concept_type,
                metadata=metadata or {}
            )
            logger.debug(f"Added concept: {concept_id} ({concept_type})")
        else:
            # Update metadata if node exists
            self.causal_graph.nodes[concept_id].update({
                "type": concept_type,
                "metadata": {**self.causal_graph.nodes[concept_id].get("metadata", {}), **(metadata or {})}
            })
    
    def add_causal_link(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        weight: float = 1.0
    ) -> None:
        """
        Add a causal/associative link between two concepts.
        
        Args:
            source: Source concept ID
            target: Target concept ID
            relation: Relationship type (e.g., "causes", "prevents", "enables", "is_a")
            weight: Strength of the connection (0.0 to 1.0)
        """
        if not self.causal_graph.has_node(source) or not self.causal_graph.has_node(target):
            logger.warning(f"Cannot add link: {source} or {target} not in graph")
            return
        
        self.causal_graph.add_edge(source, target, relation=relation, weight=weight)
        logger.debug(f"Added link: {source} -[{relation}]-> {target} (w={weight})")
    
    # Legacy-friendly aliases
    def add_link(self, source: str, target: str, relation: str = "related_to", weight: float = 1.0) -> None:
        """Alias for add_causal_link to match the Spiderweb API."""
        self.add_causal_link(source, target, relation=relation, weight=weight)
    
    def find_causal_path(self, start: str, end: str) -> List[str]:
        """
        Find a causal path from start concept to end concept.
        Useful for tracing causality chains.
        
        Returns:
            List of concept IDs forming the path, or empty list if no path exists
        """
        try:
            path = nx.shortest_path(self.causal_graph, source=start, target=end)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def find_path(self, start: str, end: str) -> List[str]:
        """Alias for find_causal_path (Spiderweb compatibility)."""
        return self.find_causal_path(start, end)
    
    def get_causal_context(
        self,
        concept_id: str,
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get the causal context around a concept (incoming and outgoing links).
        
        Args:
            concept_id: Concept to query
            depth: How many hops to explore (currently only supports 1)
        
        Returns:
            List of dicts with 'node', 'relation', 'direction', 'weight'
        """
        if not self.causal_graph.has_node(concept_id):
            return []
        
        context = []
        
        # Outgoing edges (what this concept causes/enables)
        for neighbor in self.causal_graph.successors(concept_id):
            edge_data = self.causal_graph.get_edge_data(concept_id, neighbor)
            context.append({
                "node": neighbor,
                "relation": edge_data.get("relation"),
                "direction": "outgoing",
                "weight": edge_data.get("weight")
            })
        
        # Incoming edges (what causes/enables this concept)
        for neighbor in self.causal_graph.predecessors(concept_id):
            edge_data = self.causal_graph.get_edge_data(neighbor, concept_id)
            context.append({
                "node": neighbor,
                "relation": edge_data.get("relation"),
                "direction": "incoming",
                "weight": edge_data.get("weight")
            })
        
        return context
    
    def get_context(self, concept_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Alias for get_causal_context to match Spiderweb interface."""
        return self.get_causal_context(concept_id, depth=depth)
    
    def forget(
        self,
        decay: float = 0.99,
        boost_concepts: Optional[List[str]] = None
    ) -> None:
        """
        Entropy-driven forgetting: decay edge weights, remove weak edges and isolated nodes.
        
        Args:
            decay: Multiplier for edge weights each cycle (0.0 to 1.0)
            boost_concepts: Concepts to protect from forgetting (e.g., core memories)
        """
        if not self.causal_graph.edges:
            return
        
        boost_set = set(boost_concepts) if boost_concepts else set()
        
        # Decay all edge weights
        for u, v, data in list(self.causal_graph.edges(data=True)):
            w = float(data.get("weight", 0.0)) * decay
            
            # Protect boosted concepts
            if u in boost_set or v in boost_set:
                w = min(1.0, w * 1.1)  # Slight protection
            
            # Remove very weak edges
            if w < 1e-3:
                self.causal_graph.remove_edge(u, v)
            else:
                self.causal_graph.edges[u, v]["weight"] = w
        
        # Remove isolated nodes
        isolated = [n for n, deg in self.causal_graph.degree if deg == 0]
        if isolated:
            self.causal_graph.remove_nodes_from(isolated)
            logger.info(f"Forgot {len(isolated)} isolated concepts (entropy decay)")
    
    def prune_weakest(
        self,
        edge_fraction: float = 0.3,
        node_fraction: float = 0.3
    ) -> None:
        """
        Prune the weakest edges and least-connected nodes by fraction.
        
        Args:
            edge_fraction: Fraction of weakest edges to remove (0.0 to 1.0)
            node_fraction: Fraction of least-connected nodes to remove (0.0 to 1.0)
        """
        edges = list(self.causal_graph.edges(data=True))
        if edges and edge_fraction > 0:
            edges_sorted = sorted(edges, key=lambda e: float(e[2].get("weight", 0.0)))
            cut = max(0, min(len(edges_sorted), int(len(edges_sorted) * edge_fraction)))
            for u, v, _ in edges_sorted[:cut]:
                if self.causal_graph.has_edge(u, v):
                    self.causal_graph.remove_edge(u, v)
        
        nodes = list(self.causal_graph.degree)
        if nodes and node_fraction > 0:
            nodes_sorted = sorted(nodes, key=lambda x: x[1])  # by degree
            cut_n = max(0, min(len(nodes_sorted), int(len(nodes_sorted) * node_fraction)))
            for node, deg in nodes_sorted[:cut_n]:
                if self.causal_graph.has_node(node) and self.causal_graph.degree(node) == 0:
                    self.causal_graph.remove_node(node)
        
        logger.info(f"Pruned {edge_fraction*100:.0f}% edges, {node_fraction*100:.0f}% nodes")
    
    def prune_fraction(self, edge_fraction: float = 0.3, node_fraction: float = 0.3) -> None:
        """Alias for prune_weakest to mirror the legacy Spiderweb interface."""
        self.prune_weakest(edge_fraction=edge_fraction, node_fraction=node_fraction)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            "context_keys": len(self.context_graph),
            "total_turns": sum(len(turns) for turns in self.context_graph.values()),
            "causal_nodes": self.causal_graph.number_of_nodes(),
            "causal_edges": self.causal_graph.number_of_edges()
        }

    def get_phase_tags(self) -> List[Dict[str, Any]]:
        """Collect phase metadata attached to causal nodes (if present)."""
        tags = []
        for node, data in self.causal_graph.nodes(data=True):
            meta = data.get("metadata", {})
            phase = meta.get("phase")
            if phase:
                tags.append({"node": node, "phase": phase})
        return tags

    def query_by_phase(
        self,
        min_mastery: float = 0.0,
        min_entropy: float = 0.0
    ) -> List[str]:
        """
        Return node ids whose phase metadata meets mastery/entropy thresholds.
        """
        matched = []
        for node, data in self.causal_graph.nodes(data=True):
            phase = data.get("metadata", {}).get("phase", {})
            q_state = phase.get("quaternion", {})
            qubit = phase.get("qubit", {})
            mastery = abs(q_state.get("w", 0.0))
            entropy = 0.0
            if qubit:
                total = sum(qubit.values())
                if total > 0:
                    norm = [p / total for p in qubit.values() if p > 0]
                    import math
                    entropy = -sum(p * math.log(p, 2) for p in norm)
            if mastery >= min_mastery and entropy >= min_entropy:
                matched.append(node)
        return matched
