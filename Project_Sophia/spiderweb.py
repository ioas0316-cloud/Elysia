
import json
import logging
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

class Spiderweb:
    """
    The Spiderweb is a causal reasoning graph that structures Elysia's memories and concepts.
    It moves beyond simple logging to understanding the 'why' and 'how' of experiences.
    It manages a graph where nodes are concepts/events and edges are causal/associative links.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.graph = nx.DiGraph()
        self.logger = logger or logging.getLogger("Spiderweb")
        self.logger.info("Spiderweb initialized.")

    def add_node(self, node_id: str, type: str, metadata: Dict[str, Any] = None):
        """
        Adds a node to the Spiderweb.
        
        Args:
            node_id (str): Unique identifier for the node (e.g., "fire", "burn_event_123").
            type (str): Type of the node (e.g., "concept", "event", "emotion").
            metadata (dict): Additional data associated with the node.
        """
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=type, metadata=metadata or {})
            self.logger.debug(f"Added node: {node_id} ({type})")
        else:
            # Update metadata if node exists
            self.graph.nodes[node_id].update({"type": type, "metadata": metadata or {}})

    def add_link(self, source: str, target: str, relation: str, weight: float = 1.0):
        """
        Adds a directed link between two nodes.
        
        Args:
            source (str): Source node ID.
            target (str): Target node ID.
            relation (str): Type of relationship (e.g., "causes", "is_a", "related_to").
            weight (float): Strength of the connection (0.0 to 1.0).
        """
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            self.logger.warning(f"Cannot add link: {source} or {target} not in graph.")
            return

        self.graph.add_edge(source, target, relation=relation, weight=weight)
        self.logger.debug(f"Added link: {source} -[{relation}]-> {target} (w={weight})")

    def find_path(self, start: str, end: str) -> List[str]:
        """
        Finds a path between two nodes, useful for tracing causality.
        """
        try:
            path = nx.shortest_path(self.graph, source=start, target=end)
            return path
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def get_context(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves the context (neighbors) of a node.
        """
        if not self.graph.has_node(node_id):
            return []

        context = []
        # Get successors (outgoing)
        for neighbor in self.graph.successors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            context.append({
                "node": neighbor,
                "relation": edge_data.get("relation"),
                "direction": "outgoing",
                "weight": edge_data.get("weight")
            })
        
        # Get predecessors (incoming)
        for neighbor in self.graph.predecessors(node_id):
            edge_data = self.graph.get_edge_data(neighbor, node_id)
            context.append({
                "node": neighbor,
                "relation": edge_data.get("relation"),
                "direction": "incoming",
                "weight": edge_data.get("weight")
            })
            
        return context

    def serialize(self) -> Dict[str, Any]:
        """Converts the graph to a dictionary for saving."""
        return nx.node_link_data(self.graph)

    def deserialize(self, data: Dict[str, Any]):
        """Loads the graph from a dictionary."""
        self.graph = nx.node_link_graph(data)

    def prune_fraction(self, edge_fraction: float = 0.3, node_fraction: float = 0.3):
        """
        Prune weakest edges/nodes by fraction of counts.
        - edge_fraction: remove this fraction of lowest-weight edges.
        - node_fraction: remove this fraction of lowest-degree nodes (after edge prune).
        """
        edges = list(self.graph.edges(data=True))
        if edges and edge_fraction > 0:
            edges_sorted = sorted(edges, key=lambda e: float(e[2].get("weight", 0.0)))
            cut = max(0, min(len(edges_sorted), int(len(edges_sorted) * edge_fraction)))
            for u, v, _ in edges_sorted[:cut]:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)

        nodes = list(self.graph.degree)
        if nodes and node_fraction > 0:
            nodes_sorted = sorted(nodes, key=lambda x: x[1])  # by degree
            cut_n = max(0, min(len(nodes_sorted), int(len(nodes_sorted) * node_fraction)))
            for node, _deg in nodes_sorted[:cut_n]:
                if self.graph.has_node(node) and self.graph.degree(node) == 0:
                    self.graph.remove_node(node)

        self.logger.info(
            f"Fractional prune applied: edges={edge_fraction*100:.0f}%, nodes={node_fraction*100:.0f}% (low-weight/low-degree)."
        )

    def forget(self, decay: float = 0.99, boost_nodes: Optional[List[str]] = None):
        """
        Entropy-driven forgetting: decay edge weights, remove near-zero edges, and drop isolated nodes.
        boost_nodes: nodes to protect/boost (e.g., reinforced by love events).
        """
        if not self.graph.edges:
            return

        for u, v, data in list(self.graph.edges(data=True)):
            w = float(data.get("weight", 0.0)) * decay
            if boost_nodes and (u in boost_nodes or v in boost_nodes):
                w = min(1.0, w * 1.1)  # slight protection
            if w < 1e-3:
                self.graph.remove_edge(u, v)
            else:
                self.graph.edges[u, v]["weight"] = w

        # Drop isolated nodes
        isolated = [n for n, deg in self.graph.degree if deg == 0]
        if isolated:
            self.graph.remove_nodes_from(isolated)
            self.logger.info(f"Forgot {len(isolated)} isolated nodes (entropy decay).")

    def prune_fraction(self, edge_fraction: float = 0.3, node_fraction: float = 0.3):
        """
        Prune weakest edges/nodes by fraction of counts.
        - edge_fraction: remove this fraction of lowest-weight edges.
        - node_fraction: remove this fraction of lowest-degree nodes (after edge prune).
        """
        edges = list(self.graph.edges(data=True))
        if edges and edge_fraction > 0:
            edges_sorted = sorted(edges, key=lambda e: float(e[2].get("weight", 0.0)))
            cut = max(0, min(len(edges_sorted), int(len(edges_sorted) * edge_fraction)))
            for u, v, _ in edges_sorted[:cut]:
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)

        # Remove nodes with lowest degree (and only if now isolated)
        nodes = list(self.graph.degree)
        if nodes and node_fraction > 0:
            nodes_sorted = sorted(nodes, key=lambda x: x[1])  # by degree
            cut_n = max(0, min(len(nodes_sorted), int(len(nodes_sorted) * node_fraction)))
            for node, _deg in nodes_sorted[:cut_n]:
                if self.graph.has_node(node) and self.graph.degree(node) == 0:
                    self.graph.remove_node(node)

        self.logger.info(
            f"Fractional prune applied: edges={edge_fraction*100:.0f}%, nodes={node_fraction*100:.0f}% (low-weight/low-degree)."
        )
