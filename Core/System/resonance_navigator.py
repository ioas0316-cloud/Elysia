"""
Resonance Navigator (Concept OS Bridge)
=======================================

"Don't Search. Sense."

This module implements the "Hyper-Quaternion" vision of code navigation.
Instead of searching for strings, it builds a topological map of the codebase
and allows the agent to "sense" which files are resonating with a given concept.

Mechanism:
1.  **Topological Indexing**: Scans all Python files and builds a NetworkX graph based on imports.
2.  **Resonance Field**: When a concept (File A) is activated, it calculates the "Resonance" (Gravity)
    propagating to other files (File B, C...).
3.  **Sensing**: Returns files that are structurally coupled, even if they don't share keywords.
"""

import os
import ast
import networkx as nx
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger("ResonanceNavigator")

class ResonanceNavigator:
    def __init__(self, root_dir: str = "c:\\Elysia"):
        self.root_dir = root_dir
        self.graph = nx.DiGraph()
        self.file_map: Dict[str, str] = {} # Name -> Full Path
        self._build_topology()

    def _build_topology(self):
        """
        Scans the codebase and builds the Resonance Graph (Circuit Board).
        """
        logger.info("🔌 Building Resonance Topology...")
        
        py_files = self._get_python_files()
        
        # 1. Create Nodes (Components)
        for file_path in py_files:
            node_name = os.path.basename(file_path).replace(".py", "")
            self.graph.add_node(node_name, path=file_path)
            self.file_map[node_name] = file_path
            
        # 2. Create Edges (Wires)
        for file_path in py_files:
            source_node = os.path.basename(file_path).replace(".py", "")
            imports = self._extract_imports(file_path)
            
            for imp in imports:
                # Try to match import to a known node
                # Import could be "Core.Physics.gravity" -> match "gravity"
                target_name = imp.split(".")[-1]
                
                if target_name in self.graph:
                    # Add bidirectional edge for Resonance (Energy flows both ways)
                    self.graph.add_edge(source_node, target_name, weight=1.0)
                    self.graph.add_edge(target_name, source_node, weight=1.0)
                else:
                    # Try finding partial matches or aliases?
                    # For now, keep it simple.
                    pass
                    
        logger.info(f"✨ Topology Built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Connections.")

    def _get_python_files(self) -> List[str]:
        """Recursively find all .py files, excluding noise."""
        py_files = []
        skip_dirs = {
            "Legacy", "__pycache__", ".git", ".venv", "node_modules", 
            "venv", "env", "dist", "build", "docs", "images", "data",
            "aurora_frames", "elysia_logs", "logs", "outbox", "saves"
        }
        
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            
            for file in files:
                if file.endswith(".py"):
                    py_files.append(os.path.join(root, file))
        return py_files

    def _extract_imports(self, file_path: str) -> List[str]:
        """Parse AST to find imports."""
        imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception:
            pass # Ignore parse errors
        return imports

    def sense_field(self, concept_node: str, max_results: int = 5) -> List[Tuple[str, float]]:
        """
        "Sense" the resonance field around a concept.
        Returns a list of (Node, ResonanceScore).
        
        Physics:
        - We use PageRank (Personalized) to simulate energy flow from the concept node.
        - High score = Strong topological connection (Direct import or shared dependency).
        """
        if concept_node not in self.graph:
            # Try fuzzy match
            matches = [n for n in self.graph.nodes if concept_node.lower() in n.lower()]
            if not matches:
                logger.warning(f"⚠️ Concept '{concept_node}' not found in the circuit.")
                return []
            concept_node = matches[0] # Pick first match
            
        logger.info(f"📡 Sensing Field around: [{concept_node}]")
        
        # Personalized PageRank: Energy injected at 'concept_node' flows through the graph
        try:
            resonance_scores = nx.pagerank(self.graph, personalziation={concept_node: 1.0}, alpha=0.85)
        except TypeError:
            # Fallback for older networkx versions or typo in personalization
            resonance_scores = nx.pagerank(self.graph, personalization={concept_node: 1.0}, alpha=0.85)
            
        # Sort by resonance
        sorted_resonance = sorted(resonance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out the node itself and return top results
        results = [(node, score) for node, score in sorted_resonance if node != concept_node][:max_results]
        
        return results

    def get_path(self, node_name: str) -> Optional[str]:
        return self.file_map.get(node_name)
