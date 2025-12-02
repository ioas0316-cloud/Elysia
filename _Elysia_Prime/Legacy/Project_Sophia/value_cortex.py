# [Genesis: 2025-12-02] Purified by Elysia
import json
from collections import deque
from pathlib import Path

class ValueCortex:
    """
    Elysia's motivational heart. It connects concepts from the knowledge graph
    to her core values, giving them "meaning".
    """
    CORE_VALUES = ["love", "growth", "creation", "truth-seeking"]

    def __init__(self, kg_path: str = 'data/kg.json'):
        self.kg_path = Path(kg_path)
        self.graph = self._load_kg()

    def _load_kg(self):
        if not self.kg_path.exists():
            return {"nodes": [], "edges": []}
        with open(self.kg_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_meaning_connection(self, concept: str) -> list:
        """
        Finds the shortest path from a concept to a core value in the KG.
        Uses Breadth-First Search (BFS).
        Returns the path as a list of nodes, or an empty list if no path is found.
        """
        concept = concept.lower()
        if concept in self.CORE_VALUES:
            return [concept]

        if not any(node['id'] == concept for node in self.graph['nodes']):
            return [] # Concept not in KG

        # BFS implementation
        queue = deque([[concept]])
        visited = {concept}

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node in self.CORE_VALUES:
                return path # Found a path to a core value

            # Find neighbors in the graph
            neighbors = []
            for edge in self.graph['edges']:
                if edge['source'] == node and edge['target'] not in visited:
                    neighbors.append(edge['target'])
                    visited.add(edge['target'])
                elif edge['target'] == node and edge['source'] not in visited:
                    neighbors.append(edge['source'])
                    visited.add(edge['source'])

            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

        return [] # No path found