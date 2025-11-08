import json
from collections import deque
from tools.kg_manager import KGManager

class ValueCortex:
    """
    Elysia's motivational heart. It connects concepts from the knowledge graph
    to her core values, giving them "meaning".
    """
    CORE_VALUES = ["love", "growth", "creation", "truth-seeking"]

    def __init__(self, kg_manager: KGManager):
        self.kg_manager = kg_manager

    def find_meaning_connection(self, concept: str) -> list:
        """
        Finds the shortest path from a concept to a core value in the KG.
        Uses Breadth-First Search (BFS).
        Returns the path as a list of nodes, or an empty list if no path is found.
        """
        concept = concept.lower()
        if concept in self.CORE_VALUES:
            return [concept]

        graph = self.kg_manager.kg
        if not self.kg_manager.get_node(concept):
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
            for edge in graph.get('edges', []):
                if edge.get('source') == node and edge.get('target') not in visited:
                    neighbors.append(edge['target'])
                    visited.add(edge['target'])
                elif edge.get('target') == node and edge.get('source') not in visited:
                    neighbors.append(edge['source'])
                    visited.add(edge['source'])

            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

        return [] # No path found
