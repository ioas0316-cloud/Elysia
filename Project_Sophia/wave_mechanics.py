from collections import deque
from tools.kg_manager import KGManager
from Project_Sophia.vector_utils import cosine_sim # Import cosine_sim

class WaveMechanics:
    def __init__(self, kg_manager: KGManager):
        self.kg_manager = kg_manager

    def spread_activation(self, start_node_id: str, initial_energy: float = 1.0, decay_factor: float = 0.9, threshold: float = 0.2):
        """
        Spreads activation energy from a starting node through the knowledge graph,
        using embedding similarity to determine energy transfer.

        Returns a dictionary of nodes with activation energy above the threshold.
        """
        activated_nodes = {}

        start_node = self.kg_manager.get_node(start_node_id)
        if not start_node or 'embedding' not in start_node:
            return activated_nodes

        # Queue for BFS-like spreading: (node_id, energy)
        queue = deque([(start_node_id, initial_energy)])
        visited = {start_node_id}

        while queue:
            current_id, current_energy = queue.popleft()

            if current_energy < threshold:
                continue

            activated_nodes[current_id] = current_energy

            current_node = self.kg_manager.get_node(current_id)
            if not current_node or 'embedding' not in current_node:
                continue

            current_embedding = current_node['embedding']

            # Find neighbors
            neighbors = []
            for edge in self.kg_manager.kg['edges']:
                if edge['source'] == current_id and edge['target'] not in visited:
                    neighbors.append(edge['target'])
                elif edge['target'] == current_id and edge['source'] not in visited:
                    neighbors.append(edge['source'])

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    neighbor_node = self.kg_manager.get_node(neighbor_id)
                    if not neighbor_node or 'embedding' not in neighbor_node:
                        continue
                        
                    neighbor_embedding = neighbor_node['embedding']
                    
                    # Calculate similarity and transfer energy
                    similarity = cosine_sim(current_embedding, neighbor_embedding)
                    
                    # Ensure similarity is non-negative
                    similarity = max(0, similarity)

                    new_energy = current_energy * similarity * decay_factor
                    
                    if new_energy >= threshold:
                        queue.append((neighbor_id, new_energy))

        return activated_nodes
