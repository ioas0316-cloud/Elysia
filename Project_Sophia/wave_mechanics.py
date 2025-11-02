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
        if not start_node:
            return activated_nodes

        # Queue for BFS-like spreading: (node_id, energy)
        queue = deque([(start_node_id, initial_energy)])
        visited = {start_node_id}

        while queue:
            current_id, current_energy = queue.popleft()

            if current_energy < threshold:
                continue

            activated_nodes[current_id] = current_energy

            # Find neighbors and spread energy based on relation type
            for edge in self.kg_manager.kg['edges']:
                source, target = edge['source'], edge['target']
                neighbor_id = None
                if source == current_id and target not in visited:
                    neighbor_id = target
                elif target == current_id and source not in visited:
                    neighbor_id = source

                if neighbor_id:
                    relation_type = edge.get('relation', 'related_to')
                    
                    # Energy transfer logic based on relation
                    if relation_type == 'activates':
                        transfer_factor = 1.0
                    else:
                        transfer_factor = 0.5 # Other relations transfer less
                        
                    new_energy = current_energy * transfer_factor * decay_factor
                    
                    if new_energy >= threshold:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, new_energy))

        return activated_nodes
