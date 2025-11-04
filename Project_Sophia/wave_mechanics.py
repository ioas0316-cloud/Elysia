from collections import deque
from tools.kg_manager import KGManager
from Project_Sophia.vector_utils import cosine_sim # Import cosine_sim
try:
    from infra.telemetry import Telemetry
except Exception:
    Telemetry = None

class WaveMechanics:
    def __init__(self, kg_manager: KGManager, telemetry: Telemetry | None = None):
        self.kg_manager = kg_manager
        self.telemetry = telemetry

    def spread_activation(
        self,
        start_node_id: str,
        initial_energy: float = 1.0,
        decay_factor: float = 0.9,
        threshold: float = 0.2,
        lens_weights=None,
        emotion_gain: float = 1.0,
        top_k: int = 8,
        energy_cap: float = 1.5,
        refractory_hops: int = 2,
    ):
        """
        Spreads activation energy from a starting node through the knowledge graph,
        using embedding similarity to determine energy transfer.

        Returns a dictionary of nodes with activation energy above the threshold.
        """
        activated_nodes = {}
        hop_of = {}

        start_node = self.kg_manager.get_node(start_node_id)
        if not start_node:
            return activated_nodes

        use_embeddings = 'embedding' in start_node

        # Queue for BFS-like spreading: (node_id, energy, hop)
        queue = deque([(start_node_id, initial_energy * max(0.5, min(2.0, float(emotion_gain))), 0)])
        visited = {start_node_id}
        hop_of[start_node_id] = 0

        while queue:
            current_id, current_energy, hop = queue.popleft()

            if current_energy < threshold:
                continue

            # cap energy per node
            current_energy = min(current_energy, energy_cap)
            activated_nodes[current_id] = max(activated_nodes.get(current_id, 0.0), current_energy)

            current_node = self.kg_manager.get_node(current_id)
            if not current_node:
                continue

            # Find neighbors
            all_neighbors = []
            for edge in self.kg_manager.kg['edges']:
                # Allow revisiting if refractory satisfied
                nxt = None
                if edge['source'] == current_id:
                    nxt = edge['target']
                elif edge['target'] == current_id:
                    nxt = edge['source']
                if nxt is None:
                    continue
                # refractory: skip if seen within refractory_hops
                if nxt in hop_of and (hop_of[nxt] - hop) <= refractory_hops:
                    continue

                neighbor_node = self.kg_manager.get_node(nxt)
                if not neighbor_node:
                    continue

                sim = 1.0  # Default similarity for non-embedding graphs
                if use_embeddings:
                    if 'embedding' not in current_node or 'embedding' not in neighbor_node:
                        continue
                    sim = max(0.0, cosine_sim(current_node['embedding'], neighbor_node['embedding']))

                lens = 1.0
                if lens_weights and isinstance(lens_weights, dict):
                    lens = float(lens_weights.get(nxt, 1.0))
                score = sim * lens
                all_neighbors.append((nxt, sim, lens, score))

            # select top_k neighbors by similarity*lens
            if top_k and len(all_neighbors) > top_k:
                all_neighbors.sort(key=lambda x: x[3], reverse=True)
                selected = all_neighbors[:top_k]
            else:
                selected = all_neighbors

            for neighbor_id, similarity, lens, _score in selected:
                visited.add(neighbor_id)
                new_energy = current_energy * similarity * decay_factor * lens
                accepted = new_energy >= threshold
                if accepted:
                    hop_of[neighbor_id] = hop + 1
                    queue.append((neighbor_id, new_energy, hop + 1))
                # Emit a per-edge step (even if below threshold) for traceability
                if self.telemetry:
                    try:
                        self.telemetry.emit(
                            'activation_spread_step',
                            {
                                'from': current_id,
                                'to': neighbor_id,
                                'hop': int(hop),
                                'energy_in': float(current_energy),
                                'similarity': float(similarity),
                                'decay_factor': float(decay_factor),
                                'lens_weight': float(lens),
                                'energy_out': float(new_energy),
                                'accepted': bool(accepted),
                            }
                        )
                    except Exception:
                        pass

        return activated_nodes
