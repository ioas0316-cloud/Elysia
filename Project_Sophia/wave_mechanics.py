from collections import deque
from tools.kg_manager import KGManager
from Project_Sophia.vector_utils import cosine_sim
from Project_Sophia.core.tensor_wave import Tensor3D, propagate_wave, FrequencyWave
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
        Spreads activation energy (Scalar) from a starting node through the knowledge graph.
        Legacy method kept for backward compatibility.
        """
        activated_nodes = {}
        hop_of = {}

        start_node = self.kg_manager.get_node(start_node_id)
        if not start_node or 'embedding' not in start_node:
            return activated_nodes

        queue = deque([(start_node_id, initial_energy * max(0.5, min(2.0, float(emotion_gain))), 0)])
        visited = {start_node_id}
        hop_of[start_node_id] = 0

        while queue:
            current_id, current_energy, hop = queue.popleft()

            if current_energy < threshold:
                continue

            current_energy = min(current_energy, energy_cap)
            activated_nodes[current_id] = max(activated_nodes.get(current_id, 0.0), current_energy)

            current_node = self.kg_manager.get_node(current_id)
            if not current_node or 'embedding' not in current_node:
                continue

            current_embedding = current_node['embedding']
            all_neighbors = []
            for edge in self.kg_manager.kg['edges']:
                nxt = None
                if edge['source'] == current_id:
                    nxt = edge['target']
                elif edge['target'] == current_id:
                    nxt = edge['source']
                if nxt is None:
                    continue
                if nxt in hop_of and (hop_of[nxt] - hop) <= refractory_hops:
                    continue
                neighbor_node = self.kg_manager.get_node(nxt)
                if not neighbor_node or 'embedding' not in neighbor_node:
                    continue
                sim = max(0.0, cosine_sim(current_embedding, neighbor_node['embedding']))
                lens = 1.0
                if lens_weights and isinstance(lens_weights, dict):
                    lens = float(lens_weights.get(nxt, 1.0))
                score = sim * lens
                all_neighbors.append((nxt, sim, lens, score))

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

                if self.telemetry:
                    try:
                        self.telemetry.emit('activation_spread_step', {
                            'from': current_id, 'to': neighbor_id, 'hop': int(hop),
                            'energy_in': float(current_energy), 'similarity': float(similarity),
                            'energy_out': float(new_energy), 'accepted': bool(accepted)
                        })
                    except Exception:
                        pass

        return activated_nodes

    def propagate_tensor_wave(
        self,
        start_node_id: str,
        initial_tensor: Tensor3D,
        decay_factor: float = 0.9,
        threshold_mag: float = 0.2,
        max_hops: int = 3
    ) -> dict:
        """
        Propagates a 3D Tensor Wave through the Knowledge Graph.
        This models not just 'activation' but the 'quality' of the thought (Structure, Emotion, Identity).
        """
        activated_tensors = {} # node_id -> Tensor3D
        queue = deque([(start_node_id, initial_tensor, 0)])
        visited_hops = {start_node_id: 0}

        while queue:
            current_id, current_tensor, hop = queue.popleft()

            if current_tensor.magnitude() < threshold_mag or hop >= max_hops:
                continue

            # Merge tensor state if already visited (constructive interference)
            if current_id in activated_tensors:
                activated_tensors[current_id] = activated_tensors[current_id] + current_tensor
            else:
                activated_tensors[current_id] = current_tensor

            # Get current node data to check for intrinsic resonance
            current_node = self.kg_manager.get_node(current_id)
            if not current_node:
                continue

            # If node has its own tensor state, resonate with it
            node_tensor_data = current_node.get('tensor_state')
            if node_tensor_data:
                node_tensor = Tensor3D.from_dict(node_tensor_data)
                # Resonance: The wave picks up properties from the node
                current_tensor = propagate_wave(node_tensor, current_tensor, decay=1.0)

            # Find neighbors
            neighbors = self.kg_manager.get_neighbors(current_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited_hops or visited_hops[neighbor_id] > hop:
                    neighbor_node = self.kg_manager.get_node(neighbor_id)
                    if not neighbor_node: continue

                    # Propagate: neighbor gets a decayed version of current tensor
                    # Future improvement: Use edge properties (e.g., relationship type) to filter axes
                    # e.g., 'felt' edges propagate Y-axis (Emotion), 'implies' edges propagate X-axis (Logic)
                    next_tensor = current_tensor * decay_factor

                    visited_hops[neighbor_id] = hop + 1
                    queue.append((neighbor_id, next_tensor, hop + 1))

        return activated_tensors

    def get_resonance_between(self, start_node_id: str, end_node_id: str) -> float:
        activated_nodes = self.spread_activation(start_node_id=start_node_id, threshold=0.1)
        return activated_nodes.get(end_node_id, 0.0)

    def inject_stimulus(self, concept_id: str, energy_boost: float, tensor_state: dict = None):
        """
        Injects energy. Now supports optional 3D tensor state injection.
        """
        node = self.kg_manager.get_node(concept_id)
        if node:
            # Update scalar energy
            current_energy = node.get('activation_energy', 0.0)
            new_energy = current_energy + energy_boost
            updates = {'activation_energy': new_energy}

            # Update tensor state if provided
            if tensor_state:
                current_tensor_data = node.get('tensor_state', {})
                current_tensor = Tensor3D.from_dict(current_tensor_data)
                input_tensor = Tensor3D.from_dict(tensor_state)
                new_tensor = current_tensor + input_tensor
                updates['tensor_state'] = new_tensor.to_dict()

            self.kg_manager.update_node(concept_id, updates)

            if self.telemetry:
                try:
                    payload = {
                        'concept_id': concept_id,
                        'energy_boost': float(energy_boost),
                        'new_total_energy': float(new_energy)
                    }
                    if tensor_state:
                        payload['tensor_state'] = tensor_state
                    self.telemetry.emit('stimulus_injected', payload)
                except Exception:
                    pass
