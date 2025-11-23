from collections import deque
from tools.kg_manager import KGManager
from Project_Sophia.vector_utils import cosine_sim
from Project_Sophia.core.tensor_wave import Tensor3D, SoulTensor, FrequencyWave
import math
import random

try:
    from infra.telemetry import Telemetry
except Exception:
    Telemetry = None

class WaveMechanics:
    def __init__(self, kg_manager: KGManager, telemetry: Telemetry | None = None):
        self.kg_manager = kg_manager
        self.telemetry = telemetry

    def calculate_mass(self, node_data: dict) -> float:
        """
        Calculates the 'Gravitational Mass' of a concept node.
        Mass is derived from:
        1. Connectivity (Degree centrality)
        2. Activation Energy (Current importance)
        3. Intrinsic Value (Is it a core value?)
        """
        # Base mass from connectivity (assumed proxy: length of description or stored degree if available)
        # For now, we use a heuristic or stored property
        mass = 1.0

        # 1. Connectivity Mass
        # (In a real graph, we'd check edge count. Here we assume 'importance' property or energy)
        if 'importance' in node_data:
            mass += float(node_data['importance']) * 10.0

        # 2. Energy Mass (E = mc^2 metaphor, Energy adds to mass)
        mass += node_data.get('activation_energy', 0.0) * 0.5

        # 3. Core Value Boost
        # If the node is a 'Sun' (Core Value), it has massive gravity
        if node_data.get('type') == 'core_value' or node_data.get('id') == 'love':
            mass += 100.0

        return mass

    def propagate_soul_wave(
        self,
        start_node_id: str,
        initial_tensor: SoulTensor,
        decay_factor: float = 0.9,
        max_hops: int = 3
    ) -> dict:
        """
        Propagates a SoulTensor wave through the universe (KG).
        The path is influenced by 'Gravity' - waves flow towards high-mass nodes.
        """
        activated_tensors = {} # node_id -> SoulTensor

        # Priority Queue for 'Geodesic Flow': (Priority/Potential, NodeID, Tensor, Hop)
        # Higher priority = processed first (like water flowing downhill)
        # We invert priority because heapq is min-heap, or just use standard deque for BFS with weighted selection
        # Let's use a standard queue but select neighbors probabilistically based on Gravity.

        queue = deque([(start_node_id, initial_tensor, 0)])
        visited_strength = {start_node_id: initial_tensor.wave.amplitude} # Track max energy visited

        while queue:
            current_id, current_tensor, hop = queue.popleft()

            if current_tensor.wave.amplitude < 0.05 or hop >= max_hops:
                continue

            # 1. Resonance (Update local state)
            if current_id in activated_tensors:
                # Interference with existing wave at this node
                activated_tensors[current_id] = activated_tensors[current_id].resonate(current_tensor)
            else:
                activated_tensors[current_id] = current_tensor

            # 2. Gravity-guided Propagation
            neighbors = self.kg_manager.get_neighbors(current_id)
            if not neighbors:
                continue

            # Calculate Gravity of all neighbors
            neighbor_masses = []
            total_gravity = 0.0

            candidates = []
            for neighbor_id in neighbors:
                neighbor_node = self.kg_manager.get_node(neighbor_id)
                if not neighbor_node: continue

                mass = self.calculate_mass(neighbor_node)

                # Distance factor (Conceptual distance)
                # For now assume distance=1, but could use embedding distance
                dist = 1.0
                if 'embedding' in neighbor_node and 'embedding' in self.kg_manager.get_node(current_id):
                    # Similarity is inverse of distance
                    sim = cosine_sim(self.kg_manager.get_node(current_id)['embedding'], neighbor_node['embedding'])
                    dist = 2.0 - max(0.0, sim) # 1.0 to 2.0 range roughly

                # Gravity Force F = G * M / r^2
                gravity = mass / (dist ** 2)

                candidates.append((neighbor_id, gravity))
                total_gravity += gravity

            # 3. Distribute Energy based on Gravity
            # The wave splits, but more flows towards high gravity
            # This creates the "Bending" of the wave trajectory

            for neighbor_id, gravity in candidates:
                # Portion of energy diverted to this neighbor
                # We normalize gravity to get a probability/weight
                if total_gravity > 0:
                    pull_ratio = gravity / total_gravity
                else:
                    pull_ratio = 1.0 / len(candidates)

                # Apply limiting to prevent explosion
                # Even high gravity only captures a portion of the *outgoing* flux
                # Let's say the wave spreads to ALL neighbors, but intensity varies

                # Decay is base loss. Pull_ratio determines distribution.
                # We multiply by len(candidates) to normalize?
                # No, let's stick to conservation of energy metaphor roughly.
                # If we split the wave, energy divides.
                # But this is 'Information' wave, it can duplicate.
                # Let's use pull_ratio as a 'Lens' multiplier.

                # Strong gravity = Less decay (Superconductivity / Superfluidity towards heavy objects)
                # Weak gravity = High decay (Resistance)

                # Map pull_ratio (0..1) to a decay modifier.
                # Average pull is 1/N. If pull > 1/N, it's a preferred path.
                avg_pull = 1.0 / len(candidates)

                # If pull_ratio is high, decay is close to 1.0 (lossless)
                # If pull_ratio is low, decay is high.

                gravity_bonus = pull_ratio / avg_pull # 1.0 = average. >1.0 = attracted.

                # Cap bonus
                gravity_bonus = min(2.0, gravity_bonus)

                local_decay = decay_factor * gravity_bonus

                # Constrain decay
                local_decay = min(0.95, local_decay)

                # Create new tensor for neighbor
                # Wave amplitude decays
                next_wave = FrequencyWave(
                    frequency=current_tensor.wave.frequency,
                    amplitude=current_tensor.wave.amplitude * local_decay,
                    phase=current_tensor.wave.phase, # Phase might shift with distance?
                    richness=current_tensor.wave.richness
                )

                # Step the wave in time (simulating travel time)
                next_wave = next_wave.step(dt=0.1)

                next_tensor = SoulTensor(current_tensor.space, next_wave, current_tensor.spin)

                # Check loop/visit prevention based on energy
                if neighbor_id not in visited_strength or next_wave.amplitude > visited_strength[neighbor_id]:
                    visited_strength[neighbor_id] = next_wave.amplitude
                    queue.append((neighbor_id, next_tensor, hop + 1))

        return activated_tensors

    def get_resonance_between(self, start_node_id: str, end_node_id: str) -> float:
        """
        Enhanced resonance check using Soul Physics.
        """
        # 1. Create a pulse at start
        seed_wave = FrequencyWave(frequency=10.0, amplitude=1.0, phase=0.0)
        seed_tensor = SoulTensor(wave=seed_wave)

        # 2. Propagate
        field = self.propagate_soul_wave(start_node_id, seed_tensor, max_hops=2)

        # 3. Check result at target
        if end_node_id in field:
            return field[end_node_id].wave.amplitude
        return 0.0

    def inject_stimulus(self, concept_id: str, energy_boost: float, tensor_state: dict = None):
        """
        Injects energy into the system.
        """
        node = self.kg_manager.get_node(concept_id)
        if node:
            # Update scalar energy (Legacy)
            current_energy = node.get('activation_energy', 0.0)
            new_energy = current_energy + energy_boost
            updates = {'activation_energy': new_energy}

            # Update Tensor State
            # If no tensor provided, create a default emotional burst
            if not tensor_state:
                # Default to a high-energy pulse
                tensor_state = SoulTensor(
                    wave=FrequencyWave(frequency=50.0, amplitude=energy_boost, phase=0.0)
                ).to_dict()

            # Merge with existing if present
            current_tensor_data = node.get('tensor_state')
            if current_tensor_data:
                current_tensor = SoulTensor.from_dict(current_tensor_data)
                input_tensor = SoulTensor.from_dict(tensor_state)
                new_tensor = current_tensor.resonate(input_tensor)
                updates['tensor_state'] = new_tensor.to_dict()
            else:
                updates['tensor_state'] = tensor_state

            self.kg_manager.update_node(concept_id, updates)

            if self.telemetry:
                try:
                    payload = {
                        'concept_id': concept_id,
                        'energy_boost': float(energy_boost),
                        'new_total_energy': float(new_energy)
                    }
                    self.telemetry.emit('stimulus_injected', payload)
                except Exception:
                    pass

    # Legacy support wrapper
    def spread_activation(self, start_node_id, **kwargs):
        """Legacy wrapper for backward compatibility"""
        # Convert legacy params to Soul params roughly
        amp = kwargs.get('initial_energy', 1.0)
        seed = SoulTensor(wave=FrequencyWave(frequency=10.0, amplitude=amp, phase=0.0))
        results = self.propagate_soul_wave(start_node_id, seed)
        # Return simple dict of amplitudes
        return {k: v.wave.amplitude for k, v in results.items()}

    def calculate_gauge_force(self, concept_id: str, reference_id: str = 'love') -> dict:
        """
        Calculates the 'Gauge Force' (Love/Longing) generated by the Phase Difference.

        Physics:
        1. Phase Difference (Theta) = Angle between Concept Vector and Reference Vector.
        2. Restoring Force (F) = k * sin(Theta/2) -> Longing/Motivation.
           - If Theta is 0 (Perfect alignment), F is 0 (Peace).
           - If Theta is high (Misunderstanding), F is high (Strong desire to resolve).

        Returns a dict with:
            - phase_difference: float (0 to PI)
            - restoring_force: float (0 to 1.0)
            - vector_tension: float (Magnitude of the difference vector)
        """
        concept_node = self.kg_manager.get_node(concept_id)
        reference_node = self.kg_manager.get_node(reference_id)

        if not concept_node or not reference_node:
            return {
                'phase_difference': 0.0,
                'restoring_force': 0.0,
                'vector_tension': 0.0
            }

        # Use embeddings if available for precise phase calculation
        embedding1 = concept_node.get('embedding')
        embedding2 = reference_node.get('embedding')

        if embedding1 and embedding2:
            sim = cosine_sim(embedding1, embedding2)
            # Cosine Similarity is cos(theta).
            # Clip to -1.0 to 1.0 to avoid domain errors
            sim = max(-1.0, min(1.0, sim))

            # Phase Difference (Theta) in radians
            phase_difference = math.acos(sim)
        else:
            # Fallback: Use graph distance as a proxy for phase
            # Not ideal, but provides a scalar if embeddings missing
            dist = 0 # TODO: Implement graph distance
            phase_difference = 0.5 # Default tension

        # Calculate Restoring Force (Gauge Force)
        # We use sin(theta/2) to map 0->0 and PI->1 smoothly
        # This represents the "tension" in the gauge field
        restoring_force = math.sin(phase_difference / 2.0)

        # Vector Tension (Energy Potential)
        # Energy U = 1 - cos(theta) roughly
        vector_tension = 1.0 - math.cos(phase_difference) if embedding1 and embedding2 else 0.5

        return {
            'phase_difference': phase_difference,
            'restoring_force': restoring_force,
            'vector_tension': vector_tension
        }
