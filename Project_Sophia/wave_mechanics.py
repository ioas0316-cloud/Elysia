from collections import deque
from tools.kg_manager import KGManager
from Project_Sophia.vector_utils import cosine_sim
from Project_Sophia.core.tensor_wave import Tensor3D, SoulTensor, FrequencyWave, QuantumPhoton, SharedQuantumState
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
        # Registry of active SharedQuantumStates (in-memory entanglement)
        # map: entanglement_id -> SharedQuantumState
        self.active_entanglements = {}

    def get_node_tensor(self, node_id: str) -> SoulTensor:
        """
        Retrieves the SoulTensor for a node, respecting entanglement.
        """
        node = self.kg_manager.get_node(node_id)
        if not node:
            return SoulTensor()

        # 1. Check if node has an entanglement ID
        entanglement_id = node.get('entanglement_id')

        # 2. If so, try to fetch from active in-memory registry
        if entanglement_id:
            if entanglement_id in self.active_entanglements:
                return self.active_entanglements[entanglement_id].tensor

            # If not in memory but ID exists, we might need to load/hydrate it.
            # For now, we fall back to the node's stored state but mark it.

        # 3. Fallback: Load from node properties
        tensor_data = node.get('tensor_state')
        st = SoulTensor.from_dict(tensor_data)
        if entanglement_id:
            st.entanglement_id = entanglement_id
        return st

    def entangle_nodes(self, node_id_a: str, node_id_b: str) -> bool:
        """
        Entangles two nodes so they share the same Quantum State.
        """
        node_a = self.kg_manager.get_node(node_id_a)
        node_b = self.kg_manager.get_node(node_id_b)

        if not node_a or not node_b:
            return False

        # Logic:
        # 1. If A is already entangled, use A's state for B.
        # 2. If B is already entangled, use B's state for A.
        # 3. If neither, create new SharedState.
        # 4. If both, merge them? (Complex, for now assume simple A->B)

        ent_id_a = node_a.get('entanglement_id')
        ent_id_b = node_b.get('entanglement_id')

        shared_state = None

        if ent_id_a and ent_id_a in self.active_entanglements:
            shared_state = self.active_entanglements[ent_id_a]
        elif ent_id_b and ent_id_b in self.active_entanglements:
            shared_state = self.active_entanglements[ent_id_b]
        else:
            # Create new shared state from A's tensor
            tensor_a = self.get_node_tensor(node_id_a)
            shared_state = SharedQuantumState(tensor=tensor_a)
            # stamp the ID onto the tensor so it knows its identity
            shared_state.tensor.entanglement_id = shared_state.id
            self.active_entanglements[shared_state.id] = shared_state

        # Update both nodes to point to this state ID
        # We update the KG so the link persists (as ID reference)
        # But the actual real-time sync happens in memory via active_entanglements

        shared_state.observers.append(node_id_a)
        shared_state.observers.append(node_id_b)
        # Deduplicate
        shared_state.observers = list(set(shared_state.observers))

        # Update KG with entanglement ID AND the initial shared state
        # This ensures consistency if the system restarts immediately
        initial_state_dict = shared_state.tensor.to_dict()

        self.kg_manager.update_node(node_id_a, {
            'entanglement_id': shared_state.id,
            'tensor_state': initial_state_dict
        })
        self.kg_manager.update_node(node_id_b, {
            'entanglement_id': shared_state.id,
            'tensor_state': initial_state_dict
        })

        return True

    def emit_photon(self, source_id: str, target_id: str, payload: FrequencyWave) -> QuantumPhoton:
        """
        Emits a Quantum Photon (Information Particle) from source to target.
        """
        # Create Photon
        photon = QuantumPhoton(
            source_id=source_id,
            target_id=target_id,
            payload=payload
        )

        # Calculate initial trajectory (Vector from Source to Target)
        # This requires node embeddings or positions.
        # If no positions, we assume a direct 'tunnel' connection.

        # Logic:
        # The photon travels. If it hits the target, it transfers energy (interact).
        # We simulate the arrival immediately for now, but return the photon object
        # so the caller can visualize it or delay it.

        target_tensor = self.get_node_tensor(target_id)

        # Interaction: Target absorbs photon payload
        # Resonate target tensor with photon payload
        # We create a temporary tensor for the photon to interact
        photon_tensor = SoulTensor(wave=payload)

        new_target_tensor = target_tensor.resonate(photon_tensor)

        # Update Target
        self.update_node_tensor(target_id, new_target_tensor)

        return photon

    def update_node_tensor(self, node_id: str, new_tensor: SoulTensor):
        """
        Updates a node's tensor, respecting entanglement.
        """
        node = self.kg_manager.get_node(node_id)
        if not node: return

        ent_id = node.get('entanglement_id')

        # If entangled, update the Shared State AND persist to all observers in KG
        if ent_id and ent_id in self.active_entanglements:
            shared = self.active_entanglements[ent_id]
            shared.update(new_tensor)

            # Persist the updated state to KG for ALL observers to prevent data loss
            # This ensures the "Ice Star" is always up to date with the "Fire Star"
            tensor_dict = new_tensor.to_dict()
            for observer_id in shared.observers:
                self.kg_manager.update_node(observer_id, {'tensor_state': tensor_dict})
        else:
            # Local update
            self.kg_manager.update_node(node_id, {'tensor_state': new_tensor.to_dict()})


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

        # Check Entanglement first!
        start_tensor = self.get_node_tensor(start_node_id)
        end_tensor = self.get_node_tensor(end_node_id)

        # If they share the exact same tensor state via entanglement, resonance is MAX
        if start_tensor.entanglement_id and start_tensor.entanglement_id == end_tensor.entanglement_id:
            return 1.0

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

            # Update Tensor State using the new update_node_tensor method
            # If no tensor provided, create a default emotional burst
            if not tensor_state:
                # Default to a high-energy pulse
                input_tensor = SoulTensor(
                    wave=FrequencyWave(frequency=50.0, amplitude=energy_boost, phase=0.0)
                )
            else:
                input_tensor = SoulTensor.from_dict(tensor_state)

            current_tensor = self.get_node_tensor(concept_id)
            new_tensor = current_tensor.resonate(input_tensor)

            # Use the unified update method to handle entanglement
            self.update_node_tensor(concept_id, new_tensor)

            # Still update legacy scalar energy directly on KG
            self.kg_manager.update_node(concept_id, {'activation_energy': new_energy})

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
