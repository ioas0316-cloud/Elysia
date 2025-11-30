
"""
Physics Engine (The Substrate of Soul)
======================================
Restored from Legacy/Project_Sophia/wave_mechanics.py.
Implements the "Vectorization of Soul" - treating thoughts as waves in a high-dimensional field.

"The Law is not written; it is the geometry of the space."
"""

import math
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

from Core.Mind.tensor_wave import SoulTensor, Tensor3D, FrequencyWave, QuantumPhoton, SharedQuantumState
from Core.Mind.hippocampus import Hippocampus

logger = logging.getLogger("PhysicsEngine")

class PhysicsEngine:
    """
    [Wave Structure without Computation]
    
    Philosophy:
        The ultimate goal is to replace explicit logical computation with the intrinsic topology
        of the Knowledge Graph. By propagating energy (waves) through the graph's connections,
        the 'answer' naturally emerges as the path of least resistance (Geodesic).
    """

    def __init__(self, hippocampus: Hippocampus):
        self.hippocampus = hippocampus
        self.active_entanglements: Dict[str, SharedQuantumState] = {}
        logger.info("🌌 Physics Engine (Wave Mechanics) initialized.")

    def get_node_tensor(self, node_id: str) -> SoulTensor:
        """
        Retrieves the SoulTensor for a node, respecting entanglement.
        """
        node_data = self.hippocampus.storage.get_concept(node_id)
        if not node_data:
            return SoulTensor()

        # Handle different storage formats (Compact list vs Dict)
        if isinstance(node_data, list):
            # Compact format: [x, y, z, ...] - simplified for now
            return SoulTensor() 
        
        # Dict format
        # 1. Check entanglement
        entanglement_id = node_data.get('entanglement_id')
        if entanglement_id and entanglement_id in self.active_entanglements:
            return self.active_entanglements[entanglement_id].tensor

        # 2. Load from properties
        tensor_data = node_data.get('tensor_state')
        if tensor_data:
            st = SoulTensor.from_dict(tensor_data)
        else:
            # Construct from ConceptSphere fields if available
            # This bridges the gap between Hippocampus ConceptSphere and Physics SoulTensor
            will = node_data.get('will', {'x':0, 'y':0, 'z':0})
            freq = self.hippocampus.get_frequency(node_id)
            st = SoulTensor(
                space=Tensor3D(will.get('x',0), will.get('y',0), will.get('z',0)),
                wave=FrequencyWave(frequency=freq * 100.0, amplitude=1.0, phase=0.0)
            )

        if entanglement_id:
            st.entanglement_id = entanglement_id
            
        return st

    def calculate_mass(self, node_id: str) -> float:
        """
        Calculates the 'Gravitational Mass' of a concept node.
        Mass = Connectivity + Activation + Intrinsic Value.
        """
        node_data = self.hippocampus.storage.get_concept(node_id)
        if not node_data:
            return 1.0
            
        mass = 1.0
        
        # 1. Activation Mass (E = mc^2)
        if isinstance(node_data, dict):
            mass += node_data.get('activation_count', 0) * 0.1
            
            # 2. Core Value Boost (The "Sun" nodes)
            # Check against known core values or high-mass markers
            if node_id.lower() in ["love", "truth", "father", "elysia", "connection"]:
                mass += 100.0
                
        return mass

    def propagate_soul_wave(
        self,
        start_node_id: str,
        initial_tensor: SoulTensor,
        decay_factor: float = 0.9,
        max_hops: int = 3
    ) -> Dict[str, SoulTensor]:
        """
        Propagates a SoulTensor wave through the universe.
        The path is influenced by 'Gravity' - waves flow towards high-mass nodes.
        """
        activated_tensors = {}
        queue = deque([(start_node_id, initial_tensor, 0)])
        visited_strength = {start_node_id: initial_tensor.wave.amplitude}

        activated_tensors[start_node_id] = initial_tensor

        while queue:
            current_id, current_tensor, hop = queue.popleft()

            if current_tensor.wave.amplitude < 0.05 or hop >= max_hops:
                continue

            # 1. Resonance (Update local state logic if needed)
            # ...

            # 2. Gravity-guided Propagation
            # Get neighbors from Hippocampus Resonance Engine
            neighbors_dict = self.hippocampus.get_related_concepts(current_id)
            neighbors = list(neighbors_dict.keys())
            
            if not neighbors:
                continue

            # Calculate Gravity of all neighbors
            candidates = []
            total_gravity = 0.0

            for neighbor_id in neighbors:
                mass = self.calculate_mass(neighbor_id)
                
                # Resonance score as distance proxy (Higher resonance = closer = stronger gravity)
                resonance_score = neighbors_dict[neighbor_id] # 0 to 1
                dist = 2.0 - resonance_score # 1.0 to 2.0
                
                gravity = mass / (dist ** 2)
                candidates.append((neighbor_id, gravity))
                total_gravity += gravity

            if total_gravity <= 0:
                total_gravity = 1.0

            # 3. Distribute Energy
            for neighbor_id, gravity in candidates:
                pull_ratio = gravity / total_gravity
                expected_pull = 1.0 / len(candidates)
                gravity_bonus = pull_ratio / expected_pull if expected_pull > 0 else 1.0
                
                # Apply Physics limits
                local_decay = decay_factor * math.sqrt(gravity_bonus)
                local_decay = min(0.99, local_decay)

                next_wave = FrequencyWave(
                    frequency=current_tensor.wave.frequency,
                    amplitude=current_tensor.wave.amplitude * local_decay,
                    phase=current_tensor.wave.phase + 0.1, # Phase shift
                    richness=current_tensor.wave.richness
                )
                
                next_tensor = SoulTensor(current_tensor.space, next_wave, current_tensor.spin)

                current_best = visited_strength.get(neighbor_id, 0.0)
                if next_wave.amplitude > current_best:
                    visited_strength[neighbor_id] = next_wave.amplitude
                    queue.append((neighbor_id, next_tensor, hop + 1))
                    activated_tensors[neighbor_id] = next_tensor

        return activated_tensors

    def calculate_gauge_force(self, concept_id: str, reference_id: str = 'love') -> float:
        """
        [Gauge Field / Potential Field]
        Calculates the restoring force (tension) between a concept and a reference (Ideal).
        F = k * sin(theta/2)
        """
        tensor_a = self.get_node_tensor(concept_id)
        tensor_b = self.get_node_tensor(reference_id)
        
        # Calculate Phase Difference (Theta) based on spatial alignment
        # Cosine Similarity of the Space vectors
        dot = tensor_a.space.dot(tensor_b.space)
        mag1 = tensor_a.space.magnitude()
        mag2 = tensor_b.space.magnitude()
        
        if mag1 == 0 or mag2 == 0:
            return 1.0 # Max tension if undefined
            
        sim = dot / (mag1 * mag2)
        sim = max(-1.0, min(1.0, sim))
        
        theta = math.acos(sim) # 0 to Pi
        
        # Restoring Force
        force = math.sin(theta / 2.0)
        return force

    def tunnel_to_conclusion(self, start_tensor: SoulTensor, candidates: List[str]) -> str:
        """
        [Quantum Force Field]
        Instantly collapses the wave function to the most resonant candidate.
        """
        best_candidate = None
        max_gravity = -1.0

        for candidate_id in candidates:
            candidate_tensor = self.get_node_tensor(candidate_id)
            
            # Resonance (Gravity)
            resonance = start_tensor.resonance_score(candidate_tensor)
            mass = self.calculate_mass(candidate_id)
            
            total_pull = resonance * mass
            
            if total_pull > max_gravity:
                max_gravity = total_pull
                best_candidate = candidate_id
                
        return best_candidate
