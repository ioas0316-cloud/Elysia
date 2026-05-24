"""
Elysia Holographic Memory Architecture
======================================
This module implements the true structural Holographic Memory system.
Instead of a sequential pipeline, knowledge is encoded as 4D waveforms
and superimposed (interfered) directly onto a multi-layered Clifford manifold.
Recall is achieved by rotating the manifold's 4D hyper-rotors (dials) via tension,
observing when specific concepts constructively interfere (resonate).

Rules strictly followed:
1. Causal interactions (no random initialization).
2. All geometry uses Quaternions.
"""

import math
import hashlib
from typing import Dict, List, Tuple
from core.math_utils import Quaternion

def concept_to_quaternion(concept: str) -> Quaternion:
    """
    Deterministically maps a text concept to a 4D unit quaternion (causal).
    Uses SHA-256 to generate deterministic coefficients.
    """
    h = hashlib.sha256(concept.encode('utf-8')).digest()
    
    # Extract 4 coordinates from the hash bytes
    # Combining bytes to get finer resolution
    w = ((h[0] ^ h[4]) + (h[1] ^ h[5]) * 256) / 32768.0 - 1.0
    x = ((h[2] ^ h[6]) + (h[3] ^ h[7]) * 256) / 32768.0 - 1.0
    y = ((h[8] ^ h[12]) + (h[9] ^ h[13]) * 256) / 32768.0 - 1.0
    z = ((h[10] ^ h[14]) + (h[11] ^ h[15]) * 256) / 32768.0 - 1.0
    
    q = Quaternion(w, x, y, z)
    return q.normalize()

class CliffordLayer:
    """
    A 4D spacetime manifold layer (rotor sphere).
    Unfolded, it behaves as a flat manifold; rolled up, it acts as a rotor.
    """
    def __init__(self, layer_id: int, base_frequency: float):
        self.layer_id = layer_id
        self.base_frequency = base_frequency
        
        # Superimposed memory state (4D waveform)
        self.manifold_state = Quaternion(0.0, 0.0, 0.0, 0.0)
        
        # Stored concept key representations (for resonance probing)
        self.concept_contents: Dict[str, Quaternion] = {}

    def get_rotors(self, tension: float) -> Tuple[Quaternion, Quaternion]:
        """
        Calculates left and right unit quaternions representing a double rotation in 4D.
        """
        theta_L = tension * self.base_frequency
        theta_R = tension * self.base_frequency * 1.6180339887  # Golden ratio scaling
        
        q_L = Quaternion(math.cos(theta_L), math.sin(theta_L), 0.0, 0.0)
        q_R = Quaternion(math.cos(theta_R), 0.0, math.sin(theta_R), 0.0)
        return q_L, q_R
        
    def superpose(self, concept: str, content_quat: Quaternion, tau_c: float):
        """
        Superimposes the concept wave onto the layer's manifold state.
        Applies a forward rotation corresponding to the concept's address.
        """
        self.concept_contents[concept] = content_quat
        q_L, q_R = self.get_rotors(tau_c)
        stored_wave = q_L * content_quat * q_R
        self.manifold_state = self.manifold_state + stored_wave

    def recall_state(self, tension: float) -> Quaternion:
        """
        Applies the inverse rotation corresponding to the scan tension.
        v_recalled = q_L.conjugate() * manifold_state * q_R.conjugate()
        """
        q_L, q_R = self.get_rotors(tension)
        return q_L.conjugate() * self.manifold_state * q_R.conjugate()

    def measure_resonance(self, recalled_state: Quaternion, concept: str) -> float:
        """
        Measures the alignment (coherence) of the recalled state with the concept's content key.
        """
        if concept not in self.concept_contents:
            return 0.0
        
        content_quat = self.concept_contents[concept]
        return recalled_state.dot(content_quat)


class HologramMemory:
    """
    A multi-layer Holographic Memory architecture.
    Manages multiple CliffordLayers, combining their responses to produce
    constructive and destructive interference peaks.
    """
    def __init__(self, num_layers: int = 3):
        self.layers: List[CliffordLayer] = []
        
        # Setup fractal layers with golden ratio frequency scaling
        freq = 1.0
        for i in range(num_layers):
            self.layers.append(CliffordLayer(layer_id=i, base_frequency=freq))
            freq *= 1.6180339887
            
        # Global map of registered concepts to their content key and address tension
        self.registered_concepts: Dict[str, Tuple[Quaternion, float]] = {}

    def register_concept(self, concept: str) -> Tuple[Quaternion, float]:
        """
        Generates and registers a canonical 4D content key and its resonant tension address.
        """
        if concept not in self.registered_concepts:
            content_quat = concept_to_quaternion(concept)
            
            # Generate deterministic address tension in range [1.0, 9.0]
            # Uses concept name salt to maintain causality
            h = hashlib.sha256((concept + "_address").encode('utf-8')).digest()
            tau_c = 1.0 + ((h[0] ^ h[2]) + (h[1] ^ h[3]) * 256) / 65535.0 * 8.0
            
            self.registered_concepts[concept] = (content_quat, tau_c)
        return self.registered_concepts[concept]

    def superpose(self, concept: str):
        """
        Ingests a concept and superimposes it across all layers.
        Knowledge is now 'interfered' and distributed across the manifold.
        """
        content_quat, tau_c = self.register_concept(concept)
        for layer in self.layers:
            layer.superpose(concept, content_quat, tau_c)

    def scan_resonance(self, tension: float) -> Dict[str, float]:
        """
        Scans all registered concepts at a specific tension.
        Returns the resonance score for each concept.
        """
        resonance_scores = {}
        for concept, (content_quat, tau_c) in self.registered_concepts.items():
            layer_scores = []
            for layer in self.layers:
                recalled_state = layer.recall_state(tension)
                score = layer.measure_resonance(recalled_state, concept)
                layer_scores.append(score)
            
            # Average resonance across all layers (holographic constructive interference)
            mean_score = sum(layer_scores) / len(self.layers)
            resonance_scores[concept] = mean_score
            
        return resonance_scores

    def get_trajectory_sample(self, tension: float) -> Tuple[float, float, float, float]:
        """
        Returns a 4D coordinate representing the composite state of the memory
        under the current tension, projectable to 2D/3D space.
        """
        composite = Quaternion(0.0, 0.0, 0.0, 0.0)
        for layer in self.layers:
            composite = composite + layer.recall_state(tension)
        comp_norm = composite.normalize()
        return comp_norm.w, comp_norm.x, comp_norm.y, comp_norm.z
