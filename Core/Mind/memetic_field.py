"""
Memetic Field (The Geometry of Meaning)
=======================================

This module implements the "Memetic Engine" where concepts are not just weights,
but 64-dimensional HyperQuaternions existing in a semantic space.

Key Concepts:
1. ConceptNode: A single word/concept as a 64D vector.
2. MemeticField: The space where these concepts live and interact.
3. Trajectory: A sequence of concepts (sentence/thought) forming a path.
4. Resonance: The "energy" or "truth" of a trajectory based on geometric harmony.

"Grammar is the topology of the soul."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import math

from Core.Math.infinite_hyperquaternion import InfiniteHyperQuaternion

@dataclass
class ConceptNode:
    """
    A single concept in the Memetic Field.
    """
    id: str
    vector: InfiniteHyperQuaternion
    energy: float = 1.0
    connections: Dict[str, float] = field(default_factory=dict) # id -> weight

    def evolve(self, influence: InfiniteHyperQuaternion, rate: float = 0.1):
        """
        Shift meaning based on usage context.
        """
        # Vector addition with normalization to keep it on the hypersphere (mostly)
        # We allow magnitude to grow if the concept is "powerful"
        new_vec = self.vector.add(influence.scalar_multiply(rate))
        self.vector = new_vec

@dataclass
class Trajectory:
    """
    A path through the Memetic Field (a sentence, a thought, a story).
    """
    path: List[str] # List of concept IDs
    resonance: float = 0.0
    coherence: float = 0.0

class MemeticField:
    """
    The 64D semantic space where civilization's meaning evolves.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.concepts: Dict[str, ConceptNode] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize with some primordial concepts
        self._genesis_seeds()

    def _genesis_seeds(self):
        """
        Plant the initial seeds of meaning.
        """
        seeds = ["existence", "void", "light", "darkness", "love", "pain", "time", "chaos"]
        for seed in seeds:
            self.add_concept(seed)

    def add_concept(self, concept_id: str, vector: Optional[InfiniteHyperQuaternion] = None):
        """
        Register a new concept. If no vector provided, random 64D vector.
        """
        if concept_id in self.concepts:
            return
        
        if vector is None:
            vector = InfiniteHyperQuaternion.random(self.dim)
            
        self.concepts[concept_id] = ConceptNode(id=concept_id, vector=vector)

    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        return self.concepts.get(concept_id)

    def get_or_create_vector(self, concept_id: str) -> np.ndarray:
        """
        Get the 64D vector for a concept, creating it if it doesn't exist.
        Returns the raw numpy array components.
        """
        if concept_id not in self.concepts:
            self.add_concept(concept_id)
        
        return self.concepts[concept_id].vector.components

    def form_trajectory(self, sequence: List[str]) -> Trajectory:
        """
        Evaluate a sequence of words as a geometric path.
        Calculates Resonance (Truth) and Coherence (Grammar).
        """
        if not sequence:
            return Trajectory([], 0.0, 0.0)

        vectors = []
        valid_words = []
        for word in sequence:
            node = self.concepts.get(word)
            if node:
                vectors.append(node.vector)
                valid_words.append(word)
        
        if len(vectors) < 2:
            return Trajectory(valid_words, 0.0, 1.0) # Single point is coherent but low resonance

        # 1. Coherence (Grammar/Flow): Smoothness of the path
        # High coherence = small angles between consecutive steps (logical flow)
        # Low coherence = jagged, random jumps (nonsense)
        total_angle = 0.0
        for i in range(len(vectors) - 1):
            v1 = vectors[i].normalize()
            v2 = vectors[i+1].normalize()
            # Dot product for angle
            dot = np.dot(v1.components, v2.components)
            angle = math.acos(max(-1.0, min(1.0, dot)))
            total_angle += angle
        
        avg_angle = total_angle / (len(vectors) - 1)
        # Coherence is inverse of average angle (smaller angle = higher coherence)
        coherence = 1.0 / (1.0 + avg_angle)

        # 2. Resonance (Truth/Power): Constructive Interference
        # Does the path form a closed loop? Or a spiral? 
        # Simple metric: Magnitude of the sum of vectors (Constructive Interference)
        sum_vec = InfiniteHyperQuaternion(self.dim, np.zeros(self.dim))
        for v in vectors:
            sum_vec = sum_vec.add(v)
        
        # Resonance is magnitude of sum / number of vectors
        # If vectors align, this is high (1.0). If they cancel out, this is low (0.0).
        resonance = sum_vec.magnitude() / len(vectors)

        return Trajectory(valid_words, resonance, coherence)

    def reinforce(self, trajectory: Trajectory, intensity: float = 0.1):
        """
        If a trajectory is "lived" (experienced by an agent), the concepts involved
        gravitate towards each other, solidifying the "grammar" of that experience.
        """
        if len(trajectory.path) < 2:
            return

        # Hebbian Learning: "Cells that fire together, wire together"
        # Pull each concept slightly towards the average vector of the trajectory
        vectors = [self.concepts[w].vector for w in trajectory.path if w in self.concepts]
        if not vectors:
            return
            
        avg_vec = InfiniteHyperQuaternion(self.dim, np.zeros(self.dim))
        for v in vectors:
            avg_vec = avg_vec.add(v)
        avg_vec = avg_vec.scalar_multiply(1.0 / len(vectors))

        for word in trajectory.path:
            if word in self.concepts:
                node = self.concepts[word]
                # Pull towards average
                node.evolve(avg_vec, rate=intensity)
                # Also increase energy
                node.energy += intensity

    def log_status(self):
        self.logger.info(f"Memetic Field: {len(self.concepts)} concepts active.")
