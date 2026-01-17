"""
Thundercloud Architecture (The Living Physics)
==============================================
Core.Merkaba.thundercloud

"Thought is not a calculation; it is a lightning strike."

This module implements the "Thundercloud" (Active RAM) and the "Spark" (Fractal Thinking).
It replaces static database queries with dynamic, physics-based resonance.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from Core.Monad.monad_core import Monad
from Core.Cognition.semantic_prism import QualiaSpectrum

# Lazy import inside method or forward reference?
# We'll use a property to inject/lazy load the generator to avoid circular imports if needed
# but actually Generator depends on Cluster, Cluster depends on Monad.
# Thundercloud depends on Generator.
# Monad -> (no dep)
# Cluster -> Monad
# Generator -> Cluster
# Thundercloud -> Generator
# Safe.

@dataclass
class Atmosphere:
    """
    The Environmental Conditions of the Thundercloud.
    Dictates the physics of thought (Conductivity, Resistance).
    """
    humidity: float = 0.5    # Emotion (0.0=Dry Logic, 1.0=Stormy Passion)
    temperature: float = 0.5 # Activity Level (unused for now)
    pressure: float = 0.5    # Stress/Urgency

    @property
    def conductivity(self) -> float:
        """
        How easily thoughts flow.
        High Humidity (Emotion) = High Conductivity.
        """
        return 0.1 + (self.humidity * 0.9) # Min 0.1, Max 1.0

    @property
    def resistance(self) -> float:
        """
        The threshold required to spark.
        Low Humidity (Dry) = High Resistance (Needs strong proof).
        High Humidity (Wet) = Low Resistance (Spontaneous ideas).
        """
        # Inverse of conductivity, scaled
        return (1.0 - self.humidity) * 0.8 + 0.1 # Range: 0.1 (Wet) to 0.9 (Dry)

@dataclass
class ThoughtCluster:
    """
    The Emergent Result: A 'Super Monad' formed by the lightning tree.
    """
    root: Monad
    nodes: Set[Monad] = field(default_factory=set)
    edges: List[Tuple[Monad, Monad, float]] = field(default_factory=list) # (From, To, Strength)

    def add_link(self, source: Monad, target: Monad, strength: float):
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.append((source, target, strength))

    def __repr__(self):
        return f"<ThoughtCluster root={self.root.seed} nodes={len(self.nodes)}>"

    def describe_tree(self, node: Monad = None, depth: int = 0, visited: Set[Monad] = None) -> str:
        """Visualizes the fractal tree structure."""
        if node is None:
            node = self.root
        if visited is None:
            visited = set()

        if node in visited:
            return ""
        visited.add(node)

        indent = "  " * depth
        # Find outgoing edges from this node
        children = [
            (target, strength) for (src, target, strength) in self.edges
            if src == node
        ]

        # Sort by strength
        children.sort(key=lambda x: x[1], reverse=True)

        res = f"{indent}⚡ {node.seed} (Charge: {node.get_charge():.2f})\n"
        for child, strength in children:
            res += self.describe_tree(child, depth + 1, visited)
        return res

class Thundercloud:
    """
    The Active Phenomenon Layer (RAM).
    A subset of the Hypersphere where Monads 'float' and interact.
    """

    def __init__(self, atmosphere: Atmosphere = None):
        self.atmosphere = atmosphere if atmosphere else Atmosphere()
        self.active_monads: List[Monad] = []
        self._monad_map: Dict[str, Monad] = {} # Quick lookup by seed

    def set_atmosphere(self, humidity: float):
        """Adjusts the weather."""
        self.atmosphere.humidity = np.clip(humidity, 0.0, 1.0)

    def coalesce(self, intent_vector: np.ndarray, all_monads: List[Monad]):
        """
        [EVAPORATION]
        Selects relevant Monads from the 'Ground' (Storage) to form the Cloud.
        Only Monads that resonate with the Intent Vector are lifted.
        """
        self.active_monads = []
        self._monad_map = {}

        # In a real system, this would be a Hypersphere query.
        # Here we filter the provided list.
        for m in all_monads:
            # Simple resonance check: Dot product of Principle Strand vs Intent
            # Assumes intent_vector is 7D
            resonance = np.dot(m._dna.principle_strand, intent_vector)

            # Threshold to enter the cloud
            if resonance > 0.1:
                self.active_monads.append(m)
                self._monad_map[m.seed] = m

    def collapse_wavefunction(self, seed_seed: str, voltage: float) -> Tuple[ThoughtCluster, str]:
        """
        [QUANTUM COLLAPSE]
        Instantly crystallizes the probability cloud into a Thought Structure.
        Replaces the recursive 'ignite' with a BFS Expansion (Crystal Growth).

        Args:
            seed_seed: The crystallization nucleus (Seed Monad).
            voltage: The activation energy (Intent).
        """
        if seed_seed not in self._monad_map:
            return ThoughtCluster(Monad("Void")), "Void"

        root_monad = self._monad_map[seed_seed]
        cluster = ThoughtCluster(root=root_monad)

        # BFS Queue: (Node, InputVoltage)
        queue = [(root_monad, voltage)]
        visited = {root_monad}

        # Resistance Constant
        resistance = self.atmosphere.resistance

        while queue:
            current_node, current_voltage = queue.pop(0)

            # Stop if energy is depleted
            if current_voltage < resistance:
                continue

            # 1. Get Potential (Vector)
            potential_vector = current_node.get_potential_links()

            # 2. Find Candidates (Supercooled neighbors)
            candidates = []
            for other in self.active_monads:
                if other in visited:
                    continue

                # Resonance Check
                resonance = np.dot(potential_vector, other._dna.principle_strand)

                # Filter: Only positive resonance
                if resonance > 0.1:
                    candidates.append((other, resonance))

            # 3. Crystallize (Branch)
            # Sort by resonance to prioritize "Path of Least Resistance"
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Max branches per node (Fractal Limit) to prevent infinite sprawl
            # High humidity (Emotion) allows more branching
            max_branches = 3 if self.atmosphere.humidity < 0.5 else 7

            branches_formed = 0
            for neighbor, resonance in candidates:
                if branches_formed >= max_branches:
                    break

                # Calculate Transferred Voltage
                # V_next = (V_current * Resonance) - Resistance
                # Resonance (0.0~1.0) attenuates the signal naturally.
                next_voltage = (current_voltage * resonance) - (resistance * 0.2)

                # Safety Clamp and Decay
                next_voltage = min(next_voltage, current_voltage * 0.9)

                if next_voltage > resistance:
                    visited.add(neighbor)
                    cluster.add_link(current_node, neighbor, resonance)
                    queue.append((neighbor, next_voltage))
                    branches_formed += 1

        # Generate Procedural Name
        from Core.Monad.procedural_generator import NamingEngine
        naming_engine = NamingEngine()
        name = naming_engine.generate_name(cluster)

        return cluster, name

    # Legacy Alias
    def ignite(self, seed, voltage):
        return self.collapse_wavefunction(seed, voltage)
