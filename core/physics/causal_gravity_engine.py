import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class CausalNode:
    id: str
    content: str
    causal_links: List[str]  # IDs of other nodes it depends on
    mass: float = 0.0
    position: np.ndarray = None  # Position in N-dimensional tension space

class CausalGravityEngine:
    """
    [Causal Gravity Engine]
    Calculates the 'Causal Mass' of information and simulates gravitational attraction
    based on causal resonance rather than arbitrary weights.
    """
    def __init__(self, dimensions: int = 5):
        self.dimensions = dimensions
        self.nodes: Dict[str, CausalNode] = {}
        self.G = 0.1  # Universal Causal Gravitational Constant

    def add_node(self, node_id: str, content: str, links: List[str]):
        """Adds a node and initializes its position and mass."""
        # Initial mass is based on the number of causal links (intrinsic complexity)
        initial_mass = 1.0 + len(links) * 0.5
        position = np.random.rand(self.dimensions).astype(np.float32)

        node = CausalNode(id=node_id, content=content, causal_links=links, mass=initial_mass, position=position)
        self.nodes[node_id] = node
        self._update_all_masses()

    def _update_all_masses(self):
        """
        Recalculates mass based on recursive connectivity.
        A node's mass increases if many things depend on it (it becomes a 'Constant').
        """
        # Simple iterative mass amplification based on dependency
        for _ in range(3): # Converge mass
            new_masses = {nid: 1.0 + len(node.causal_links)*0.5 for nid, node in self.nodes.items()}
            for nid, node in self.nodes.items():
                for link_id in node.causal_links:
                    if link_id in new_masses:
                        # If I depend on link_id, link_id's mass increases because it's a foundational cause
                        new_masses[link_id] += node.mass * 0.2

            for nid, m in new_masses.items():
                self.nodes[nid].mass = m

    def calculate_attraction(self, node_a_id: str, node_b_id: str) -> np.ndarray:
        """
        Calculates the gravitational force vector between two nodes.
        F = G * (m1 * m2) / r^2
        """
        node_a = self.nodes[node_a_id]
        node_b = self.nodes[node_b_id]

        direction = node_b.position - node_a.position
        distance = np.linalg.norm(direction)

        # Softening factor to prevent infinite force at zero distance
        softening = 0.5

        if distance < 0.001:
            return np.zeros(self.dimensions)

        # Causal Resonance Factor: If they share links, the gravity is stronger
        # Also check if one depends on the other directly
        resonance = 1.0
        shared_links = set(node_a.causal_links).intersection(set(node_b.causal_links))
        resonance += len(shared_links) * 2.0

        if node_a_id in node_b.causal_links or node_b_id in node_a.causal_links:
            resonance += 5.0

        force_magnitude = self.G * (node_a.mass * node_b.mass * resonance) / (distance**2 + softening)
        return (direction / distance) * force_magnitude

    def step(self, dt: float = 0.1):
        """Simulates one step of gravitational movement."""
        forces = {nid: np.zeros(self.dimensions) for nid in self.nodes}

        ids = list(self.nodes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                f = self.calculate_attraction(ids[i], ids[j])
                forces[ids[i]] += f
                forces[ids[j]] -= f

        # Update positions based on forces (simple Euler integration)
        for nid, node in self.nodes.items():
            # Acceleration = F / m
            acceleration = forces[nid] / node.mass
            node.position += acceleration * dt
            # Optional: Add damping to reach equilibrium
            node.position *= 0.95

    def get_equilibrium_state(self) -> Dict[str, Any]:
        return {nid: {"pos": node.position.tolist(), "mass": node.mass, "content": node.content}
                for nid, node in self.nodes.items()}

if __name__ == "__main__":
    engine = CausalGravityEngine()
    engine.add_node("apple", "A red fruit", ["evolution", "photosynthesis", "seed"])
    engine.add_node("red", "A color", ["light", "wavelength"])
    engine.add_node("light", "Electromagnetic radiation", ["physics"])

    print("Initial Masses:")
    for nid, node in engine.nodes.items():
        print(f"Node {nid}: Mass {node.mass:.2f}")

    for _ in range(10):
        engine.step()

    print("\nState after 10 steps:")
    for nid, node in engine.nodes.items():
        print(f"Node {nid}: Position {node.position}")
