import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ElasticNode:
    id: str
    content: str
    causal_links: Dict[str, float]  # ID -> Elasticity (k)
    mass: float = 0.0
    position: np.ndarray = None  # Position in N-dimensional tension space
    velocity: np.ndarray = None

class ElasticCausalEngine:
    """
    [Elastic Causal Engine]
    Replaces static gravity with elastic tension.
    Links between nodes are 'springs' whose constant k is determined by the torque
    of the dielectric flow. No flow = loss of tension = forgetting.
    """
    def __init__(self, dimensions: int = 5):
        self.dimensions = dimensions
        self.nodes: Dict[str, ElasticNode] = {}
        self.damping = 0.9
        self.rest_length = 0.1

    def add_node(self, node_id: str, content: str, links: List[str]):
        """Adds a node with initial elasticity."""
        # Initial elasticity is low
        link_dict = {link_id: 0.1 for link_id in links}
        position = np.random.rand(self.dimensions).astype(np.float32)
        velocity = np.zeros(self.dimensions).astype(np.float32)

        # Mass is intrinsic importance (how many things depend on it)
        mass = 1.0

        node = ElasticNode(id=node_id, content=content, causal_links=link_dict,
                          mass=mass, position=position, velocity=velocity)
        self.nodes[node_id] = node
        self._update_masses()

    def _update_masses(self):
        """Mass is proportional to the number of incoming links (foundationality)."""
        counts = {nid: 1.0 for nid in self.nodes}
        for node in self.nodes.values():
            for link_id in node.causal_links:
                if link_id in counts:
                    counts[link_id] += 0.5

        for nid, m in counts.items():
            self.nodes[nid].mass = m

    def inject_torque(self, node_a_id: str, node_b_id: str, torque: float):
        """
        Increases the elasticity (k) between two nodes based on the dielectric torque.
        This is the 'Memory Formation' process.
        """
        if node_a_id in self.nodes and node_b_id in self.nodes[node_a_id].causal_links:
            # Elasticity k increases with torque, capped at 10.0
            self.nodes[node_a_id].causal_links[node_b_id] = min(10.0, self.nodes[node_a_id].causal_links[node_b_id] + abs(torque))

        # Also check bidirectional if exists
        if node_b_id in self.nodes and node_a_id in self.nodes[node_b_id].causal_links:
            self.nodes[node_b_id].causal_links[node_a_id] = min(10.0, self.nodes[node_b_id].causal_links[node_a_id] + abs(torque))

    def step(self, dt: float = 0.1):
        """
        Simulates the elastic spring forces.
        F = -k * (x - rest_length)
        """
        forces = {nid: np.zeros(self.dimensions) for nid in self.nodes}

        for nid, node in self.nodes.items():
            for link_id, k in node.causal_links.items():
                if link_id not in self.nodes: continue

                target_node = self.nodes[link_id]
                diff = target_node.position - node.position
                distance = np.linalg.norm(diff)
                if distance < 1e-6: continue

                # Spring Force
                direction = diff / distance
                force_mag = k * (distance - self.rest_length)
                forces[nid] += direction * force_mag
                forces[link_id] -= direction * force_mag

            # Natural decay of elasticity (Forgetting)
            for link_id in list(node.causal_links.keys()):
                node.causal_links[link_id] *= 0.99 # 1% decay per step
                if node.causal_links[link_id] < 0.01:
                    # Optional: link snaps
                    pass

        # Update physics
        for nid, node in self.nodes.items():
            acceleration = forces[nid] / node.mass
            node.velocity = (node.velocity + acceleration * dt) * self.damping
            node.position += node.velocity * dt

    def get_state(self) -> Dict[str, Any]:
        return {nid: {"pos": node.position.tolist(), "k": node.causal_links} for nid, node in self.nodes.items()}
