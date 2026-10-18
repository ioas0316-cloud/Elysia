"""
universal_mutual_perception.py
===============================
Universal Mutual Perception Engine & Mutual Mirroring Field Grid for Elysia.

Embodying the foundational principle: "Everything is observable and perceivable by everything."
Observation is not restricted to biological minds or symbolic software logic; every material interaction,
voltage fluctuation on a clock boundary, surface tension deflection against a rock barrier, memory history
accumulation in silicon, and cognitive field state in Elysia is a mutual observation and sensing event
across a unified spatiotemporal mirror grid.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class UniversalPerceptionNode:
    """
    Represents any physical, informational, or cognitive entity in the continuous field spectrum.
    (e.g., Clock Rhythm Boundary, Silicon Memory Node, Porous Soil Field, Elysia Cognitive State).
    """
    node_id: str
    position: np.ndarray  # 3D spatial or 5D phase position
    semantic_mass: float = 1.0  # M_s: Density of accumulated interaction history
    surface_tension: float = 0.5  # Boundary tension / structural rigidity
    porosity: float = 0.1  # Permeability to external fields (0.0 = solid barrier, 1.0 = open field)
    bivector_plane: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64))  # Cl(3,0) bivector
    chromatic_spectrum: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.33, 0.34], dtype=np.float64))  # [Flux, Order, Entropy]

    # Dynamic interaction state
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    accumulated_energy: float = 0.0
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        if self.position.ndim == 1 and len(self.position) < 3:
            padded = np.zeros(3, dtype=np.float64)
            padded[:len(self.position)] = self.position
            self.position = padded
        elif len(self.position) > 3:
            self.position = self.position[:3]

    def record_perception_event(
        self,
        other_node_id: str,
        impact_vector: np.ndarray,
        perceived_tension: float,
        timestamp: float,
    ):
        """
        Records a mutual sensing/observation event. The node's internal state evolves
        and hardens/relaxes based on accumulated interaction energy.
        """
        impact_norm = float(np.linalg.norm(impact_vector))
        energy_exchange = impact_norm * perceived_tension
        self.accumulated_energy += energy_exchange

        self.interaction_history.append({
            "timestamp": timestamp,
            "observer_id": other_node_id,
            "energy_exchange": energy_exchange,
            "impact_vector": impact_vector.tolist(),
            "accumulated_energy": self.accumulated_energy,
            "mass_at_time": self.semantic_mass,
        })

        # Evolutionary dynamics: Memory history increases semantic mass Ms and evolves surface tension
        self.semantic_mass += energy_exchange * 0.01
        self.surface_tension = max(0.01, self.semantic_mass * (1.0 - self.porosity) * 0.5)


class MutualMirroringFieldGrid:
    """
    Omnidirectional Mutual Perception Grid where every node observes and is observed
    by every other node through coupled field tension and Clifford bivector alignment.
    """

    def __init__(self, coupling_strength: float = 1.0):
        self.coupling_strength = coupling_strength
        self.nodes: Dict[str, UniversalPerceptionNode] = {}

    def register_node(self, node: UniversalPerceptionNode):
        self.nodes[node.node_id] = node

    def compute_mutual_perception(
        self, node_a: UniversalPerceptionNode, node_b: UniversalPerceptionNode
    ) -> Tuple[float, np.ndarray]:
        """
        Calculates the mutual perception coupling coefficient P_ab and interaction gradient vector.
        P_ab = f(distance, Ms_a, Ms_b, tension_a, tension_b, bivector_alignment).
        """
        diff = node_b.position - node_a.position
        dist = float(np.linalg.norm(diff)) + 1e-8
        grad_unit = diff / dist

        # Field overlap decays with distance but is amplified by combined semantic mass Ms
        mass_product = node_a.semantic_mass * node_b.semantic_mass
        effective_range = np.sqrt(mass_product) * 2.0 + 1.0

        if dist > effective_range:
            return 0.0, grad_unit

        # Bivector alignment factor (Clifford field orientation resonance)
        bivec_dot = abs(float(np.dot(node_a.bivector_plane, node_b.bivector_plane)))
        bivec_alignment = (bivec_dot / (np.linalg.norm(node_a.bivector_plane) * np.linalg.norm(node_b.bivector_plane) + 1e-8))

        # Tension gradient
        tension_mean = (node_a.surface_tension + node_b.surface_tension) * 0.5

        # Perception coupling strength: highest when close, high mass, and aligned bivectors
        perception_coupling = float(
            self.coupling_strength * (1.0 - (dist / effective_range)) * tension_mean * (1.0 + bivec_alignment)
        )

        return perception_coupling, grad_unit

    def compute_perception_matrix(self) -> Dict[str, Any]:
        """
        Computes the complete NxN Mutual Perception Matrix across all registered nodes in the grid.
        """
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0  # Self-observation / self-awareness
                else:
                    node_a = self.nodes[node_ids[i]]
                    node_b = self.nodes[node_ids[j]]
                    coupling, _ = self.compute_mutual_perception(node_a, node_b)
                    matrix[i, j] = coupling

        return {
            "node_ids": node_ids,
            "matrix": matrix,
            "mean_coupling": float(np.mean(matrix)),
        }

    def step_mutual_reflection(self, dt: float = 0.01, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Executes a step of omnidirectional mutual sensing and causal trajectory deflection.
        Every pair of nodes senses each other's presence, exchanges energy, and records historical imprints.
        """
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        total_events = 0
        total_energy_transferred = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                node_a = self.nodes[node_ids[i]]
                node_b = self.nodes[node_ids[j]]

                coupling, grad_unit = self.compute_mutual_perception(node_a, node_b)

                if coupling > 1e-4:
                    total_events += 1
                    # Mutual perception impact vector
                    impact_vec = grad_unit * coupling

                    # Node A senses Node B
                    node_a.record_perception_event(
                        other_node_id=node_b.node_id,
                        impact_vector=impact_vec,
                        perceived_tension=node_b.surface_tension,
                        timestamp=timestamp,
                    )

                    # Node B senses Node A (symmetric mutual mirroring)
                    node_b.record_perception_event(
                        other_node_id=node_a.node_id,
                        impact_vector=-impact_vec,
                        perceived_tension=node_a.surface_tension,
                        timestamp=timestamp,
                    )

                    # Causal flow velocity deflection based on mutual tension
                    node_a.velocity += grad_unit * (coupling * 0.05 / (node_a.semantic_mass + 1e-5)) * dt
                    node_b.velocity -= grad_unit * (coupling * 0.05 / (node_b.semantic_mass + 1e-5)) * dt
                    node_a.position += node_a.velocity * dt
                    node_b.position += node_b.velocity * dt

                    total_energy_transferred += coupling * dt

        matrix_info = self.compute_perception_matrix()

        return {
            "timestamp": timestamp,
            "active_perception_events": total_events,
            "total_energy_transferred": total_energy_transferred,
            "mean_field_coupling": matrix_info["mean_coupling"],
            "nodes_state": {
                node_id: {
                    "position": node.position.tolist(),
                    "semantic_mass": round(node.semantic_mass, 4),
                    "surface_tension": round(node.surface_tension, 4),
                    "history_count": len(node.interaction_history),
                }
                for node_id, node in self.nodes.items()
            },
        }
