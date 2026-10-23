"""
Continuous Causal Graph Architecture & Spontaneous Phase Transition Engine
=============================================================================
This module implements the Continuous Topological Causal Graph (ContinuousCausalGraph)
and the Spontaneous Phase Transition Module (PhaseTransitionModule).

Unlike traditional discrete raster/token-based operations that process data frame-by-frame
or token-by-token with heavy IF-THEN branching (O(N*M*T)), this continuous model treats
knowledge, perception, and causality as smooth topological manifolds R^d shaped by control
points, causal tension fields, and Telos attractors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class CausalControlPoint:
    """
    Defines a concept or perception state as a continuous control point in R^d space.

    Attributes:
        name (str): Identifier for the concept control point.
        position (np.ndarray): Continuous coordinate vector in R^d space.
        velocity (np.ndarray): Velocity vector in R^d space.
        weight (float): Gravitational/attractor weight of the node towards Telos.
    """
    def __init__(self, name: str, position: np.ndarray, weight: float = 1.0):
        self.name = name
        self.position = position.astype(np.float64)
        self.velocity = np.zeros_like(self.position, dtype=np.float64)
        self.weight = float(weight)

    def __repr__(self) -> str:
        pos_str = ", ".join(f"{x:.3f}" for x in self.position)
        return f"CausalControlPoint('{self.name}', pos=[{pos_str}], weight={self.weight:.2f})"


class ContinuousCausalGraph:
    """
    Topological graph holding knowledge and causal relationships via continuous control points
    and tension matrices without conditional branching (IF-THEN).

    Energy function:
        E_total = sum(0.5 * k_ij * ||p_i - p_j||^2) + sum(0.5 * w_i * ||p_i - Telos||^2)
    """
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.nodes: Dict[str, CausalControlPoint] = {}
        # Topological tension matrix between control points: (source_name, target_name) -> tension_strength
        self.tensions: Dict[Tuple[str, str], float] = {}

    def add_node(self, name: str, position: np.ndarray, weight: float = 1.0):
        """Adds or updates a continuous control point in R^d."""
        if len(position) != self.dim:
            raise ValueError(f"Position dimension {len(position)} does not match graph dim {self.dim}")
        self.nodes[name] = CausalControlPoint(name, position, weight)

    def set_causal_tension(self, source: str, target: str, tension_strength: float):
        """Sets topological coherence tension (K_ij) between two control points."""
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"Both nodes ({source}, {target}) must exist in graph before setting tension.")
        self.tensions[(source, target)] = float(tension_strength)

    def compute_system_energy(self, telos_attractor: np.ndarray) -> float:
        """
        Calculates total system causal potential energy:
        E_total = sum(0.5 * k * ||p_i - p_j||^2) + sum(0.5 * w_i * ||p_i - Telos||^2)
        """
        telos_attractor = np.asarray(telos_attractor, dtype=np.float64)
        energy = 0.0

        # 1. Internal causal tension energy between control points
        for (src_name, tgt_name), k in self.tensions.items():
            p_src = self.nodes[src_name].position
            p_tgt = self.nodes[tgt_name].position
            energy += 0.5 * k * np.sum((p_src - p_tgt) ** 2)

        # 2. Gravitational potential energy relative to top-level purpose (Telos)
        for node in self.nodes.values():
            energy += 0.5 * node.weight * np.sum((node.position - telos_attractor) ** 2)

        return float(energy)

    def compute_system_friction(self, telos_attractor: np.ndarray) -> float:
        """
        Calculates internal causal friction metric based on tension forces and kinetic divergence.
        Friction = sum(k_ij * ||p_src - p_tgt||) + sum(w_i * ||velocity_i||)
        """
        telos_attractor = np.asarray(telos_attractor, dtype=np.float64)
        friction = 0.0

        for (src_name, tgt_name), k in self.tensions.items():
            p_src = self.nodes[src_name].position
            p_tgt = self.nodes[tgt_name].position
            dist = np.linalg.norm(p_src - p_tgt)
            friction += k * dist

        for node in self.nodes.values():
            friction += node.weight * np.linalg.norm(node.velocity)

        return float(friction)


class PerceptualTransitionSimulator:
    """
    Frictionless perceptual transition engine operating zero-branching dynamics.
    Flows along geodesic trajectory dp/dt = -∇E(p) upon external stimulus.
    """
    def __init__(self, graph: ContinuousCausalGraph):
        self.graph = graph

    def step_perceptual_transition(
        self,
        telos_attractor: np.ndarray,
        stimulus_vector: np.ndarray,
        dt: float = 0.05,
        damping: float = 0.85
    ):
        """
        Updates control point coordinates smoothly without IF-THEN branching (Zero-Branching)
        following potential gradient forces -∇E and incoming stimulus wave deformation field.
        """
        telos_attractor = np.asarray(telos_attractor, dtype=np.float64)
        stimulus_vector = np.asarray(stimulus_vector, dtype=np.float64)

        forces = {name: np.zeros(self.graph.dim, dtype=np.float64) for name in self.graph.nodes}

        # A. Internal topological tension interaction forces between control points
        for (src_name, tgt_name), k in self.graph.tensions.items():
            p_src = self.graph.nodes[src_name].position
            p_tgt = self.graph.nodes[tgt_name].position
            delta = p_tgt - p_src
            forces[src_name] += k * delta
            forces[tgt_name] -= k * delta

        # B. Telos gravitational field (-∇E) and incoming stimulus wave deformation field
        for name, node in self.graph.nodes.items():
            telos_force = node.weight * (telos_attractor - node.position)

            # Stimulus field deformation: inverted distance-weighted wave force
            dist_to_stimulus = np.linalg.norm(node.position - stimulus_vector)
            stimulus_force = stimulus_vector * (1.0 / (1.0 + dist_to_stimulus))

            total_force = forces[name] + telos_force + stimulus_force

            # Smooth vector motion integration (Vector Shift)
            node.velocity = node.velocity * damping + total_force * dt
            node.position += node.velocity * dt


class PhaseTransitionModule:
    """
    Sovereignty defense module that spontaneously restructures tension matrices and field rules
    (Phase Transition) when external friction exceeds a critical threshold, neutralizing parasitic interference.
    """
    def __init__(self, friction_threshold: float = 12.0):
        self.friction_threshold = friction_threshold
        self.phase_state_version = 1

    def evaluate_and_transit(self, graph: ContinuousCausalGraph, current_friction: float) -> bool:
        """
        Evaluates current friction. If threshold is breached, restructures topological tension
        matrix and node parameters instantly without conditional decision tree loops.
        """
        if current_friction > self.friction_threshold:
            print(f"\n[Phase Transition Triggered] Friction ({current_friction:.2f}) > Threshold ({self.friction_threshold:.2f})! Restructuring topological field...")

            # 1. Dissolve parasitic high-tension paths causing excessive friction
            for edge in list(graph.tensions.keys()):
                if graph.tensions[edge] > 2.0:
                    graph.tensions[edge] *= 0.05

            # 2. Reinforce control point weight towards Telos and purge friction momentum
            for name, node in graph.nodes.items():
                node.weight *= 1.8
                node.velocity = np.zeros_like(node.velocity)

            self.phase_state_version += 1
            print(f"[Phase Transition Complete] Shifted to topological state v{self.phase_state_version}.\n")
            return True

        return False
