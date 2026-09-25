"""
Adaptive Cognitive Lens Engine & Collective Resonance Network
============================================================
Implements biological adaptation and structural plasticity for real-time
cognitive retuning without discrete data resets or static weight multipliers.

Core Components:
1. AutonomousCognitiveNode: A self-tuning biological receptor cell that perceives
   stimulus/wave coupling, detects internal phase tension, retunes its posture and sensitivity gain,
   and accumulates hysteresis baseline residual state.
2. CollectiveResonanceNetwork: Multi-node network demonstrating wave propagation,
   local tension rippling, global coherence synchronization, and contextual backgrounding.
"""

import math
from typing import List, Dict, Tuple, Optional
import torch


class AutonomousCognitiveNode:
    """
    elysia_engine: Autonomous Cognitive Node (자율 인지 노드)
    A minimal autonomous receptor cell that adapts its cognitive posture and sensitivity gain
    in response to external waves and internal tension without static matrix multiplication.
    """

    def __init__(self, node_id: int, dim: int = 3, hysteresis_rate: float = 0.85):
        self.node_id = node_id
        self.dim = dim
        self.hysteresis_rate = hysteresis_rate

        # Baseline posture (direction vector on unit sphere) and sensitivity gain
        posture = torch.randn(dim)
        self.posture = posture / (torch.norm(posture) + 1e-8)
        self.sensitivity_gain: float = 1.0

        # Adjacency topology links
        self.neighbors: List['AutonomousCognitiveNode'] = []
        self.coupling_weights: Dict[int, float] = {}

    def connect(self, other_node: 'AutonomousCognitiveNode', weight: float = 0.5):
        """Establishes bidirectional or directional resonance coupling with an adjacent node."""
        if other_node not in self.neighbors and other_node.node_id != self.node_id:
            self.neighbors.append(other_node)
            self.coupling_weights[other_node.node_id] = weight

    def perceive(self, stimulus: torch.Tensor, wave_influence: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor]:
        """
        [Stages 1 & 2] Receptor Coupling & Internal Tension Sensing
        Couples external stimulus and neighboring wave influences with current baseline posture,
        detecting phase mismatch tension.
        """
        if wave_influence is None:
            wave_influence = torch.zeros(self.dim, dtype=stimulus.dtype, device=stimulus.device)

        total_input = stimulus + wave_influence
        input_norm_val = torch.norm(total_input)
        if input_norm_val < 1e-8:
            return 0.0, torch.zeros(self.dim, dtype=stimulus.dtype, device=stimulus.device)

        input_norm = total_input / input_norm_val
        coupling = torch.dot(self.posture, input_norm)

        # Internal tension: Phase discrepancy between baseline posture and incoming wave
        internal_tension = float((1.0 - coupling) * self.sensitivity_gain)
        return internal_tension, input_norm

    def retune_lens(self, internal_tension: float, input_norm: torch.Tensor):
        """
        [Stage 3] Posture & Gain Retuning
        Spontaneously rotates cognitive posture towards incoming wave direction to relieve tension,
        and adjusts sensitivity gain to avoid oversaturation.
        """
        if internal_tension > 0.01:
            shift_amount = 0.25 * internal_tension
            new_posture = self.posture + shift_amount * input_norm
            self.posture = new_posture / (torch.norm(new_posture) + 1e-8)
            self.sensitivity_gain = float(1.0 / (1.0 + 0.4 * internal_tension))

    def accumulate_hysteresis(self):
        """
        [Stage 4] Hysteresis Accumulation & Baseline Shift
        Retains retuned posture as the new baseline residual state for future interactions
        instead of resetting memory.
        """
        self.posture = self.hysteresis_rate * self.posture + (1.0 - self.hysteresis_rate) * self.posture
        self.posture = self.posture / (torch.norm(self.posture) + 1e-8)

    def step(self, stimulus: torch.Tensor, wave_influence: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        """Performs a complete 4-stage biological adaptation cycle for the autonomous node."""
        tension, input_norm = self.perceive(stimulus, wave_influence)
        self.retune_lens(tension, input_norm)
        self.accumulate_hysteresis()

        emitted_wave = self.posture * self.sensitivity_gain * (1.0 + tension)
        return emitted_wave, tension


class CollectiveResonanceNetwork:
    """
    elysia_engine: Collective Resonance Network (집단 공명 네트워크)
    Coordinates local tension rippling, wave propagation, global coherence synchronization,
    and contextual backgrounding across an ensemble of autonomous cognitive nodes.
    """

    def __init__(self, num_nodes: int = 6, dim: int = 3):
        self.num_nodes = num_nodes
        self.dim = dim
        self.nodes = [AutonomousCognitiveNode(node_id=i, dim=dim) for i in range(num_nodes)]
        self._build_topology()

    def _build_topology(self):
        """Constructs a ring topology with cross-mesh resonance bridges."""
        for i in range(self.num_nodes):
            next_node = self.nodes[(i + 1) % self.num_nodes]
            prev_node = self.nodes[(i - 1) % self.num_nodes]
            self.nodes[i].connect(next_node, weight=0.6)
            self.nodes[i].connect(prev_node, weight=0.6)

            # Cross bridge link
            cross_node = self.nodes[(i + self.num_nodes // 2) % self.num_nodes]
            self.nodes[i].connect(cross_node, weight=0.3)

    def calculate_global_coherence(self) -> float:
        """
        Calculates global order parameter / phase synchronization across all node postures.
        1.0 represents perfect phase alignment; ~0.0 represents diffuse disorder.
        """
        mean_posture = torch.zeros(self.dim)
        for node in self.nodes:
            mean_posture += node.posture
        return float((torch.norm(mean_posture) / self.num_nodes).item())

    def propagate_resonance(
        self,
        target_node_id: int,
        direct_stimulus: torch.Tensor,
        steps: int = 5
    ) -> List[Dict[str, float]]:
        """
        Injects a localized wave stimulus into target node, propagating tension ripples
        across adjacent nodes and inducing collective resonance alignment.
        """
        history = []

        for step_idx in range(1, steps + 1):
            wave_buffers = {i: torch.zeros(self.dim, dtype=direct_stimulus.dtype) for i in range(self.num_nodes)}

            # Gather emitted waves from neighbors
            for node in self.nodes:
                emitted = node.posture * node.sensitivity_gain
                for neighbor in node.neighbors:
                    weight = node.coupling_weights[neighbor.node_id]
                    wave_buffers[neighbor.node_id] += emitted * weight

            # Execute adaptation step for each node
            step_tensions = []
            for node in self.nodes:
                node_stimulus = direct_stimulus if node.node_id == target_node_id else torch.zeros(self.dim, dtype=direct_stimulus.dtype)
                wave_in = wave_buffers[node.node_id]

                _, tension = node.step(node_stimulus, wave_in)
                step_tensions.append(tension)

            coherence = self.calculate_global_coherence()
            avg_tension = float(sum(step_tensions) / len(step_tensions))

            history.append({
                "step": step_idx,
                "avg_tension": avg_tension,
                "global_coherence": coherence
            })

        return history
