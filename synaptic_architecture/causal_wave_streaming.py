r"""
Causal Wave Streaming Engine & Video Decoder
===========================================

Implements the wave-like causal playback paradigm:
- Contiguous Causal Stream Buffer [T, N, D]
- Wave Propagation & Interference Engine (Damping, Inter-thread/node interference)
- Causal Video Decoder Architecture:
  1. Header Parser & Frame Stream Manager
  2. I-Frame (Key-Cognition Baseline State Snapshot)
  3. P-Frame (\Delta C Causal Motion Vector Stream)
  4. Latent Causal Renderer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class CausalFrameHeader:
    """Header metadata for Causal Wave Video Frames."""
    frame_type: str  # "I-FRAME" or "P-FRAME"
    timestamp: float
    num_nodes: int
    dim: int
    active_mask: torch.Tensor  # Boolean bitmask for active nodes [N]


class ContiguousCausalStreamBuffer:
    """
    Contiguous memory buffer layout for zero-copy ring buffer operations.
    Maintains [T, N, D] contiguous causal state streams.
    """

    def __init__(self, capacity_t: int = 100, num_nodes: int = 128, dim: int = 64, device: torch.device = None):
        self.capacity_t = capacity_t
        self.num_nodes = num_nodes
        self.dim = dim
        self.device = device or torch.device("cpu")

        # Contiguous allocation [Capacity_T, Num_Nodes, Dim]
        self.buffer = torch.zeros((capacity_t, num_nodes, dim), dtype=torch.float32, device=self.device)
        self.head_ptr = 0
        self.tail_ptr = 0
        self.current_count = 0

    def push_frame(self, state_frame: torch.Tensor):
        """Pushes a single frame [N, D] into the contiguous ring buffer."""
        self.buffer[self.head_ptr].copy_(state_frame)
        self.head_ptr = (self.head_ptr + 1) % self.capacity_t
        if self.current_count < self.capacity_t:
            self.current_count += 1
        else:
            self.tail_ptr = (self.tail_ptr + 1) % self.capacity_t

    def get_latest_frame(self) -> torch.Tensor:
        """Returns the most recent frame [N, D]."""
        latest_idx = (self.head_ptr - 1 + self.capacity_t) % self.capacity_t
        return self.buffer[latest_idx]

    def get_history_block(self, steps: int) -> torch.Tensor:
        """Returns the last `steps` frames as a contiguous block [Steps, N, D]."""
        actual_steps = min(steps, self.current_count)
        indices = [(self.head_ptr - actual_steps + i + self.capacity_t) % self.capacity_t for i in range(actual_steps)]
        return self.buffer[indices]


class CausalWaveStreamingEngine(nn.Module):
    """
    Simulates hardware shared memory wave propagation & interference kernel.
    Performs vectorized wave transfers along topological causal edges with damping filters.
    """

    def __init__(self, num_nodes: int = 128, dim: int = 64, k_neighbors: int = 8, damping_decay: float = 0.9):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.k_neighbors = k_neighbors
        self.damping_decay = damping_decay

        # Pre-allocated Edge matrix & Damping factors
        self.register_buffer("causal_edges", torch.randint(0, num_nodes, (num_nodes, k_neighbors)))
        self.register_buffer("damping_factors", torch.rand(num_nodes, k_neighbors) * 0.5 + 0.4)

    def propagate_wave_step(self, current_wavefront: torch.Tensor) -> torch.Tensor:
        """
        Executes one wave-propagation step across nodes.

        Args:
            current_wavefront: [Num_Nodes, Dim] current node wave energy states

        Returns:
            next_wavefront: [Num_Nodes, Dim] updated wave energy states after propagation & interference
        """
        num_nodes, dim = current_wavefront.shape

        # Gather incoming wave energy from neighbor indices [Num_Nodes, K_Neighbors, Dim]
        neighbor_indices = self.causal_edges[:num_nodes, :self.k_neighbors]  # [N, K]
        incoming_waves = current_wavefront[neighbor_indices]  # [N, K, Dim]

        # Apply damping factors [N, K, 1]
        damps = self.damping_factors[:num_nodes, :self.k_neighbors].unsqueeze(-1)  # [N, K, 1]
        accumulated_wave_energy = (incoming_waves * damps).sum(dim=1)  # [N, Dim]

        # Wave state update with 90% incoming wave accumulation + 10% self-energy retention
        next_energy = current_wavefront * (1.0 - self.damping_decay) + accumulated_wave_energy * self.damping_decay

        # Local interference smoothing (simulating Warp Shuffle/adjacent tile coupling)
        if num_nodes >= 2:
            rolled_left = torch.roll(next_energy, shifts=-1, dims=0)
            rolled_right = torch.roll(next_energy, shifts=1, dims=0)
            interference = (rolled_left + rolled_right) * 0.05
            next_energy = next_energy * 0.9 + interference

        return next_energy


class CausalVideoDecoder(nn.Module):
    """
    Causal Video Decoder Architecture for decoding thoughts/causal states.
    Recreates thought progression via I-Frames (Key-Cognition Snapshots) and P-Frames (Causal Deltas).
    """

    def __init__(self, num_nodes: int = 128, dim: int = 64, k_neighbors: int = 8):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim

        # Core Wave Streaming Engine & Ring Buffer
        self.wave_engine = CausalWaveStreamingEngine(num_nodes=num_nodes, dim=dim, k_neighbors=k_neighbors)
        self.stream_buffer = ContiguousCausalStreamBuffer(capacity_t=100, num_nodes=num_nodes, dim=dim)

        # Baseline I-Frame key-cognition bank [Num_Keys, N, D]
        self.num_keys = 10
        self.key_cognition_bank = nn.Parameter(torch.randn(self.num_keys, num_nodes, dim) * 0.1)

        # Latent Renderer Projection
        self.latent_renderer = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

        # Active current decoded state [N, D]
        self.register_buffer("current_decoded_state", torch.zeros(num_nodes, dim))

    def decode_i_frame(self, key_frame_idx: int) -> Tuple[torch.Tensor, CausalFrameHeader]:
        """
        Decodes an I-Frame (Key-Cognition Snapshot) to dump full baseline state from key bank.
        """
        idx = max(0, min(key_frame_idx, self.num_keys - 1))
        key_state = self.key_cognition_bank[idx].clone()  # [N, D]
        self.current_decoded_state.copy_(key_state)

        # Push to stream buffer
        self.stream_buffer.push_frame(self.current_decoded_state)

        active_mask = (key_state.norm(dim=-1) > 1e-4)
        header = CausalFrameHeader(
            frame_type="I-FRAME",
            timestamp=0.0,
            num_nodes=self.num_nodes,
            dim=self.dim,
            active_mask=active_mask
        )
        return self.current_decoded_state, header

    def decode_p_frame(self, delta_stream: torch.Tensor, timestamp: float = 0.0) -> Tuple[torch.Tensor, CausalFrameHeader]:
        r"""
        Decodes a P-Frame (\Delta C Causal Motion Vector Stream).
        Applies causal delta vector on top of wave-propagated current state.

        Args:
            delta_stream: [N, D] causal delta values
            timestamp: frame timestamp

        Returns:
            updated_state: [N, D] decoded frame
            header: frame header
        """
        # 1. Wave propagation on current state
        propagated_state = self.wave_engine.propagate_wave_step(self.current_decoded_state)

        # 2. Add delta stream (\Delta C)
        updated_state = propagated_state + delta_stream
        self.current_decoded_state.copy_(updated_state)

        # 3. Store into ring buffer
        self.stream_buffer.push_frame(self.current_decoded_state)

        active_mask = (delta_stream.norm(dim=-1) > 1e-4)
        header = CausalFrameHeader(
            frame_type="P-FRAME",
            timestamp=timestamp,
            num_nodes=self.num_nodes,
            dim=self.dim,
            active_mask=active_mask
        )
        return self.current_decoded_state, header

    def render_output(self, node_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Latent Causal Renderer (thought display / token rasterization).
        Renders continuous latent state output from the current wave energy state.
        """
        if node_weights is not None:
            # Weighted rasterization across nodes [N, D] x [N, 1] -> [D]
            norm_w = node_weights / (node_weights.sum() + 1e-8)
            aggregated = (self.current_decoded_state * norm_w.unsqueeze(-1)).sum(dim=0)  # [D]
        else:
            # Mean pool across nodes
            aggregated = self.current_decoded_state.mean(dim=0)  # [D]

        return self.latent_renderer(aggregated)
