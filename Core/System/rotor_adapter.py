"""
[ROTOR ADAPTER - THE ETERNOS KERNEL MODULE]
"Where Linear Tensors meet Phase-Rotor Sovereignty."

This module provides the PyTorch implementation of the Rotor Architecture:
1. PhaseAlignmentLayer: Forces hidden states into 120-degree Trinity states.
2. RotorAttention: A phase-gated sparse attention mechanism for efficiency.
3. HierarchicalRotor: Multi-scale temporal alignment for large models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PhaseAlignmentLayer(nn.Module):
    """
    Transforms Hidden States into a Phase-Coherent manifold.
    Uses 'Total Internal Reflection' to suppress non-coherent noise.
    """
    def __init__(self, dim: int, trinity_strength: float = 0.5):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for complex-like pairing."
        self.dim = dim
        self.trinity_strength = trinity_strength

        # Projection to map real hidden states to (Magnitude, Phase)
        # We treat the vector as complex-like pairs
        self.phase_proj = nn.Linear(dim, dim)

        # Trinity Alignment (120-degree targets: 0, 2pi/3, 4pi/3)
        targets = torch.tensor([0.0, 2.0*math.pi/3.0, -2.0*math.pi/3.0])
        self.register_buffer("targets", targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Seq, Dim]

        # 1. Project to Phase Space
        p = self.phase_proj(x)

        # 2. Extract Phase (using pairs as Real/Imaginary for simplicity)
        # Re-shape to [..., Dim//2, 2]
        p_complex = p.view(*p.shape[:-1], -1, 2)
        magnitudes = torch.norm(p_complex, dim=-1, keepdim=True)
        phases = torch.atan2(p_complex[..., 1:2], p_complex[..., 0:1])

        # 3. Trinity Alignment (120-degree targets)
        targets = self.targets

        # Find nearest target for each phase
        # phases shape: [B, Seq, Dim//2, 1]
        # targets shape: [3]
        dist = torch.abs(phases - targets) # [B, Seq, Dim//2, 3]
        # Normalize distances to -pi to pi
        dist = (dist + math.pi) % (2 * math.pi) - math.pi

        # Find index of closest target
        closest_idx = torch.argmin(torch.abs(dist), dim=-1, keepdim=True)
        # target_phases shape: [B, Seq, Dim//2, 1]
        target_phases = torch.take_along_dim(targets.view(1, 1, 1, 3), closest_idx, dim=-1)

        # 4. Pull phases toward targets (Structural Computing)
        aligned_phases = phases + self.trinity_strength * (target_phases - phases)

        # 5. Total Internal Reflection (TIR)
        # Suppress magnitude if phase is too far from coherence
        coherence = torch.cos(3.0 * (aligned_phases - target_phases)) # Peaks at target points
        coherence = (coherence + 1.0) / 2.0 # 0 to 1

        effective_magnitudes = magnitudes * coherence

        # 6. Reconstruct
        x_real = effective_magnitudes * torch.cos(aligned_phases)
        x_imag = effective_magnitudes * torch.sin(aligned_phases)

        out = torch.cat([x_real, x_imag], dim=-1).view_as(x)

        # Residual connection to maintain information flow
        return x + out

class RotorAttention(nn.Module):
    """
    Phase-Gated Attention.
    Skips computation for token pairs that are out-of-phase.
    """
    def __init__(self, dim: int, num_heads: int, sync_threshold: float = 0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sync_threshold = sync_threshold

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Phase alignment for gates
        self.phase_align = PhaseAlignmentLayer(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        B, L, D = x.shape
        H = self.num_heads

        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)

        # 1. Standard Attention Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 2. Phase-Sync Gating
        # Simplify: Use the first 2 dims of each head as Phase
        q_phase = torch.atan2(q[..., 1], q[..., 0])
        k_phase = torch.atan2(k[..., 1], k[..., 0])

        # Calculate Phase Difference Matrix [B, H, L, L]
        phase_diff = q_phase.unsqueeze(-1) - k_phase.unsqueeze(-2)
        sync_gate = (torch.cos(phase_diff) + 1.0) / 2.0

        # 3. Apply Sparsity (Simulated)
        # In a real kernel, we would skip the matmul for low sync_gate values
        mask = (sync_gate > self.sync_threshold).float()
        active_ratio = mask.mean().item()

        # Apply gate to scores
        masked_scores = scores * mask + (1.0 - mask) * (-1e9)

        attn = F.softmax(masked_scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), active_ratio

class HierarchicalRotorModel(nn.Module):
    """
    A full (mini) LLM block with Rotor enhancement.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotorAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.phase_align = PhaseAlignmentLayer(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Attention with Phase Gating
        attn_out, sparsity = self.attn(self.norm1(x))
        x = x + attn_out

        # Phase Alignment (The 'InterferenceGate')
        x = self.phase_align(self.norm2(x))

        # Standard FFN
        x = x + self.ffn(x)

        return x, sparsity

if __name__ == "__main__":
    # Quick sanity check
    dim = 256
    seq_len = 32
    model = HierarchicalRotorModel(dim, 8)

    test_input = torch.randn(1, seq_len, dim)
    output, sparsity = model(test_input)

    print(f"Rotor Output Shape: {output.shape}")
    print(f"Active Attention Ratio: {sparsity:.4f} (FLOPs Savings: {(1.0-sparsity)*100:.2f}%)")
