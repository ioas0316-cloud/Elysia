import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class PhaseRectificationFunction(nn.Module):
    """
    Complex Phase Rectification Function (CPRF)
    Collapses low-level continuous tensors into high-level discrete symbolic basis nodes
    via Heaviside amplitude gating, phase quantization, and temperature-based Gumbel-Softmax phase transition.
    """
    def __init__(self, embed_dim: int, num_symbols: int, tau: float = 0.5, num_phases: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_symbols = num_symbols
        self.tau = tau  # Amplitude gating threshold
        self.num_phases = num_phases  # Quantization steps M

        # High-level symbolic basis dictionary
        self.symbol_bases = nn.Parameter(torch.randn(num_symbols, embed_dim))

    def _quantize_phase(self, phase: torch.Tensor) -> torch.Tensor:
        """Quantizes continuous phase angles into M discrete quantization basins."""
        step = 2.0 * torch.pi / self.num_phases
        quantized = torch.round(phase / step) * step
        return quantized

    def forward(self, h: torch.Tensor, temperature: float = 0.01) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            h: [Batch, embed_dim] continuous input tensor
            temperature: Gumbel-Softmax temperature parameter (T -> 0 for discrete collapse)
        Returns:
            rectified_h: [Batch, embed_dim] discrete symbolic rectified tensor
            logs: dictionary with diagnostic metrics
        """
        # 1. Hilbert space projection: Resonance amplitude calculation
        norm_h = F.normalize(h, p=2, dim=-1)
        norm_bases = F.normalize(self.symbol_bases, p=2, dim=-1)

        amplitudes = torch.matmul(norm_h, norm_bases.T)  # [B, K]

        # 2. Heaviside amplitude gating (Hard Sparsification)
        gated_amplitudes = torch.where(amplitudes > self.tau, amplitudes, torch.zeros_like(amplitudes))

        # 3. Gumbel-Softmax discrete transition (T -> 0)
        symbol_probs = F.gumbel_softmax(gated_amplitudes, tau=temperature, hard=True)  # [B, K] One-Hot

        # 4. Discrete basis reconstruction
        rectified_h = torch.matmul(symbol_probs, self.symbol_bases)  # [B, embed_dim]

        active_symbols = torch.argmax(symbol_probs, dim=-1)
        sparsity_ratio = (gated_amplitudes == 0).float().mean().item()

        return rectified_h, {
            "active_symbol_id": active_symbols,
            "sparsity_ratio": sparsity_ratio
        }
