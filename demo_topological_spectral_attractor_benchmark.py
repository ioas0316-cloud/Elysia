"""
Demo & Benchmark Script: Topological Spectral Attractor Relaxation vs Standard Attention
Compares Transformer O(N^2) Attention with Spectral Attractor Relaxation O(N log N) in memory and execution speed.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from synaptic_architecture.topological_spectral_engine import SpectralAttractorRelaxation


class StandardTransformerAttention(nn.Module):
    """Standard Scaled Dot-Product Attention (O(N^2) memory and compute)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (D ** 0.5)  # (B, N, N)
        attn_weights = F.softmax(scores, dim=-1)                   # (B, N, N)
        return torch.matmul(attn_weights, V)                        # (B, N, D)


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"  ELYSIUM ENGINE: SPECTRAL ATTRACTOR RELAXATION BENCHMARK")
    print(f"  Running on execution device: {device.type.upper()}")
    print("=" * 90)

    batch_size = 2
    embed_dim = 128
    seq_lengths = [512, 1024, 2048, 4096, 8192]

    transformer = StandardTransformerAttention(embed_dim).to(device)
    spectral = SpectralAttractorRelaxation(embed_dim).to(device)

    header = f"{'Seq Length (N)':<15} | {'Trans. Mem (MB)':<18} | {'FFT Mem (MB)':<16} | {'Trans. Time (ms)':<18} | {'FFT Time (ms)':<15}"
    print(header)
    print("-" * len(header))

    for N in seq_lengths:
        x = torch.randn(batch_size, N, embed_dim, device=device)

        # 1. Standard Transformer Attention
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        try:
            _ = transformer(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_trans = (time.time() - t0) * 1000
            mem_trans = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
            mem_t_str = f"{mem_trans:.2f}"
            t_t_str = f"{t_trans:.2f}"
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            mem_t_str = "OOM"
            t_t_str = "N/A"
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # 2. Spectral Attractor Relaxation
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        _ = spectral(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_spec = (time.time() - t0) * 1000
        mem_spec = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

        print(f"{N:<15} | {mem_t_str:<18} | {mem_spec:<16.2f} | {t_t_str:<18} | {t_spec:<15.2f}")

    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
