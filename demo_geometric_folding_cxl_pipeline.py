#!/usr/bin/env python3
"""
Integration Demo: Geometric Product Folding & Hardware-Aware CXL/GDS Phase-Fault Pipeline

This demo demonstrates the full end-to-end cognitive memory cycle:
1. Substrate 1-Vector signals folding via Geometric Product and Clifford Rotors in Cl(3,0).
2. Phase coherence evaluation & promotion to upper virtual memory volume.
3. Hardware-aware latency mapping across VRAM, System RAM, and NVMe SSD.
4. Phase-Fault detection and zero-copy virtual memory swapping.
"""

import torch
import math
from core.memory.geometric_folding_engine import GeometricFoldingEngine
from core.memory.hardware_aware_clifford_pipeline import HardwareAwareCliffordPipeline

def run_geometric_folding_cxl_demo():
    print("=========================================================================")
    print("  Elysia Cognitive Engine: Geometric Product Folding & CXL/GDS Pipeline  ")
    print("=========================================================================\n")

    # 1. Initialize PyTorch Engines
    folding_engine = GeometricFoldingEngine(dim=8)
    pipeline = HardwareAwareCliffordPipeline()

    # 2. Simulate Substrate 1-Vector Signals (Amino Acid Fragments)
    batch_size = 4
    v1 = torch.zeros(batch_size, 8)
    v2 = torch.zeros(batch_size, 8)

    # Populate 1-vector components (indices 1, 2, 3)
    v1[:, 1:4] = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.5, 0.8, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.5]
    ])
    v2[:, 1:4] = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.8, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.5]
    ])

    theta = torch.tensor([math.pi / 4.0, math.pi / 6.0, math.pi / 3.0, math.pi / 2.0])

    print("[STEP 1] Substrate Signals -> Geometric Folding Operation")
    V_promoted = folding_engine(v1, v2, theta)

    for i in range(batch_size):
        print(f"  Signal {i}:")
        print(f"    - Grade-0 (Scalar / Contraction) : {V_promoted[i, 0].item():.4f}")
        print(f"    - Grade-1 (Rotated Vector Field) : {V_promoted[i, 1:4].tolist()}")
        print(f"    - Grade-2 (Bivector Area Volume) : {V_promoted[i, 4:7].tolist()}")
        print(f"    - Grade-3 (Top Pseudoscalar Vol) : {V_promoted[i, 7].item():.4f}")

    print("\n[STEP 2] Hardware-Aware Latency Mapping Across Tiers")
    vram_block = V_promoted
    ram_block = V_promoted * 0.9
    ssd_block = V_promoted * 0.5

    memory_blocks = {
        'vram': vram_block,
        'ram': ram_block,
        'ssd': ssd_block
    }

    psi_total = pipeline(memory_blocks)

    for tier in ['vram', 'ram', 'ssd']:
        metric_scale = pipeline.compute_hardware_metric(tier)
        print(f"  Tier [{tier.upper()}] - Metric Distance g_kk: {metric_scale:.2f}")

    print(f"\n  Combined Active Multivector Field Shape: {psi_total.shape}")
    print(f"  Active Field Energy Norm: {torch.norm(psi_total).item():.4f}")

    print("\n[STEP 3] Phase Coherence & Phase-Fault Evaluation")
    gamma_threshold = 0.5
    for i in range(batch_size):
        coherence = folding_engine.compute_coherence(vram_block[i], ssd_block[i]).item()
        faulted = coherence < gamma_threshold
        status = "PHASE FAULT TRIGGERED -> Swapping Required" if faulted else "PHASE LOCK OK"
        print(f"  Block {i} Coherence Score: {coherence:.4f} | Status: {status}")

    print("\n=========================================================================")
    print("  Integration Demo Successfully Executed!")
    print("=========================================================================")

if __name__ == "__main__":
    run_geometric_folding_cxl_demo()
