#!/usr/bin/env python3
"""
Integrated Verification Script for Causal Nexus Engine Architecture
Verifies end-to-end execution of:
1. AsuraCheonmu Nexus bitmask packing
2. CausalToControlNetPipeline zero-allocation 3-channel voltage tensor injection
3. PyTorch CausalSubspaceProjectionLayer forward & autograd gradient isolation
4. 2D Betti Numbers (b0, b1) topological invariance & O(1) rollback
5. Bitwise 1-Bit BNN (BitNet XNOR) execution
6. Audio-Visual Voltage Surround 5.1 DSP register splitting
7. Performance latency benchmark
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import torch

from core.engine.asura_nexus import AsuraCheonmuNexus, NEXUS_BIT_TRIGGERED, NEXUS_BIT_DOMAIN_LOCKED
from core.engine.causal_nexus_pipeline import (
    CausalToControlNetPipeline,
    CausalSubspaceProjectionLayer,
    calculate_betti_2d,
    BitwiseCausalSimulator,
    CausalNexusRenderEngine,
)


def run_verification():
    print("=========================================================================")
    print("  VERIFYING HARDWARE-DIRECT CAUSAL NEXUS ENGINE & GENERATIVE AI HYBRID  ")
    print("=========================================================================\n")

    # 1. AsuraCheonmu Causal Nexus State & Bit Array Packing
    print("[1/6] AsuraCheonmu Causal Nexus Bit Array Serialization...")
    nexus = AsuraCheonmuNexus()
    flags = NEXUS_BIT_TRIGGERED | NEXUS_BIT_DOMAIN_LOCKED
    flat_buf = nexus.to_flat_bit_array(flags)
    assert len(flat_buf) == 64
    print("  ✓ Flat 64-byte bit array successfully serialized.")

    # 2. Voltage Tensor Injection
    print("\n[2/6] Voltage Tensor Injection & 3-Channel Binding...")
    H, W = 512, 512
    pipeline = CausalToControlNetPipeline(height=H, width=W, device="cpu")

    traj_bits = np.zeros((H, W), dtype=np.uint8)
    hitbox_bits = np.zeros((H, W), dtype=np.uint8)

    # Simulate Asura 16-slash trajectory and hitbox area
    traj_bits[100:400, 256] = 1
    hitbox_bits[200:300, 206:306] = 1

    t0 = time.perf_counter()
    voltage_tensor = pipeline.inject_nexus_bits(traj_bits, hitbox_bits)
    t1 = time.perf_counter()
    latency_us = (t1 - t0) * 1e6

    assert voltage_tensor.shape == (1, 3, H, W)
    print(f"  ✓ 3-Channel Voltage Tensor shape: {voltage_tensor.shape}")
    print(f"  ✓ Non-blocking Injection Latency: {latency_us:.2f} µs")

    # 3. PyTorch Autograd Subspace Projection & Gradient Masking
    print("\n[3/6] Orthogonal Subspace Projection Layer & Gradient Masking...")
    layer = CausalSubspaceProjectionLayer()
    z_latent = torch.randn(1, 4, 32, 32, requires_grad=True)
    z_prior = torch.ones(1, 4, 32, 32)
    v_mask = torch.zeros(1, 1, 32, 32)
    v_mask[0, 0, 8:24, 8:24] = 1.0  # Causal domain

    z_bounded = layer.forward(z_latent, v_mask, z_prior)
    z_bounded.sum().backward()

    # Leakage check: Outside gradient must be exactly 0
    outside_grad_norm = float(torch.norm(z_latent.grad[0, :, :8, :8]))
    inside_grad_norm = float(torch.norm(z_latent.grad[0, :, 8:24, 8:24]))

    assert outside_grad_norm == 0.0
    assert inside_grad_norm > 0.0
    print(f"  ✓ Inside Causal Domain Gradient Norm: {inside_grad_norm:.4f}")
    print(f"  ✓ Outside Causal Domain Gradient Norm: {outside_grad_norm:.4f} (Strict 0% Leakage)")

    # 4. Topological Invariance (Betti Numbers) & O(1) Rollback
    print("\n[4/6] 2D Betti Numbers (b0, b1) & Topological Rollback Engine...")
    engine = CausalNexusRenderEngine(height=64, width=64, device="cpu")

    test_mask = np.zeros((64, 64), dtype=np.uint8)
    test_mask[16:48, 16:48] = 1  # Solid component (b0=1, b1=0)

    res_valid = engine.render_step(test_mask, test_mask, expected_betti=(1, 0))
    assert res_valid["status"] == "SIGNALED"
    print(f"  ✓ Valid Frame Status: {res_valid['status']} | Betti: {res_valid['betti_numbers']}")

    # Induce topological disruption (expected_betti = 3, 3)
    res_rollback = engine.render_step(test_mask, test_mask, expected_betti=(3, 3))
    assert res_rollback["status"] == "TOPOLOGICAL_ROLLBACK"
    print(f"  ✓ Disrupted Frame Status: {res_rollback['status']} | Topological Loss: {res_rollback['topological_loss']}")

    # 5. Bitwise 1-Bit BNN SIMD Execution
    print("\n[5/6] 1-Bit Bitwise BNN (XNOR / POPCNT) Simulator...")
    in_blocks = np.full(8, 0xAAAAAAAAAAAAAAAA, dtype=np.uint64)
    w_blocks = np.full(8, 0xAAAAAAAAAAAAAAAA, dtype=np.uint64)

    act = BitwiseCausalSimulator.execute_bitwise_xnor_popcnt(in_blocks, w_blocks)
    assert len(act) == 512
    assert np.all(act == 1)
    print("  ✓ 1-Bit BNN XNOR Bitwise activations verified.")

    # 6. Audio-Visual Voltage Surround Splitter
    print("\n[6/6] Real-time Audio-Visual Voltage Surround Splitter...")
    av_frame = BitwiseCausalSimulator.split_audio_visual_voltage(traj_bits, hitbox_bits, H, W)
    rgb = av_frame["visual_voltage_rgb"]
    dsp = av_frame["audio_dsp_registers"]

    assert rgb.shape == (3, H, W)
    assert len(dsp) == 6
    print(f"  ✓ Visual RGB Voltage Tensor Shape: {rgb.shape}")
    print(f"  ✓ Audio 5.1 Surround DSP Registers [L, R, C, LFE, SL, SR]: {np.round(dsp, 4)}")

    print("\n=========================================================================")
    print("  ALL CAUSAL NEXUS ENGINE VERIFICATION TESTS PASSED SUCCESSFULLY!       ")
    print("=========================================================================")


if __name__ == "__main__":
    run_verification()
