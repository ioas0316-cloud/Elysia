"""
Unit and Minimum Acceptance Tests (T1 - T5) for Scale-Coupling System
(Principle Document v0.1 Specification)

T1. Freezing and Thawing Value Preservation within tolerance.
T2. Calculation using frozen blocks (FLOPs=0) vs without freezing error & speed metrics.
T3. Bottom-up crystallization propagation (when all children freeze, parent couples and freezes).
T4. Limit Map creation when reconstruction error exceeds threshold.
T5. Hysteresis stability preventing state flickering at freeze/thaw boundary.
"""

import os
import time
import pytest
import torch
import numpy as np

from core.physics.scale_coupling_system import (
    NodeState,
    StructureNode,
    ScaleCouplingSystem,
    DefaultCouplingFunction
)


def test_t1_freeze_thaw_value_preservation(tmp_path):
    """
    T1. Values after freeze and thaw match original values within tolerance.
    """
    storage_dir = str(tmp_path / "t1_store")
    node = StructureNode(node_id="test_node_t1", scale=0, storage_dir=storage_dir)

    original_phi = torch.randn(8, 8, dtype=torch.float32)
    original_val = torch.randn(8, 8, dtype=torch.float32)

    node.condense_from_gas(original_phi, original_val)
    node.freeze()

    assert node.state == NodeState.ICE
    assert os.path.exists(node.mmap_path)

    # Thaw node
    node.thaw()
    assert node.state == NodeState.LIQUID

    diff_phi = torch.max(torch.abs(node.phi - original_phi)).item()
    diff_val = torch.max(torch.abs(node.val - original_val)).item()

    assert diff_phi < 1e-6, f"Phi reconstructed error too high: {diff_phi}"
    assert diff_val < 1e-6, f"Val reconstructed error too high: {diff_val}"


def test_t2_frozen_block_computation_vs_full(tmp_path):
    """
    T2. Report calculation with frozen blocks vs full calculation error and speed/FLOPs gain.
    """
    sys = ScaleCouplingSystem(storage_dir=str(tmp_path / "t2_sys"), eps_freeze=0.001, eps_thaw=0.5)

    # Inject smooth observations that rapidly settle
    obs = []
    for _ in range(4):
        phi = torch.ones(8, 8) * 0.1
        val = torch.ones(8, 8) * 0.1
        obs.append((phi, val))

    sys.inject_observation(obs)

    # Run steps until system crystallizes into ICE
    steps_run = 0
    while sys.root.state != NodeState.ICE and steps_run < 50:
        sys.step(dt=0.01)
        steps_run += 1

    assert sys.root.state == NodeState.ICE

    # Measure FLOPs when frozen
    initial_flops = sum(n.flops_count for n in sys.nodes.values())
    t0 = time.perf_counter()
    for _ in range(100):
        sys.step(dt=0.01)  # All nodes ICE -> no internal PDE FLOPs executed
    t1 = time.perf_counter()
    frozen_time = t1 - t0
    frozen_flops_added = sum(n.flops_count for n in sys.nodes.values()) - initial_flops

    # Benchmark without freezing (always LIQUID)
    sys_active = ScaleCouplingSystem(storage_dir=str(tmp_path / "t2_active"), eps_freeze=-1.0)  # Never freeze
    sys_active.inject_observation(obs)

    t2 = time.perf_counter()
    for _ in range(100):
        sys_active.step(dt=0.01)
    t3 = time.perf_counter()
    active_time = t3 - t2
    active_flops = sum(n.flops_count for n in sys_active.nodes.values())

    assert frozen_flops_added == 0, "Frozen blocks must execute 0 FLOPs for internal PDE steps"
    assert active_flops > 0, "Active liquid system must execute PDE FLOPs"

    print(f"\n[T2 Report] Frozen FLOPs added: {frozen_flops_added}, Active FLOPs: {active_flops}")
    print(f"[T2 Report] Frozen execution time: {frozen_time:.6f}s, Active execution time: {active_time:.6f}s")


def test_t3_bottom_up_crystallization(tmp_path):
    """
    T3. When all children freeze, parent applies coupling function and freezes.
    """
    sys = ScaleCouplingSystem(storage_dir=str(tmp_path / "t3_sys"), eps_freeze=0.01)

    obs = [
        (torch.full((8, 8), 0.5), torch.full((8, 8), 0.2)),
        (torch.full((8, 8), 0.5), torch.full((8, 8), 0.2)),
        (torch.full((8, 8), 0.5), torch.full((8, 8), 0.2)),
        (torch.full((8, 8), 0.5), torch.full((8, 8), 0.2)),
    ]
    sys.inject_observation(obs)

    assert sys.root.state == NodeState.GAS
    assert all(child.state == NodeState.LIQUID for child in sys.leaf_nodes)

    # Step PDE until leaves freeze
    for _ in range(20):
        sys.step(dt=0.01)

    assert all(child.state == NodeState.ICE for child in sys.leaf_nodes)
    assert sys.root.state == NodeState.ICE, "Parent root must freeze automatically when all children freeze"

    # Parent phi/val should equal coupled result of children
    expected_phi, expected_val = sys.root.coupling_fn.couple(sys.leaf_nodes)
    assert torch.allclose(sys.root.phi, expected_phi, atol=1e-5)
    assert torch.allclose(sys.root.val, expected_val, atol=1e-5)


def test_t4_limit_map_on_reconstruction_failure(tmp_path):
    """
    T4. Limit Map creation when reconstruction error exceeds threshold (eps_limit).
    """
    # Create child leaf nodes with sharp non-smooth discontinuities that cannot be downsampled/upsampled accurately
    sys = ScaleCouplingSystem(
        storage_dir=str(tmp_path / "t4_sys"),
        eps_freeze=0.01,
        eps_limit=0.01  # Strict limit threshold
    )

    # Incompatible high-frequency pattern across 4 leaves (smooth enough to freeze quickly)
    leaf_obs = []
    for i in range(4):
        phi = torch.ones(8, 8) * (i + 1) * 5.0
        val = torch.ones(8, 8) * (-1.0 if i % 2 == 0 else 1.0) * 5.0
        leaf_obs.append((phi, val))

    sys.inject_observation(leaf_obs)

    for _ in range(20):
        sys.step(dt=0.01)

    assert sys.root.state == NodeState.ICE
    assert len(sys.root.limit_records) > 0, "Root must record limit map entry when reconstruction error exceeds limit threshold"

    limit_rec = sys.root.limit_records[0]
    assert limit_rec.recon_error > limit_rec.threshold
    print(f"\n[T4 Report] Limit map recorded with relative reconstruction error: {limit_rec.recon_error:.4f} > threshold {limit_rec.threshold}")


def test_t5_hysteresis_boundary_flicker_prevention(tmp_path):
    """
    T5. State does not flicker at freeze/thaw boundary due to hysteresis (eps_freeze != eps_thaw).
    """
    storage_dir = str(tmp_path / "t5_store")
    node = StructureNode(
        node_id="hysteresis_node",
        scale=0,
        storage_dir=storage_dir,
        eps_freeze=0.01,
        eps_thaw=0.20  # Significant hysteresis gap
    )

    phi = torch.ones(8, 8) * 0.5
    val = torch.ones(8, 8) * 0.5
    node.condense_from_gas(phi, val)

    # Step until frozen
    for _ in range(10):
        node.step_pde(dt=0.01)

    assert node.state == NodeState.ICE

    # Apply small stimulus between eps_freeze (0.01) and eps_thaw (0.20), e.g. norm = 0.08
    small_stimulus = torch.ones(8, 8) * 0.01  # norm = sqrt(64 * 0.0001) = 0.08
    stimulus_norm = torch.norm(small_stimulus).item()
    assert node.eps_freeze < stimulus_norm < node.eps_thaw

    # Step PDE with small stimulus
    for _ in range(10):
        node.step_pde(dt=0.01, external_stimulus=small_stimulus)

    # Node MUST remain ICE (no flickering back to LIQUID)
    assert node.state == NodeState.ICE, "Node should not thaw under stimulus below eps_thaw threshold"

    # Apply large stimulus above eps_thaw (0.20), e.g. norm = 0.8
    large_stimulus = torch.ones(8, 8) * 0.1  # norm = 0.8
    assert torch.norm(large_stimulus).item() > node.eps_thaw

    node.step_pde(dt=0.01, external_stimulus=large_stimulus)
    assert node.state == NodeState.LIQUID, "Node should thaw when stimulus exceeds eps_thaw threshold"
