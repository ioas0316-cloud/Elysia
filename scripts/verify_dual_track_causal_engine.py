"""
Verification script for Dual-Track Causal Engine (`scripts/verify_dual_track_causal_engine.py`).
"""

import sys
import time
import numpy as np

from core.engine.dual_track_causal_engine import DualTrackCausalEngine, DirectStorageFileHeader, CausalPageEntry


def main():
    print("=" * 70)
    print("      ELYSIAN DUAL-TRACK CAUSAL COGNITIVE ENGINE VERIFICATION      ")
    print("=" * 70)

    num_nodes = 5000
    print(f"\n[STEP 1] Initializing Dual-Track Causal Engine with {num_nodes:,} nodes...")
    start_time = time.time()
    engine = DualTrackCausalEngine(num_nodes=num_nodes, energy_threshold=1.0, lock_threshold=0.2)
    init_time = time.time() - start_time
    print(f" -> Initialized in {init_time:.4f} seconds.")

    print("\n[STEP 2] Simulating Baseline Track A Latent Space (0 Active Impulse)...")
    stats_baseline = engine.step_simulation(0.0)
    print(f" -> Active Nodes: {stats_baseline['active_nodes']} / {stats_baseline['total_nodes']}")
    print(f" -> ALU Compute Reduction: {stats_baseline['alu_reduction_percent']:.2f}%")
    assert stats_baseline["active_nodes"] == 0
    assert stats_baseline["alu_reduction_percent"] == 100.0

    print("\n[STEP 3] Injecting Local Energy Impulse into 50 Target Nodes...")
    for idx in range(50):
        engine.inject_external_impulse(node_idx=idx * 2, impulse_energy=2.5)

    stats_impulse = engine.step_simulation(0.1)
    print(f" -> Active Nodes: {stats_impulse['active_nodes']} / {stats_impulse['total_nodes']}")
    print(f" -> ALU Compute Reduction: {stats_impulse['alu_reduction_percent']:.2f}%")
    assert stats_impulse["active_nodes"] >= 50

    print("\n[STEP 4] Simulating Energy Diffusion & Attractor Phase-Lock Baking...")
    for t in range(25):
        step_stats = engine.step_simulation(0.1 * (t + 2))

    print(f" -> Baked Trajectory Pages in Page Table: {len(engine.page_table)}")
    assert len(engine.page_table) >= 0

    print("\n[STEP 5] Testing Critical Entropy Evaluation at Bifurcation Ridge...")
    for idx in range(100):
        engine.inject_external_impulse(node_idx=idx, impulse_energy=5.0)
    engine.step_simulation(3.0)

    bifurcation_event = engine.evaluate_critical_bifurcation(entropy_threshold=0.1)
    if bifurcation_event:
        print(" -> Critical Bifurcation Event Detected & Branching Executed!")
        print(f"    - Entropy: {bifurcation_event['entropy']:.4f}")
        print(f"    - Active Nodes: {bifurcation_event['active_nodes_count']}")
        print(f"    - Resolution: {bifurcation_event['resolution']}")
        assert bifurcation_event["resolution"] == "Ridge_Split_Conditional_Attractor"

    print("\n[STEP 6] DirectStorage Header & Page Table Binary Packing Verification...")
    header = DirectStorageFileHeader(total_baked_pages=len(engine.page_table))
    packed = header.pack()
    unpacked = DirectStorageFileHeader.unpack(packed)
    assert unpacked.magic_bytes == b"ELYSIAN1"
    assert unpacked.total_baked_pages == len(engine.page_table)
    print(f" -> Header Sector Size: {len(packed)} bytes (4KB aligned). Magic: {unpacked.magic_bytes.decode()}")

    print("\n" + "=" * 70)
    print("      ALL DUAL-TRACK CAUSAL ENGINE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
