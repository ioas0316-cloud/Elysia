"""
Unified Causal Pipeline Benchmark & Demonstration
===================================================
Compares traditional OOP polling pipeline vs Unified Causal Pipeline
across key metrics:
1. Frame Latency & Execution Speed (ms per 1,000 frames)
2. Zero-Polling Culling Efficiency (Evaluated Nodes / Skipped Branches)
3. Snapshot Rollback & Fast-Forward Resimulation Overhead
4. Zero-Copy Network & Tensor Dispatch Overhead
"""

import time
import numpy as np
from synaptic_architecture.unified_causal_pipeline import (
    UnifiedCausalPipeline,
    SignalCartridge,
    ENTITY_STATE_DTYPE
)


class OOPEntity:
    """Traditional OOP Entity with individual heap allocations and un-aligned structs."""
    def __init__(self, entity_id):
        self.entity_id = entity_id
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [1.0, 2.0, 3.0]
        self.rotation = [0.0, 0.0, 0.0, 1.0]
        self.latent_param = 0.0
        self.flags = 0

    def update_physics(self, dt=0.016):
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.position[2] += self.velocity[2] * dt
        self.velocity[0] *= 0.95
        self.velocity[1] *= 0.95
        self.velocity[2] *= 0.95

    def update_anim(self):
        self.latent_param += 0.05

    def serialize(self):
        return {
            'id': self.entity_id,
            'pos': list(self.position),
            'vel': list(self.velocity),
            'rot': list(self.rotation),
            'param': self.latent_param
        }


def run_benchmark():
    num_entities = 10000
    num_frames = 500

    print("==========================================================================")
    print("      UNIFIED CAUSAL PIPELINE vs TRADITIONAL OOP BENCHMARK")
    print(f"      Entity Count: {num_entities:,} | Benchmark Frames: {num_frames:,}")
    print("==========================================================================")

    # -------------------------------------------------------------------------
    # 1. Traditional OOP Polling Pipeline Simulation
    # -------------------------------------------------------------------------
    print("\n[1/3] Benchmarking Traditional OOP Polling Pipeline...")
    oop_entities = [OOPEntity(i) for i in range(num_entities)]

    t0 = time.perf_counter()
    oop_evals = 0
    for frame in range(num_frames):
        # Polling every entity every frame regardless of change
        for ent in oop_entities:
            ent.update_physics()
            ent.update_anim()
            oop_evals += 2
        # Mock serialization copy
        _ = [ent.serialize() for ent in oop_entities[:10]]
    t1 = time.perf_counter()
    oop_ms = (t1 - t0) * 1000.0

    print(f"    - Execution Time: {oop_ms:.2f} ms")
    print(f"    - Total Component Updates Evaluated: {oop_evals:,}")

    # -------------------------------------------------------------------------
    # 2. Unified Causal Pipeline Simulation (Signal-Driven & Zero-Polling)
    # -------------------------------------------------------------------------
    print("\n[2/3] Benchmarking Unified Causal Pipeline...")
    causal_pipeline = UnifiedCausalPipeline(capacity=num_entities)

    # Inject signals only on intermittent frames (e.g., every 50 frames)
    t0 = time.perf_counter()
    causal_evals = 0
    for frame in range(num_frames):
        if frame % 50 == 0:
            cartridge = SignalCartridge.pack(protocol_id=1, phase=0, param=100, seed=42)
            causal_pipeline.inject_signal(cartridge)

        res = causal_pipeline.step_frame()
        causal_evals += res['nodes_evaluated']

        # Zero-Copy dispatch
        _, _ = causal_pipeline.dispatch_outputs()
    t1 = time.perf_counter()
    causal_ms = (t1 - t0) * 1000.0

    print(f"    - Execution Time: {causal_ms:.2f} ms")
    print(f"    - Total Causal Nodes Evaluated: {causal_evals:,}")
    print(f"    - Performance Speedup: {oop_ms / causal_ms:.2f}x Faster")
    print(f"    - Node Evaluation Reduction: {(1.0 - causal_evals / oop_evals) * 100:.2f}% Culled")

    # -------------------------------------------------------------------------
    # 3. Snapshot Rollback & Fast-Forward Resimulation Test
    # -------------------------------------------------------------------------
    print("\n[3/3] Benchmarking Snapshot Rollback & Fast-Forward Resimulation...")
    current_f = causal_pipeline.current_frame
    rollback_target = current_f - 30  # Rollback 30 frames (~0.5 sec)

    write_buf = causal_pipeline.triple_buffer.get_write_buffer()
    t0 = time.perf_counter()
    success = causal_pipeline.ring_buffer.rollback_and_resimulate(
        target_frame=rollback_target,
        current_frame=current_f,
        state_buffer=write_buf,
        dag=causal_pipeline.dag,
        solver=causal_pipeline.solver
    )
    t1 = time.perf_counter()
    rollback_us = (t1 - t0) * 1_000_000.0

    print(f"    - Rollback 30 Frames Resimulation Time: {rollback_us:.2f} μs ({rollback_us/1000.0:.3f} ms)")
    print(f"    - Rollback Status: {'SUCCESS' if success else 'FAILED'}")

    print("\n==========================================================================")
    print("      UNIFIED CAUSAL PIPELINE BENCHMARK COMPLETE")
    print("==========================================================================")


if __name__ == "__main__":
    run_benchmark()
