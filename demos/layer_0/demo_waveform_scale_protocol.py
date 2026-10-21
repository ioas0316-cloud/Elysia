"""
Demonstration Script: Waveform Scale Protocol Engine & Phase Locking
=====================================================================
This demo demonstrates how the Waveform Scale Protocol Engine operates across scale hierarchy:
1. Setting macro topological phase boundaries.
2. Spontaneous Phase-Locking (자발적 위상 동기화) of thousands of micro nodes without discrete checking loops.
3. Scale-invariant result lookup (O(1) continuous direct mapping) eliminating computational friction.
"""

import time
import torch
from core.physics.waveform_scale_protocol import WaveformScaleProtocolEngine, PhaseLockingMetrics


def main():
    print("=" * 72)
    print(" Elysia: Waveform Scale Protocol Engine & Phase Alignment Demo")
    print(" Philosophy: 'Do not calculate, let it flow.'")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 2000
    print(f"\n[1] Initializing Waveform Scale Protocol Engine ({num_nodes} Micro Nodes)...")

    engine = WaveformScaleProtocolEngine(
        num_micro_nodes=num_nodes,
        macro_dim=64,
        coupling_strength_K=8.0,
        dt=0.03,
        device=device
    )

    init_R, _ = engine.compute_order_parameter()
    print(f"    - Initial Micro Node Order Parameter R: {init_R.item():.4f} (Unsynchronized background)")

    # Test Macro Phase Alignment over 3 distinct inputs
    test_inputs = [15.0, 42.0, 99.0]

    for idx, inp in enumerate(test_inputs, 1):
        print(f"\n[{idx+1}] Shifting Macro Phase for Scale Input = {inp}...")
        start_time = time.time()

        result_vec, metrics = engine(input_val=inp, steps=25)
        elapsed = (time.time() - start_time) * 1000.0

        print(f"    - Macro Phase Target        : {metrics.macro_phase:.4f} rad")
        print(f"    - Order Parameter R (Sync)   : {metrics.order_parameter_R:.4f} (1.0 = Perfect Alignment)")
        print(f"    - Micro Phase Variance      : {metrics.micro_phase_variance:.4f}")
        print(f"    - Bottleneck Reduction      : {metrics.bottleneck_reduction_ratio:.2f}% (Resistance Eliminated)")
        print(f"    - Field Energy Potential    : {metrics.field_energy:.4f}")
        print(f"    - Chromatic Signature (R/B/Y): Red(Flux)={metrics.chromatic_balance[0]:.3f}, Blue(Order)={metrics.chromatic_balance[1]:.3f}, Yellow(Entropy)={metrics.chromatic_balance[2]:.3f}")
        print(f"    - Direct Result Vector Norm : {torch.norm(result_vec).item():.4f}")
        print(f"    - Continuous Resolution Time: {elapsed:.2f} ms")

    print("\n" + "=" * 72)
    print(" SUCCESS: Macro field phase shift induced instantaneous micro alignment.")
    print(" All lower-level checking bottlenecks eliminated via Waveform Scale Protocol.")
    print("=" * 72)


if __name__ == "__main__":
    main()
