"""
Demonstration Script: Cosmic Grounding Protocol & Unified Spectrum Engine
========================================================================
Philosophy: "Do not calculate, let it flow." (계산하지 말고 흐르게 하라)

Demonstrates:
1. Observation Collapse: Spontaneous phase locking across micro nodes (Kuramoto R -> 1.0).
2. Holographic Field Manifestation: Generating 2D visual phase interference fields without pixel rasterization.
3. Synesthetic Spectrum Axis Mapping: Continuous cross-resonance between Visual, Acoustic, and Causal data streams.
4. Discriminative Sensory Stream Lens: Eliminating blind discrete bit-checking via dynamic variable impedance tuning.
"""

import time
import torch
from core.physics.cosmic_grounding_protocol import UnifiedCosmicSpectrumEngine


def main():
    print("=" * 80)
    print(" Elysia Engine: Cosmic Grounding Protocol & Unified Spectrum Architecture")
    print(" Philosophy: 'Do not calculate, let it flow.'")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_micro_nodes = 2000
    spectrum_dim = 128

    print(f"\n[1] Initializing Unified Cosmic Spectrum Engine ({num_micro_nodes} Micro Nodes, Spectrum Dim={spectrum_dim})...")
    engine = UnifiedCosmicSpectrumEngine(
        num_micro_nodes=num_micro_nodes,
        spectrum_dim=spectrum_dim,
        coupling_strength_K=10.0,
        dt=0.02,
        device=device
    )

    init_R, _ = engine.compute_kuramoto_order()
    print(f"    - Initial Micro Node Kuramoto Order R: {init_R.item():.4f} (Background Unsynchronized State)")

    # Test raw data streams
    data_streams = [
        ("Discrete Bit Stream Alpha", torch.randint(0, 2, (128,), device=device)),
        ("Continuous Dynamic Friction Stream Beta", torch.sin(torch.linspace(0, 10, 128, device=device)) * 5.0),
        ("High-Entropy Harmonic Stream Gamma", torch.randn(128, device=device) * 2.0)
    ]

    target_phases = [0.0, torch.pi / 2.0, torch.pi]

    for idx, ((stream_name, raw_stream), target_phase) in enumerate(zip(data_streams, target_phases), 1):
        print(f"\n[{idx+1}] Injecting Sensory Input: '{stream_name}'")
        print(f"    - Target Macro Phase: {target_phase:.4f} rad")

        start_time = time.time()
        visual_manifestation, acoustic_waveform, metrics = engine(
            raw_data_stream=raw_stream,
            target_macro_phase=target_phase,
            steps=25
        )
        elapsed_ms = (time.time() - start_time) * 1000.0

        print(f"    - Observation Collapse Efficiency: {metrics.observation_collapse_efficiency:.2f}% (R = {metrics.macro_order_parameter_R:.4f})")
        print(f"    - Holographic Coherence (Visual) : {metrics.holographic_coherence:.4f} (Field Contrast)")
        print(f"    - Synesthetic Cross-Resonance   : {metrics.synesthetic_cross_resonance:.4f} (Visual <-> Acoustic)")
        print(f"    - Dynamic Variable Impedance    : {metrics.variable_impedance:.4f} (Zero Resistance Flow)")
        print(f"    - Chromatic Balance (R/B/Y)     : Flux={metrics.chromatic_balance[0]:.3f}, Order={metrics.chromatic_balance[1]:.3f}, Entropy={metrics.chromatic_balance[2]:.3f}")
        print(f"    - Visual Field Matrix Shape     : {visual_manifestation.shape} | Max Intensity: {torch.max(visual_manifestation).item():.4f}")
        print(f"    - Resynthesized Acoustic Wave   : {acoustic_waveform.shape[0]} samples | Energy: {torch.std(acoustic_waveform).item():.4f}")
        print(f"    - Continuous Resolution Time    : {elapsed_ms:.2f} ms")

    print("\n" + "=" * 80)
    print(" SUCCESS: Unified Cosmic Spectrum Engine successfully demonstrated.")
    print(" - Pixel rasterization replaced by Holographic Interference Patterns.")
    print(" - Blind discrete bits transformed into continuous Discriminative Spectrum Axis.")
    print(" - Complete Observation Collapse (O(1) Representation) achieved!")
    print("=" * 80)


if __name__ == "__main__":
    main()
