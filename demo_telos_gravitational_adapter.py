#!/usr/bin/env python3
"""
[Demo] Telos-Gravitational Boundary Adapter
Demonstrates:
1. Entry into a primary Telos execution stream (e.g. Real-Time Game / Elysia Core Loop).
2. Establishment of a Teleological Gravitational Center (Semantic Mass Ms & Curvature Kc).
3. Topological Gravitational Attenuation of incoming background OS noise, telemetry, and interrupts.
4. Zero-copy high-speed shortcut routing for core execution flow and SNR amplification.
"""

import numpy as np
from core.physics.telos_gravitational_adapter import (
    TelosGravitationalBoundaryAdapter,
    TelosSignal,
    ExternalNoiseStream
)


def run_demo():
    print("=" * 80)
    print("      ELYSIA: TELOS-GRAVITATIONAL BOUNDARY ADAPTER DEMO")
    print("=" * 80)

    adapter = TelosGravitationalBoundaryAdapter(dimensions=16, base_gravitational_constant=2.5)

    print("\n[Phase 1] Baseline Operating State (No Active Telos)")
    topo_base = adapter.evaluate_system_topology()
    print(f"  Telos Active: {topo_base['telos_active']}")
    print(f"  Baseline Semantic Mass: {topo_base['telos_semantic_mass']:.4f}")
    print(f"  Baseline Curvature (Kc): {topo_base['field_curvature']:.4f}")

    # Simulate incoming background OS noise before Telos activation
    raw_os_noise_1 = ExternalNoiseStream(
        stream_id="STREAM_WIN_TELEMETRY_01",
        source_type="Windows_Telemetry",
        signal_vector=np.array([0.1, 0.9, -0.4, 0.8] + [0.2] * 12, dtype=np.float32),
        amplitude=8.5
    )
    res_base_noise = adapter.filter_background_noise(raw_os_noise_1)
    print(f"\n  Incoming Background Noise (Pre-Telos): {raw_os_noise_1.source_type}")
    print(f"    - Original Amplitude: {res_base_noise['original_amplitude']:.2f}")
    print(f"    - Retained Amplitude: {res_base_noise['retained_amplitude']:.2f} (Attenuation: {res_base_noise['attenuation_factor'] * 100:.1f}%)")

    print("\n[Phase 2] Triggering Core Intention Stream (Telos Activation)")
    game_intent_vec = np.array([1.0, 0.1, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    game_telos_signal = TelosSignal(
        signal_id="TELOS_GAME_ENGINE_LOOP",
        intent_description="Real-Time Immersive Game Execution Core",
        telos_vector=game_intent_vec,
        priority_weight=3.0
    )

    activation_res = adapter.activate_telos(game_telos_signal)
    print(f"  >>> TELOS ACTIVATED: '{activation_res['intent']}'")
    print(f"  Teleological Semantic Mass (Ms): {activation_res['semantic_mass']:.4f}")
    print(f"  Gravitational Spacetime Curvature (Kc): {activation_res['field_curvature']:.4f}")

    print("\n[Phase 3] Background OS Noise & Telemetry Gravitational Attenuation")
    noise_streams = [
        ExternalNoiseStream("NOISE_OS_TELEMETRY", "OS_Telemetry_Upload", np.array([0.0, 1.0, 0.2, 0.5] + [0.1] * 12, dtype=np.float32), amplitude=12.0),
        ExternalNoiseStream("NOISE_POLLING_INTERRUPT", "Background_Hardware_Polling", np.array([-0.3, 0.8, 1.0, 0.0] + [-0.1] * 12, dtype=np.float32), amplitude=15.0),
        ExternalNoiseStream("NOISE_DISCORD_OVERLAY", "External_App_Overlay_Draw", np.array([0.5, 0.5, -0.8, 0.2] + [0.3] * 12, dtype=np.float32), amplitude=9.0),
    ]

    for stream in noise_streams:
        res = adapter.filter_background_noise(stream)
        print(f"  * Noise Source: {stream.source_type:<30} | Orig Amp: {res['original_amplitude']:5.1f} -> Retained Amp: {res['retained_amplitude']:5.2f} (Attenuated: {res['attenuation_factor'] * 100:5.1f}%)")

    print("\n[Phase 4] Primary Execution Stream Zero-Copy Shortcut & SNR Amplification")
    primary_frame_payload = {"frame_number": 6001, "causal_nodes_count": 1250, "dt": 0.016}
    route_res = adapter.route_primary_stream(raw_data=primary_frame_payload)
    print(f"  Route Pipeline Status: {route_res['route_status']}")
    print(f"  Signal-to-Noise Ratio (SNR) Gain Factor: {route_res['snr_gain_factor']:.2f}x")
    print(f"  Latency Overhead: {route_res['latency_overhead_ms']} ms")

    print("\n[Phase 5] Final System Topological Evaluation")
    topo_final = adapter.evaluate_system_topology()
    print(f"  Active Intent: {topo_final['active_intent']}")
    print(f"  Total Filtered External Noise Streams: {topo_final['total_filtered_streams']}")
    print(f"  Average System Noise Attenuation Rate: {topo_final['average_noise_attenuation'] * 100:.2f}%")

    print("\n" + "=" * 80)
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
