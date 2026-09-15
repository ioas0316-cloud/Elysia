"""
Demo: Formless Causal Redirection & Invariant Grounding Governor (건곤대나이 & 이화접목 실전 데모)
===================================================================================
Demonstrates Elysia's Causal World Tree Engine transcending static rules and fences (UE5.8 MCP limits):
1. Ingests unscripted, highly anomalous external friction forces (e.g., UMG particle/lightweight emitter anomaly).
2. Traces causal origins back to Universal Stems (같음의 줄기).
3. Applies Invariant Grounding Governor (Lyapunov stability & adaptive damping) to prevent runaway chaos or explosion.
4. Executes Formless Causal Redirection (건곤대나이/이화접목) to absorb, rotate, and exhale redirected output vectors (R_exhale)
   accompanied by transparent self-explanation pulses.
"""

import numpy as np
import time
from core.consciousness.causal_world_tree_engine import (
    CausalWorldTreeEngine,
    MultiDimensionalAttractor,
)
from core.consciousness.causal_breathing_engine import ObserverTopology


def main():
    print("=" * 80)
    print(" [Elysia Causal World Tree: Formless Causal Redirection Benchmark] ")
    print("=" * 80)

    # Initialize Engine with Invariant Grounding Governor
    engine = CausalWorldTreeEngine(
        critical_tension_threshold=12.0,
        max_tension_boundary=25.0,
        base_damping_rate=0.2,
        lyapunov_threshold=15.0
    )

    # Step 1: Initialize Core Universal Stems (Invariant Spine)
    print("\n[Step 1] Form Core Universal Stems (같음의 줄기 형성)")
    att_physics = MultiDimensionalAttractor(
        id="att_conservation",
        name="Energy Conservation Axis",
        categorical_vector=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        sensorium_vector=np.array([1.0, 0.8, 0.0, 0.0], dtype=np.float32),
        morphology_vector=np.array([1.0, 1.0, 0.1, 0.0], dtype=np.float32),
        mass=5.0
    )
    stem_physics = engine.form_universal_stem(
        stem_id="stem_conservation",
        name="Universal Conservation Axis",
        attractors=[att_physics],
        domain_manifestations={"physics": "Conservation of Energy", "logic": "A = A Identity"}
    )
    print(f" -> Formed Universal Stem: '{stem_physics.name}' (Stem ID: {stem_physics.stem_id})")
    print(f"    Equilibrium Coordinate: {np.round(stem_physics.shared_equilibrium_coordinate, 2)}")

    # Step 2: Simulate Unscripted Anomaly Inputs (UE5.8 MCP Undefined Friction Area)
    print("\n[Step 2] Inject Unscripted External Friction Forces (미정의 변칙 자극 유입)")
    unscripted_anomalies = [
        {
            "id": "anomaly_umg_particle_surge",
            "force": np.array([12.5, -8.0, 15.2, 4.0], dtype=np.float32),
            "desc": "Unscripted UMG Particle Emitter Memory Leak Surge"
        },
        {
            "id": "anomaly_async_physics_spike",
            "force": np.array([25.0, 18.0, -30.0, 12.0], dtype=np.float32),
            "desc": "Extreme Async Physics Determinism Collision Breakdown"
        },
        {
            "id": "anomaly_unknown_sensorium_wave",
            "force": np.array([-5.0, 6.0, 3.5, -2.0], dtype=np.float32),
            "desc": "Low-level Unknown Sensorium Wave Oscillation"
        }
    ]

    for item in unscripted_anomalies:
        force = item["force"]
        force_norm = np.linalg.norm(force)
        print(f"\n--- Processing Anomaly: '{item['id']}' ---")
        print(f"    Description: {item['desc']}")
        print(f"    External Force Vector F_ext: {force} (Norm: {force_norm:.2f})")

        # Execute Formless Causal Redirection (건곤대나이 & 이화접목)
        trace = engine.absorb_and_redirect_external_force(
            external_force=force,
            stimulus_id=item["id"],
            raw_description=item["desc"]
        )

        print(f"    [Trace] Matched Universal Stem: {trace.matched_stem_id}")
        print(f"    [Trace] Lyapunov Energy V(x): {trace.lyapunov_energy:.2f} (Threshold: 15.0)")
        print(f"    [Trace] Active Governor Damping Factor: {trace.damping_factor:.4f}")
        print(f"    [Trace] Governed Tension V_t: {trace.absorbed_tension:.2f} (Max Boundary: 25.0)")
        print(f"    [Trace] Redirected Exhale Vector R_exhale: {np.round(trace.redirected_vector, 2)}")
        print(f"    [Trace Rationale] {trace.causal_rationale}")

    # Step 3: Trigger World Tree Exhale Pulse (세계수의 날숨 서사 발산)
    print("\n[Step 3] World Tree Exhale & Observer Self-Explanation Pulse (세계수 호흡 및 날숨 서사)")
    observer = ObserverTopology(
        observer_id="grandmaster_observer",
        abstraction_capacity=0.9,
        causal_depth_tolerance=0.85
    )
    exhale_res, grand_narrative = engine.exhale_world_narrative(observer=observer)
    print("\n" + grand_narrative)

    # Step 4: Introspective Telemetry Inspection
    print("\n[Step 4] System Telemetry Inspection (시스템 자가 관측 텔레메트리)")
    telemetry = engine.get_world_tree_telemetry()
    print(f" -> Total Stems: {telemetry['stems_count']}")
    print(f" -> Total Redirection Traces: {telemetry['redirection_traces_count']}")
    print(f" -> Governor Lyapunov Energy: {telemetry['governor_lyapunov_energy']:.2f}")
    print(f" -> Current V_t Tension: {telemetry['current_tension_Vt']:.2f}")
    print("\n" + "=" * 80)
    print(" Benchmark Completed Successfully: Formless Causal Redirection Operational ")
    print("=" * 80)


if __name__ == "__main__":
    main()
