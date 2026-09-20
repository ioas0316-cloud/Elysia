#!/usr/bin/env python3
"""
Elysia Pure Sensory-Phase Core (Phase 1) & Cognitive Digital Twin Demonstration Script
"""

import time
import math
import torch
import elysia_phase_lock_cuda as eplc

def main():
    print("=========================================================================")
    print("  ELYASIA: Pure Sensory-Phase Core & Cognitive Digital Twin Demonstration")
    print("=========================================================================\n")

    # 1. Pipeline Initialization
    cfg = eplc.SensoryPhaseConfig()
    cfg.phi_solid = 0.8
    cfg.phi_liquid = 0.4
    cfg.tau_shear = 2.0
    pipeline = eplc.SensoryPhaseCorePipeline(cfg)

    print("[1] Initialized C++/CUDA Pure Sensory-Phase Pipeline")
    print(f"    Config -> phi_solid: {cfg.phi_solid}, phi_liquid: {cfg.phi_liquid}, tau_shear: {cfg.tau_shear}\n")

    # 2. Build 4 Distinct Sensory Stream Inputs S_i(t)
    inputs = []

    # Frame 0: High Motion-Audio Alignment -> Solid Crystallization
    s0 = eplc.SensoryStreamInput()
    s0.position_tension = eplc.Float4(0.0, 0.0, 0.0, 0.6)       # x, y, z, tau
    s0.velocity_dtension = eplc.Float4(1.5, 0.0, 0.0, 0.0)      # vx, vy, vz, dtau/dt
    s0.audio_spectrum = eplc.Float4(0.4, 0.8, 0.0, 120.0)       # AL, AM, AH, w0
    s0.acceleration_grad = eplc.Float4(1.5, 0.0, 0.0, 1.0)     # ax, ay, az, |grad AM|
    inputs.append(s0)

    # Frame 1: Medium Alignment -> Liquid Vortex Flow
    s1 = eplc.SensoryStreamInput()
    s1.position_tension = eplc.Float4(-1.0, 0.5, 0.0, 0.4)
    s1.velocity_dtension = eplc.Float4(0.5, 0.5, 0.0, 0.3)
    s1.audio_spectrum = eplc.Float4(0.3, 0.4, 0.2, 220.0)
    s1.acceleration_grad = eplc.Float4(0.2, 0.2, 0.0, 0.5)
    inputs.append(s1)

    # Frame 2: High Audio High-Frequency Noise -> Gas Random Floating
    s2 = eplc.SensoryStreamInput()
    s2.position_tension = eplc.Float4(1.0, -0.5, 0.0, 0.1)
    s2.velocity_dtension = eplc.Float4(0.1, -0.1, 0.0, 0.1)
    s2.audio_spectrum = eplc.Float4(0.1, 0.1, 1.5, 800.0)       # High AH
    s2.acceleration_grad = eplc.Float4(0.0, 0.0, 0.0, 0.0)
    inputs.append(s2)

    # Frame 3: Violent Tension Fluctuation -> Shear Breakdown
    s3 = eplc.SensoryStreamInput()
    s3.position_tension = eplc.Float4(0.5, 1.0, 0.0, 0.9)
    s3.velocity_dtension = eplc.Float4(0.0, 2.0, 0.0, 3.2)      # dtau/dt = 3.2 > tau_shear
    s3.audio_spectrum = eplc.Float4(0.5, 0.5, 0.5, 440.0)
    s3.acceleration_grad = eplc.Float4(0.0, 3.0, 0.0, 0.0)
    inputs.append(s3)

    print("[2] Processing 4 Sensory Stream Frames through Pure Sensory-Phase Core...")
    nodes, diagnostics = pipeline.process_frame_cpu(inputs)

    phase_names = {0: "Gas (Diffusion)", 1: "Liquid (Vortex)", 2: "Solid (Crystallization)", 3: "Shear (Breakdown)"}

    for i in range(len(inputs)):
        diag = diagnostics[i]
        node = nodes[i]
        state_id = diag.classification_st[0]
        state_str = phase_names.get(state_id, "Unknown")
        phase_val = node.metric_offdiag[3]
        gamma_val = node.metric_diag[3]
        gdi_val = diag.position_gdi[3]
        hess_det = diag.saddle_hessian[0]

        print(f"  Node [{i}]:")
        print(f"    Phase Coherence (Φ_i): {phase_val:.3f} | Time Dilation (γ_i): {gamma_val:.3f}")
        print(f"    GDI: {gdi_val:.3f} | Saddle Det(H): {hess_det:.3f}")
        print(f"    Auto Phase State: [{state_str}]\n")

    # 3. Game Mechanics & Riemannian Navigation Simulation
    print("[3] Simulating Riemannian NavMesh & Agent Phase Jump (Tunneling)...")
    agent_x = -0.2
    agent_v = 1.0
    agent_energy = 50.0

    print(f"    Agent Initial State -> Position X: {agent_x:.2f}, Velocity: {agent_v:.2f}, Energy: {agent_energy:.1f}")

    # Approach Separatrix wall at X=0
    wall_repulsion = 15.0 / (abs(agent_x) + 0.1)
    barrier_cost = 25.0

    if agent_energy >= barrier_cost:
        # Quantum Phase Tunneling
        agent_x += 0.8
        agent_energy -= barrier_cost
        print(f"    -> [Phase Jump Triggered] Quantum Tunneling across Separatrix Wall!")
        print(f"    Agent Post-Tunneling State -> Position X: {agent_x:.2f}, Energy Left: {agent_energy:.1f}\n")

    # 4. Cognitive Digital Twin Knowledge Injection
    print("[4] Demonstrating Cognitive Digital Twin Knowledge Spatial Anchors:")
    anchors = [
        ("Chemistry Workbench", "Molecular reaction energy -> Solid crystallization & GDI convergence"),
        ("Math Proof Scroll", "Unfolding manifold -> Geodesic highway smoothing"),
        ("Music Instrument", "Audio score -> Phase wave Separatrix resonance"),
        ("Literature Book", "Narrative theme -> Spacetime curvature climate shift")
    ]

    for name, desc in anchors:
        print(f"  Anchor [{name}]: {desc}")

    print("\n=========================================================================")
    print("  Sensory-Phase Core Demonstration Completed Successfully!")
    print("=========================================================================")

if __name__ == "__main__":
    main()
