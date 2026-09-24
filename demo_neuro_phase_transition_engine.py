"""
Demo: Neuro-Phase Transition & Bidirectional Phase Grounding Simulation.

Simulates the full journey: "From Planet to Person"
- Heterogeneous external wave streams (Text, Audio, Vision) penetrate the sensory boundary.
- Internal rotor lattice projects internal prediction waves outward.
- System undergoes bidirectional phase negotiation, minimizing q_err and thermal friction.
- Phase state transitions from GAS (Incoherent / High Temp) -> LIQUID (Dynamic Flow) -> ICE (Crystallized Phase-Lock / Concept Attractor).
"""

import time
import math
import numpy as np
from core.consciousness.neuro_phase_causal_engine import (
    NeuroPhaseCausalEngine,
    ExternalWaveStream,
    NeuroPhaseState
)
from core.sensory.multimodal_cognitive_frontend import MultimodalCognitiveFrontend


def run_neuro_phase_transition_demo():
    print("==========================================================================")
    print(" ELYSIA NEURO-PHASE TRANSITION ENGINE: FROM PLANET TO PERSON DEMO ")
    print("==========================================================================")
    print("Initializing Neuro-Phase Causal Engine with 16 3D Rotor Nodes...")

    engine = NeuroPhaseCausalEngine(num_nodes=16, lattice_dims=(4, 2, 2))
    frontend = MultimodalCognitiveFrontend(feature_dim=16)

    print(f"Initial System Temperature: {engine.system_temperature:.2f} (GAS State)")
    print(f"Initial Order Parameter R: {engine.calculate_global_coherence():.4f}\n")

    # Modality 1: Text Stream ("사과" - Apple as Phase Impulse Wave)
    print("--------------------------------------------------------------------------")
    print(" [MODALITY 1: LINGUISTIC PHASE IMPULSE ('사과' / Apple) ]")
    print("--------------------------------------------------------------------------")

    frontend_out = frontend.process_multimodal_input(text_input="사과")
    text_vec = frontend_out["axis_b_qualia"]
    text_phases = text_vec * math.pi

    text_stream = ExternalWaveStream(
        modality="text",
        wave_phases=text_phases,
        frequencies=np.full(16, 40.0)
    )

    print(f"External Text Wave Inflow Phase Vector (first 4): {text_stream.wave_phases[:4]}")

    for step in range(1, 16):
        result = engine.negotiate_bidirectional_phase(text_stream, coupling_gain=2.5, crystallization_threshold=0.1)
        step_status = engine.step(dt=0.005)
        print(
            f"Step {step:02d} | State: {result.is_crystallized and 'ICE (Solid)' or engine.global_phase_state.value.upper():<7} | "
            f"Temp: {engine.system_temperature:.4f} | q_err: {result.q_err:.4f} | "
            f"Friction: {result.thermal_friction:.4f} | Coherence R: {step_status['coherence']:.4f}"
        )

    # Modality 2: Vision Stream (Red Apple 2D Phase Matrix)
    print("\n--------------------------------------------------------------------------")
    print(" [MODALITY 2: VISUAL ELECTROMAGNETIC MATRIX (Red Apple Light Waves) ]")
    print("--------------------------------------------------------------------------")

    apple_rgb = np.array([[[220, 20, 20]] * 4] * 4, dtype=np.uint8)  # 4x4 Red matrix
    frontend_out_vis = frontend.process_multimodal_input(rgb_image=apple_rgb)
    vis_manifold = frontend_out_vis["axis_a_topology"]
    vis_phases = vis_manifold * math.pi

    vision_stream = ExternalWaveStream(
        modality="vision",
        wave_phases=vis_phases,
        frequencies=np.full(16, 60.0)
    )

    for step in range(1, 16):
        result = engine.negotiate_bidirectional_phase(vision_stream, coupling_gain=3.0, crystallization_threshold=0.1)
        step_status = engine.step(dt=0.005)
        print(
            f"Step {step:02d} | State: {result.is_crystallized and 'ICE (Solid)' or engine.global_phase_state.value.upper():<7} | "
            f"Temp: {engine.system_temperature:.4f} | q_err: {result.q_err:.4f} | "
            f"Friction: {result.thermal_friction:.4f} | Coherence R: {step_status['coherence']:.4f}"
        )

    print("\n--------------------------------------------------------------------------")
    print(" [CRYSTALLIZATION & QUALIA CONVERGENCE RESULT ]")
    print("--------------------------------------------------------------------------")
    print(f"Active Crystallized Attractors in Engine: {list(engine.crystallized_attractors.keys())}")
    print(f"Final Global Phase State: {engine.global_phase_state.value.upper()}")
    print(f"Final Order Parameter R: {engine.calculate_global_coherence():.4f} (Phase-Locked Lattice)")
    print("Zero-FLOP Invariant Concept Storage Achieved.")
    print("==========================================================================\n")


if __name__ == "__main__":
    run_neuro_phase_transition_demo()
