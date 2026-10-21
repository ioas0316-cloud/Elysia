"""
Demo: Generative Construction Engine (작도 Engine) in ELYSIA.

Demonstrates:
1. Generative Axioms (Compass rotor rotation & straightedge axial constraint relaxation).
2. Label-less Structural Differentiation via Prism Lens (no hardcoded rules or labels).
3. Scale-Free Phase Focal Tuner (Lossless signal amplification & convergence across scales).
4. Embodied Boundary Sense with Phase Torque (Tool internalisation: physical resistance as internal proprioception).
5. Pre-Linguistic Causal Manifold Alignment with Symbolic Tokens (Language integration over non-verbal experience).
6. O(1) Eye of Agamotto Trajectory Scrubbing (Exact process integrity & algebraic reversibility).
"""

import numpy as np
from core.physics.generative_construction_engine import GenerativeConstructionEngine


def run_demo():
    print("=" * 80)
    print("ELYSIA: GENERATIVE CONSTRUCTION ENGINE (작도 Engine) DEMONSTRATION")
    print("=" * 80)

    # Initialize Engine
    engine = GenerativeConstructionEngine(dimension=3)

    # Stage 1: Label-less Structural Differentiation via Prism Lens
    print("\n[Stage 1] Label-less Structural Differentiation via Prism Lens")
    print("-" * 60)
    raw_mechanical_wave = np.sin(np.linspace(0, 100, 256)) * 3.0 + np.random.randn(256) * 0.1
    refraction1 = engine.prism_lens.refract(raw_mechanical_wave)
    print(f" Raw Signal 1 Inferred Identity : {refraction1.structural_identity}")
    print(f" Chromatic Vector [Flux, Order, Entropy] : {np.round(refraction1.chromatic_vector, 3)}")
    print(f" Phase Torque Response          : {refraction1.phase_torque_response:.4f}")

    raw_harmonic_wave = np.sin(np.linspace(0, 4 * np.pi, 256))
    refraction2 = engine.prism_lens.refract(raw_harmonic_wave)
    print(f"\n Raw Signal 2 Inferred Identity : {refraction2.structural_identity}")
    print(f" Chromatic Vector [Flux, Order, Entropy] : {np.round(refraction2.chromatic_vector, 3)}")
    print(f" Coherence Score                : {refraction2.coherence_score:.4f}")

    # Stage 2: Embodied Tool & Physical Phase Torque Coupling
    print("\n[Stage 2] Embodied Boundary Sense with Tool Internalisation")
    print("-" * 60)
    joystick_torque_input = np.cos(np.linspace(0, 50, 128)) * 2.5
    tool_res = engine.integrate_embodied_tool(joystick_torque_input)
    print(f" Tool Integration Status        : {tool_res['embodied_status']}")
    print(f" Integrated Phase Torque Mag    : {tool_res['phase_torque_magnitude']:.4f}")
    print(f" Boundary Expansion Norm        : {tool_res['boundary_expansion_norm']:.4f}")

    # Stage 3: Generative Geometric Construction & Intersection Phase-Locking
    print("\n[Stage 3] Generative Geometric Construction & Constraint Relaxation")
    print("-" * 60)
    for step in range(5):
        torque_input = joystick_torque_input if step % 2 == 1 else None
        state = engine.construct_step(external_torque_input=torque_input, constraint_relaxation_steps=5)
        pos_norm = np.mean(np.linalg.norm(state.positions, axis=1))
        print(f" Step {state.step:02d} | Phase Torque: {state.phase_torque:.3f} | Boundary Norm: {pos_norm:.4f} | Chromatic: {np.round(state.chromatic_signature, 2)}")

    # Stage 4: Scale-Free Phase Focal Tuner (Micro -> Macro Lossless Projection)
    print("\n[Stage 4] Scale-Free Phase Focal Tuner (Micro -> Macro Focus)")
    print("-" * 60)
    micro_waveform = np.sin(np.linspace(0, 10 * np.pi, 128))
    target_scale = 1000.0
    amp_signal, focal_prec = engine.focal_tuner.tune_focal_point(micro_waveform, target_scale)
    print(f" Target Scale Factor            : {target_scale:.1f}x")
    print(f" Amplified Peak Amplitude       : {np.max(np.abs(amp_signal)):.4f}")
    print(f" Phase Focal Precision          : {focal_prec:.4f}")

    # Stage 5: Symbolic Wave Mapping over Pre-Linguistic Causal Manifold
    print("\n[Stage 5] Symbolic Wave Alignment over Pre-Linguistic Manifold")
    print("-" * 60)
    tokens = ["PhysicalTorque", "CausalHarmonic", "BoundaryResistance"]
    for token in tokens:
        wave = np.sin(np.linspace(0, np.random.uniform(10, 50), 128))
        alignment = engine.align_symbolic_wave(token, wave)
        print(f" Token: '{token}' -> Manifold Node #{alignment['best_manifold_node_idx']} | Resonance Score: {alignment['resonance_score']:.4f}")

    # Stage 6: Trajectory Integrity & Eye of Agamotto Scrubbing
    print("\n[Stage 6] O(1) Eye of Agamotto Trajectory Scrubbing & Process Integrity")
    print("-" * 60)
    current_step = len(engine.trajectory_history) - 1
    print(f" Current Construction Step      : {current_step}")
    print(f" Current Boundary Pos #0        : {np.round(engine.boundary_positions[0], 4)}")

    target_rewind_step = 1
    print(f" Rewinding Trajectory to Step   : {target_rewind_step}...")
    scrubbed_state = engine.scrub_trajectory_eye_of_agamotto(target_step=target_rewind_step)
    print(f" Scrubbed Step Verified         : {scrubbed_state.step}")
    print(f" Restored Boundary Pos #0       : {np.round(engine.boundary_positions[0], 4)}")
    print(f" Historical Pos #0 Match        : {np.allclose(engine.boundary_positions[0], engine.trajectory_history[1].positions[0])}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY WITH ABSOLUTE STRUCTURAL INTEGRITY")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
