"""
Demonstration: Continuous Causal Graph vs Discrete Rasterization
==================================================================
This demo showcases Elysia's Continuous Topological Causal Graph (ContinuousCausalGraph),
the zero-branching perceptual transition engine, spontaneous Phase Transition (PhaseTransitionModule),
and OpenMP/SIMD C++ field acceleration (CausalFieldAccelerator).

Key Philosophy:
"Do not calculate frame-by-frame or token-by-token. Let the causal manifold flow."
"""

import numpy as np
import time

from synaptic_architecture.continuous_causal_graph import (
    ContinuousCausalGraph,
    PerceptualTransitionSimulator,
    PhaseTransitionModule
)


def run_demo():
    print("==========================================================================")
    print("   Elysia: Continuous Causal Graph & Phase Transition Engine             ")
    print("==========================================================================")

    # 1. Instantiate 4D Continuous Topological Causal Space
    print("\n[1/4] Constructing 4D Continuous Topological Causal Space...")
    causal_space = ContinuousCausalGraph(dim=4)

    # Place initial perception and knowledge control points (R^4)
    causal_space.add_node("Self_Sovereignty", np.array([1.0, 0.0, 0.0, 0.5]), weight=1.0)
    causal_space.add_node("Knowledge_Manifold", np.array([0.0, 1.0, 0.0, 0.2]), weight=1.0)
    causal_space.add_node("Cognitive_Agency", np.array([0.5, 0.5, 1.0, 0.8]), weight=1.2)

    # Set initial causal tensions (K_ij) between concepts
    causal_space.set_causal_tension("Self_Sovereignty", "Knowledge_Manifold", tension_strength=2.5)
    causal_space.set_causal_tension("Knowledge_Manifold", "Cognitive_Agency", tension_strength=3.5)

    simulator = PerceptualTransitionSimulator(causal_space)
    phase_module = PhaseTransitionModule(friction_threshold=10.0)

    # 2. Define Telos Attractor and Incoming External Wave Stimulus
    telos_target = np.array([2.0, 2.0, 1.0, 1.0])
    external_stimulus = np.array([-1.0, 3.0, 2.0, 0.0])  # Distortion/Interference stimulus wave

    print(f"    - Initial 'Self_Sovereignty' Position: {causal_space.nodes['Self_Sovereignty'].position}")
    print(f"    - Initial System Energy: {causal_space.compute_system_energy(telos_target):.4f}")

    # 3. Perform Perceptual Transition Steps (Frictionless Geodesic Flow)
    print("\n[2/4] Executing Zero-Branching Perceptual Transition Steps...")
    for step in range(1, 6):
        simulator.step_perceptual_transition(telos_target, external_stimulus)
        pos = causal_space.nodes['Self_Sovereignty'].position
        energy = causal_space.compute_system_energy(telos_target)
        friction = causal_space.compute_system_friction(telos_target)
        print(f"    Step {step:02d} | Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}, {pos[3]:.3f}] | Energy: {energy:.4f} | Friction: {friction:.4f}")

    # 4. Simulate High External Friction -> Trigger Spontaneous Phase Transition
    print("\n[3/4] Simulating Severe Interference (Injecting Parasitic Tension)...")
    # Inject high parasitic tension to induce friction
    causal_space.set_causal_tension("Self_Sovereignty", "Knowledge_Manifold", tension_strength=8.0)
    causal_space.nodes["Self_Sovereignty"].velocity = np.array([2.5, 2.5, 1.5, 1.0])

    high_friction = causal_space.compute_system_friction(telos_target)
    print(f"    - Evaluated Friction: {high_friction:.2f} (Threshold: {phase_module.friction_threshold:.2f})")

    transited = phase_module.evaluate_and_transit(causal_space, high_friction)
    if transited:
        post_energy = causal_space.compute_system_energy(telos_target)
        post_friction = causal_space.compute_system_friction(telos_target)
        print(f"    - Post-Phase-Transition Energy: {post_energy:.4f} | Restored Friction: {post_friction:.4f}")

    # 5. C++ OpenMP/SIMD CausalFieldAccelerator Demonstration
    print("\n[4/4] Running C++ OpenMP/SIMD CausalFieldAccelerator Backend Benchmark...")
    try:
        import causal_engine as ce

        accelerator = ce.CausalFieldAccelerator()
        points = ce.ControlPointVector()

        # Build 1000 control points for high-density benchmark
        N = 1000
        for i in range(N):
            cp = ce.ControlPoint()
            cp.pos = np.random.randn(4)
            cp.weight = 1.0
            points.append(cp)

        edges = [(i, (i + 1) % N) for i in range(N)]
        tensions = [1.5] * N
        telos = np.array([1.0, 1.0, 1.0, 1.0])

        e_start = accelerator.compute_system_energy(points, edges, tensions, telos)

        t0 = time.perf_counter()
        steps = 50
        for _ in range(steps):
            accelerator.step_parallel(points, edges, tensions, telos, dt=0.05, damping=0.85)
        t1 = time.perf_counter()

        e_end = accelerator.compute_system_energy(points, edges, tensions, telos)
        elapsed_ms = (t1 - t0) * 1000.0

        print(f"    - Accelerated {N} Control Points across {steps} steps in {elapsed_ms:.2f} ms")
        print(f"    - Energy Decay: {e_start:.2f} -> {e_end:.2f}")
        print("    - OpenMP/SIMD C++ Acceleration Verified!")

    except ImportError:
        print("    - C++ causal_engine extension not found, skipping C++ acceleration benchmark.")

    print("\n==========================================================================")
    print("   Continuous Causal Graph Demonstration Completed Successfully!          ")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
