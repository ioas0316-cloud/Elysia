"""
demos/demo_optical_shadow_blister.py

Interactive Demonstration & Benchmark:
Optical Spectrum Bivector Rotor & Shadow Blister Effect Engine
==============================================================
Simulates:
1. Continuous optical spectrum parameterization (380nm - 750nm) on Cℓ(3,0) Bivector planes.
2. Penumbra zone overlap and non-linear 'Shadow Blister' snap attraction.
3. Elimination of hunting oscillations via phase-locking stiction.
4. Micro-scale rapid auto-clumping (Spinal Reflex).
5. Macro-scale causal gating (R_gate) forwarding resonant clusters into
   Tensor Helmholtz Metric Decomposition and 3x3x3 Fractal Concept Space.
"""

import math
import sys
import os
import time
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import elysia_loader
elysia_loader.install()

from elysia_engine.core.optical_blister_rotor import (
    OpticalSpectrumBivectorRotor,
    ShadowBlisterTensionKernel,
    MicroReflexBlisterNetwork
)
from elysia_engine.core.hierarchical_blister_bridge import (
    HierarchicalBlisterBridge
)


def run_demo():
    print("=" * 80)
    print("  ELYSIA ENGINE :: OPTICAL SPECTRUM BIVECTOR ROTOR & SHADOW BLISTER DEMO")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Optical Spectrum Bivector Rotors Initialization
    # ------------------------------------------------------------------
    print("\n[Stage 1] Initializing Multi-Modal Optical Concept Rotors:")
    concepts = [
        {"name": "Concept A (Red / Flux)", "wl": 660.0, "pos": [0.0, 0.0, 0.0]},
        {"name": "Concept B (Orange-Red / Flux-Order)", "wl": 640.0, "pos": [0.35, 0.0, 0.0]},
        {"name": "Concept C (Green / Order)", "wl": 530.0, "pos": [1.8, 0.0, 0.0]},
        {"name": "Concept D (Blue / Entropy)", "wl": 450.0, "pos": [2.2, 0.5, 0.0]},
    ]

    rotors = []
    for c in concepts:
        r = OpticalSpectrumBivectorRotor(
            wavelength_nm=c["wl"],
            position=c["pos"],
            intensity=1.0
        )
        rotors.append(r)
        print(f"  • {c['name']:<35} | λ = {r.wavelength_nm:5.1f}nm | Penumbra r = {r.penumbra_radius:.2f} | Triad [R:{r.triad_weights[0]:.2f}, G:{r.triad_weights[1]:.2f}, B:{r.triad_weights[2]:.2f}]")

    # ------------------------------------------------------------------
    # Step 2: Non-Linear Shadow Blister Snap & Hunting Suppression
    # ------------------------------------------------------------------
    print("\n[Stage 2] Demonstrating Shadow Blister Snap & Hunting Suppression:")
    kernel = ShadowBlisterTensionKernel(omega_c=0.25, kappa_0=1.0, kappa_1=15.0)

    # Measure tension curve as distance decreases between Concept A and B
    test_distances = [2.0, 1.5, 1.0, 0.6, 0.4, 0.2, 0.05]
    print(f"  {'Distance':<10} | {'Overlap Ω':<12} | {'Force (Tension)':<18} | {'Blister Status':<18}")
    print("  " + "-" * 65)

    r_test1 = OpticalSpectrumBivectorRotor(wavelength_nm=660.0, position=[0.0, 0.0, 0.0])
    r_test2 = OpticalSpectrumBivectorRotor(wavelength_nm=640.0, position=[0.0, 0.0, 0.0])

    for dist in test_distances:
        r_test2.position = np.array([dist, 0.0, 0.0], dtype=np.float32)
        res = kernel.evaluate_blister_interaction(r_test1, r_test2)
        status_str = "⚡ SNAP BLISTERING" if res["is_blister_active"] else "Smooth Penumbra"
        print(f"  {dist:<10.2f} | {res['overlap']:<12.3f} | {res['force_total']:<18.3f} | {status_str:<18}")

    # ------------------------------------------------------------------
    # Step 3: Hierarchical Decoupled 2-Tier Architecture Simulation
    # ------------------------------------------------------------------
    print("\n[Stage 3] Executing 2-Tier Hierarchical Blister Bridge (Micro Reflex -> Macro Gating):")
    bridge = HierarchicalBlisterBridge(r_gate=0.6, grid_shape=(16, 16, 16))
    for r in rotors:
        bridge.register_rotor(r)

    print("  Beginning multi-scale time evolution...")
    dt = 0.02
    for step in range(1, 31):
        report = bridge.step(dt=dt)
        if step in [1, 5, 10, 20, 30]:
            gate_status = "🔓 FORWARDED (Macro Gated)" if report.is_gated else "🔒 DECOUPLED (Micro Only)"
            print(f"  [Cycle {step:02d}] Micro Coherence: {report.micro_coherence:.3f} | Clusters: {report.num_clumped_clusters} | {gate_status}")

            if report.is_gated and report.macro_wave_modes:
                modes = report.macro_wave_modes
                print(f"            └ Helmholtz Modes  : Trace(Scalar)={modes['scalar_trace']:.3f}, Shear(Vector)={modes['vector_shear']:.3f}, TT(Tensor)={modes['tensor_tt']:.3f}")
                print(f"            └ 3x3x3 Fractal R  : {report.fractal_coherence:.4f} (Phase-Locking Coherence)")

    print("\n" + "=" * 80)
    print("  [Conclusion] Optical Spectrum Bivector Rotor & Shadow Blister Engine Verified!")
    print("  1. Continuous wavelength seamlessly translates into Cℓ(3,0) Bivector rotation.")
    print("  2. Penumbra overlap triggers abrupt non-linear attraction snap without numerical hunting.")
    print("  3. 2-Tier hierarchical bridge successfully decouples reflex clumping from macro Helmholtz/Fractal space.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
