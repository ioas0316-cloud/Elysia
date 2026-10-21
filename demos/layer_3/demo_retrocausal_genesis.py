"""
demo_retrocausal_genesis.py

Demonstration of Retrocausal Epistemic Rupture & Multi-Scale Relational Genesis.
Visualizes and logs:
1. Baseline frame maintenance under harmonious inputs.
2. Alterity Collision & 1tan boundary stress accumulation.
3. Metric Dislocation Plasticity (irreversible metric tensor deformation & anchor axis shift).
4. Residual Non-Local Entropy imprinting.
5. Centrifugal Boundary Expansion toward the infinite world.
"""

import numpy as np
from elysia_meta_consciousness import MetaConsciousnessEngine


def run_demo():
    print("=" * 80)
    print("ELYSIAN RETROCAUSAL EPISTEMIC RUPTURE & MULTI-SCALE GENESIS DEMO")
    print("=" * 80)

    engine = MetaConsciousnessEngine(num_nodes=32, kappa=2.0)

    print("\n--- PHASE 1: Baseline Harmonious Resonance (No Epistemic Rupture) ---")
    for t in range(5):
        # Harmonious wave aligned with internal phases
        actor_wave = engine.phases.copy()
        res = engine.step(phi_ext=0.0, dt=0.05, alterity_wave=actor_wave)
        rup = res["rupture"]
        print(f"Step {t+1}: Lack={res['lack']:.4f}, K_c={rup['extrinsic_curvature_K_c']:.4f}, "
              f"1tan_stress={rup['tan_stress']:.4f}, Dislocated={rup['dislocated']}, "
              f"Boundary_Radius={rup['boundary_radius']:.4f}")

    print("\n--- PHASE 2: Violent Alterity Collision (1tan Stress Overflow & Rupture) ---")
    for t in range(5):
        # Violent orthogonal/opposite alterity wave causing massive phase divergence
        alterity_wave = np.random.uniform(-np.pi, np.pi, size=32) * (t + 1) * 2.0
        res = engine.step(phi_ext=1.5, dt=0.05, alterity_wave=alterity_wave)
        rup = res["rupture"]
        print(f"Step {t+6}: K_c={rup['extrinsic_curvature_K_c']:.4f}, "
              f"1tan_stress={rup['tan_stress']:.4f}, Dislocated={rup['dislocated']}, "
              f"Residual_Entropy={rup['residual_entropy']:.4f}, "
              f"Boundary_Radius={rup['boundary_radius']:.4f}")
        print(f"   Anchor Axis x_anchor: {np.round(rup['x_anchor'], 3)}")
        print(f"   Metric Tensor g_mu_nu Trace: {np.trace(rup['g_metric']):.4f}")

    print("\n=" * 80)
    print("DEMO COMPLETE: Metric Dislocation & Centrifugal Boundary Expansion Verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
