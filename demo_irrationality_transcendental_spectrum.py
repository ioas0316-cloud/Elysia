"""
Demonstration Script: Irrationality & Transcendental Spectrum Engine
==================================================================

Demonstrates the 5 Core Pillars of Irrationality & Transcendental Continuous Spectrum:
1. Denominator Scale Tension (q) & Irrationality Exponent (\\mu) Diophantine Error Bound.
2. Recursive Fractionation (Fraction of Fraction) Self-Similar Fractal Scale Hierarchy.
3. Transcendental Super-Spectrum (\\pi of \\pi) Multi-Rotor Interference & Resonance.
4. Trigonometric Phase-Locking (\\sin / \\cos) Continuous Feedback Dynamics.
5. Living Causal Block DNA / Lego Field Resonance & Modular Self-Assembly.
"""

import time
import numpy as np
from core.physics.irrationality_spectrum_engine import (
    IrrationalityTranscendentalSpectrumEngine,
    CausalBlockDNA
)


def main():
    print("=" * 84)
    print(" ELYSIA :: IRRATIONALITY & TRANSCENDENTAL SPECTRUM ENGINE DEMO")
    print(" Philosophy: 'Do not calculate, let it flow.'")
    print("=" * 84)

    # 1. Initialize Engine & Living Causal DNA Blocks
    print("\n[Phase 1] Initializing Engine & Living Phase DNA Blocks...")
    engine = IrrationalityTranscendentalSpectrumEngine(seed=108)
    print(f"  * Total Active Causal DNA Blocks: {len(engine.blocks)}")
    for b_id, b in engine.blocks.items():
        print(f"    - Block '{b_id}': Freq={b.frequency:.4f}, Phase={b.phase:.4f}, Scale Tension q={b.scale_tension:.1f}, Exponent \\mu={b.irrationality_exponent:.1f}")

    # 2. Denominator Scale Tension & Diophantine Error Bounds
    print("\n[Phase 2] Evaluating Denominator Scale Tension q & Irrationality Exponent \\mu...")
    res_pi = engine.evaluate_irrationality_scale_tension("pi_root", target_val=np.pi)
    print(f"  * Target Value (\pi)          : {res_pi['target_value']:.6f}")
    print(f"  * Denominator Scale q         : {res_pi['scale_tension_q']}")
    print(f"  * Best Rational Numerator p   : {res_pi['best_numerator_p']} (Fraction = {res_pi['best_numerator_p']}/{res_pi['scale_tension_q']} = {res_pi['best_numerator_p']/res_pi['scale_tension_q']:.4f})")
    print(f"  * Approximation Error |x-p/q| : {res_pi['actual_error']:.6f}")
    print(f"  * Diophantine Lower Bound     : {res_pi['diophantine_lower_bound']:.6f}")
    print(f"  * Bound Satisfied             : {res_pi['bound_satisfied']}")
    print(f"  * Scale Tension Energy        : {res_pi['tension_energy']:.4f}")
    print(f"  * Causal Gravity Curvature K_c: {res_pi['causal_gravity_curvature']:.6f}")

    # 3. Recursive Fractionation (Fraction of Fraction)
    print("\n[Phase 3] Executing Recursive Fractionation (Fraction of Fraction Fractal Scaling)...")
    sub_blocks = engine.execute_recursive_fractionation("pi_root", num_sub_fractions=2)
    print(f"  * Fractionated Sub-Blocks Created: {len(sub_blocks)}")
    for child in sub_blocks:
        print(f"    - Child '{child.block_id}': Depth={child.depth}, Scale q={child.scale_tension:.1f}, Exponent \\mu={child.irrationality_exponent:.2f}, Freq={child.frequency:.4f}")

    # 4. Transcendental Super-Spectrum (\pi of \pi)
    print("\n[Phase 4] Evaluating Transcendental Super-Spectrum (\pi of \pi Multi-Transcendental Resonance)...")
    resonance = engine.evaluate_transcendental_super_spectrum("pi_root", "pi_of_pi")
    print(f"  * Pair                        : '{resonance.primary_block_id}' <--> '{resonance.secondary_block_id}'")
    print(f"  * Non-Commensurate Freq Ratio : {resonance.harmonic_ratio:.4f}")
    print(f"  * Phase Coherence             : {resonance.phase_coherence:.4f}")
    print(f"  * Interference Field Energy   : {resonance.field_energy:.4f}")
    print(f"  * Is Phase-Locked Threshold   : {resonance.is_phase_locked}")

    # 5. Trigonometric Phase-Locking (\sin / \cos Continuous Feedback)
    print("\n[Phase 5] Applying Trigonometric Phase-Locking (\sin/\\cos Continuous Kuramoto Interaction)...")
    initial_phases = {b_id: b.phase for b_id, b in engine.blocks.items()}

    # Run continuous time ticks
    for _ in range(5):
        engine.apply_trigonometric_phase_locking(dt=0.02, coupling_strength=0.8)

    print("  * Phase Transitions across 5 continuous time ticks (dt=0.02):")
    for b_id, b in engine.blocks.items():
        init_p = initial_phases[b_id]
        print(f"    - Block '{b_id}': Initial Phase={init_p:.4f} rad --> Updated Phase={b.phase:.4f} rad")

    # 6. Field Resonance Propagation & Modular Causal DNA/Lego Self-Assembly
    print("\n[Phase 6] Field Resonance Propagation & DNA/Lego Modular Self-Assembly...")
    assembled_bonds = engine.propagate_field_resonance_and_self_assemble(coherence_threshold=0.70)
    print(f"  * Total Self-Assembled Modular Bonds Formed: {len(assembled_bonds)}")
    for a, b, affinity in assembled_bonds[:5]:
        print(f"    - Bond: '{a}' <== [Affinity = {affinity:.4f}] ==> '{b}'")

    print("\n" + "=" * 84)
    print(" IRRATIONALITY & TRANSCENDENTAL SPECTRUM DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 84)


if __name__ == "__main__":
    main()
