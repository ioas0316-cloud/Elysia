#!/usr/bin/env python3
"""
Demo: Unified System Architecture (5-Stage Wavefield Computing Paradigm)
========================================================================
Demonstrates the practical assembly of the 5-stage wavefield architecture:
1. Scalar Bit -> Complex Phasor Field (A * e^(i*phi))
2. Prism Bifurcation Operator (N-Phase Spectral Projection & Causal Tagging)
3. Multi-Layered Wave Memory & Dynamic Macro-Rotor Stack (Dynamic Chunking)
4. Retrocausal Execution Loop & Phase Error Self-Correction (Moire Pattern Overlap)
5. Recursive Fractal Scaling & Wave Collapse Convergence
"""

import numpy as np
from elysia_core.unified_system_architecture import (
    ComplexPhasor,
    PhasorField,
    PrismBifurcationOperator,
    DynamicMacroRotorStack,
    RetrocausalExecutionLoop,
    FractalResonanceUnit,
    UnifiedSystemArchitecture,
)


def main():
    print("\n" + "=" * 80)
    print(" PROJECT ELYSIA :: UNIFIED SYSTEM ARCHITECTURE (5-STAGE WAVEFIELD DEMO) ")
    print("=" * 80 + "\n")

    # Initialize 5-stage orchestrator
    usa = UnifiedSystemArchitecture(num_prism_phases=3, fractal_depth=2)

    # -------------------------------------------------------------------------
    # STAGE 1: Basic Computational Unit Redefinition
    # -------------------------------------------------------------------------
    print(">>> STAGE 1: Redefining Computational Unit (Scalar Bit -> Complex Phasor Field)")
    shape = (4, 4)
    root_field = usa.initialize_root_field(shape=shape, random_seed=42)
    s1_coherence = root_field.calculate_coherence()
    print(f"  * Created Root Phasor Field with shape {shape}")
    print(f"  * Sample Mini Rotor at (0,0): {ComplexPhasor.from_complex(root_field.field[0,0], root_field.frequencies[0,0])}")
    print(f"  * Global Phase Coherence (Order Parameter R): {s1_coherence:.4f}\n")

    # -------------------------------------------------------------------------
    # STAGE 2: Prism Bifurcation Operator
    # -------------------------------------------------------------------------
    print(">>> STAGE 2: Prism Bifurcation Operator (N-Phase Spectral Projection)")
    refracted_branches = usa.stage2_prism.refract(root_field, root_id="root_cortex")
    print(f"  * Refracted into {len(refracted_branches)} sub-spectra with causal lineage tags:")
    for branch_field, tag in refracted_branches:
        b_coherence = branch_field.calculate_coherence()
        print(f"    - Branch {tag.branch_index}: Phase Shift={tag.phase_shift:.3f} rad | Passed Threshold={tag.threshold_passed} | Coherence={b_coherence:.4f}")
    print()

    # -------------------------------------------------------------------------
    # STAGE 3: Multi-Layered Wave Memory & Dynamic Macro-Rotor Stack
    # -------------------------------------------------------------------------
    print(">>> STAGE 3: Multi-Layered Wave Memory & Dynamic Macro-Rotor Stack")
    memory_stack = usa.stage3_memory
    promoted_rotors = []
    for branch_field, _ in refracted_branches:
        chunked = memory_stack.process_and_chunk(branch_field)
        promoted_rotors.extend(chunked)

    print(f"  * Dynamic Chunking Completed. Promoted Macro-Rotors: {len(promoted_rotors)}")
    for lvl, rotors in memory_stack.layers.items():
        if rotors:
            print(f"    - Layer {lvl}: {len(rotors)} Macro-Rotors bound. Example: {rotors[0]}")
    print()

    # -------------------------------------------------------------------------
    # STAGE 4: Retrocausal Execution Loop & Phase Error Self-Correction
    # -------------------------------------------------------------------------
    print(">>> STAGE 4: Retrocausal Execution Loop & Phase Error Self-Correction")
    retro_loop = usa.stage4_retro
    rot_transform = RetrocausalExecutionLoop.create_phase_rotation_transform(np.pi / 6)
    scale_transform = RetrocausalExecutionLoop.create_amplitude_scaling_transform(1.8)
    transforms = [rot_transform, scale_transform]

    transformed = rot_transform.apply_forward(root_field)
    final_field = scale_transform.apply_forward(transformed)

    valid, entropy_loss, recovered = retro_loop.retrocausal_trace(root_field, final_field, transforms)
    print(f"  * Retrocausal Back-Trace Validated: {valid}")
    print(f"  * Entropy Loss Delta S: {entropy_loss:.8e} (approx 0)")

    # Moire Pattern Geometric Phase Self-Correction
    noise_phase = np.random.normal(0, 0.35, size=shape)
    distorted_field = root_field.rotate(noise_phase)
    rectified_field, mismatch_map, global_distortion = retro_loop.detect_and_rectify_phase_errors(
        distorted_field, root_field
    )
    print(f"  * Phase Distortion Injected -> Global Mismatch Level: {global_distortion:.4f}")
    print(f"  * Moire Overlap Self-Correction Applied -> Post-Correction Coherence: {rectified_field.calculate_coherence():.4f}")
    print()

    # -------------------------------------------------------------------------
    # STAGE 5: Recursive Fractal Scaling
    # -------------------------------------------------------------------------
    print(">>> STAGE 5: Recursive Fractal Scaling & Rapid Wave Collapse")
    fractal_root = usa.stage5_fractal
    initial_macro_phasor = fractal_root.aggregate_macro_phasor()
    print(f"  * Initial Fractal Macro Phasor: {initial_macro_phasor}")

    final_fractal_coherence = fractal_root.wave_collapse(coupling_strength=0.85, max_steps=12)
    final_macro_phasor = fractal_root.aggregate_macro_phasor()
    print(f"  * Wave Collapse Executed across recursive fractal levels.")
    print(f"  * Final Global Resonant Coherence: {final_fractal_coherence:.4f}")
    print(f"  * Final Resonant Macro Phasor: {final_macro_phasor}")
    print()

    print("=" * 80)
    print(" UNIFIED SYSTEM ARCHITECTURE (5-STAGE WAVEFIELD) DEMONSTRATION COMPLETE ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
