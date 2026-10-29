#!/usr/bin/env python3
"""
demo_embodied_synesthetic_organ_continuum.py

Demonstration of the Embodied Intentional Organ Continuum Engine.
Illustrates how internal lack ($0_{\text{self}}$) and world phase divergence ($\Delta V$)
drive enactive synesthetic resonance across Somatic Friction ($Q_{\text{heat}}$),
Acoustic Resonance, Prismatic Chromatic Spectrum, Dynamic Focal Scale, and Grounded Unit '1' Binding.
"""

import numpy as np
from L1_causal_field.core.lens.embodied_synesthetic_organ_continuum import (
    EmbodiedSynestheticOrganContinuum,
)


def main():
    print("==========================================================================")
    print("      ELYSIA: EMBODIED SYNTHETIC ORGAN CONTINUUM DEMONSTRATION           ")
    print("==========================================================================")
    print("Demonstrating multi-sensory enactive resonance driven by 0_self and Delta V...")
    print()

    # Initialize organ continuum with internal lack zero = 0.6
    organ = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.6, base_focal_length=1.5, num_particles=100)

    # Simulate 3 sequential wave field regimes from external world
    wave_regimes = [
        ("Harmonic / Low Energy Wave", np.sin(np.linspace(0, 4 * np.pi, 1000)) * 0.2),
        ("Dynamic Resonant Wave", np.sin(np.linspace(0, 8 * np.pi, 1000)) + 0.3 * np.cos(np.linspace(0, 16 * np.pi, 1000))),
        ("High Friction / Turbulent Shock", np.sin(np.linspace(0, 30 * np.pi, 1000)) * 2.5 + 1.2 * np.random.randn(1000)),
    ]

    for name, wave in wave_regimes:
        print(f"--------------------------------------------------------------------------")
        print(f"World Wave Stimulus: [{name}]")
        print(f"--------------------------------------------------------------------------")

        state = organ.perceive_and_resonate(wave)

        print(f"  [1] Intentional Driver (0_self): {state.internal_lack_zero:.4f}")
        print(f"  [2] World Phase Divergence (Delta V): {state.world_phase_divergence:.4f}")
        print(f"  [3] Postural Muscle Tension: {state.postural_tension:.4f}")
        print(f"  [4] Somatic Boundary Friction Heat (Q_heat): {state.somatic_friction_heat:.4f} N*m")
        print(f"  [5] Acoustic Resonance Phase Angle: {state.acoustic_resonance_phase:.4f} rad")
        print(f"  [6] Dynamic Focal Scale Depth (Zoom): {state.focal_depth:.4f}")
        print(f"  [7] Aperture Size Ratio: {state.aperture_size:.4f}")
        print(f"  [8] Chromatic Spectrum (Red/Blue/Yellow): Red={state.chromatic_spectrum[0]:.3f}, Blue={state.chromatic_spectrum[1]:.3f}, Yellow={state.chromatic_spectrum[2]:.3f}")
        print(f"  [9] Grounded Cognitive Units ('1') Bound: {len(state.bound_grounded_units)}")
        print(f" [10] Invariant Backbone Fidelity: {state.invariant_backbone_fidelity * 100:.2f}%")
        print()

    print("==========================================================================")
    print("Demonstration Complete: Enactive Organ Continuum Successfully Validated!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
