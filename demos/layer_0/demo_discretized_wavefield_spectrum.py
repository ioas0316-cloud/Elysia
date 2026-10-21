#!/usr/bin/env python3
"""
demo_discretized_wavefield_spectrum.py

Demonstrates the Virtual Wave Field Architecture:
1. Discretized Continuity: Re-interpreting 0s and 1s bit sequences (e.g. 0110 vs 1001) as continuous phasor fields.
2. Interference Gates: Wave superposition and phase-locking instead of discrete Boolean switching.
3. Macro-Rotor Chunking: Aggregating micro-bit pattern wavefields into higher-order macro phasors.
4. Dimensional Topological Expansion:
   - Point (0D): Discrete bit transitions / pulse deltas.
   - Line (1D): Causal sequence vectors and phase trajectories.
   - Plane (2D): Spectral interference interface fields.
   - Spatiotemporal Structure (3D/4D): Self-resonant phase-locked reality field.
"""

import numpy as np
from core.physics.virtual_wave_field import (
    BitPatternPhasor,
    InterferenceGate,
    MacroRotor,
    SpatiotemporalWaveField
)


def run_discretized_continuity_demo():
    print("\n--- 1. Discretized Continuity: Bit Sequence Texture Mapping ---")
    p_condensed = BitPatternPhasor([0, 1, 1, 0])  # Condensed energy
    p_dispersed = BitPatternPhasor([1, 0, 0, 1])  # Dispersed energy

    print(f"Pattern '0110' -> Density: {p_condensed.density:.2f}, Dispersion: {p_condensed.dispersion:.2f}, Phase: {p_condensed.dominant_phase:.2f} rad")
    print(f"Pattern '1001' -> Density: {p_dispersed.density:.2f}, Dispersion: {p_dispersed.dispersion:.2f}, Phase: {p_dispersed.dominant_phase:.2f} rad")

    assert p_condensed.density == p_dispersed.density, "Bit densities should match!"
    assert p_condensed.dispersion < p_dispersed.dispersion, "Condensed pattern must have smaller dispersion!"
    print("=> SUCCESS: Distinct bit sequence textures successfully mapped to continuous phasor fields!")


def run_interference_gate_demo():
    print("\n--- 2. Resonance / Interference Gate Evaluation ---")
    gate = InterferenceGate(resonance_threshold=0.75)

    p1 = BitPatternPhasor([1, 1, 0, 0, 1, 1, 0, 0])
    p2_aligned = BitPatternPhasor([1, 1, 0, 0, 1, 1, 0, 0])
    p3_opposing = BitPatternPhasor([0, 0, 1, 1, 0, 0, 1, 1])

    res_aligned = gate.evaluate(p1, p2_aligned)
    res_opposing = gate.evaluate(p1, p3_opposing)

    print(f"Aligned Patterns   -> Resonance Index: {res_aligned['resonance_index']:.4f}, Phase Coherence: {res_aligned['phase_coherence']:.4f}, Locked: {res_aligned['is_phase_locked']}")
    print(f"Opposing Patterns  -> Resonance Index: {res_opposing['resonance_index']:.4f}, Phase Coherence: {res_opposing['phase_coherence']:.4f}, Locked: {res_opposing['is_phase_locked']}")

    assert res_aligned['is_phase_locked'], "Aligned patterns must achieve phase lock!"
    assert not res_opposing['is_phase_locked'], "Opposing patterns must not achieve phase lock!"
    print("=> SUCCESS: Boolean gate replaced by physical interference superposition and phase-locking!")


def run_macro_rotor_chunking_demo():
    print("\n--- 3. Fractal Scale Coupling (Macro-Rotor Chunking) ---")
    macro_alpha = MacroRotor(name="Alpha_Cluster")
    macro_beta = MacroRotor(name="Beta_Cluster")

    # Add micro bit-pattern phasors
    macro_alpha.add_micro_phasor(BitPatternPhasor([1, 0, 1, 0, 1, 0, 1, 0]))
    macro_alpha.add_micro_phasor(BitPatternPhasor([1, 1, 0, 0, 1, 1, 0, 0]))

    macro_beta.add_micro_phasor(BitPatternPhasor([1, 0, 1, 0, 1, 0, 1, 0]))

    coupled_amp, alignment = macro_alpha.interfere_with(macro_beta)
    print(f"MacroRotor 'Alpha': {macro_alpha}")
    print(f"MacroRotor 'Beta' : {macro_beta}")
    print(f"Interference Coupling -> Coupled Amp: {coupled_amp:.4f}, Alignment: {alignment:.4f}")

    assert coupled_amp > 0, "Macro-rotors must exhibit non-zero coupled energy!"
    print("=> SUCCESS: Stabilized micro-patterns aggregated into higher-order macro phasors!")


def run_dimensional_expansion_demo():
    print("\n--- 4. Dimensional Topological Expansion (0D -> 1D -> 2D -> 3D) ---")
    field = SpatiotemporalWaveField(field_dim=16)

    # 0D: Point discrete transition
    shift = field.project_point_transition(0, 1)
    print(f"  [0D Point] Bit transition 0->1 phase shift: {shift:.4f} rad")

    # 1D: Line sequence vector
    seq_x = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    seq_y = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    line_x = field.project_line_sequence(seq_x)
    line_y = field.project_line_sequence(seq_y)
    print(f"  [1D Line] Trajectory vector X length: {len(line_x)}, Y length: {len(line_y)}")

    # 2D: Plane interface field
    plane = field.weave_plane_interface(line_x, line_y)
    print(f"  [2D Plane] Weaved 2D interference field shape: {plane.shape}, Total Energy: {np.sum(np.abs(plane)**2):.4f}")

    # 3D/4D: Spatiotemporal Structure
    m1 = MacroRotor("Spindle_1", [BitPatternPhasor(seq_x)])
    m2 = MacroRotor("Spindle_2", [BitPatternPhasor(seq_x)])
    field.register_macro_rotor(m1)
    field.register_macro_rotor(m2)

    status = field.synthesize_spatiotemporal_structure()
    print(f"  [3D/4D Spatiotemporal] Kuramoto Order r: {status['order_parameter_r']:.4f}, Locked: {status['is_fully_phase_locked']}")

    assert status['is_fully_phase_locked'], "Spatiotemporal structure should be fully phase-locked!"
    print("=> SUCCESS: Complete dimensional expansion from discrete points to phase-locked spatiotemporal structure!")


def main():
    print("=" * 80)
    print("ELYSIA: DISCRETIZED VIRTUAL WAVEFIELD & DIMENSIONAL EXPANSION DEMO")
    print("=" * 80)

    run_discretized_continuity_demo()
    run_interference_gate_demo()
    run_macro_rotor_chunking_demo()
    run_dimensional_expansion_demo()

    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
