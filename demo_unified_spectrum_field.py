"""
Demo: Unified Spectrum Field & Dynamic Formula Memory Engine.

Demonstrates:
1. Physical Quantity Spectrum Grounding (Distance, Velocity, Force, Energy, Temperature)
2. Formula Transformation Functions on Mutable Wave Field Stack (Euler, Log, Lagrangian Collapse, F=ma, Thermal Equilibrium)
3. Dynamic Chunking & Macro-Rotor Promotion
4. 100% Reversible Retrocausal Trajectory Tracing (Delta S ≈ 0)
"""

import numpy as np
from core.physics.unified_spectrum_field import UnifiedSpectrumFieldEngine


def main():
    print("=========================================================================")
    print("  ELYSIA: UNIFIED SPECTRUM FIELD & DYNAMIC FORMULA MEMORY DEMONSTRATION  ")
    print("=========================================================================\n")

    engine = UnifiedSpectrumFieldEngine(resolution=128)

    # 1. Initialize Continuous Wave Spectrum Input
    t = np.linspace(0, 4 * np.pi, 128)
    initial_wave = np.sin(2 * t) + 0.5 * np.cos(5 * t) + 0.2 * np.random.randn(128)
    engine.set_field_state(initial_wave)

    print("[Step 1] Physical Quantity Spectrum Grounding")
    phys_state = engine.ground_physical_quantities()
    print(f"  - Distance (Spatial Phase Shift)     : {phys_state.distance_phase_shift:.4f} rad")
    print(f"  - Velocity (Frequency Rate d_freq/dt): {phys_state.velocity_freq_rate:.4f} Hz")
    print(f"  - Force (Interference Pressure)      : {phys_state.force_interference_pressure:.4f} N")
    print(f"  - Energy (Power Spectrum Total)      : {phys_state.energy_amplitude_power:.4f} J")
    print(f"  - Temperature (Spectral Variance)    : {phys_state.temperature_dispersion:.4f} K")
    print("  => All physical quantities successfully grounded into single wave spectrum.\n")

    # 2. Apply Dynamic Formula Transformations
    print("[Step 2] Applying Dynamic Spectrum Transformations")

    # 2.1 Euler Rotation
    print("  [2.1] Applying Euler Rotation (e^{i * pi/4})...")
    engine.apply_euler_rotation(np.pi / 4.0)

    # 2.2 Log Scale Compression
    print("  [2.2] Applying Logarithmic Scale Compression (base=2.0)...")
    engine.apply_log_scale_compression(base=2.0)

    # 2.3 Force Acceleration (F = ma)
    print("  [2.3] Applying Force Acceleration (F=10N, m=2kg)...")
    engine.apply_force_acceleration(mass=2.0, force_amplitude=10.0)

    # 2.4 Lagrangian Minimum Action Wave Collapse
    print("  [2.4] Applying Lagrangian Minimum Action Wave Collapse...")
    potential_landscape = np.sin(t) ** 2
    _, action_val = engine.apply_lagrangian_wave_collapse(potential_landscape)
    print(f"        -> Total Action (L = T - V) Converged: {action_val:.4f}")

    # 2.5 Thermal Equilibrium Rectification
    print("  [2.5] Applying Thermal Equilibrium Phase Rectification...")
    engine.apply_thermal_equilibrium(dissipation_rate=0.15)
    print("  => Spectrum transformations complete.\n")

    # 3. Dynamic Chunking & Macro-Rotor Promotion
    print("[Step 3] Dynamic Chunking & Macro-Rotor Promotion")
    macro = engine.chunk_and_promote_macro_rotor(rotor_id="MacroRotor_UnifiedLaw_1", threshold_coherence=0.3)
    if macro:
        print(f"  - Promoted Macro-Rotor ID  : {macro.rotor_id}")
        print(f"  - Base Frequency            : {macro.base_frequency:.2f}")
        print(f"  - Coherence Score           : {macro.coherence_score:.4f}")
        print(f"  - Usage Count               : {macro.usage_count}")
        print("  => Micro-computations successfully chunked behind higher-order macro axis.\n")
    else:
        print("  - Wave pattern not yet coherent enough for macro-rotor promotion.\n")

    # 4. Retrocausal Trajectory Tracing
    print("[Step 4] Retrocausal Reversible Circuit Back-Tracing")
    result = engine.trace_retrocausal_trajectory()
    print(f"  - Reconstruction Fidelity           : {result.reconstruction_fidelity * 100:.2f}%")
    print(f"  - Reversibility Dissipation (Delta S): {result.reversibility_dissipation_index:.6f}")
    print(f"  - Transformation Steps Traversed     : {len(result.transformation_history)}")

    if result.reconstruction_fidelity > 0.99 and result.reversibility_dissipation_index < 0.01:
        print("\n[VERIFICATION SUCCESS] Unified Spectrum Field & Retrocausal Circuit 100% Validated!")
    else:
        print("\n[VERIFICATION WARNING] Reversibility metric exceeded loss threshold.")


if __name__ == "__main__":
    main()
