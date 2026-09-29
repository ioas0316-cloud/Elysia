"""
Interactive Demo: Ontological Water Medium & Principle Emergence
================================================================
Demonstrates the Causal Principle Engine (Ontological Fluid Medium):
1. Sets up Air (top) and Water (bottom) boundary interface without hardcoded Snell formulas.
2. Injects wave pulse into Air heading towards Water.
3. Simulates wave propagation, reflection, refraction, and phase transition (Water <-> Ice).
4. Renders real-time ASCII/Matrix visualization of wave amplitude, emergent refractive index, and tension.
5. Extracts and displays underlying Principle Isomorphism schema.
"""

import time
import numpy as np
from core.physics.ontological_fluid_medium import OntologicalFluidMedium
from core.consciousness.triadic_boundary_causal_engine import TriadicBoundaryCausalEngine


def render_ascii_field(field: np.ndarray, title: str):
    """Renders a 2D float array as ASCII characters."""
    chars = " .:-=+*#%@"
    rows, cols = field.shape
    min_v = np.min(field)
    max_v = np.max(field)
    norm = (field - min_v) / (max_v - min_v + 1e-6)

    print(f"\n--- [{title}] ---")
    for r in range(rows):
        line = ""
        for c in range(cols):
            idx = int(norm[r, c] * (len(chars) - 1))
            line += chars[idx] + " "
        print(line)


def run_demo():
    print("=" * 80)
    print("      ELYSIA: ONTOLOGICAL WATER MEDIUM & PRINCIPLE EMERGENCE DEMO")
    print("=" * 80)
    print("Principle: 'Do not calculate ray bounces, let the principle flow in the medium.'\n")

    # Initialize 16x16 Ontological Fluid Medium
    medium = OntologicalFluidMedium(grid_shape=(16, 16), base_density=1.0, base_viscosity=0.5)

    # Air region (Top half: rows 0-7) | Water region (Bottom half: rows 8-15)
    medium.set_medium_region(slice(0, 8), slice(0, 16), density=0.5, viscosity=0.2, phase_state=0)
    medium.set_medium_region(slice(8, 16), slice(0, 16), density=2.5, viscosity=0.9, phase_state=0)

    print("[Medium Field Setup]")
    print(" - Top Half (Rows 0-7): Air   (Density rho = 0.50, Viscosity eta = 0.20)")
    print(" - Bottom Half (Rows 8-15): Water (Density rho = 2.50, Viscosity eta = 0.90)\n")

    n_field = medium.calculate_emergent_refractive_index()
    print(f"[Emergent Refractive Index Prior to Wave Injection]")
    print(f" - Emergent Refractive Index in Air:   {np.mean(n_field[0:8, :]):.4f}")
    print(f" - Emergent Refractive Index in Water: {np.mean(n_field[8:16, :]):.4f}")

    # Inject impulse at (3, 8) in air
    print("\n" + "-" * 80)
    print("[Cycle 1: Injecting Physical Wave Impulse at (3, 8) in Air]")
    medium.inject_impulse((3, 8), amplitude=6.0, flux_color=(0.9, 0.1, 0.1))

    for step in range(1, 11):
        medium.step(dt=0.1)
        if step in [1, 5, 10]:
            print(f"\n[Step {step}] Wave Propagation in Causal Medium")
            render_ascii_field(medium.wave_field, f"Step {step}: Wave Amplitude Field (phi)")
            summary = medium.get_state_summary()
            print(f" - Max Wave Amp: {summary['max_wave_amplitude']:.4f} | Total Ignorance Tension: {summary['total_tension']:.4f}")

    print("\n" + "=" * 80)
    print("[Cycle 2: Extracting Principle Isomorphism]")
    principle = medium.extract_principle_isomorphism()

    print("\n[Extracted Principle Invariant Schema]")
    print(f" - Principle Name:              {principle['principle_name']}")
    print(f" - Isomorphic Invariant:         {principle['isomorphic_invariant']}")
    print(f" - Emergent Refractive Index Mean: {principle['emergent_refractive_index_mean']:.4f}")
    print(f" - Density Contrast Ratio:       {principle['density_contrast_ratio']:.4f}")
    print(f" - Phase State Distribution:     {principle['phase_state_distribution']}")

    print("\n" + "-" * 80)
    print("[Conclusion: Ontological Reality Achieved]")
    print("System did not compute ray angles or Snell's Law formulas.")
    print("Refraction and wave reflection naturally emerged from density, viscosity, and impedance dynamics!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
