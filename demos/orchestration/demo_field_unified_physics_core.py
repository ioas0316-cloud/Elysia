"""
Demo: Field-Based Unified Physics Core.

Demonstrates the Field-Based Unified Physics Core weaving Space (curvature), Time (dilation),
Gravity (refraction), Light (standing waves & energy conservation),
Matter States (Solid, Fluid, Gas, Plasma), and Organic Systems (Kuramoto Phase Skeleton & Gait).
"""

import numpy as np
from core.physics.clifford_spacetime_rotor import CliffordSpacetimeRotor
from core.physics.field_unified_physics_core import (
    FieldUnifiedPhysicsCore,
    MatterState,
    KuramotoOscillatorNetwork
)


def main():
    print("========================================================================")
    print("       ELYSIA: FIELD-BASED UNIFIED PHYSICS CORE RUNTIME DEMO           ")
    print("========================================================================\n")

    core = FieldUnifiedPhysicsCore(resolution=256, memory_budget_mb=3000.0)

    # 1. Space & Time Dilation
    rotor_z = CliffordSpacetimeRotor.create_rotor_3d(np.array([0.0, 0.0, 1.0]), np.pi / 4.0) # 45 deg
    friction = 0.5 # Medium friction tension
    field_space, time_dilation = core.compute_space_time_curvature(rotor_z, friction)
    print(f"[1. Space Curvature & Time Dilation]")
    print(f"   - Phase Friction Tension: {friction}")
    print(f"   - Time Dilation Factor (dt'/dt): {time_dilation:.4f}")
    print(f"   - Field Power Integral: {np.sum(np.abs(field_space)**2) / core.resolution:.4f}")

    # 2. Gravity as Refractive Phase Tension
    density_grad = np.linspace(0.1, 1.0, core.resolution)
    field_grav = core.compute_gravity_refraction(density_grad)
    print(f"\n[2. Gravity as Refractive Phase Tension]")
    print(f"   - Spatial Density Gradient Min/Max: {density_grad.min():.2f} / {density_grad.max():.2f}")
    print(f"   - Refracted Wave Power: {np.sum(np.abs(field_grav)**2) / core.resolution:.4f}")

    # 3. Light Standing Wave Propagation & Energy Conservation
    field_light, energy = core.propagate_light_standing_wave(frequency=10.0)
    print(f"\n[3. Light & Exact Energy Conservation]")
    print(f"   - Standing Wave Frequency: 10 Hz")
    print(f"   - Conserved Total Field Energy E: {energy:.6f} (Strict 1.0000 Target)")

    # 4. Unified Matter Spectrum Transitions
    print(f"\n[4. Unified Matter Spectrum Transitions]")
    for state in MatterState:
        result = core.simulate_matter_spectrum(state)
        print(f"   - State [{result['matter_state'].upper()}]: "
              f"Friction={result['friction_tension']:.2f}, "
              f"Cost={result['computation_cost']:.3f}, "
              f"Power={result['field_power']:.4f}")

    # 5. Organic Systems: Kuramoto Phase Skeleton Gait Dynamics
    print(f"\n[5. Organic Systems: Kuramoto Phase Skeleton Gait Dynamics]")
    kuramoto_net = KuramotoOscillatorNetwork(
        num_nodes=4,
        phases=np.array([0.0, np.pi/2, np.pi, 3*np.pi/2]),
        natural_freqs=np.array([1.0, 1.0, 1.0, 1.0]),
        coupling_matrix=np.ones((4, 4)) * 0.5,
        friction_homeostasis=0.1
    )
    for step in range(3):
        kuramoto_net = core.simulate_kuramoto_organic_gait(kuramoto_net, dt=0.05)
        print(f"   - Step {step+1}: Phases={kuramoto_net.phases.round(3)}, "
              f"Homeostasis Damping={kuramoto_net.friction_homeostasis:.4f}")

    # 6. Hardware Resilience: Dynamic Precision Throttling
    print(f"\n[6. Dynamic Precision Throttling (GTX 1060 Benchmark)]")
    print(f"   - Initial Spatial Resolution: {core.resolution}")
    core.dynamic_precision_throttle(estimated_vram_mb=2800.0) # > 80% budget
    print(f"   - Throttled Spatial Resolution: {core.resolution} (Throttled={core.throttled})")
    core.dynamic_precision_throttle(estimated_vram_mb=1200.0) # Normal load
    print(f"   - Restored Spatial Resolution: {core.resolution} (Throttled={core.throttled})")

    print("\n========================================================================")
    print("         FIELD-BASED UNIFIED PHYSICS CORE DEMO SUCCESSFUL                ")
    print("========================================================================")


if __name__ == "__main__":
    main()
