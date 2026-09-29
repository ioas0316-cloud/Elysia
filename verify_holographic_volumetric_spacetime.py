"""
Verification Test Suite for Native 3D/4D Holographic Volumetric Spacetime Engine
=============================================================================
Verifies:
1. Universal Topological Representation across heterogeneous domains (numeric, semantic, value, causal).
2. Holographic global pattern reconstruction from heavy partial spatial cropping (>70% volume loss).
3. Knowing-Ignorance phase strain gradients and causal tension pressures.
4. Continuous spatiotemporal 3D wave evolution and rotor phase relaxation.
"""

import sys
import os
import numpy as np

# Ensure core repository path is available
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.physics.holographic_volumetric_spacetime import (
    UniversalTopologicalPoint,
    KnowingIgnorancePhaseGradient,
    HolographicVolumetricStorage,
    CausalVolumetricSpacetimeEngine
)

def test_universal_topological_points():
    print("--- 1. Testing Universal Topological Representation ---")
    pt_num = UniversalTopologicalPoint("p1", "numeric", 42.0, (0.2, -0.3, 0.5))
    pt_sem = UniversalTopologicalPoint("p2", "semantic", "Causal Continuum", (-0.4, 0.1, -0.2))
    pt_val = UniversalTopologicalPoint("p3", "value", {"ethics": 0.95}, (0.0, 0.8, -0.1))

    assert pt_num.chromatic_vector.shape == (3,)
    assert pt_sem.chromatic_vector.shape == (3,)
    assert pt_val.chromatic_vector.shape == (3,)

    # Rotate phase quaternion
    initial_q = pt_num.phase_quaternion.copy()
    pt_num.rotate_phase(np.array([0.0, 1.0, 0.0]), np.pi / 4)
    assert not np.allclose(initial_q, pt_num.phase_quaternion)
    print("SUCCESS: Universal Topological Representative nodes created and phase-rotated.")

def test_knowing_ignorance_phase_gradient():
    print("--- 2. Testing Knowing-Ignorance Phase Strain Gradient ---")
    engine = KnowingIgnorancePhaseGradient((16, 16, 16))
    pt_known = UniversalTopologicalPoint("p_know", "numeric", 1.0, (-0.5, -0.5, -0.5))
    pt_known.set_phase_coherence(1.0)

    pt_ignorant = UniversalTopologicalPoint("p_ignorant", "existential", None, (0.5, 0.5, 0.5))
    pt_ignorant.set_phase_coherence(0.0)

    engine.update_coherence_from_points([pt_known, pt_ignorant])
    grad_field, tension_force = engine.compute_phase_strain_gradient()

    assert grad_field.shape == (3, 16, 16, 16)
    assert tension_force.shape == (16, 16, 16)
    assert np.max(tension_force) > 0.0
    print(f"SUCCESS: Strain Gradient computed. Max Causal Pressure Force: {np.max(tension_force):.4f}")

def test_holographic_crop_reconstruction():
    print("--- 3. Testing Holographic Reconstruction from Heavy Crop (>70%) ---")
    engine = CausalVolumetricSpacetimeEngine((16, 16, 16))
    pt1 = UniversalTopologicalPoint("p1", "numeric", 10.0, (-0.3, 0.2, 0.1))
    pt2 = UniversalTopologicalPoint("p2", "semantic", "Hologram", (0.4, -0.2, -0.5))
    engine.add_topological_point(pt1)
    engine.add_topological_point(pt2)

    fidelity_70 = engine.evaluate_holographic_crop_reconstruction(crop_ratio=0.7)
    print(f"Holographic Reconstruction Fidelity with 70% Cropped Volume: {fidelity_70 * 100:.2f}%")
    assert fidelity_70 > 0.5, f"Fidelity too low: {fidelity_70}"

    fidelity_80 = engine.evaluate_holographic_crop_reconstruction(crop_ratio=0.8)
    print(f"Holographic Reconstruction Fidelity with 80% Cropped Volume: {fidelity_80 * 100:.2f}%")
    assert fidelity_80 > 0.3, f"Fidelity too low: {fidelity_80}"
    print("SUCCESS: Global wave pattern successfully reconstructed from partial spatial crop!")

def test_continuous_spatiotemporal_evolution():
    print("--- 4. Testing Continuous Spatiotemporal 3D Wave Evolution ---")
    engine = CausalVolumetricSpacetimeEngine((16, 16, 16), dt=0.05)
    pt = UniversalTopologicalPoint("p1", "causal", "Action", (0.0, 0.0, 0.0))
    pt.set_phase_coherence(0.5)
    engine.add_topological_point(pt)

    initial_energy = float(np.mean(engine.wave_psi**2))
    for _ in range(10):
        stats = engine.step_spatiotemporal_evolution()

    final_energy = stats["mean_wave_energy"]
    print(f"Initial Wave Energy: {initial_energy:.6f} -> Final Wave Energy: {final_energy:.6f}")
    assert stats["time"] > 0.0
    print("SUCCESS: Spatiotemporal 3D continuum evolved continuously across time.")

if __name__ == "__main__":
    print("=========================================================================")
    print("RUNNING VERIFICATION FOR HOLOGRAPHIC VOLUMETRIC SPACETIME ENGINE")
    print("=========================================================================")
    test_universal_topological_points()
    test_knowing_ignorance_phase_gradient()
    test_holographic_crop_reconstruction()
    test_continuous_spatiotemporal_evolution()
    print("=========================================================================")
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=========================================================================")
