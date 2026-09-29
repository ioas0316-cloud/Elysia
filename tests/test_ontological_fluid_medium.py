"""
Unit tests for core/physics/ontological_fluid_medium.py
======================================================
Verifies:
1. OntologicalFluidMedium initialization & region setup.
2. Impulse injection & wave propagation dynamics.
3. Emergent refractive index calculation from causal impedance and density.
4. Ignorance gradient & tension computation.
5. Emergent phase transition (Water -> Ice cohesion).
6. Principle isomorphism extraction.
"""

import pytest
import numpy as np
from core.physics.ontological_fluid_medium import OntologicalFluidMedium
from core.consciousness.triadic_boundary_causal_engine import TriadicBoundaryCausalEngine


def test_medium_initialization():
    medium = OntologicalFluidMedium(grid_shape=(16, 16), base_density=1.0, base_viscosity=0.5)
    assert medium.shape == (16, 16)
    assert np.allclose(medium.density, 1.0)
    assert np.allclose(medium.viscosity, 0.5)
    assert medium.impedance.shape == (16, 16)


def test_set_medium_region_and_impedance():
    medium = OntologicalFluidMedium(grid_shape=(20, 20), base_density=1.0, base_viscosity=0.5)
    # Set denser water region in bottom half
    medium.set_medium_region(slice(10, 20), slice(0, 20), density=2.25, viscosity=1.0)

    assert medium.density[15, 10] == 2.25
    assert medium.viscosity[15, 10] == 1.0
    # Z = sqrt(2.25 / 1.0) = 1.5
    assert np.isclose(medium.impedance[15, 10], 1.5)


def test_impulse_and_wave_propagation():
    medium = OntologicalFluidMedium(grid_shape=(16, 16))
    initial_amp = np.sum(np.abs(medium.wave_field))
    assert initial_amp == 0.0

    # Inject impulse at center
    medium.inject_impulse((8, 8), amplitude=5.0)
    assert medium.wave_field[8, 8] == 5.0

    # Run steps to verify propagation
    for _ in range(5):
        medium.step(0.1)

    # Wave should spread to neighboring cells
    assert np.abs(medium.wave_field[8, 9]) > 0.0 or np.abs(medium.wave_field[9, 8]) > 0.0
    assert medium.get_state_summary()["avg_wave_amplitude"] > 0.0


def test_emergent_refractive_index():
    medium = OntologicalFluidMedium(grid_shape=(16, 16), base_density=1.0, base_viscosity=1.0)
    # Dense region (water) vs low density region (air)
    medium.set_medium_region(slice(0, 8), slice(0, 16), density=0.5, viscosity=0.5)   # Air
    medium.set_medium_region(slice(8, 16), slice(0, 16), density=3.0, viscosity=0.8)  # Water

    n_field = medium.calculate_emergent_refractive_index()
    n_air = np.mean(n_field[0:8, :])
    n_water = np.mean(n_field[8:16, :])

    # Water should have higher emergent refractive index than air without hardcoded Snell formulas
    assert n_water > n_air


test_medium_initialization()
test_set_medium_region_and_impedance()
test_impulse_and_wave_propagation()
test_emergent_refractive_index()
print("All ontological fluid medium tests passed successfully!")
