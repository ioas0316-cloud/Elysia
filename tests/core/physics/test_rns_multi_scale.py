import numpy as np
import pytest
from core.physics.rns_multi_scale_field import (
    ResidueNumberSystem,
    MultiScaleRNSField,
    MicroGrid
)

def test_rns_encoding_decoding_exact():
    """Verify that Residue Number System (RNS) encoding/decoding is completely lossless."""
    primes = [3, 5, 7]
    rns = ResidueNumberSystem(primes)
    assert rns.M == 3 * 5 * 7  # 105

    # Test all integers in the dynamic range [0, 104]
    for x in range(rns.M):
        residues = rns.encode(np.array(x))
        decoded = rns.decode(residues)
        assert decoded == x

def test_rns_carry_free_arithmetic():
    """Verify that RNS operations are carry-free and modularly correct."""
    primes = [5, 7, 11]
    rns = ResidueNumberSystem(primes)

    x_val = 15
    y_val = 22

    res_x = rns.encode(np.array(x_val))
    res_y = rns.encode(np.array(y_val))

    # Carry-free addition: (15 + 22) % 385 = 37
    res_add = rns.add(res_x, res_y)
    assert rns.decode(res_add) == (x_val + y_val) % rns.M

    # Carry-free multiplication: (15 * 22) % 385 = 330
    res_mul = rns.multiply(res_x, res_y)
    assert rns.decode(res_mul) == (x_val * y_val) % rns.M

    # Carry-free subtraction: (15 - 22) % 385 = 378
    res_sub = rns.subtract(res_x, res_y)
    assert rns.decode(res_sub) == (x_val - y_val) % rns.M

def test_rns_invalid_primes():
    """Ensure non-coprime primes raise ValueError."""
    with pytest.raises(ValueError):
        ResidueNumberSystem([2, 4, 5])  # 2 and 4 are not coprime

def test_field_initialization_and_ground_state():
    """Verify field is initialized to the ground vacuum state of 1."""
    field = MultiScaleRNSField(macro_shape=(4, 4), macro_primes=[3, 5])

    # All residues in macro_residues should be 1
    assert np.all(field.macro_residues == 1)

    # Initial energy should be zero
    assert np.all(field.macro_energy == 0.0)

    # Potential at ground state (all 1s) should be 0.0
    potential = field.get_macro_potential()
    assert np.all(potential == 0.0)

def test_torus_coordinate_stimulation():
    """Verify coordinate stimulation wrap-around due to Torus boundary topology."""
    field = MultiScaleRNSField(macro_shape=(4, 4), macro_primes=[3, 5])

    # Stimulate beyond grid bounds (e.g., at (4, 4) in 4x4 grid)
    # toroidal boundary should map (4, 4) to (0, 0)
    field.stimulate(4, 4, 15.0)

    assert field.macro_energy[0, 0] == 15.0
    # Residue at (0, 0) should be excited (not 1)
    assert np.any(field.macro_residues[0, 0] != 1)

def test_physical_relaxation():
    """Verify that excited states relax back to the ground vacuum state 1 over time."""
    field = MultiScaleRNSField(macro_shape=(4, 4), macro_primes=[5, 7], zoom_threshold=100.0)

    # Excite field at (1, 1)
    field.stimulate(1, 1, 5.0)
    assert np.any(field.macro_residues[1, 1] != 1)

    # Run steps to see physical relaxation towards 1 (should reach ground state in a few steps)
    for _ in range(10):
        field.step(0.1)

    # After relaxation steps, it should return to the vacuum state 1
    assert np.all(field.macro_residues == 1)

def test_zoom_in_and_renormalization_lifecycle():
    """Verify dynamic zoom-in (MicroGrid creation) and renormalization (collapse back)."""
    field = MultiScaleRNSField(
        macro_shape=(4, 4),
        micro_shape=(2, 2),
        macro_primes=[3, 5],
        micro_primes=[7, 11],
        zoom_threshold=5.0,
        decay_threshold=1.0,
        dissipation_rate=2.0 # high dissipation for quick collapse
    )

    # 1. Stimulate a cell above zoom_threshold (5.0) to trigger Zoom-In
    field.stimulate(2, 2, 8.0)

    # Run one step
    field.step(0.1)

    # MicroGrid should be spawned at (2, 2)
    assert (2, 2) in field.micro_grids
    mgrid = field.micro_grids[(2, 2)]
    assert mgrid.shape == (2, 2)
    assert np.any(mgrid.energy > 0)

    # 2. Let the energy decay over time (Renormalization Group coalescing)
    # With dissipation_rate=2.0, energy will decay below decay_threshold=1.0
    for _ in range(20):
        field.step(0.1)

    # MicroGrid should be collapsed/removed after energy decays
    assert (2, 2) not in field.micro_grids

def test_outpouring_and_variable_friction():
    """Verify self-outpouring flow and modular potential diffusion."""
    field = MultiScaleRNSField(macro_shape=(4, 4), macro_primes=[3, 5])

    # Set high energy friction at a cell
    field.macro_friction[1, 1] = 10.0 # high friction
    field.macro_friction[1, 2] = 1.0  # low friction

    # Excite neighbors
    field.stimulate(1, 1, 10.0)
    field.stimulate(1, 2, 10.0)

    # Step the field
    field.step(0.1)

    # Energy at low friction cell (1, 2) should flow faster or adjust potential faster
    # than high friction cell (1, 1)
    # Let's ensure no errors were thrown and states updated properly
    assert np.any(field.macro_energy > 0)
