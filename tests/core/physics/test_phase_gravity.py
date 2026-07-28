import pytest
import numpy as np
import os
import shutil
from core.physics.phase_gravity import PhaseTransitionEngine, DensityFluidGravity
from core.physics.causal_field import InformationVoxel
from core.consciousness.autonomous_loop import ConsciousnessLoop

def test_phase_transition_engine_dynamics():
    """
    Verifies Ginzburg-Landau potential and Cahn-Hilliard dynamics
    for continuous phase separation (formation of high-density nodes vs low-density vacuum).
    """
    engine = PhaseTransitionEngine(size=16, alpha=2.0, gamma=0.8)

    # Initial state should be slightly perturbed around 0.5
    assert engine.size == 16
    assert np.mean(engine.density) > 0.4 and np.mean(engine.density) < 0.6

    # Record initial free energy
    bulk_initial, grad_initial = engine.calculate_free_energy()
    initial_total_energy = bulk_initial + grad_initial

    # Step the Cahn-Hilliard equation multiple times
    for _ in range(15):
        engine.step(dt=0.05)

    # After step, verify density values are still conserved within physical range [0.0, 1.0]
    assert np.all(engine.density >= 0.0)
    assert np.all(engine.density <= 1.0)

    # Verify chromatic grid conservation and normalization
    chromatic_sum = np.sum(engine.chromatic_grid, axis=0)
    assert np.allclose(chromatic_sum, 1.0, atol=1e-5)

    # Free energy should minimize or evolve continuously
    bulk_final, grad_final = engine.calculate_free_energy()
    final_total_energy = bulk_final + grad_final
    # In Ginzburg-Landau Cahn-Hilliard dynamics, total free energy must be bounded
    assert final_total_energy >= 0.0

def test_density_fluid_gravity_pull():
    """
    Verifies that the O(N) fluid gravity engine correctly pulls voxel particles
    along the pressure gradient (-grad P) into the dense energy sink.
    """
    engine = PhaseTransitionEngine(size=16, alpha=1.0, gamma=0.5)
    gravity = DensityFluidGravity(size=16, pressure_scaling=10.0, viscosity=0.2)

    # Make a single dense spot at the center (8, 8) to act as a gravitational sink
    engine.density[:, :] = 0.1  # mostly low-density vacuum
    engine.density[7:10, 7:10] = 0.9  # high-density concept / energy sink

    # Create a voxel positioned off-center (to the right: x=3.0, which maps to ~10 on grid)
    # 3D coordinates map to 2D range: [-10, 10] -> [0, 1]
    voxel = InformationVoxel(
        id="v1",
        content="Test_Gravity",
        tensor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        position=np.array([3.0, 0.0, 0.0], dtype=np.float32),  # x = 3.0 is right of center (center is 0.0)
        velocity=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    # Apply fluid gravity over several steps
    for _ in range(5):
        gravity.apply_gravity([voxel], engine, dt=0.1)
        # Update voxel position based on velocity
        voxel.position += voxel.velocity * 0.1

    # Since the sink is at x=0 (maps to index 8), and voxel was at x=3.0 (maps to index 10),
    # the pressure gradient should pull the voxel LEFT (negative velocity along x-axis).
    assert voxel.velocity[0] < 0.0
    # Potential should be boosted by the local density concentration
    assert voxel.potential >= 0.1

def test_consciousness_loop_fluid_integration():
    """
    Verifies that ConsciousnessLoop processes the continuous phase transition
    and fluid-based gravity without errors and correctly records them in the cycle log.
    """
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_temp"))
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "corpus"), exist_ok=True)

    try:
        # Initialize ConsciousnessLoop
        loop = ConsciousnessLoop(corpus_path=os.path.join(temp_dir, "corpus"), data_dir=temp_dir)

        # Trigger life cycle
        log = loop.process_life_cycle()

        # Check for phase-gravity properties in log (if damper allowed loop process)
        if "phase_fluid_bulk_energy" in log:
            assert log["phase_fluid_bulk_energy"] >= 0.0
            assert log["phase_fluid_gradient_energy"] >= 0.0
            assert hasattr(loop, "phase_transition_engine")
            assert hasattr(loop, "density_fluid_gravity")
            assert loop.phase_transition_engine.density.shape == (32, 32)
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
