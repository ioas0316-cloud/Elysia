"""
test_procedural_causal_world.py
================================
Unit tests for ProceduralCausalWorld generator and components.
"""

import pytest
import numpy as np
from core.physics.procedural_causal_world import (
    ProceduralCausalWorld,
    CosmicMacroRotor,
    FractalRotorCascade,
    PlanetaryDynamo,
    TopologicalPhaseValidator
)
from core.physics.causal_mmorpg_sandbox import ContinuousWorldManifold


def test_cosmic_macro_rotor():
    rotor = CosmicMacroRotor(day_length=86400.0)
    state0 = rotor.get_cosmic_state(0.0)
    state12h = rotor.get_cosmic_state(43200.0)

    assert "diurnal_phase" in state0
    assert "sun_vector" in state0
    assert len(state0["sun_vector"]) == 3

    # Sun elevation should vary across day/night
    assert state0["insolation"] >= 0.0
    assert state12h["insolation"] >= 0.0

    # Sun vector norm must be ~1.0
    norm0 = np.linalg.norm(state0["sun_vector"])
    assert pytest.approx(norm0, abs=1e-5) == 1.0


def test_fractal_rotor_cascade():
    cascade = FractalRotorCascade(seed=42)
    x = np.linspace(0, 10, 10, dtype=np.float32)
    y = np.linspace(0, 10, 10, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)

    wave1 = cascade.evaluate_rotor_wave(x_grid, y_grid, t=0.0, scale_depth=3)
    wave2 = cascade.evaluate_rotor_wave(x_grid, y_grid, t=0.0, scale_depth=3)

    # 100% deterministic reproducibility
    np.testing.assert_allclose(wave1, wave2)

    # Test domain warping
    wx, wy = cascade.evaluate_domain_warping(x_grid, y_grid, t=0.0)
    assert wx.shape == x_grid.shape
    assert wy.shape == y_grid.shape


def test_topological_phase_validator():
    validator = TopologicalPhaseValidator(max_slope_threshold=1.5)

    # Create artificial heightmap with steep cliff
    h = np.zeros((10, 10), dtype=np.float32)
    h[:, 5:] = 100.0 # Extreme cliff jump
    flow = np.ones((10, 10), dtype=np.float32)
    chroma = np.ones((10, 10, 3), dtype=np.float32) / 3.0

    healed_h, report = validator.validate_and_heal_field(h, flow, chroma)

    assert "invalid_slope_count" in report
    assert report["invalid_slope_count"] > 0
    assert report["healed_cells_count"] > 0
    # Healed heightmap should reduce extreme jump
    assert np.max(np.abs(np.diff(healed_h, axis=1))) < np.max(np.abs(np.diff(h, axis=1)))


def test_procedural_world_seed_determinism():
    world_a1 = ProceduralCausalWorld(seed=12345)
    world_a2 = ProceduralCausalWorld(seed=12345)
    world_b = ProceduralCausalWorld(seed=99999)

    chunk_a1 = world_a1.generate_chunk(0, 0, resolution=16)
    chunk_a2 = world_a2.generate_chunk(0, 0, resolution=16)
    chunk_b = world_b.generate_chunk(0, 0, resolution=16)

    # Same seed -> identical chunk data
    h_a1 = chunk_a1["field_data"]["heightmap"]
    h_a2 = chunk_a2["field_data"]["heightmap"]
    h_b = chunk_b["field_data"]["heightmap"]

    np.testing.assert_allclose(h_a1, h_a2)

    # Different seed -> different chunk data
    assert not np.allclose(h_a1, h_b)


def test_emergent_structure_placement():
    world = ProceduralCausalWorld(seed=42)
    chunk = world.generate_chunk(1, 2, chunk_size=64.0, resolution=32)

    structures = chunk["emergent_structures"]
    assert isinstance(structures, list)

    # Check Poisson Disk distance constraint (min_distance = 8.0)
    positions = [s["position"] for s in structures]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos_i = np.array(positions[i][:2])
            pos_j = np.array(positions[j][:2])
            dist = np.linalg.norm(pos_i - pos_j)
            assert dist >= 7.9, f"Poisson disk min distance violated: {dist}"


def test_populate_manifold_integration():
    world = ProceduralCausalWorld(seed=100)
    manifold = ContinuousWorldManifold(size=200.0)

    initial_nodes_count = len(manifold.potential_nodes)
    res = world.populate_manifold(manifold, chunk_x=0, chunk_y=0, resolution=16)

    assert res["potential_nodes_added"] > 0
    assert len(manifold.potential_nodes) > initial_nodes_count

    # Verify potential field can be sampled at a generated node position
    sample_pos = manifold.potential_nodes[0]["pos"]
    pot_val = manifold.get_potential_at(sample_pos)
    assert pot_val > 0.0


def test_causal_lod_depth():
    world = ProceduralCausalWorld(seed=77)
    x_grid, y_grid = np.meshgrid(np.linspace(0, 5, 8), np.linspace(0, 5, 8))

    field_lod2 = world.evaluate_coupled_equilibrium(x_grid, y_grid, causal_lod_depth=2)
    field_lod6 = world.evaluate_coupled_equilibrium(x_grid, y_grid, causal_lod_depth=6)

    # Higher LOD depth resolves finer details
    assert not np.allclose(field_lod2["heightmap"], field_lod6["heightmap"])
