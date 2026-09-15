r"""
Unit tests for Isometric Causal Engine, Tilemap Adapter, and Elysian Intervention Framework (`tests/test_isometric_causal_engine.py`).
"""

import numpy as np
import pytest

from modules.causal_game_engine.isometric_adapter import (
    IsoTileConfig,
    IsometricProjection,
    CC0AssetRegistry,
    IsometricTilemap,
    AssetMetadata
)
from modules.causal_game_engine.isometric_causal_engine import (
    IsometricCausalEngine,
    CognitiveLensState,
    MicroSimsState,
    ElysianInterventionEngine,
    InterventionResult
)


def test_isometric_projection_forward_and_inverse():
    config = IsoTileConfig(tile_width=64.0, tile_height=32.0, elevation_scale=16.0, origin_x=100.0, origin_y=50.0)
    proj = IsometricProjection(config)

    grid_x, grid_y, elevation = 5.0, 3.0, 2.0
    iso_x, iso_y = proj.grid_to_iso(grid_x, grid_y, elevation)

    # Inverse projection check
    calc_gx, calc_gy = proj.iso_to_grid(iso_x, iso_y, elevation)
    assert pytest.approx(calc_gx, abs=1e-5) == grid_x
    assert pytest.approx(calc_gy, abs=1e-5) == grid_y


def test_cc0_asset_registry():
    registry = CC0AssetRegistry()
    grass = registry.get_asset("grass")
    assert grass is not None
    assert grass.category == "terrain"
    assert grass.passable is True

    town_hall = registry.get_asset("town_hall")
    assert town_hall is not None
    assert town_hall.category == "building"
    assert town_hall.passable is False


def test_isometric_tilemap_multi_layer_and_friction():
    tilemap = IsometricTilemap(width=10, height=10)
    tilemap.set_building(2, 2, "town_hall")

    # Town hall is impassable -> friction should be infinite
    f_building = tilemap.compute_cell_friction(2, 2)
    assert np.isinf(f_building)

    # Grass tile friction should be finite baseline
    f_grass = tilemap.compute_cell_friction(0, 0)
    assert pytest.approx(f_grass, abs=1e-2) == 1.0


def test_isometric_causal_engine_scale_lens_and_cognitive_phases():
    engine = IsometricCausalEngine(width=10, height=10)

    # Test initial Scale Lens
    engine.set_scale_lens(0.1)
    assert engine.get_scale_category() == "MICRO_SIMS"

    engine.set_scale_lens(0.5)
    assert engine.get_scale_category() == "MID_CITY"

    engine.set_scale_lens(0.9)
    assert engine.get_scale_category() == "MACRO_CIV"

    # Spawn NPC
    engine.spawn_npc("npc_test", 5.0, 5.0, role="villager", is_anchor=False)
    assert "npc_test" in engine.npc_voxels
    assert "npc_test" in engine.micro_states

    # Step engine
    log = engine.step(delta_time=0.1)
    assert "cognitive_lens" in log
    assert "npc_positions" in log
    assert "npc_test" in log["npc_positions"]


def test_cognitive_lens_switching_mechanics():
    engine = IsometricCausalEngine(width=10, height=10)

    # Force low food -> Survival lens
    engine.lens_state.resource_reserves["food"] = 5.0
    engine.step(0.1)
    assert engine.lens_state.current_lens == "Survival"

    # Restore food but low wood -> Gathering lens
    engine.lens_state.resource_reserves["food"] = 50.0
    engine.lens_state.resource_reserves["wood"] = 10.0
    engine.step(0.1)
    assert engine.lens_state.current_lens == "Gathering"

    # Restore resources -> Construction lens
    engine.lens_state.resource_reserves["wood"] = 80.0
    engine.lens_state.resource_reserves["stone"] = 50.0
    engine.step(0.1)
    assert engine.lens_state.current_lens == "Construction"

    # High macro geopolitical tension -> Empire_Expansion lens
    engine.macro_geopolitical_tension = 60.0
    engine.step(0.1)
    assert engine.lens_state.current_lens == "Empire_Expansion"


def test_elysian_intervention_engine_constraints():
    engine = IsometricCausalEngine(width=10, height=10)
    engine.spawn_npc("anchor_saint", 3.0, 3.0, role="saint", is_anchor=True, sync_rate=0.8)

    # Indirect intervention via Anchor (Cheap cost & low rebound)
    res_indirect = engine.apply_elysian_intervention(
        action_type="INJECT_IDEA",
        target_x=3, target_y=3,
        anchor_npc_id="anchor_saint",
        is_indirect=True
    )
    assert res_indirect.success is True
    assert res_indirect.cost_spent < 10.0

    # Direct miracle intervention (High cost & high rebound side effects)
    res_direct = engine.apply_elysian_intervention(
        action_type="SPAWN_BUILDING",
        target_x=5, target_y=5,
        is_indirect=False
    )
    assert res_direct.success is True
    assert res_direct.cost_spent >= 90.0
    assert len(res_direct.side_effects) > 0
    assert "NPC_MASS_PANIC" in res_direct.side_effects
