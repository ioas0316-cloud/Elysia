"""
Unit tests for Causal Powder Field & Spatiotemporal Weaving Engine
(core/physics/causal_powder_field_weaving.py)
"""

import numpy as np
from core.physics.causal_powder_field_weaving import (
    MicroscopicPowderField,
    CausalManifoldWeaver,
    PowderWorldCausalEngine
)


def test_microscopic_powder_field_seeding_and_step():
    width, height = 10, 10
    field = MicroscopicPowderField(width=width, height=height)

    # Test initial state
    assert field.earth.shape == (height, width)
    assert field.thermal.shape == (height, width)
    assert field.fluid.shape == (height, width)

    # Seed elements
    field.seed_powder(5, 5, amount=2.0)
    field.seed_thermal(5, 5, intensity=3.0)
    field.seed_fluid(5, 2, amount=1.5)

    assert field.earth[5, 5] == 2.0
    assert field.thermal[5, 5] == 3.0
    assert field.fluid[2, 5] == 1.5

    # Step field physics
    field.step(dt=0.1)

    # Thermal diffusion should spread heat to neighboring cells
    assert field.thermal[5, 4] > 0.0
    assert field.thermal[5, 6] > 0.0
    # Chromatic vector should remain normalized
    np.testing.assert_allclose(np.sum(field.chromatic, axis=2), 1.0, rtol=1e-5)
    print("test_microscopic_powder_field_seeding_and_step PASSED")


def test_causal_manifold_weaver_time_sliding():
    width, height = 8, 8
    weaver = CausalManifoldWeaver(width=width, height=height)

    # Record two snapshots at t=0.0 and t=1.0
    snap0 = {
        "earth": np.ones((height, width), dtype=np.float32) * 1.0,
        "thermal": np.zeros((height, width), dtype=np.float32),
        "fluid": np.zeros((height, width), dtype=np.float32),
        "elevation": np.ones((height, width), dtype=np.float32) * 0.5,
        "chromatic": np.full((height, width, 3), 0.33, dtype=np.float32)
    }

    snap1 = {
        "earth": np.ones((height, width), dtype=np.float32) * 3.0,
        "thermal": np.ones((height, width), dtype=np.float32) * 2.0,
        "fluid": np.ones((height, width), dtype=np.float32) * 1.0,
        "elevation": np.ones((height, width), dtype=np.float32) * 1.5,
        "chromatic": np.full((height, width, 3), 0.33, dtype=np.float32)
    }

    weaver.record_snapshot(0.0, snap0)
    weaver.record_snapshot(1.0, snap1)
    weaver.weave_causal_manifold()

    assert weaver.is_woven
    assert weaver.causal_tensor_4d.shape == (2, height, width, 7)

    # Query t=0.0
    q0 = weaver.query_causal_state(0.0)
    np.testing.assert_allclose(q0["earth"], 1.0, rtol=1e-5)

    # Query t=1.0
    q1 = weaver.query_causal_state(1.0)
    np.testing.assert_allclose(q1["earth"], 3.0, rtol=1e-5)

    # Query interpolated t=0.5 (Smooth Hermite midpoint for linear endpoint delta is midpoint)
    q_mid = weaver.query_causal_state(0.5)
    np.testing.assert_allclose(q_mid["earth"], 2.0, rtol=1e-4)
    np.testing.assert_allclose(q_mid["thermal"], 1.0, rtol=1e-4)
    print("test_causal_manifold_weaver_time_sliding PASSED")


def test_powder_world_causal_engine():
    engine = PowderWorldCausalEngine(width=12, height=12)
    engine.seed_initial_powder_world()

    # Initial snapshot at t=0
    assert len(engine.weaver.timestamps) == 1

    # Run simulation phase
    engine.run_simulation_phase(ticks=10, dt=0.1)

    assert len(engine.weaver.timestamps) == 11
    assert engine.weaver.is_woven

    # Slide back to t=0.5 and t=1.0 without running micro physics loop
    state_05 = engine.slide_time(0.5)
    state_10 = engine.slide_time(1.0)

    assert state_05["earth"].shape == (12, 12)
    assert state_10["thermal"].shape == (12, 12)
    print("test_powder_world_causal_engine PASSED")


if __name__ == "__main__":
    test_microscopic_powder_field_seeding_and_step()
    test_causal_manifold_weaver_time_sliding()
    test_powder_world_causal_engine()
    print("ALL CAUSAL POWDER FIELD WEAVING TESTS PASSED SUCCESSFULLY!")
