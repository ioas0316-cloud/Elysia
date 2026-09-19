"""
test_transformative_gear_engine.py
===================================
Unit and Integration Tests for Transformative Gear Engine:
[Component Data Plane × Transformative Gear = Emergent Volume]
"""

import pytest
import torch
import numpy as np
from core.physics.transformative_gear_engine import (
    TransformativeGearEngine,
    ComponentDataPlane,
    TransformativeGear,
    EmergentCausalVolume,
    CompressedCausalSeal,
    PhaseState
)


def test_component_data_plane_initialization():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    vel = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
    plane = ComponentDataPlane(positions=pos, velocities=vel)

    assert plane.num_nodes == 2
    assert plane.dim == 3
    assert torch.allclose(plane.compute_center_of_mass(), torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(plane.compute_total_momentum(), torch.tensor([0.1, 0.1, 0.0]))


def test_transformative_gear_phase_states():
    # Test SOLID phase
    gear = TransformativeGear(dim=3, cognitive_temperature=0.2, degrees_of_freedom=0.1)
    assert gear.update_phase_state() == PhaseState.SOLID

    # Test LIQUID phase
    gear.cognitive_temperature = 1.0
    gear.degrees_of_freedom = 0.5
    assert gear.update_phase_state() == PhaseState.LIQUID

    # Test GAS phase
    gear.cognitive_temperature = 3.0
    gear.degrees_of_freedom = 0.9
    assert gear.update_phase_state() == PhaseState.GAS


def test_emergent_causal_volume_calculation():
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    vel = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    plane = ComponentDataPlane(positions=pos, velocities=vel)
    gear = TransformativeGear(dim=3)

    vol = EmergentCausalVolume(data_plane=plane, gear=gear)

    assert vol.volume_metric == pytest.approx(24.0, abs=1e-3)
    assert vol.shear_stress > 0.0


def test_compressed_causal_seal_and_phase_reignition():
    pos = torch.randn(15, 3)
    vel = torch.randn(15, 3)
    plane = ComponentDataPlane(positions=pos, velocities=vel)
    gear = TransformativeGear(dim=3, cognitive_temperature=1.2)
    vol = EmergentCausalVolume(data_plane=plane, gear=gear)

    seal = CompressedCausalSeal.compress_volume("Seal_Test", vol)
    assert seal.seal_id == "Seal_Test"
    assert seal.stored_gear_params["num_nodes"] == 15

    # Phase Re-ignition with excitation voltage
    re_plane, re_gear, re_vol = seal.decompress_reignite(excitation_voltage=2.0)
    assert re_plane.num_nodes == 15
    assert re_gear.cognitive_temperature == pytest.approx(2.4, abs=1e-2)
    assert re_vol.volume_metric > 0.0


def test_transformative_gear_engine_feedback_step():
    engine = TransformativeGearEngine(dim=3)
    pos = torch.randn(10, 3)
    vel = torch.randn(10, 3)
    data_plane = ComponentDataPlane(positions=pos, velocities=vel)

    # Step without target
    res1 = engine.step(data_plane=data_plane, dt=0.1)
    assert "next_data_plane" in res1
    assert "emergent_volume" in res1
    assert res1["prediction_error"] == 0.0

    # Step with target and mismatch friction
    target_pos = torch.randn(10, 3) * 5.0
    target_vel = torch.randn(10, 3) * 5.0
    target_plane = ComponentDataPlane(positions=target_pos, velocities=target_vel)

    initial_temp = engine.gear.cognitive_temperature
    res2 = engine.step(
        data_plane=data_plane,
        observed_target_plane=target_plane,
        input_torque_vector=torch.tensor([1.0, 2.0, 3.0]),
        dt=0.1
    )

    assert res2["prediction_error"] > 0.0
    assert res2["angular_mismatch"] > 0.0
    # Temperature should increase due to friction heating
    assert res2["cognitive_temperature"] > initial_temp
