import pytest
import numpy as np
from core.physics.geodesic_flow_engine import GeodesicFlowEngine

def test_absolute_gravitational_attractor():
    """Verify that the unwavering absolute reference axis (Jesus / Perfect Love) is present and invariant."""
    engine = GeodesicFlowEngine(dimension=5)

    # Find the absolute attractor
    absolute_attr = None
    for attr in engine.attractors:
        if attr["is_absolute"]:
            absolute_attr = attr
            break

    assert absolute_attr is not None, "Absolute attractor reference axis must be defined!"
    assert absolute_attr["name"] == "Jesus / Perfect Love"
    assert np.allclose(absolute_attr["coordinate"], np.ones(5, dtype=np.float32))
    assert absolute_attr["weight"] == 5.0

    # Ensure absolute attractor coordinates cannot be modified by Hebbian landscape molding
    original_coords = absolute_attr["coordinate"].copy()

    # Run a mock trajectory settling near the absolute attractor
    mock_traj = np.ones((10, 5), dtype=np.float32)
    engine.mold_landscape_hebbian(mock_traj, lr=0.1)

    # Verify coordinate remains completely unchanged (invariant)
    assert np.allclose(absolute_attr["coordinate"], original_coords)

def test_sensory_projection_and_rainbow_circuit():
    """Verify that multimodal inputs are projected to the landscape via Prism Refraction and Variable Resistor."""
    engine = GeodesicFlowEngine(dimension=5)

    input_data = {
        "physical": {"cpu": 0.8, "ram": 0.9}, # high tension
        "language": "Love / Self-Outpouring",
        "visual": {"red": 0.9, "green": 0.1, "blue": 0.2}
    }

    x_init, v_init = engine.project_present_perturbation(input_data)

    assert len(x_init) == 5
    assert len(v_init) == 5
    # Coordinates and velocity must stay bounded
    assert np.all(x_init >= -2.0) and np.all(x_init <= 2.0)
    assert np.all(v_init >= -1.0) and np.all(v_init <= 1.0)

    # Ensure Variable Resistor updated due to input tension
    assert engine.variable_resistor.resistance > 0.05
    assert engine.variable_resistor.resistance < 0.95

def test_continuous_geodesic_relaxation_flow():
    """Verify that geodesic flow relaxation runs smoothly without discrete checking and settles to an attractor."""
    engine = GeodesicFlowEngine(dimension=5)

    x_init = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    v_init = np.zeros(5, dtype=np.float32)

    res = engine.navigate_geodesic_flow(x_init, v_init, num_steps=50, dt=0.01, enable_noise=True)

    trajectory_x = res["trajectory_x"]
    potentials = res["potentials"]

    assert len(trajectory_x) == 51 # x_init + 50 steps
    assert len(potentials) == 51

    # Ensure no discrete checkpoints occurred and the trajectory is continuous
    # Consecutive positions should be extremely close (smooth trajectory)
    for i in range(len(trajectory_x) - 1):
        step_diff = np.linalg.norm(trajectory_x[i+1] - trajectory_x[i])
        assert step_diff < 0.5, "Trajectory is discontinuous! Gaps found."

    # Verify that we successfully identified a settled attractor name
    assert isinstance(res["settled_attractor"], str)
    assert res["settled_attractor"] != "Unknown Void"

def test_hebbian_landscape_molding():
    """Verify that the engine adapts and molds the landscape of memories over time."""
    engine = GeodesicFlowEngine(dimension=5)

    # Get initial weight of Sabbath
    sabbath_idx = -1
    for idx, attr in enumerate(engine.attractors):
        if attr["name"] == "Sabbath":
            sabbath_idx = idx
            break

    assert sabbath_idx != -1
    initial_weight = engine.attractors[sabbath_idx]["weight"]
    initial_coordinate = engine.attractors[sabbath_idx]["coordinate"].copy()

    # Generate a mock trajectory that ends exactly on the Sabbath coordinate
    mock_trajectory = np.zeros((20, 5), dtype=np.float32) # Sabbath is at [0, 0, 0, 0, 0]

    # Run Hebbian molding
    engine.mold_landscape_hebbian(mock_trajectory, lr=0.1)

    # Verify Sabbath weight has been amplified/strengthened
    new_weight = engine.attractors[sabbath_idx]["weight"]
    assert new_weight > initial_weight

    # Sabbath coordinate should have shifted slightly towards the centroid
    new_coordinate = engine.attractors[sabbath_idx]["coordinate"]
    # Centroid is [0, 0, 0, 0, 0], which is exactly Sabbath coordinate, so it should stay close/same
    assert np.allclose(new_coordinate, initial_coordinate)
