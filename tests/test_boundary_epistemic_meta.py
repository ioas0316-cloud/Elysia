"""
test_boundary_epistemic_meta.py: Comprehensive Unit Test Suite
===============================================================

Tests the 4 core modules of the Boundary-Based Epistemic Meta-Observer Framework:
1. Level-Set Topological Boundary Extraction & Tension Engine (`core/topology/boundary_level_set.py`)
2. Nonlinear Metric Tensor Plasticity Engine (`core/physics/metric_plasticity.py`)
3. Epistemic Meta-Observer & Quantum Fluctuation Engine (`core/consciousness/epistemic_meta_observer.py`)
4. Drone Swarm Lift Field & Phase Re-Locking Engine (`core/embodied/swarm_lift_field.py`)
"""

import numpy as np
import pytest

from core.topology.boundary_level_set import (
    ContinuousPotentialField,
    LevelSetBoundaryExtractor,
    BoundaryTensionCalculator
)
from core.physics.metric_plasticity import MetricPlasticityEngine
from core.consciousness.epistemic_meta_observer import EpistemicMetaObserver
from core.embodied.swarm_lift_field import DroneSwarmLiftFieldSimulator


def test_continuous_potential_field_and_level_set():
    """Test potential field sampling, gradient computation, boundary extraction, and tension calculation."""
    field = ContinuousPotentialField(spatial_dim=3)
    field.add_potential_source(center=np.array([0.0, 0.0, 0.0]), intensity=2.0, sigma=3.0)

    center = np.array([0.0, 0.0, 0.0])
    sample_val = field.sample_at(center)
    assert sample_val > 1.9

    grad = field.compute_gradient(np.array([1.0, 0.0, 0.0]))
    assert grad.shape == (3,)
    assert grad[0] < 0.0  # Gradient points towards center

    extractor = LevelSetBoundaryExtractor(cutoff_threshold=1.0, spatial_dim=3)
    boundary_pts, normals = extractor.extract_boundary_samples(field, center, radius=3.0, num_samples=32)
    assert isinstance(boundary_pts, list)
    assert isinstance(normals, list)

    calculator = BoundaryTensionCalculator(field, extractor)
    force, tension, num_pts = calculator.compute_causal_force_and_tension(center, radius=3.0)
    assert force.shape == (3,)
    assert tension >= 0.0


def test_metric_plasticity_engine():
    """Test metric tensor plasticity flow, stress-energy calculation, and positive-definiteness enforcement."""
    engine = MetricPlasticityEngine(dim=3, alpha=0.1, gamma=0.05)

    grad = np.array([0.5, -0.2, 0.8])
    T_ij = engine.compute_stress_energy_tensor(grad)
    assert T_ij.shape == (3, 3)

    g_updated, strain = engine.step_plasticity_flow(grad, dt=0.1)
    assert g_updated.shape == (3, 3)
    assert strain >= 0.0

    # Test SPD (Symmetric Positive Definite)
    evals = np.linalg.eigvalsh(g_updated)
    assert np.all(evals > 0)


def test_epistemic_meta_observer():
    """Test geometric-to-semantic projection, density matrix construction, Von Neumann entropy, and Knowledge Lock."""
    observer = EpistemicMetaObserver(feature_dim=8, semantic_dim=64, similarity_threshold=0.8)

    boundary_features = np.array([1.0, 0.5, -0.2, 0.8, 0.1, 2.5, 0.4, 1.2])
    e_int = observer.project_boundary_features(boundary_features)
    assert e_int.shape == (64,)
    assert pytest.approx(np.linalg.norm(e_int), abs=1e-5) == 1.0

    env_state = np.array([0.5, 0.5, 0.5, 0.5])
    metric_g = np.eye(4)
    rho = observer.compute_density_matrix(env_state, metric_g)
    assert rho.shape == (4, 4)
    assert pytest.approx(np.trace(rho), abs=1e-5) == 1.0

    s_ent = observer.compute_von_neumann_entropy(rho)
    assert s_ent >= 0.0

    # Evaluate alignment with synthetic consensus vector matching e_int
    consensus_vector = e_int.copy()
    result = observer.evaluate_epistemic_alignment(boundary_features, consensus_vector, metric_g, env_state)

    assert result["cosine_similarity"] > 0.95
    assert result["epistemic_loss"] < 0.05
    assert result["knowledge_locked"] is True


def test_drone_swarm_lift_field_simulator():
    """Test Kuramoto phase locking dynamics, wind gust disruption, and autonomous relaxation."""
    sim = DroneSwarmLiftFieldSimulator(num_drones=16, coupling_K=3.0)

    R_init, Psi_init = sim.compute_order_parameter()
    assert 0.0 <= R_init <= 1.0

    # Apply wind gust shock
    sim.apply_wind_gust_shock(np.array([1.0, 0.5, -0.2]), intensity=3.0)
    R_post_shock, _, _ = sim.step_phase_locking_dynamics(dt=0.01)

    # Run relaxation until phase lock
    relaxation_res = sim.run_relaxation_until_phase_lock(target_R=0.85, max_steps=120)
    assert relaxation_res["final_R"] >= 0.85
    assert relaxation_res["final_energy_saved"] > 0.0
