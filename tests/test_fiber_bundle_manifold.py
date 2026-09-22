import pytest
import torch
import numpy as np
from core.topology.fiber_bundle_manifold import FiberBundleManifold, CausalSectionCache, SENSORY_PORTS, NUM_SENSORY_PORTS


def test_fiber_bundle_manifold_initialization():
    manifold = FiberBundleManifold(num_points=100, device="cpu")
    assert manifold.coords.shape == (100, 4)
    assert manifold.velocity.shape == (100, 4)
    assert manifold.h_metric.shape == (100, 3, 3)
    assert manifold.sensory_weights.shape == (100, NUM_SENSORY_PORTS)
    # Check that initial temporal velocity is strictly positive
    assert (manifold.velocity[:, 0] > 0).all()


def test_temporal_monotonicity_and_no_collision():
    """
    Verify Foliation & Temporal Monotonicity:
    Even if two points share identical 3D fiber coordinates x1 = x2,
    as long as t1 != t2, they exist on distinct fiber leaves F_t1 != F_t2
    and do not collide or interfere.
    """
    manifold = FiberBundleManifold(num_points=2, device="cpu")

    # Force identical 3D fiber spatial coordinates for both points
    spatial_point = torch.tensor([1.5, -0.5, 2.0])
    manifold.coords[0, 1:] = spatial_point
    manifold.coords[1, 1:] = spatial_point

    # Set distinct temporal coordinates t1 = 0.2, t2 = 0.8
    manifold.coords[0, 0] = 0.2
    manifold.coords[1, 0] = 0.8

    # Verify 4D points are strictly non-equal due to temporal leaf separation
    assert not torch.allclose(manifold.coords[0], manifold.coords[1])

    # Step geodesic flow
    manifold.step_geodesic_flow(d_tau=0.05)

    # Verify time t progressed monotonically for both
    assert manifold.coords[0, 0] > 0.2
    assert manifold.coords[1, 0] > 0.8


def test_geodesic_flow_step():
    manifold = FiberBundleManifold(num_points=50, device="cpu")
    initial_coords = manifold.coords.clone()

    # Apply gauge potential A_t
    gauge_A = torch.randn((50, 3)) * 0.2
    manifold.set_gauge_potential(gauge_A)

    # Compute Christoffel symbols
    gamma = manifold.compute_christoffel_symbols()
    assert gamma.shape == (50, 4, 4, 4)

    # Step geodesic flow
    manifold.step_geodesic_flow(d_tau=0.01)

    # Coords must have evolved
    assert not torch.allclose(manifold.coords, initial_coords)


def test_multi_sensory_occupancy_and_wave_impact():
    manifold = FiberBundleManifold(num_points=100, device="cpu")

    # Set custom sensory occupancy (e.g. 80% Somatosensory, 5% each for rest)
    weights = torch.tensor([0.05, 0.05, 0.80, 0.05, 0.05]).repeat(100, 1)
    manifold.update_sensory_occupancy(weights)

    # Inject SOMATOSENSORY thermal/pressure wave impact (index 2)
    resonances = manifold.inject_sensory_wave_impact(port_idx=2, impact_magnitude=1.0)

    assert len(resonances) == NUM_SENSORY_PORTS
    for port in SENSORY_PORTS:
        assert port in resonances
        assert isinstance(resonances[port].item(), float)


def test_causal_section_cache():
    manifold = FiberBundleManifold(num_points=200, device="cpu")

    cache = CausalSectionCache(manifold)
    section = cache.slice_temporal_section(t_slice=0.5, tol=0.2)

    assert "fiber_3d" in section
    assert "sensory_weights" in section
    assert section["fiber_3d"].shape[1] == 3
    assert section["sensory_weights"].shape[1] == NUM_SENSORY_PORTS

    grid_2d = cache.rasterize_section_to_2d_projection(section)
    assert grid_2d.shape == (64, 64, NUM_SENSORY_PORTS)
    assert grid_2d.sum() > 0
