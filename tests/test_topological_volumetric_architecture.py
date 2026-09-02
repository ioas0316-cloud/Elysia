"""
Unit tests for Topological Volumetric Architecture (위상적 체적 텐서 아키텍처)
"""

import math
import numpy as np
import pytest
import torch

from synaptic_architecture.topological_volumetric_architecture import (
    VolumetricPolytope,
    TopologicalSpaceEngine,
    SpacetimeTensorLayer4D,
    DynamicTopologicalRelaxationEngine,
    benchmark_flash_attention,
    benchmark_4d_spacetime_tensor
)


def test_volumetric_polytope_area_and_volume():
    """Verify Shoelace formula area calculation and 3D volume integration."""
    poly = VolumetricPolytope(
        node_id="Concept_A",
        base_footprint=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        height=5.0,
        angular_deficit=1.2
    )

    # 4x4 square base = 16.0 area, height = 5.0 -> Volume = 80.0
    assert abs(poly.volume - 80.0) < 1e-5


def test_topological_space_engine_collision_and_relaxation():
    """Verify 3D volume intersection detection and topological stress relaxation."""
    engine = TopologicalSpaceEngine()

    poly_a = VolumetricPolytope(
        node_id="Concept_A",
        base_footprint=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        height=5.0,
        position_3d=(0.0, 0.0, 0.0)
    )

    poly_b = VolumetricPolytope(
        node_id="Concept_B",
        base_footprint=[(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0)],
        height=3.0,
        position_3d=(0.0, 0.0, 1.0)
    )

    # Register A succeeds
    success_a = engine.register_information(poly_a)
    assert success_a is True

    # Initial height of B
    initial_h_b = poly_b.height

    # Register B detects 3D volume collision Vol(A ∩ B) > 0 and relaxes B's height
    success_b = engine.register_information(poly_b)
    assert success_b is False
    assert poly_b.height > initial_h_b


def test_spacetime_tensor_layer_4d():
    """Verify PyTorch SpacetimeTensorLayer4D Broad-Phase and Narrow-Phase execution."""
    layer = SpacetimeTensorLayer4D(alpha=0.05)

    # 10 AABBs [xmin, ymin, zmin, xmax, ymax, zmax]
    aabbs = torch.tensor([
        [0.0, 0.0, 0.0, 4.0, 4.0, 5.0],
        [2.0, 2.0, 1.0, 6.0, 6.0, 4.0],
        [10.0, 10.0, 10.0, 14.0, 14.0, 15.0],
        [12.0, 12.0, 11.0, 16.0, 16.0, 14.0],
        [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
    ], dtype=torch.float32)

    updated_aabbs, vol_overlap, repulsion = layer(aabbs, k_neighbors=4)

    assert updated_aabbs.shape == aabbs.shape
    assert vol_overlap.shape == (5, 4)
    assert repulsion.shape == (5, 1)

    # Overlap between AABB 0 and 1 should be detected (> 0)
    assert repulsion[0].item() > 0.0
    assert repulsion[1].item() > 0.0
    # AABB 4 is isolated, so its repulsion force should be 0
    assert repulsion[4].item() == 0.0


def test_dynamic_topological_relaxation_engine():
    """Verify dynamic momentum, elastic potential energy, and 4D worldline step evolution."""
    dyn_engine = DynamicTopologicalRelaxationEngine(elasticity_k=1.0, viscosity_gamma=0.1, mass=1.0)

    p1 = VolumetricPolytope(
        node_id="Node1",
        base_footprint=[(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)],
        height=4.0,
        position_3d=(0.0, 0.0, 0.0)
    )

    p2 = VolumetricPolytope(
        node_id="Node2",
        base_footprint=[(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)],
        height=4.0,
        position_3d=(0.0, 0.0, 0.5)
    )

    dyn_engine.add_polytope(p1)
    dyn_engine.add_polytope(p2)

    initial_pos_p1 = p1.position_3d
    initial_pos_p2 = p2.position_3d

    # Step forward in 4D spacetime
    record = dyn_engine.step(time_delta=0.1)

    assert record["time_t"] == 0.1
    assert record["total_potential_energy"] > 0.0
    assert record["total_overlap_volume"] > 0.0

    # Polytopes should move apart due to repulsion forces
    assert p1.position_3d != initial_pos_p1
    assert p2.position_3d != initial_pos_p2


def test_benchmark_functions():
    """Verify benchmark utility functions execute cleanly."""
    lat_fa, mem_fa = benchmark_flash_attention(N=100, device="cpu")
    lat_st, mem_st = benchmark_4d_spacetime_tensor(N=100, device="cpu")

    assert lat_fa >= 0.0
    assert lat_st >= 0.0
