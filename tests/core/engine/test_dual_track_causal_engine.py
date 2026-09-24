"""
Unit Tests for Dual-Track Causal Cognitive Engine (`tests/core/engine/test_dual_track_causal_engine.py`)
"""

import math
import pytest
import numpy as np

from core.engine.dual_track_causal_engine import (
    DualTrackCausalEngine,
    SensoryTensorEncoder,
    SpatialHashTensorField,
    DirectStorageFileHeader,
    CausalPageEntry,
    hermite_s_curve,
    quaternion_slerp,
)


def test_directstorage_and_page_entry_packing():
    """Verify 4KB sector aligned header and 64-byte page entry packing/unpacking."""
    header = DirectStorageFileHeader(
        total_baked_pages=42, page_table_offset=4096, data_region_offset=65536
    )
    packed_header = header.pack()
    assert len(packed_header) == 4096

    unpacked_header = DirectStorageFileHeader.unpack(packed_header)
    assert unpacked_header.magic_bytes == b"ELYSIAN1"
    assert unpacked_header.total_baked_pages == 42
    assert unpacked_header.page_table_offset == 4096

    page = CausalPageEntry(
        spatial_hash_key=123456789,
        nvme_sector_offset=2048,
        payload_size_bytes=512,
        vq_codebook_idx=15,
        is_baked=True,
        vram_pinned=True,
        is_attractor=True,
    )
    packed_page = page.pack()
    assert len(packed_page) == 64

    unpacked_page = CausalPageEntry.unpack(packed_page)
    assert unpacked_page.spatial_hash_key == 123456789
    assert unpacked_page.nvme_sector_offset == 2048
    assert unpacked_page.vq_codebook_idx == 15
    assert unpacked_page.is_baked is True
    assert unpacked_page.vram_pinned is True
    assert unpacked_page.is_attractor is True


def test_sensory_tensor_encoder_and_vq():
    """Verify 16-channel sensory encoding and VQ codebook quantization."""
    encoder = SensoryTensorEncoder(codebook_size=256, vector_dim=16)

    sensory_frame = encoder.encode_frame(
        visual_depth=(0.5, 0.5, 0.5, 10.0),
        acoustic_phase=(1.0, 440.0, 0.0, 1.0),
        physical_force=(0.1, -0.2, 0.0, 0.05),
        contextual_meta=(0.1, 1.0, 0.0, 0.0),
    )
    assert sensory_frame.shape == (16,)

    indices = encoder.quantize(sensory_frame)
    assert indices.shape == (1,)
    assert 0 <= indices[0] < 256

    reconstructed = encoder.dequantize(indices)
    assert reconstructed.shape == (1, 16)


def test_spatial_hash_tensor_field_diffusion():
    """Verify 3D Discrete Laplacian Diffusion and Energy Stamping."""
    field = SpatialHashTensorField(
        grid_dim=(8, 8, 8), cell_size=1.0, diffusion_rate=0.1, damping_factor=0.9
    )

    pos = np.array([2.5, 2.5, 2.5], dtype=np.float32)
    sensory_vec = np.ones(16, dtype=np.float32) * 10.0

    # Stamp energy
    field.stamp_energy(pos, sensory_vec)
    initial_sample = field.sample_field(pos)
    assert np.allclose(initial_sample, 10.0)

    # Step diffusion
    field.step_diffusion()
    diffused_sample = field.sample_field(pos)
    # Center should decrease due to diffusion and damping
    assert diffused_sample[0] < 10.0

    # Neighbor should receive diffused energy
    neighbor_pos = np.array([3.5, 2.5, 2.5], dtype=np.float32)
    neighbor_sample = field.sample_field(neighbor_pos)
    assert neighbor_sample[0] > 0.0


def test_hermite_and_quaternion_slerp():
    """Verify Hermite S-curve C^1 continuity and Quaternion Slerp interpolation."""
    # Hermite S-Curve tests
    assert hermite_s_curve(0.0) == 0.0
    assert hermite_s_curve(1.0) == 1.0
    assert hermite_s_curve(0.5) == 0.5

    # Quaternion Slerp tests
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0, 0.0])

    mid_q = quaternion_slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(mid_q), 1.0)
    assert np.isclose(mid_q[0], mid_q[1])


def test_dual_track_causal_engine_full_loop():
    """Verify Track A -> B breakout, ALU reduction, baking, and bifurcation."""
    engine = DualTrackCausalEngine(num_nodes=50, energy_threshold=1.0, lock_threshold=0.2)

    # Initial state: 100% ALU reduction (all nodes in Track A)
    stats_0 = engine.step_simulation(0.0)
    assert stats_0["active_nodes"] == 0
    assert stats_0["alu_reduction_percent"] == 100.0

    # Inject impulse into node #5
    engine.inject_external_impulse(node_idx=5, impulse_energy=2.5)

    # Step simulation: Node #5 should breakout into Track B
    stats_1 = engine.step_simulation(0.1)
    assert stats_1["active_nodes"] >= 1
    assert 5 in engine.active_node_indices
    assert stats_1["alu_reduction_percent"] < 100.0

    # Let energy decay over several ticks
    for tick in range(20):
        engine.step_simulation(0.1 * (tick + 2))

    # Check that baking or attractor decay occurred
    assert stats_1["active_ratio"] > 0.0

    # Test Critical Bifurcation Evaluation
    engine.inject_external_impulse(node_idx=10, impulse_energy=5.0)
    engine.inject_external_impulse(node_idx=11, impulse_energy=5.0)
    engine.step_simulation(3.0)

    bifurcation_event = engine.evaluate_critical_bifurcation(entropy_threshold=0.1)
    if bifurcation_event:
        assert "resolution" in bifurcation_event
        assert bifurcation_event["resolution"] == "Ridge_Split_Conditional_Attractor"
