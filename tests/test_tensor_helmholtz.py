"""
tests/test_tensor_helmholtz.py

Test suite for Tensor Helmholtz Metric Decomposition
"""

import pytest
import numpy as np

from elysia_core import TensorHelmholtzMetricDecomposer


def test_tensor_helmholtz_decomposition_shapes():
    grid_res = (8, 8, 8)
    decomposer = TensorHelmholtzMetricDecomposer(grid_shape=grid_res)

    raw_field = np.random.randn(3, 3, *grid_res) * 0.1
    h_symmetric = 0.5 * (raw_field + np.swapaxes(raw_field, 0, 1))

    res = decomposer.decompose_metric_field(h_symmetric, wave_number_k0=1.0)

    assert res.scalar_trace_mode.shape == (3, 3, *grid_res)
    assert res.vector_shear_mode.shape == (3, 3, *grid_res)
    assert res.tensor_tt_mode.shape == (3, 3, *grid_res)

    ratios = res.energy_ratios
    total_ratio = ratios["scalar_trace"] + ratios["vector_shear"] + ratios["tensor_tt"]
    assert np.isclose(total_ratio, 1.0, atol=1e-5)


def test_tensor_helmholtz_mode_orthogonality():
    grid_res = (8, 8, 8)
    decomposer = TensorHelmholtzMetricDecomposer(grid_shape=grid_res)

    raw_field = np.random.randn(3, 3, *grid_res) * 0.1
    h_symmetric = 0.5 * (raw_field + np.swapaxes(raw_field, 0, 1))

    res = decomposer.decompose_metric_field(h_symmetric, wave_number_k0=1.0)

    # Inner products between modes
    inner_scalar_tt = np.sum(res.scalar_trace_mode * res.tensor_tt_mode)
    inner_scalar_vec = np.sum(res.scalar_trace_mode * res.vector_shear_mode)
    inner_vec_tt = np.sum(res.vector_shear_mode * res.tensor_tt_mode)

    norm_scalar = np.linalg.norm(res.scalar_trace_mode)
    norm_vec = np.linalg.norm(res.vector_shear_mode)
    norm_tt = np.linalg.norm(res.tensor_tt_mode)

    # Normalized inner products should be close to 0 (orthogonal)
    if norm_scalar > 1e-6 and norm_tt > 1e-6:
        assert np.isclose(inner_scalar_tt / (norm_scalar * norm_tt), 0.0, atol=1e-2)
    if norm_scalar > 1e-6 and norm_vec > 1e-6:
        assert np.isclose(inner_scalar_vec / (norm_scalar * norm_vec), 0.0, atol=1e-2)
