"""
tests/synaptic_architecture/test_causal_skeleton_projection.py

Unit tests for CausalSkeletonProjectionOperator and SymbolAttractorField.
"""

import pytest
import numpy as np
from synaptic_architecture.causal_skeleton_projection_engine import (
    CausalSkeletonProjectionOperator,
    SymbolAttractorField
)


def test_skeleton_projection_operator():
    N = 64
    K = 8
    operator = CausalSkeletonProjectionOperator(feature_dim=N, skeleton_dim=K)

    # Generate a sample field: Core signal + noise
    np.random.seed(123)
    core_signal = np.dot(np.random.randn(K), operator.P_skeleton.T)
    noise = np.random.randn(N) * 0.1
    input_field = core_signal + noise

    projected = operator.project(input_field)

    assert "skeleton_vector" in projected
    assert projected["skeleton_vector"].shape == (K,)
    assert projected["equivalence_class"].shape == (N,)
    assert projected["null_space_noise"].shape == (N,)
    assert projected["compression_efficiency_factor"] == (N ** 2) / K
    assert projected["mdl_compression_ratio"] == K / N
    assert projected["noise_norm"] > 0.0


def test_symbol_attractor_and_pareidolia():
    N = 64
    K = 8
    operator = CausalSkeletonProjectionOperator(feature_dim=N, skeleton_dim=K)
    attractor_field = SymbolAttractorField(skeleton_operator=operator, attraction_strength=3.0)

    # Register symbol 'A'
    np.random.seed(42)
    symbol_A_exemplar = np.dot(np.random.randn(K), operator.P_skeleton.T)
    attractor_field.register_symbol_attractor("Symbol_A", symbol_A_exemplar)

    # Test 1: Similar noisy input -> Pareidolia capture
    noisy_input_A = symbol_A_exemplar + np.random.randn(N) * 0.05
    res1 = attractor_field.evaluate_attractor_gravitational_pull(noisy_input_A)

    assert res1["matched_symbol"] == "Symbol_A"
    assert res1["is_pareidolia_captured"] is True
    assert res1["gravitational_potential_delta"] > 0.0

    # Test 2: Completely distant input -> Not captured
    distant_input = np.random.randn(N) * 10.0
    res2 = attractor_field.evaluate_attractor_gravitational_pull(distant_input, distance_threshold=2.0)

    assert res2["matched_symbol"] is None
    assert res2["is_pareidolia_captured"] is False
