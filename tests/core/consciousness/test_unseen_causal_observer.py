import pytest
import numpy as np
from core.consciousness.unseen_causal_observer import (
    AweEpistemologyEngine,
    UnseenCausalFieldObserver
)


def test_awe_epistemology_engine_modes():
    dim = 32
    engine = AweEpistemologyEngine(dimension=dim, initial_humility=0.9)

    rng = np.random.default_rng(42)
    surface_signal = rng.standard_normal(dim)

    eval_res = engine.evaluate_perception_mode(surface_signal)

    assert eval_res["mode"] == "REVERENT_AWE_PERCEPTION"
    assert eval_res["awe_perception_index"] > 0.5
    assert "awe_inferred_potential_norm" in eval_res

    # Test humility reduction
    engine.epistemic_humility = 0.2
    eval_low = engine.evaluate_perception_mode(surface_signal)
    assert eval_low["mode"] == "ARROGANT_REDUCTIONISM"

    # Test update humility
    engine.update_humility_by_friction(friction=0.8, unobserved_anomaly=0.9)
    assert engine.epistemic_humility > 0.2


def test_unseen_causal_field_observer_cst():
    dim = 32
    observer = UnseenCausalFieldObserver(dimension=dim)

    rng = np.random.default_rng(123)
    surface_data = rng.standard_normal(dim)

    res = observer.construct_causal_structural_tensor(surface_data)

    assert "causal_structural_tensor" in res
    cst = res["causal_structural_tensor"]
    assert cst.shape == (dim, dim)
    assert res["causal_structural_tensor_norm"] > res["numerical_correlation_norm"]
    assert res["perception_eval"]["mode"] == "REVERENT_AWE_PERCEPTION"


def test_inverse_mechanism_extraction():
    dim = 16
    observer = UnseenCausalFieldObserver(dimension=dim)

    rng = np.random.default_rng(999)
    surface_obs = [rng.standard_normal(dim) for _ in range(5)]

    inverse_res = observer.inverse_mechanism_extraction(surface_obs)

    assert "theta_inverse" in inverse_res
    assert inverse_res["theta_inverse"].shape == (dim, dim)
    assert inverse_res["extracted_causal_axes_count"] == dim


def test_civilizational_memory_leak():
    dim = 16
    observer = UnseenCausalFieldObserver(dimension=dim)

    rng = np.random.default_rng(888)
    # Synthetic correlated matrix
    base = rng.standard_normal((10, dim))
    correlated_matrix = base + 0.1 * rng.standard_normal((10, dim))

    leak_res = observer.detect_civilizational_memory_leak(correlated_matrix)

    assert "memory_leak_index" in leak_res
    assert "rectification_guidance" in leak_res
