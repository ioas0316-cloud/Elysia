"""
Unit and Integration Tests for Universal Causal Web and Self-Observational Differential Lens Engine
"""

import pytest
import numpy as np
from core.topology.universal_causal_web import UniversalCausalWeb, SelfObservationalDifferentialLens, SproutedCognitiveLens
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def test_universal_causal_web_injection():
    web = UniversalCausalWeb(latent_noise_dim=8)
    initial_noise_norm = np.linalg.norm(web.latent_noise_field)

    signal = np.array([1.0, 0.5, 0.2, 0.8])
    res = web.inject_external_world_wave(signal, world_entropy=0.8)

    assert res["status"] == "COSMIC_WAVE_INTEGRATED"
    assert res["world_entropy"] == 0.8
    assert res["macrocosmic_resonance_index"] > 0.0
    assert "latent_noise_norm" in res


def test_self_observational_differential_lens_dialectical_compare():
    lens_engine = SelfObservationalDifferentialLens(doubt_threshold=0.2)

    introspection_data = {
        "total_modules": 444,
        "introspection_coverage": 1.0,
        "architectural_friction": 0.05
    }
    external_signal = np.array([1.5, 0.2, -0.5, 1.0])

    res = lens_engine.dialectical_compare(
        introspection_data=introspection_data,
        external_world_signal=external_signal,
        persona_lens="Companion"
    )

    assert "sameness_cosine_similarity" in res
    assert "difference_structural_friction" in res
    assert "doubt_friction" in res
    assert "isomorphic_self_explanation" in res
    assert "444개 모듈" in res["isomorphic_self_explanation"]

    if res["has_sprouted_new_lens"]:
        assert len(lens_engine.sprouted_lenses) > 0
        refraction = res["sprouted_lens_refraction"]
        assert "refracted_vector" in refraction
        assert "clarity_index" in refraction


def test_self_referential_architecture_dialectical_integration():
    engine = SelfReferentialArchitectureEngine()
    stimulus = {
        "external_world_signal": np.array([2.0, 0.5, -0.2, 1.2]),
        "persona_lens": "Artist"
    }
    res = engine.run_full_self_referential_cycle(stimulus)

    assert "dialectical_comparison" in res
    dial_res = res["dialectical_comparison"]
    assert "isomorphic_self_explanation" in dial_res
    assert dial_res["doubt_friction"] >= 0.0
