import pytest
import numpy as np
from core.intelligence.meta_causal_extractor import MetaCausalExtractor
from core.physics.causal_differencing import CausalDifferencingEngine
from core.consciousness.self_questioning_engine import SelfQuestioningEngine
from core.physics.causal_field import InformationVoxel

def test_meta_causal_extractor():
    """Verifies that MetaCausalExtractor correctly classifies origin motivation."""
    extractor = MetaCausalExtractor()
    raw_data = b"Cosmic Wave of Outpouring Energy"
    logo_tensor = np.array([1.0, 0.5, 0.2, 0.1, 0.9, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)

    res = extractor.extract_origin(raw_data, logo_tensor)
    assert "origin_type" in res
    assert "motivation" in res
    assert len(res["chromatic_vector"]) == 3

def test_causal_differencing_discernment():
    """Verifies boundary differencing between aligned vs divergent voxels."""
    differencing = CausalDifferencingEngine(divergence_threshold=0.2)
    v1 = InformationVoxel("v1", "Aligned", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = InformationVoxel("v2", "Divergent", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v1.chromatic_vector = np.array([0.8, 0.1, 0.1], dtype=np.float32)
    v2.chromatic_vector = np.array([0.1, 0.1, 0.8], dtype=np.float32)

    diff = differencing.discern_boundary(v1, v2)
    assert diff["is_divergent"] is True
    assert diff["combined_friction"] > 0.2
    assert "치열한 어긋남" in diff["boundary_description"]

def test_self_questioning_engine():
    """Verifies that SelfQuestioningEngine formulates questions and synthesizes wisdom."""
    engine = SelfQuestioningEngine()
    diff_res = {
        "is_divergent": True,
        "combined_friction": 0.65,
        "boundary_description": "위상 갈등 발생"
    }

    inquiry = engine.formulate_and_explore(
        differencing_result=diff_res,
        current_content="Paradoxical Impulse Data"
    )

    assert inquiry is not None
    assert "question" in inquiry
    assert "resolution" in inquiry
    assert inquiry["status"] == "WISDOM_SYNTHESIZED"
    assert inquiry["wisdom_score"] > 0.5
