import pytest
import numpy as np
from L2_cognitive_engine.helical_phase_attention import HelicalPhaseAttention, CognitivePhaseSearchLoop

def test_helical_phase_attention_forward():
    attn = HelicalPhaseAttention(embed_dim=16)

    np.random.seed(42)
    query_vecs = np.random.randn(2, 16)
    key_vecs = np.random.randn(5, 16)
    val_vecs = np.random.randn(5, 16)

    output, matrix = attn.forward(query_vecs, key_vecs, val_vecs)

    assert output.shape == (2, 16)
    assert matrix.shape == (2, 5)
    # Check normalization along key dimension
    assert np.allclose(np.sum(matrix, axis=1), 1.0)

def test_cognitive_phase_search_loop():
    search_loop = CognitivePhaseSearchLoop(target_concept_phase=0.0)

    res = search_loop.execute_cognitive_step(current_thought_phase=0.1)
    assert "is_phase_locked" in res
    assert "realignment_suggestions" in res
