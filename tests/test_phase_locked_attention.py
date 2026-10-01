"""
Unit tests for Phase-Locked Multi-Head Self-Attention (PL-MHSA) & Multimodal Phase Lock Steering.
"""

import numpy as np
from core.consciousness.phase_locked_attention import (
    PhaseLockedMultiHeadAttention,
    RealTimeMultimodalPhaseLockInference
)


def test_phase_locked_multi_head_attention_forward():
    """Tests PL-MHSA layer forward pass, skew torque calculation, and Cayley transform rotor update."""
    d_model = 32
    n_heads = 4
    attn_layer = PhaseLockedMultiHeadAttention(d_model=d_model, n_heads=n_heads)

    np.random.seed(42)
    X = np.random.randn(2, 8, d_model)  # (Batch=2, Seq=8, d_model=32)

    out, info = attn_layer.forward(X, dt=0.01)

    assert out.shape == (2, 8, d_model)
    assert "torque_norm" in info
    assert "R_mat" in info
    assert info["R_mat"].shape == (n_heads, d_model // n_heads, d_model // n_heads)

    # Check orthogonality of Cayley rotors for each head
    d_head = d_model // n_heads
    for h in range(n_heads):
        R_h = info["R_mat"][h]
        ortho_diff = np.linalg.norm(np.dot(R_h, R_h.T) - np.eye(d_head))
        assert ortho_diff < 1e-4, f"Cayley rotor head {h} not orthogonal: {ortho_diff}"


def test_multimodal_phase_lock_inference_kv_steering():
    """Tests real-time multimodal latent recalibration and KV-cache phase steering."""
    embed_dim = 32
    recalibrator = RealTimeMultimodalPhaseLockInference(embed_dim=embed_dim)

    np.random.seed(7)
    H_text = np.random.randn(1, 10, embed_dim)
    H_modal = np.random.randn(1, 5, embed_dim)

    R_mat = recalibrator.step_recalibrator(H_text, H_modal, dt=0.01)
    assert R_mat.shape == (embed_dim, embed_dim)

    # Verify R_mat is in SO(embed_dim)
    ortho_diff = np.linalg.norm(np.dot(R_mat, R_mat.T) - np.eye(embed_dim))
    assert ortho_diff < 1e-4, f"R_mat not orthogonal: {ortho_diff}"

    # Apply phase lock to KV cache
    K_cache = np.random.randn(1, 4, 10, 8)  # (Batch, Heads, Seq, Head_Dim=8)
    V_cache = np.random.randn(1, 4, 10, 8)

    K_locked, V_locked = recalibrator.apply_phase_lock_to_kv(K_cache, V_cache, R_mat)
    assert K_locked.shape == K_cache.shape
    assert not np.allclose(K_locked, K_cache), "Key cache should be steered by Lie rotor rotation"


if __name__ == "__main__":
    test_phase_locked_multi_head_attention_forward()
    test_multimodal_phase_lock_inference_kv_steering()
    print("ALL PHASE LOCKED ATTENTION TESTS PASSED!")
