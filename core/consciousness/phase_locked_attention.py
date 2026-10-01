"""
Phase-Locked Multi-Head Self-Attention & Multimodal Steering
============================================================
Implements:
1. PhaseLockedMultiHeadAttention: Hybrid PL-MHSA with head-wise Lie algebra skew torque,
   Cayley transform or matrix exponential rotor updates, and manifold Q/K steering.
2. RealTimeMultimodalPhaseLockInference: Real-time latent recalibration across multimodal embedding streams.
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np


class PhaseLockedMultiHeadAttention:
    """
    Phase-Locked Multi-Head Self-Attention (PL-MHSA).
    Computes head-wise anti-symmetric skew torque from Query and Key interactions,
    updates Lie algebra state omega_state in so(d_head), applies Lie group rotation
    steering to Q and K, and performs scaled dot-product attention.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, gamma: float = 2.5, beta: float = 0.15):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.gamma = gamma
        self.beta = beta

        # Lie algebra state per head so(d_head): (n_heads, d_head, d_head)
        self.omega_state = np.zeros((self.n_heads, self.d_head, self.d_head), dtype=np.float64)

        # Projections initialized with seed for deterministic behavior
        np.random.seed(42)
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_out = np.random.randn(d_model, d_model) * scale

    def _cayley_transform(self, A: np.ndarray) -> np.ndarray:
        """
        Computes Cayley transform R = (I - A)^(-1) @ (I + A) for skew-symmetric matrix A in so(d_head).
        Guarantees exact orthogonal Lie group rotation matrix R in SO(d_head).
        """
        I_mat = np.eye(self.d_head, dtype=np.float64)
        inv_term = np.linalg.inv(I_mat - A)
        return np.matmul(inv_term, I_mat + A)

    def forward(
        self,
        X: np.ndarray,
        dt: float = 0.01,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        X: (Batch, Seq_Len, d_model) input sequence.
        Returns: (Output_Tensor, info_dict)
        """
        B, L, _ = X.shape

        # Linear projections
        Q_flat = np.matmul(X, self.W_q)  # (B, L, d_model)
        K_flat = np.matmul(X, self.W_k)
        V_flat = np.matmul(X, self.W_v)

        # Split heads: (B, n_heads, L, d_head)
        Q = Q_flat.reshape(B, L, self.n_heads, self.d_head).swapaxes(1, 2)
        K = K_flat.reshape(B, L, self.n_heads, self.d_head).swapaxes(1, 2)
        V = V_flat.reshape(B, L, self.n_heads, self.d_head).swapaxes(1, 2)

        # Head-wise covariance and skew torque computation
        # Q_h, K_h per head: (n_heads, B*L, d_head)
        Q_h = Q.swapaxes(1, 2).reshape(self.n_heads, B * L, self.d_head)
        K_h = K.swapaxes(1, 2).reshape(self.n_heads, B * L, self.d_head)

        # Covariance per head: (n_heads, d_head, d_head)
        scale_denom = B * L * math.sqrt(self.d_head)
        Cov = np.matmul(Q_h.swapaxes(1, 2), K_h) / max(scale_denom, 1e-8)

        # Skew-symmetric torque Q_skew = 0.5 * (Cov - Cov.T)
        Q_skew = 0.5 * (Cov - Cov.swapaxes(-1, -2))

        # Update Lie algebra state omega_state per head
        dOmega = self.gamma * Q_skew - self.beta * self.omega_state
        self.omega_state += dOmega * dt

        # Cayley transform to compute rotor matrix R_mat per head in SO(d_head)
        R_mat = np.zeros((self.n_heads, self.d_head, self.d_head), dtype=np.float64)
        for h in range(self.n_heads):
            A = 0.5 * dt * 0.5 * (self.omega_state[h] - self.omega_state[h].T)
            R_mat[h] = self._cayley_transform(A)

        # Apply manifold rotation steering: Q_steered = Q @ R_mat^T, K_steered = K @ R_mat^T
        # Q: (B, n_heads, L, d_head), R_mat: (n_heads, d_head, d_head)
        Q_steered = np.einsum('bhld,hdk->bhlk', Q, R_mat)
        K_steered = np.einsum('bhld,hdk->bhlk', K, R_mat)

        # Scaled dot-product attention
        scores = np.matmul(Q_steered, K_steered.swapaxes(-1, -2)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax over last dimension
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        # Attention output
        attn_out = np.matmul(attn_weights, V)  # (B, n_heads, L, d_head)

        # Concatenate heads and final projection
        attn_out_concat = attn_out.swapaxes(1, 2).reshape(B, L, self.d_model)
        output = np.matmul(attn_out_concat, self.W_out)

        torque_norm = float(np.linalg.norm(self.omega_state))

        return output, {
            "torque_norm": torque_norm,
            "R_mat": R_mat,
            "attn_weights": attn_weights
        }


class RealTimeMultimodalPhaseLockInference:
    """
    Real-time Latent Recalibration for Multimodal Embedding Streams.
    Measures cross-modal phase interference torque Q_cross between text hidden states
    and multimodal sensory features, updating so(d) rotors to steer attention and KV cache.
    """
    def __init__(self, embed_dim: int = 64, gamma: float = 3.0, beta: float = 0.2):
        self.d = embed_dim
        self.gamma = gamma
        self.beta = beta

        # so(d) Lie algebra state buffer: (d, d)
        self.omega = np.zeros((self.d, self.d), dtype=np.float64)

        # Cross-modal curvature tensor
        np.random.seed(42)
        self.W_cross = np.random.randn(self.d, self.d) * (1.0 / math.sqrt(embed_dim))

    def compute_cross_torque(self, H_text: np.ndarray, H_modal: np.ndarray) -> np.ndarray:
        """
        H_text:  (Batch, Seq_T, d)
        H_modal: (Batch, Seq_M, d)
        Computes skew-symmetric cross-modal torque Q_cross in so(d).
        """
        # Cross-modal projection via bilinear curvature tensor W_cross: (Batch, Seq_T, Seq_M)
        proj = np.matmul(np.matmul(H_text, self.W_cross), H_modal.swapaxes(-1, -2))
        proj_activation = np.tanh(proj)

        # Bilinear cross-covariance matrix Cov in (d, d)
        # Sum over sequence lengths L_T and L_M using bilinear alignment
        # H_text_trans: (Batch, d, Seq_T) @ proj_activation: (Batch, Seq_T, Seq_M) @ H_modal: (Batch, Seq_M, d) -> (Batch, d, d)
        H_text_trans = H_text.swapaxes(-1, -2)
        Cov_batch = np.matmul(np.matmul(H_text_trans, proj_activation), H_modal)
        Cov = np.mean(Cov_batch, axis=0) / max(H_text.shape[1] * H_modal.shape[1], 1)

        Q_cross = Cov - Cov.T
        return Q_cross

    def step_recalibrator(self, H_text: np.ndarray, H_modal: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Computes Lie algebra rotor R_mat in SO(d) to recalibrate multimodal latents.
        """
        Q_cross = self.compute_cross_torque(H_text, H_modal)
        dOmega = self.gamma * Q_cross - self.beta * self.omega
        self.omega += dOmega * dt

        # Skew-symmetric guarantee
        Omega_skew = 0.5 * (self.omega - self.omega.T)

        # Matrix exponential via Padé / Scaling and Squaring
        norm = np.linalg.norm(Omega_skew, ord=1)
        if norm == 0:
            return np.eye(self.d, dtype=np.float64)

        s = max(0, int(np.ceil(np.log2(norm))))
        A = (0.5 * dt * Omega_skew) / (2**s)

        I_mat = np.eye(self.d, dtype=np.float64)
        inv_term = np.linalg.inv(I_mat - A)
        res = np.matmul(inv_term, I_mat + A)
        for _ in range(s):
            res = np.matmul(res, res)
        return res

    def apply_phase_lock_to_kv(
        self,
        Key_cache: np.ndarray,
        Value_cache: np.ndarray,
        R_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies phase lock rotor rotation R_mat to KV cache tensors.
        Key_cache: (Batch, Heads, Seq_Len, Head_Dim)
        """
        K_shape = Key_cache.shape
        head_dim = K_shape[-1]
        K_flat = Key_cache.reshape(-1, head_dim)

        R_sub = R_mat[:head_dim, :head_dim]
        K_locked = np.matmul(K_flat, R_sub.T).reshape(K_shape)

        return K_locked, Value_cache
