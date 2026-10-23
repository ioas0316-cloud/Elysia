"""
Omni-Modal Causal Cognition Engine (OmniModalCognitionEngine)
=============================================================
Modal-Agnostic Omni-Cognition Pipeline supporting Cross-Modal Attractor Convergence.
Fuses heterogeneous modalities S = {S_visual, S_auditory, S_textual, S_proprioceptive, S_intent, S_emotion}
into a unified phase space vector P_omni, passing through System 1 O(1) Intuition / System 2 Reasoning
and Universal Scar Consolidation.
"""

import numpy as np
from typing import Dict, Any, Optional


class OmniModalCognitionEngine:
    """
    Modal-Agnostic Causal Engine supporting multi-sensory cross-modal convergence.
    """

    def __init__(
        self,
        omni_dim: int = 256,
        num_attractors: int = 1024,
        beta: float = 8.0,
        friction_threshold: float = 0.25,
        random_seed: int = 42,
    ):
        self.omni_dim = omni_dim
        self.beta = beta
        self.tau = friction_threshold

        np.random.seed(random_seed)
        # Unified Attractor Bank (Unit normalized)
        raw_bank = np.random.randn(num_attractors, omni_dim).astype(np.float32)
        norms = np.linalg.norm(raw_bank, axis=1, keepdims=True) + 1e-8
        self.attractor_bank = raw_bank / norms

        # Modality projection weights mapping various input vectors to omni_dim
        self.modality_weights = {
            "visual": np.random.randn(128, omni_dim).astype(np.float32) * 0.1,
            "auditory": np.random.randn(64, omni_dim).astype(np.float32) * 0.1,
            "textual": np.random.randn(64, omni_dim).astype(np.float32) * 0.1,
            "proprioception": np.random.randn(32, omni_dim).astype(np.float32) * 0.1,
            "intent": np.random.randn(16, omni_dim).astype(np.float32) * 0.1,
            "emotion": np.random.randn(16, omni_dim).astype(np.float32) * 0.1,
        }

        # System 2 Reasoning projection weights
        self.sys2_w1 = np.random.randn(omni_dim, omni_dim * 2).astype(np.float32) * 0.1
        self.sys2_w2 = np.random.randn(omni_dim * 2, omni_dim).astype(np.float32) * 0.1

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm < 1e-8 or v_norm < 1e-8:
            return 0.0
        return float(np.dot(u, v) / (u_norm * v_norm))

    def project_stream_to_omni_phase(
        self, stream_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Projects heterogeneous modality streams S = {S_v, S_a, S_t, ...} into single P_omni vector.
        """
        p_omni = np.zeros(self.omni_dim, dtype=np.float32)
        count = 0

        for modality_key, raw_vec in stream_dict.items():
            if raw_vec is None:
                continue
            raw_arr = np.asarray(raw_vec, dtype=np.float32).flatten()
            if modality_key in self.modality_weights:
                W = self.modality_weights[modality_key]
                if raw_arr.shape[0] != W.shape[0]:
                    if raw_arr.shape[0] < W.shape[0]:
                        raw_arr = np.pad(raw_arr, (0, W.shape[0] - raw_arr.shape[0]))
                    else:
                        raw_arr = raw_arr[: W.shape[0]]
                proj = np.dot(raw_arr, W)
                p_omni += proj
                count += 1
            else:
                if raw_arr.shape[0] < self.omni_dim:
                    padded = np.pad(raw_arr, (0, self.omni_dim - raw_arr.shape[0]))
                else:
                    padded = raw_arr[: self.omni_dim]
                p_omni += padded
                count += 1

        if count > 0:
            p_omni /= count
        norm = np.linalg.norm(p_omni)
        if norm > 1e-8:
            p_omni /= norm
        return p_omni

    def process_omni_stream(
        self, stream_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Processes multi-sensory stream through cross-modal attractor convergence.
        """
        p_omni = self.project_stream_to_omni_phase(stream_dict)

        # ----------------------------------------------------
        # Step 1. System 1: Continuous Hopfield O(1) Attractor Recall
        # ----------------------------------------------------
        dots = np.dot(self.attractor_bank, p_omni)
        logits = self.beta * dots
        exp_logits = np.exp(logits - np.max(logits))
        attn_weights = exp_logits / (np.sum(exp_logits) + 1e-8)

        z_intuition = np.dot(attn_weights, self.attractor_bank)

        # ----------------------------------------------------
        # Step 2. Cross-Modal Friction Detection
        # ----------------------------------------------------
        cos_sim = self._cosine_similarity(p_omni, z_intuition)
        delta_p = float(1.0 - cos_sim)
        target_idx = int(np.argmax(attn_weights))

        # ----------------------------------------------------
        # Step 3. Dual Loop Switching & Universal Scar Consolidation
        # ----------------------------------------------------
        if delta_p <= self.tau:
            return {
                "predicted_phase": z_intuition,
                "mode": "System_1 (Cross-Modal O(1) Intuition)",
                "friction": delta_p,
                "target_attractor_idx": target_idx,
            }
        else:
            # System 2 Reasoning
            h = np.dot(p_omni, self.sys2_w1)
            reasoned_scar = np.dot(self._gelu(h), self.sys2_w2)
            final_state = z_intuition + reasoned_scar

            # Universal Scar Consolidation: Shift target attractor towards p_omni
            lr = 0.3 * max(0.2, delta_p - self.tau)
            self.attractor_bank[target_idx] += lr * (p_omni - self.attractor_bank[target_idx])
            norm = np.linalg.norm(self.attractor_bank[target_idx])
            if norm > 1e-8:
                self.attractor_bank[target_idx] /= norm

            return {
                "predicted_phase": final_state,
                "mode": "System_2 (Cross-Modal Reasoning & Universal Scar Consolidated)",
                "friction": delta_p,
                "target_attractor_idx": target_idx,
            }
