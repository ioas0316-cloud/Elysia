"""
Intuition Dual Engine with Scar Consolidator (IntuitionDualEngine)
==================================================================
Continuous Hopfield Network-based O(1) Attractor Basin recall for System 1 (Intuition).
Friction detection (ΔP = 1 - CosineSimilarity).
System 2 reasoning fallback when ΔP > τ with Scar Tensor consolidation into Attractor Bank
for future O(1) intuition.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class ScarConsolidator:
    """
    Consolidates friction experience (Scar Vector) into Attractor Memory Bank.
    Irreversibly warps the attractor landscape so that future similar stimuli
    fall directly into System 1 O(1) intuition basin.
    """

    def __init__(self, base_learning_rate: float = 0.1):
        self.base_learning_rate = base_learning_rate

    def consolidate(
        self,
        attractor_bank: np.ndarray,
        target_idx: int,
        input_signal: np.ndarray,
        scar_vector: np.ndarray,
        friction_excess: float,
    ) -> np.ndarray:
        """
        Updates attractor bank at target_idx by embedding the scar_vector and input_signal.
        """
        lr = self.base_learning_rate * max(0.1, friction_excess)
        # Shift target attractor towards input signal + reasoned scar direction
        target_direction = input_signal + 0.1 * scar_vector
        attractor_bank[target_idx] += lr * (target_direction - attractor_bank[target_idx])

        # Re-normalize attractor vector to maintain unit norm manifold stability
        norm = np.linalg.norm(attractor_bank[target_idx])
        if norm > 1e-8:
            attractor_bank[target_idx] /= norm
        return attractor_bank


class IntuitionDualEngine:
    """
    Dual-Loop System 1 (O(1) Intuition) & System 2 (O(N) Reasoning) Engine.

    Mechanisms:
    - System 1: Continuous Hopfield style energy minimization -> O(1) Attractor recall.
    - Friction Detector: Friction ΔP = 1 - CosineSimilarity(x_causal, z_intuition).
    - System 2 (Deformation Engine): Triggered when ΔP > τ. Computes reasoning delta.
    - ScarConsolidator: Embeds reasoning delta into Attractor Bank for future O(1) intuition.
    """

    def __init__(
        self,
        dim: int = 128,
        num_attractors: int = 512,
        beta: float = 8.0,
        friction_threshold: float = 0.25,
        random_seed: int = 42,
    ):
        self.dim = dim
        self.num_attractors = num_attractors
        self.beta = beta
        self.tau = friction_threshold

        np.random.seed(random_seed)
        # Attractor Memory Bank (Unit normalized)
        raw_bank = np.random.randn(num_attractors, dim).astype(np.float32)
        norms = np.linalg.norm(raw_bank, axis=1, keepdims=True) + 1e-8
        self.attractor_bank = raw_bank / norms

        # System 2 Reasoning projection weights
        self.sys2_w1 = np.random.randn(dim, dim * 2).astype(np.float32) * 0.1
        self.sys2_w2 = np.random.randn(dim * 2, dim).astype(np.float32) * 0.1

        self.scar_consolidator = ScarConsolidator(base_learning_rate=0.15)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm < 1e-8 or v_norm < 1e-8:
            return 0.0
        return float(np.dot(u, v) / (u_norm * v_norm))

    def _system2_reason(self, x: np.ndarray) -> np.ndarray:
        """
        System 2 Deformation Engine: Computes deeper causal reasoning delta.
        """
        h = np.dot(x, self.sys2_w1)
        h_act = self._gelu(h)
        reasoned_delta = np.dot(h_act, self.sys2_w2)
        return reasoned_delta

    def forward(self, x_causal: np.ndarray) -> Dict[str, Any]:
        """
        Processes causal input vector through System 1 / System 2 dual loop.

        Args:
            x_causal: Array of shape (dim,) or (1, dim)

        Returns:
            Dict containing:
                - predicted_state: Final output state
                - mode: 'System_1 (Intuition)' or 'System_2 (Reasoning & Scar Consolidated)'
                - friction: Friction metric ΔP
                - target_attractor_idx: Index of recalled attractor
        """
        x = np.asarray(x_causal, dtype=np.float32).flatten()
        if x.shape[0] != self.dim:
            if x.shape[0] < self.dim:
                x = np.pad(x, (0, self.dim - x.shape[0]))
            else:
                x = x[: self.dim]

        # ----------------------------------------------------
        # Step 1. System 1: Continuous Hopfield O(1) Attractor Recall
        # ----------------------------------------------------
        dots = np.dot(self.attractor_bank, x)  # [num_attractors]
        logits = self.beta * dots
        exp_logits = np.exp(logits - np.max(logits))
        attn_weights = exp_logits / (np.sum(exp_logits) + 1e-8)

        z_intuition = np.dot(attn_weights, self.attractor_bank)

        # ----------------------------------------------------
        # Step 2. Friction Detection (ΔP = 1 - CosineSimilarity)
        # ----------------------------------------------------
        cos_sim = self._cosine_similarity(x, z_intuition)
        delta_p = float(1.0 - cos_sim)

        target_idx = int(np.argmax(attn_weights))

        # ----------------------------------------------------
        # Step 3. Dual Loop Switching & Scar Consolidation
        # ----------------------------------------------------
        if delta_p <= self.tau:
            # System 1: Immediate O(1) Intuition Readout
            return {
                "predicted_state": z_intuition,
                "mode": "System_1 (Intuition)",
                "friction": delta_p,
                "target_attractor_idx": target_idx,
            }
        else:
            # System 2: Reasoning & Scar Consolidation
            reasoned_scar = self._system2_reason(x)
            final_state = z_intuition + reasoned_scar

            # Scar Consolidation: Warps attractor bank at target_idx
            self.attractor_bank = self.scar_consolidator.consolidate(
                self.attractor_bank,
                target_idx=target_idx,
                input_signal=x,
                scar_vector=reasoned_scar,
                friction_excess=delta_p - self.tau,
            )

            return {
                "predicted_state": final_state,
                "mode": "System_2 (Reasoning & Scar Consolidated)",
                "friction": delta_p,
                "target_attractor_idx": target_idx,
            }
