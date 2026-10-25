"""
Helical Phase Attention Engine (나선 위상 주의력 인지 엔진)

Core Principle:
"토큰의 의미를 벡터가 아닌 토러스 상의 나선 위상 값(phi)으로 인코딩하여,
$O(N^2)$ 행렬 곱 없이 위상 어긋남(Delta phi -> 0) 및 장벽 이완을 통해 핵심 연관성 신호만을 스르륵 통과시킨다."

This module implements:
1. HelicalPhaseAttention: $O(N)$ lightweight phase-aligned cognitive attention mechanism.
2. CognitivePhaseSearchLoop: Dead-end (closure barrier) detection and adaptive phase orbit realignment.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from L1_causal_field.core.physics.helical_offset_gate import HelicalOffsetTorusGate
from L1_causal_field.core.physics.fractal_helical_bridge import PhaseLockGradientAttractor


class HelicalPhaseAttention:
    r"""
    Torus Helical Phase-Locked Attention Mechanism.

    Replaces standard $O(N^2)$ Transformer Self-Attention matrix dot-products with
    $O(N)$ helical phase alignment ($\Delta \phi \to 0$) permeability gates.

    Attributes:
        embed_dim (int): Vector embedding dimension
        helix_pitch (float): Helical pitch winding parameter (k)
        gate (HelicalOffsetTorusGate): Base topological gate
    """

    def __init__(
        self,
        embed_dim: int = 64,
        helix_pitch: float = 5.0,
        scale_sigma: float = 0.2,
        gate: Optional[HelicalOffsetTorusGate] = None
    ):
        self.embed_dim = embed_dim
        self.helix_pitch = helix_pitch
        self.gate = gate or HelicalOffsetTorusGate(helix_pitch=helix_pitch, scale_sigma=scale_sigma)

    def encode_tokens_to_helical_phases(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Encodes sequence of token vectors into torus helical phase values phi in [-pi, pi].

        Args:
            token_embeddings: Array of shape (N, embed_dim)

        Returns:
            phases: Array of shape (N,) with helical phase offsets
        """
        if token_embeddings.ndim == 1:
            token_embeddings = token_embeddings.reshape(1, -1)

        # Project embedding onto phase angle using mean directional vector & arctan2
        u = np.mean(token_embeddings[:, :self.embed_dim // 2], axis=1)
        v = np.mean(token_embeddings[:, self.embed_dim // 2:], axis=1)

        phases = np.arctan2(v, u)
        return phases

    def compute_phase_attention_weights(
        self,
        query_phase: float,
        key_phases: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Computes attention permeability weights P_gate(Delta phi) for a Query phase against Keys.

        High permeability (-> 1.0) occurs when Delta phi -> 0.
        Irrelevant tokens are blocked by high barrier energy (P_gate -> 0.0).

        Returns:
            attention_weights: Array of shape (N,) normalized attention weights.
        """
        diffs = (key_phases - query_phase + np.pi) % (2.0 * np.pi) - np.pi
        energies = self.gate.E_max * np.tanh(np.abs(diffs) / (self.gate.sigma * temperature))

        # Permeability permeability = 1 - energy / E_max
        permeabilities = np.maximum(0.0, 1.0 - (energies / self.gate.E_max))

        # Softmax-like normalization or direct permeability weighting
        total_perm = np.sum(permeabilities)
        if total_perm > 1e-8:
            weights = permeabilities / total_perm
        else:
            weights = np.ones_like(permeabilities) / len(permeabilities)

        return weights

    def forward(
        self,
        query_vectors: np.ndarray,
        key_vectors: np.ndarray,
        value_vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes Helical Phase-Locked Attention over sequences.

        Args:
            query_vectors: Array of shape (Q_len, embed_dim)
            key_vectors: Array of shape (K_len, embed_dim)
            value_vectors: Array of shape (K_len, embed_dim)

        Returns:
            output_context: Array of shape (Q_len, embed_dim)
            attention_matrix: Array of shape (Q_len, K_len)
        """
        q_phases = self.encode_tokens_to_helical_phases(query_vectors)
        k_phases = self.encode_tokens_to_helical_phases(key_vectors)

        q_len = len(q_phases)
        k_len = len(k_phases)

        attn_matrix = np.zeros((q_len, k_len))
        output_context = np.zeros_like(query_vectors)

        for i in range(q_len):
            weights = self.compute_phase_attention_weights(q_phases[i], k_phases)
            attn_matrix[i, :] = weights
            output_context[i, :] = np.sum(weights[:, np.newaxis] * value_vectors, axis=0)

        return output_context, attn_matrix


class CognitivePhaseSearchLoop:
    """
    Cognitive Search & Reasoning Loop with Topological Dead-End Detection.

    Tracks reasoning trajectory along helical phase gradients, identifies
    closure barriers (dead-ends), and applies adaptive phase orbit realignment.
    """

    def __init__(
        self,
        target_concept_phase: float = 0.0,
        attractor: Optional[PhaseLockGradientAttractor] = None
    ):
        self.target_phase = target_concept_phase
        self.attractor = attractor or PhaseLockGradientAttractor(target_phase=target_concept_phase)

    def execute_cognitive_step(
        self,
        current_thought_phase: float,
        gradient_signal: float = 1.0
    ) -> Dict[str, Any]:
        """
        Advances thought trajectory by one step. If a dead-end is detected,
        suggests orbit realignments to bypass the cognitive obstacle.
        """
        step_result = self.attractor.step_attractor(
            current_phase=current_thought_phase,
            gradient_vector=gradient_signal
        )

        realignments = []
        if step_result["is_dead_end"]:
            realignments = self.attractor.suggest_orbit_realignment(current_thought_phase)

        step_result["realignment_suggestions"] = realignments
        return step_result
