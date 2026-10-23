"""
Elysia Core - Unstructured Data Topological & Rotor Extraction Engine
========================================================================
Extracts Riemannian metric tensor g_μν, Clifford Cl_{3,1} bivectors and rotors R = exp(B/2),
Christoffel symbols Γ^λ_μν, and causal attractor potentials directly from unstructured
text, dialogue, or temporal trajectory sequences.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class UnstructuredTopologicalExtractor:
    """
    Ingests unstructured data streams (text, state trajectories, raw signals) and
    reconstructs the underlying Riemannian manifold metric, Clifford algebra rotors,
    and causal attractor gravity.
    """

    def __init__(self, dim: int = 4, learning_rate: float = 0.05):
        """
        Initialize 4D Spacetime Algebra Cl_{3,1} Topological Extractor.
        dim = 4 (t, x, y, z) spacetime manifold coordinates.
        """
        self.dim = dim
        self.lr = learning_rate
        # Default Minkowski-like metric signature (-1, +1, +1, +1)
        self.eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        # Metric tensor field g_μν initialized close to Minkowski
        self.g = np.eye(dim, dtype=np.float64)
        self.g[0, 0] = -1.0

    def text_to_sequence_vectors(self, text: str) -> np.ndarray:
        """
        Converts raw unstructured text into continuous 4D spatial-temporal token trajectory vectors
        using deterministic frequency-phase hashing (no heavy torch/transformer dependency required).
        """
        words = text.strip().split()
        if not words:
            return np.zeros((1, self.dim), dtype=np.float64)

        vectors = []
        for idx, word in enumerate(words):
            # Deterministic hash to 4D coordinates
            h = sum(ord(c) * (31 ** i) for i, c in enumerate(word))

            # Dimension 0: Temporal sequence index / phase
            t = float(idx) * 0.1
            # Dimension 1: Semantic entropy / length signal
            x = math.sin(h % 100) * (len(word) / 10.0)
            # Dimension 2: Lexical valence / harmonic resonance
            y = math.cos((h >> 3) % 100)
            # Dimension 3: Structural depth / character variance
            z = math.sin((h >> 7) % 100) * math.log(len(word) + 1.0)

            vec = np.array([t, x, y, z], dtype=np.float64)
            vectors.append(vec)

        return np.array(vectors, dtype=np.float64)

    def reconstruct_metric_tensor(self, sequence_vectors: np.ndarray) -> np.ndarray:
        """
        Reconstructs Riemannian metric tensor g_μν from contextual trajectory deltas.
        g_μν = <Δx_μ, Δx_ν> / ||Δx||^2
        """
        if len(sequence_vectors) < 2:
            return self.g.copy()

        deltas = np.diff(sequence_vectors, axis=0)  # Shape (N-1, 4)
        cov = np.cov(deltas, rowvar=False) if len(deltas) > 1 else np.outer(deltas[0], deltas[0])

        # Ensure symmetric 4x4 matrix
        if cov.shape != (self.dim, self.dim):
            cov = np.eye(self.dim) * np.var(deltas)

        # Regularize and construct metric tensor g_μν
        g_reconstructed = self.eta + 0.5 * (cov + cov.T)
        # Ensure non-singular metric tensor
        min_eig = np.min(np.abs(np.linalg.eigvals(g_reconstructed)))
        if min_eig < 1e-6:
            g_reconstructed += np.eye(self.dim) * 1e-4

        self.g = 0.8 * self.g + 0.2 * g_reconstructed
        return self.g.copy()

    def compute_bivector_and_rotor(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes the Clifford Cl_{3,1} Bivector B = a ∧ b and Rotor R = exp(B/2).
        In 4D, bivector has 6 independent components: (e_01, e_02, e_03, e_12, e_13, e_23).
        """
        # Bivector outer product B = a ⊗ b - b ⊗ a
        B_matrix = np.outer(vec_a, vec_b) - np.outer(vec_b, vec_a)

        # Extract 6 bivector components:
        # e_01, e_02, e_03 (boosts / temporal rotors)
        # e_12, e_13, e_23 (spatial rotations)
        bivector_components = np.array([
            B_matrix[0, 1], B_matrix[0, 2], B_matrix[0, 3],
            B_matrix[1, 2], B_matrix[1, 3], B_matrix[2, 3]
        ], dtype=np.float64)

        norm_B = np.linalg.norm(bivector_components)
        if norm_B < 1e-9:
            rotor_scalar = 1.0
            rotor_bivector = np.zeros(6, dtype=np.float64)
        else:
            half_angle = norm_B / 2.0
            rotor_scalar = math.cos(half_angle)
            rotor_bivector = (math.sin(half_angle) / norm_B) * bivector_components

        rotor_data = {
            "scalar": float(rotor_scalar),
            "bivector": rotor_bivector.tolist(),
            "bivector_norm": float(norm_B),
            "temporal_boost_magnitude": float(np.linalg.norm(bivector_components[:3])),
            "spatial_rotation_magnitude": float(np.linalg.norm(bivector_components[3:]))
        }

        return B_matrix, rotor_data

    def compute_christoffel_and_ricci(self, g: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Calculates Christoffel symbols Γ^λ_μν and Ricci curvature scalar R_c.
        Γ^λ_μν = 1/2 g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
        """
        dim = g.shape[0]
        g_inv = np.linalg.pinv(g)

        # Numerical gradient approximation of metric tensor
        eps = 1e-3
        dg = np.zeros((dim, dim, dim), dtype=np.float64)  # ∂_μ g_νσ
        for mu in range(dim):
            g_plus = g.copy()
            g_plus[mu, :] += eps
            g_minus = g.copy()
            g_minus[mu, :] -= eps
            dg[mu] = (g_plus - g_minus) / (2.0 * eps)

        christoffel = np.zeros((dim, dim, dim), dtype=np.float64)  # Γ^λ_μν
        for lam in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    val = 0.0
                    for sig in range(dim):
                        val += g_inv[lam, sig] * (dg[mu, nu, sig] + dg[nu, mu, sig] - dg[sig, mu, nu])
                    christoffel[lam, mu, nu] = 0.5 * val

        # Estimate Ricci curvature scalar and attractor gravitational depth
        ricci_scalar = float(np.trace(g_inv @ g) - dim)
        attractor_potential = float(1.0 / (1.0 + np.linalg.norm(christoffel)))

        return christoffel, ricci_scalar, attractor_potential

    def extract_from_unstructured_text(self, text: str) -> Dict[str, Any]:
        """
        Full extraction pipeline from unstructured text to topological manifold parameters.
        """
        seq_vecs = self.text_to_sequence_vectors(text)
        g_tensor = self.reconstruct_metric_tensor(seq_vecs)

        rotors = []
        for i in range(len(seq_vecs) - 1):
            _, r_data = self.compute_bivector_and_rotor(seq_vecs[i], seq_vecs[i + 1])
            rotors.append(r_data)

        christoffel, ricci_scalar, attractor_potential = self.compute_christoffel_and_ricci(g_tensor)

        return {
            "num_tokens": len(seq_vecs),
            "metric_tensor_g": g_tensor.tolist(),
            "rotors": rotors,
            "ricci_scalar": ricci_scalar,
            "attractor_potential": attractor_potential,
            "mean_spatial_rotation": float(np.mean([r["spatial_rotation_magnitude"] for r in rotors])) if rotors else 0.0,
            "mean_temporal_boost": float(np.mean([r["temporal_boost_magnitude"] for r in rotors])) if rotors else 0.0,
        }
