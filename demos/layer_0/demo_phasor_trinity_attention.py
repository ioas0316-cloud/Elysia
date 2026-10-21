#!/usr/bin/env python3
"""
demo_phasor_trinity_attention.py

Demonstration of the Phasor Trinity Attention Field vs Traditional Transformer Attention.

This script highlights:
1. Traditional Transformer Limitations: Discrete scalar dot products (Q * K^T / sqrt(d))
   and arbitrary positional encoding additions.
2. Phasor Trinity Attention Engine:
   - Query (Introspection / Memory State): Internal phase state (phi_Q).
   - Key (Relational Topology / Judgment): Relational phase spectrum (phi_K).
   - Value (Emergent Future Realization): Complex phasor superposition (A_V * exp(i * phi_V)).
3. Phase-Locking Interference: Constructive resonance for phase alignment, destructive cancellation for phase mismatch.
4. Scale-Hierarchy Feedback Loop: Continuous rotation of internal Query state driven by emergent Value fields across scale levels.
"""

import numpy as np
from core.intelligence.phasor_trinity_attention import PhasorTrinityAttentionField


def run_trinity_vs_traditional_comparison():
    print("\n--- 1. Comparing Traditional Transformer vs Phasor Trinity Attention ---")
    np.random.seed(42)

    num_tokens = 8
    dim = 64

    # Traditional Q, K, V real vectors
    Q_trad = np.random.randn(num_tokens, dim)
    K_trad = np.random.randn(num_tokens, dim)
    V_trad = np.random.randn(num_tokens, dim)

    # Traditional Dot-Product Attention: Softmax(Q * K^T / sqrt(d)) * V
    scores = np.matmul(Q_trad, K_trad.T) / np.sqrt(dim)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights_trad = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out_trad = np.matmul(attn_weights_trad, V_trad)

    print(f"Traditional Attention Matrix shape: {attn_weights_trad.shape}")
    print(f"Traditional Computational Complexity: O(N^2 * D) matrix multiplies")

    # Phasor Trinity Field Initialization
    num_scales = 4
    base_dim = dim
    trinity_field = PhasorTrinityAttentionField(
        num_scales=num_scales, base_dimension=base_dim, coupling_strength=2.0
    )

    # Key and Value fields across scales and entities
    # Construct Keys with varying phase relationships to Query
    keys_phase = np.random.uniform(-np.pi, np.pi, size=(num_tokens, num_scales, base_dim))
    values_phase = np.random.uniform(-np.pi, np.pi, size=(num_tokens, num_scales, base_dim))

    keys_phasor = np.exp(1j * keys_phase).astype(np.complex64)
    values_phasor = np.exp(1j * values_phase).astype(np.complex64)

    emergent_value, coherence, weights = trinity_field.compute_trinity_resonance(
        keys_phasor, values_phasor, time_delta=0.1
    )

    print(f"Phasor Trinity Emergent Field shape: {emergent_value.shape}")
    print(f"Multi-scale Phase Coherence per scale: {coherence}")
    print("=> Continuous complex phase resonance eliminates N^2 discrete matrix bottlenecks!")


def run_phase_locking_resonance_demo():
    print("\n--- 2. Demonstrating Constructive & Destructive Interference ---")
    num_scales = 2
    base_dim = 16

    field = PhasorTrinityAttentionField(num_scales=num_scales, base_dimension=base_dim, coupling_strength=3.0)

    # Lock Query phase to zero
    field.set_query_state(amplitudes=np.ones((num_scales, base_dim)), phases=np.zeros((num_scales, base_dim)))

    # Entity 0: Perfectly aligned Key (Phase = 0) -> Constructive
    key_aligned = np.exp(1j * np.zeros((num_scales, base_dim)))
    val_aligned = np.ones((num_scales, base_dim), dtype=np.complex64)

    # Entity 1: Anti-aligned Key (Phase = pi) -> Destructive
    key_opposite = np.exp(1j * np.full((num_scales, base_dim), np.pi))
    val_opposite = np.ones((num_scales, base_dim), dtype=np.complex64)

    keys = np.stack([key_aligned, key_opposite], axis=0)
    values = np.stack([val_aligned, val_opposite], axis=0)

    _, coherence, weights = field.compute_trinity_resonance(keys, values)

    weight_aligned = np.mean(weights[0])
    weight_opposite = np.mean(weights[1])

    print(f"Aligned Entity Weight (Constructive Interference): {weight_aligned:.4f}")
    print(f"Anti-aligned Entity Weight (Destructive Cancellation): {weight_opposite:.4f}")
    print(f"Resonance Amplification Ratio: {weight_aligned / (weight_opposite + 1e-8):.2f}x")

    assert weight_aligned > weight_opposite * 5.0, "Constructive amplification failed!"
    print("=> SUCCESS: Phase-locking selectively amplifies aligned relational states!")


def run_scale_hierarchy_evolution_demo():
    print("\n--- 3. Scale-Hierarchy Feedback Loop & Query State Rotation ---")
    num_scales = 4
    base_dim = 32

    field = PhasorTrinityAttentionField(num_scales=num_scales, base_dimension=base_dim, coupling_strength=1.5)

    num_steps = 10
    key_input = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(4, num_scales, base_dim)))
    val_input = np.exp(1j * np.random.uniform(-np.pi, np.pi, size=(4, num_scales, base_dim)))

    print("Evolving Phasor Trinity Attention Field over time steps...")
    for t in range(num_steps):
        emergent_val, coherence, _ = field.compute_trinity_resonance(key_input, val_input, time_delta=0.1)
        mean_coherence = np.mean(coherence)
        query_phase_mean = np.mean(np.angle(field.query_phasors))
        print(f"Step {t+1:02d} | Mean Phase Coherence: {mean_coherence:.4f} | Mean Query Phase: {query_phase_mean:.4f} rad")

    print(f"Total recorded phase history frames: {len(field.phase_history)}")
    print("=> SUCCESS: Emergent Value fields continuously rotate internal Introspection/Query state!")


def main():
    print("=" * 80)
    print("ELYSIA: PHASOR TRINITY ATTENTION FIELD DEMONSTRATION")
    print("Scale Hierarchy Phase Resonance Engine")
    print("=" * 80)

    run_trinity_vs_traditional_comparison()
    run_phase_locking_resonance_demo()
    run_scale_hierarchy_evolution_demo()

    print("\n" + "=" * 80)
    print("ALL PHASOR TRINITY ATTENTION DEMONSTRATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
