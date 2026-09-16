"""
Demonstration Script: Causal DNA Helix Engine
=============================================
This script demonstrates:
1. Logarithmic dynamic range compression (log(A*B) = log A + log B) & octave quantization.
2. Double helix complementary duality: continuous variable wave strand vs invariant identity shell strand.
3. 2x2 4-Base Atomic Causal Tensor & Kronecker Product expansion.
4. Multi-LOD octave dial indexing and block slicing.
5. Korean Phoneme (한글 자모 '가', '한글') Phase Manifold & Context Plane Transformation.
"""

import math
import torch
from core.topology.causal_dna_helix_engine import (
    CausalDNAHelixEngine,
    WordPhaseManifold,
    LogarithmicMapper,
    DoubleHelixTopology,
    CausalAtomicTensor,
    MultiLODDialIndexer
)


def encode_jamo_to_tensor(jamo_char: str) -> torch.Tensor:
    """Encode Korean phoneme jamo into a 16-dimensional FP32 tensor."""
    jamo_map = {
        'ㄱ': [1.0, 0.0, 0.0, 0.5,  0.0, 1.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,  0.5, 0.5, 0.0, 1.0],
        'ㄴ': [0.0, 1.0, 0.0, 0.2,  1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.2, 0.8, 0.0, 0.5],
        'ㄷ': [0.0, 1.0, 0.0, 0.8,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,  0.8, 0.2, 0.0, 1.0],
        'ㅏ': [0.0, 0.0, 1.0, 1.0,  0.0, 0.0, 1.0, 0.0,  1.0, 1.0, 0.0, 0.0,  1.0, 0.0, 1.0, 0.0],
        'ㅣ': [0.0, 0.0, 1.0, 0.1,  0.0, 0.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0,  0.1, 0.9, 0.0, 0.0]
    }
    vec = jamo_map.get(jamo_char, [0.1] * 16)
    return torch.tensor(vec, dtype=torch.float32)


def main():
    print("=========================================================================")
    print("          Elysia Causal DNA Helix Engine Demonstration Script           ")
    print("=========================================================================\n")

    engine = CausalDNAHelixEngine(vram_budget_mb=2000)

    # 1. Logarithmic Range Compression & Additive Property
    print("--- 1. Logarithmic Dynamic Range Compression ---")
    mapper = LogarithmicMapper()
    val_a = torch.tensor([1e2, 1e4, 1e6], dtype=torch.float32)
    val_b = torch.tensor([1e1, 1e2, 1e3], dtype=torch.float32)

    log_a = mapper.to_log_space(val_a)
    log_b = mapper.to_log_space(val_b)
    log_ab = mapper.to_log_space(val_a * val_b)

    print(f"Value A: {val_a.tolist()}")
    print(f"Value B: {val_b.tolist()}")
    print(f"log(A)  : {log_a.tolist()}")
    print(f"log(B)  : {log_b.tolist()}")
    print(f"log(A*B): {log_ab.tolist()}")
    print(f"Additive check (log(A) + log(B) vs log(A*B)):")
    print(f"Difference: {(log_a + log_b - log_ab).abs().max().item():.6f}\n")

    # 2. Double Helix Duality: Continuous Wave vs Invariant Shell
    print("--- 2. Double Helix Duality (Variable Wave vs Invariant Shell) ---")
    signal = encode_jamo_to_tensor('ㄱ')
    topo = DoubleHelixTopology(dimension=16)

    wave_0, shell_0 = topo.generate_helix_pair(signal, theta=0.0)
    wave_pi2, shell_pi2 = topo.generate_helix_pair(signal, theta=math.pi / 2)

    print(f"Rotation θ = 0  rad -> Wave sample: {wave_0[:4].tolist()}")
    print(f"Rotation θ = π/2 rad -> Wave sample: {wave_pi2[:4].tolist()}")
    print(f"Invariant Shell θ=0  : {shell_0.mean().item():.4f}")
    print(f"Invariant Shell θ=π/2: {shell_pi2.mean().item():.4f}\n")

    # 3. 4-Base Atomic Causal Tensor & Quarter-Phase Transitions
    print("--- 3. 4-Base Atomic Causal States & Quarter-Phase Transitions ---")
    atomic = CausalAtomicTensor()
    angles = [
        (0.0, "0° (0 rad) - Direct Action (1,0)"),
        (math.pi / 2, "90° (π/2 rad) - Phase-Lock (1,1)"),
        (math.pi, "180° (π rad) - Feedback Action (0,1)"),
        (3 * math.pi / 2, "270° (3π/2 rad) - Orthogonality (0,0)")
    ]

    for theta, label in angles:
        mat = atomic.get_atomic_state_by_phase(theta)
        print(f"▶ {label}:")
        for row in mat.tolist():
            print(f"   {row}")
    print()

    # 4. Multi-LOD Indexing & Word Phase Manifold
    print("--- 4. Multi-LOD Indexing & Word '한글' Phase Manifold ---")
    word_builder = WordPhaseManifold(engine)
    han_thetas = (math.pi / 2, math.pi / 2)
    geul_thetas = (math.pi / 2, math.pi / 2)

    word_noun = word_builder.construct_word_manifold([han_thetas, geul_thetas], context_type="NOUN_FIELD")
    word_verb = word_builder.construct_word_manifold([han_thetas, geul_thetas], context_type="VERB_FIELD")

    print("Word '한글' under NOUN_FIELD Context:")
    for row in word_noun["word_manifold"].tolist():
        formatted_row = [f"{val:6.3f}" for val in row]
        print(f"   {formatted_row}")

    print("\nWord '한글' under VERB_FIELD Context:")
    for row in word_verb["word_manifold"].tolist():
        formatted_row = [f"{val:6.3f}" for val in row]
        print(f"   {formatted_row}")

    # Multi-LOD Slicing Demo
    lod_indexer = MultiLODDialIndexer()
    sliced_lod0 = lod_indexer.slice_by_lod(word_noun["word_manifold"], lod_level=0)
    sliced_lod1 = lod_indexer.slice_by_lod(word_noun["word_manifold"], lod_level=1)

    print(f"\nMulti-LOD Slicing (LOD 0 -> {sliced_lod0.shape}, LOD 1 -> {sliced_lod1.shape})")

    print("\n=========================================================================")
    print("               Demonstration Completed Successfully!                     ")
    print("=========================================================================")


if __name__ == "__main__":
    main()
