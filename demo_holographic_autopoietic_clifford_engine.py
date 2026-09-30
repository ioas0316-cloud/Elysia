"""
Demo: Holographic Autopoietic Clifford Engine (demo_holographic_autopoietic_clifford_engine.py)
==========================================================================================
Demonstrates the full holographic cognitive pipeline:
1. Transformer embedding vector -> Cl(3,0) Clifford Multivector projection
2. Clifford Phase-Coherence Attention & Phase-Locking Gate
3. Spontaneous Symbol Emergence & Dynamic Compatibility Matrix Expansion
4. MERA Hierarchical RG Coarse-Graining Flow
5. Non-Equilibrium Spacetime Field Dynamics & Dual Action Engine Wave Emission
"""

import torch
from core.physics.holographic_clifford_engine import HolographicCognitiveEngine


def main():
    print("=" * 80)
    print("DEMO: Holographic Autopoietic Clifford Engine Pipeline")
    print("=" * 80)

    device = 'cpu'
    batch_size = 2
    seq_len = 16
    d_model = 128
    num_heads = 4

    print(f"\n1. Initializing Engine (d_model={d_model}, num_heads={num_heads})...")
    engine = HolographicCognitiveEngine(d_model=d_model, num_heads=num_heads, device=device)

    # Simulated Transformer Output Embeddings
    x_emb = torch.randn(batch_size, seq_len, d_model)
    print(f"Input Transformer Embedding Tensor: {x_emb.shape}")

    # Forward Step 1
    print("\n2. Executing Forward Pass Step 1 (Normal Alignment State)...")
    res1 = engine(x_emb)

    print(f"   - Multivector Field Shape    : {res1['psi_multivector'].shape}")
    print(f"   - Attended Multivector Shape : {res1['psi_attended'].shape}")
    print(f"   - Phase Attention Weights    : {res1['attn_weights'].shape}")
    print(f"   - Symbol Emergence Sprouted? : {res1['emergence']['sprouted']}")
    print(f"   - Mismatch Score             : {res1['emergence']['mismatch_score']:.4f}")
    print(f"   - Macro Field Shape (MERA-RG): {res1['macro_field'].shape}")
    print(f"   - Causal Potential Energy V  : {res1['causal_potential']:.6f}")

    # Forward Step 2: Inject extreme orthogonal noise to force symbol emergence
    print("\n3. Executing Forward Pass Step 2 (High Mismatch / Tension Injection)...")
    x_orthogonal = torch.randn(batch_size, seq_len, d_model) * 15.0
    res2 = engine(x_orthogonal)

    print(f"   - Mismatch Score             : {res2['emergence']['mismatch_score']:.4f}")
    print(f"   - Symbol Emergence Sprouted? : {res2['emergence']['sprouted']}")
    print(f"   - New Number of Tiles        : {res2['emergence']['new_num_tiles']}")
    print(f"   - Compatibility Matrix Shape : {res2['emergence']['compatibility_matrix'].shape}")

    # Dual Action Wave Emission
    print("\n4. Executing Dual Action Engine (Expansion Phase Emission Wave)...")
    emitted_wave = res2['emitted_wave']
    print(f"   - Emitted Causal Wave Shape  : {emitted_wave.shape}")
    print(f"   - Emitted Wave Norm Mean     : {torch.mean(torch.norm(emitted_wave, dim=-1)).item():.4f}")

    print("\n" + "=" * 80)
    print("Holographic Autopoietic Clifford Engine Demo Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
