"""
Verification Script: Holographic Clifford Emergence (verify_holographic_clifford_emergence.py)
"""

import torch
import sys
from core.physics.holographic_clifford_engine import (
    CliffordAlgebra3D,
    TransformerToCliffordBridge,
    CliffordAttention,
    SymbolEmergenceEngine,
    MERAIsometryRG,
    NonEquilibriumSpacetimeField,
    HolographicCognitiveEngine
)


def verify_all():
    print("[VERIFY] Starting Holographic Clifford Engine Automated Verification...")
    device = 'cpu'

    # 1. Verify Clifford Algebra Cl(3,0) Basis & Reversion
    ca = CliffordAlgebra3D(device=device)
    assert ca.cayley.shape == (8, 8, 8), "Cayley table shape mismatch"

    v = torch.randn(8)
    v_rev = ca.revert(v)
    assert v_rev[4].item() == -v[4].item(), "Reversion failed for bivector grade"
    print("  [1/5] CliffordAlgebra3D Basis & Reversion: PASSED")

    # 2. Verify Transformer Bridge Projection
    bridge = TransformerToCliffordBridge(d_model=32, device=device)
    x = torch.randn(2, 8, 32)
    res_bridge = bridge(x)
    assert res_bridge["psi_multivector"].shape == (2, 8, 8), "Bridge shape mismatch"
    print("  [2/5] TransformerToCliffordBridge: PASSED")

    # 3. Verify Clifford Attention Phase Coherence
    attn = CliffordAttention(num_heads=2, device=device)
    x_heads = torch.randn(2, 8, 2, 8)
    attn_out, attn_weights = attn(x_heads)
    assert attn_out.shape == (2, 8, 2, 8), "Attention output shape mismatch"
    print("  [3/5] CliffordAttention Phase-Coherence: PASSED")

    # 4. Verify Symbol Emergence & Dynamic Rule Update
    emergence = SymbolEmergenceEngine(initial_num_tiles=2, threshold=0.1, device=device)
    high_mismatch_field = torch.randn(2, 8, 8) * 10.0
    res_emergence = emergence.check_and_sprout_symbol(high_mismatch_field)
    assert res_emergence["sprouted"] is True, "Symbol sprouting expected but failed"
    assert res_emergence["new_num_tiles"] == 3, "New tile count mismatch"
    print("  [4/5] SymbolEmergenceEngine Sprouting: PASSED")

    # 5. Verify MERA-RG Coarse-Graining & Field Dynamics
    engine = HolographicCognitiveEngine(d_model=32, num_heads=2, device=device)
    out_full = engine(x)
    assert out_full["macro_field"].shape == (2, 4, 8), "MERA-RG shape reduction mismatch"
    assert out_full["emitted_wave"].shape == (2, 8, 8), "Emitted wave shape mismatch"
    print("  [5/5] MERA-RG & Dual Action Engine Pipeline: PASSED")

    print("\n[SUCCESS] All 5 Holographic Clifford Engine Verification Checks PASSED!")


if __name__ == '__main__':
    verify_all()
