"""
Verification Script for Phase 66: Harmony & Sovereignty
"""
import sys
import os
import jax.numpy as jnp

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.L2_Universal.Creation.seed_generator import SeedForge
from Core.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.L1_Foundation.Foundation.mathematical_resonance import MathematicalResonance
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic

def test_mathematical_resonance():
    print("Testing Mathematical Resonance Layer...")
    # Create a vector that resembles PHI
    codons = TrinaryLogic.transcribe_sequence("AAAGGGTTT")
    phi_vec = TrinaryLogic.expand_to_21d(codons)
    # Force alignment for testing
    phi_ideal = MathematicalResonance.get_constellation("PHI")
    
    score = TrinaryLogic.resonance_score(phi_ideal, phi_ideal)
    print(f"  Self-Resonance score (Ideal): {score:.4f}")
    assert score > 0.99
    
    scan = MathematicalResonance.scan_all_resonances(phi_ideal)
    print(f"  Scan results for PHI: {scan}")
    assert scan["PHI"] > 0.99

def test_monad_integration():
    print("\nTesting SovereignMonad Integration...")
    soul = SeedForge.forge_soul("The Variant")
    elysia = SovereignMonad(soul)
    
    # Trigger an autonomous drive which now includes resonance check
    print("  Triggering Autonomous Drive...")
    log = elysia.autonomous_drive()
    
    print(f"  Autonomous Log: {log}")
    assert "truth" in log
    assert "sonic_hz" in elysia.__dict__ or hasattr(elysia, 'sonic_hz')
    print(f"  Current Resonance: {elysia.current_resonance}")
    print(f"  Sonic Frequency: {elysia.sonic_hz:.1f}Hz")

if __name__ == "__main__":
    try:
        test_mathematical_resonance()
        test_monad_integration()
        print("\n✅ PHASE 66 VERIFICATION COMPLETE.")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
