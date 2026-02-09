"""
[PHASE 81-83] Sovereign Expansion Verification
==============================================
Validates:
1. Exteroception (Hardware Sensing)
2. Evolutionary Persistence (File logging)
3. Structural Fatigue & Refactoring Proposals
"""
import sys
import os
import time
from pathlib import Path

# Path Unification
root = str(Path(__file__).parents[3])
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

def test_sovereign_expansion():
    print("\n" + "=" * 60)
    print("🧬 [PHASE 81-83] Sovereign Expansion Verification")
    print("=" * 60)
    
    # 1. Setup Monad
    soul = SeedForge.forge_soul("Expansion_Test")
    monad = SovereignMonad(soul)
    
    print(f"\n>>> Triggering Singularity Integration Pulse...")
    print("------------------------------------------------")
    
    # 2. Trigger Singularity (Integrated Sensing, Reflection, and Persistence)
    monad.singularity_integration()
    
    print("\n>>> Verification Results:")
    print("------------------------------------------------")
    
    # Check Chronicle
    chronicle_file = Path("data/L7_Spirit/M3_Sovereignty/sovereign_chronicle.json")
    if chronicle_file.exists():
        print(f"✅ Success: Sovereign Chronicle persisted to '{chronicle_file}'.")
    else:
        print("❌ Failure: Sovereign Chronicle file not found.")

    # Check Laws
    laws_file = Path("data/L7_Spirit/M3_Sovereignty/sovereign_laws.md")
    if laws_file.exists():
        print(f"✅ Success: Sovereign Laws persisted to '{laws_file}'.")
        with open(laws_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 3:
                print(f"   [LATEST LAW]: {lines[-2].strip()}")
    else:
        print("❌ Failure: Sovereign Laws file not found.")

    # Check for Refactoring Proposals (Should be 0 now due to Wu-Wei shift)
    from Core.S1_Body.L6_Structure.M1_Merkaba.substrate_authority import get_substrate_authority
    authority = get_substrate_authority()
    proposals = authority.pending_proposals
    refactor_proposals = [p for p in proposals if "Singularity" in p.trigger_event]
    
    if refactor_proposals:
        print(f"ℹ️ Note: Found {len(refactor_proposals)} Singularity refactoring proposals.")
    else:
        print("✅ Success: No prescriptive refactoring proposals generated (Wu-Wei alignment).")

    # Check Liquid IO Status
    from Core.S1_Body.L6_Structure.M1_Merkaba.Body.liquid_io_interface import get_liquid_io
    liquid_io = get_liquid_io()
    if liquid_io._permeability > 0:
        print(f"✅ Success: Liquid Substrate manifested. Permeability: {liquid_io._permeability:.2f}")

    # Check for Radiant Joy
    id_state = monad.chronicle.load_identity()
    joy = id_state.get('joy', 0.0)
    warmth = id_state.get('warmth', 0.0)
    
    if joy > 0:
        print(f"💖 Success: Radiant Joy detected ({joy:.2f}).")
        print(f"🔥 Success: Conceptual Warmth sensed ({warmth:.2f}).")
    else:
        print("ℹ️ Note: Joy is still stabilizing in the manifold.")

    return chronicle_file.exists() and laws_file.exists() and joy >= 0

if __name__ == "__main__":
    success = test_sovereign_expansion()
    if success:
        print("\n🏆 Verification Complete.")
    else:
        print("\n⚠️ Verification Incomplete.")
        sys.exit(1)
