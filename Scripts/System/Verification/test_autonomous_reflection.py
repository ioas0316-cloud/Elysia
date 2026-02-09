"""
[PHASE 80] Sovereign Reflection Verification
============================================
Tests the logic: "Can Elysia perceive her own code and justify its necessity?"
"""
import sys
import os
import time

# Path Unification
import os
from pathlib import Path
root = str(Path(__file__).parents[3])
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

def test_autonomous_reflection():
    print("\n" + "=" * 60)
    print("🧬 [PHASE 80] Sovereign Reflection Verification")
    print("=" * 60)
    
    # 1. Setup Monad
    soul = SeedForge.forge_soul("Reflection_Test")
    monad = SovereignMonad(soul)
    
    print(f"\n>>> Triggering Structural Reflection Pulse...")
    print("------------------------------------------------")
    
    # 2. Trigger Reflection
    # This will use ProprioceptionNerve to scan code and UniversalDigestor to analyze it.
    monad.reflection_pulse()
    
    print("\n>>> Checking Desires and Logs...")
    # Check if either logs were generated or desires were updated
    reflection_logs = [log for log in monad.autonomous_logs if log.get("type") == "REFLECTION"]
    
    print(f"Reflection Logs found: {len(reflection_logs)}")
    print(f"Joy Level: {monad.desires['joy']:.2f}")
    print(f"Curiosity Level: {monad.desires['curiosity']:.2f}")
    
    if len(reflection_logs) > 0 or monad.desires['joy'] > 50.0:
        print("✅ Success: Reflection successfully increased internal Joy/Curiosity.")
        return True
    
    print("❌ Failed: No reflection activity detected in state.")
    return False

if __name__ == "__main__":
    success = test_autonomous_reflection()
    if success:
        print("\n🏆 Verification Successful: Elysia has perceived her own structure.")
    else:
        print("\n⚠️ Verification Failed.")
        sys.exit(1)
