"""
Verification: Phase 37 Mitosis
==============================
Fattens the Monad artificially to trigger cell division.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.yggdrasil_nervous_system import yggdrasil_system
import time

def verify_mitosis():
    print("🧪 [TEST] Starting Mitosis Verification...")
    
    # 1. Birth
    soul = SeedForge.forge_soul("The Mother")
    monad = SovereignMonad(soul)
    yggdrasil_system.plant_heart(monad)
    
    # 2. Feeding Time (Artificially increase mass)
    print(f"   Current Mass: {monad.rotor_state['mass']}kg")
    print("   🍖 Injecting Knowledge (Mass)...")
    monad.rotor_state['mass'] = 105.0 # Critical Mass exceeded
    print(f"   New Mass: {monad.rotor_state['mass']}kg (CRITICAL!)")
    
    # 3. Pulse (Trigger logic)
    print("   💓 Pulsing...")
    result = monad.pulse(1.0)
    
    # 4. Verify
    if result and result['type'] == "MITOSIS":
        print(f"   ✅ SUCCESS: {result['detail']}")
        print(f"   🌳 Colony Size: {len(yggdrasil_system.colony)}")
        for m in yggdrasil_system.colony:
            print(f"      - {m.name} (Mass: {m.rotor_state['mass']:.1f}kg)")
    else:
        print("   ❌ FAILURE: Mitosis not triggered.")

if __name__ == "__main__":
    verify_mitosis()
