
import sys
import os
import time

# 1. Path Unification
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def verify_functional_doctrines():
    print("🌀 [VERIFICATION] Testing Functional Cognitive Integration...")
    
    # 1. Setup Monad
    soul = SeedForge.forge_soul("TestElysia")
    monad = SovereignMonad(soul)
    
    # 2. Verify Double Helix Ownership
    if hasattr(monad, 'helix'):
        print("✅ Success: SovereignMonad owns a DoubleHelixEngine.")
    else:
        print("❌ Failure: SovereignMonad is missing DoubleHelixEngine.")
        return

    # 3. Simulate Cycles and check Interference -> Resonance modulation
    print("\n--- Phase 1: Rotor Interference Influence ---")
    interference_readings = []
    resonance_readings = []
    
    for i in range(10):
        # We manually excite the helix to see changes
        monad.helix.modulate(1.0)
        report = monad.pulse(dt=0.1)
        
        # We need to peek into the monad's internal state for verification
        interf = monad.rotor_state.get('interference', 0.0)
        # Note: report might be None if autonomy trigger isn't reached, 
        # so we look at monad.engine.pulse report via a mock or actual call
        
        interference_readings.append(interf)
        print(f"Cycle {i}: Interference={interf:.3f}")
        time.sleep(0.1)

    # 4. Verify Vocabulary (Many-Worlds)
    print("\n--- Phase 2: Many-Worlds Vocabulary Integration ---")
    # We force an autonomy trigger to get a narrative
    monad.wonder_capacitor = 100.0 # Force trigger
    
    # Force specific conditions for vocabulary check
    # Many-Worlds: Low intensity, low resonance
    monad.desires['curiosity'] = 10.0 # Low intensity
    # We mock the engine report to bypass random logic if needed, 
    # but let's try a natural pulse first.
    
    action = monad.autonomous_drive()
    print(f"Narrative Output: \"{action.get('narrative', '...')}\"")
    
    narrative = action.get('narrative', "").lower()
    keywords = ["quantum sea", "many worlds", "miracle", "collapse", "void"]
    found = [k for k in keywords if k in narrative]
    
    if found:
        print(f"✅ Success: Found Many-Worlds vocabulary: {found}")
    else:
        # Retry with different conditions (Collapse state)
        monad.wonder_capacitor = 100.0
        monad.desires['curiosity'] = 150.0 # High intensity
        action = monad.autonomous_drive()
        print(f"Narrative Output (Collapse): \"{action.get('narrative', '...')}\"")
        found = [k for k in keywords if k in action.get('narrative', "").lower()]
        if found:
            print(f"✅ Success: Found Many-Worlds vocabulary: {found}")
        else:
            print("❌ Failure: Many-Worlds vocabulary not detected in the narrative Loom.")

    print("\n[VERIFICATION COMPLETE]")

if __name__ == "__main__":
    verify_functional_doctrines()
