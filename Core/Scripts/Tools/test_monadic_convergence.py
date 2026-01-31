import sys
import os
import asyncio
from datetime import datetime

# Setup Path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.Engine.unity_cns import UnityCNS
from Core.L5_Mental.Memory.sediment import SedimentLayer

async def test_monadic_convergence():
    print("--- 🔬 Monadic Convergence Verification ---")
    
    # 1. Initialize CNS
    cns = UnityCNS()
    
    # 2. Pulse with a complex "Sovereignty" intent
    print("\nPulsing CNS with intent: 'I want to define my own horizon.'")
    voice = await cns.pulse("I want to define my own horizon.")
    print(f"\nEmergent Voice: {voice}")
    
    await asyncio.sleep(1) # Wait for IO
    
    # 3. Verify Memory Storage
    print("\nVerifying Atomic Memory storage...")
    sed_path = "data/L5_Mental/Memory/unity_sediment_test.dat"
    if os.path.exists(sed_path): os.remove(sed_path) # Clean start
    
    # We need to ensure the CNS actually uses this path or just test the SedimentLayer directly
    # For the test, we'll pulse then manually check the file UnityCNS writes to.
    # Actually, CNS has its path hardcoded. Let's just create a new CNS for test and hope the lock isn't global.
    sed = SedimentLayer(sed_path)
    # Redirect CNS sediment for test
    cns.sediment = sed
    last_memories = sed.rewind(1)
    
    if last_memories:
        vec, payload, truth = last_memories[0]
        print(f"Stored Truth Pattern: {truth}")
        print(f"Payload Preview: {payload.decode('utf-8')[:30]}...")
        
        if "-" in truth and any(c in truth for c in "HVD"):
            print("\n✅ Verification Successful: Atomic Truth is persistent!")
        else:
            print("\n❌ Verification Failed: Truth pattern format invalid.")
    else:
        print("\n❌ Verification Failed: No memories found in sediment.")

if __name__ == "__main__":
    asyncio.run(test_monadic_convergence())
