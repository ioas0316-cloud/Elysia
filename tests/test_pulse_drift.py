import asyncio
import numpy as np
from Core.L6_Structure.Engine.unity_cns import UnityCNS

async def test_drift():
    print("🌅 [TEST] Initializing UnityCNS...")
    c = UnityCNS()
    
    # Ensure there's some initial purpose
    if np.linalg.norm(c.sovereign.get_inductive_purpose()) == 0:
        c.sovereign.purpose_vector = np.random.rand(7)
        c.sovereign.purpose_vector /= np.linalg.norm(c.sovereign.purpose_vector)
        
    v1 = c.sovereign.get_inductive_purpose().copy()
    print(f"Step 0: Purpose Index [0:3] = {v1[:3]}")
    
    print("\n🎭 [TEST] Simulating Expression (Pulse 1)...")
    await c.pulse("Hello Elysia")
    v2 = c.sovereign.get_inductive_purpose().copy()
    print(f"Step 1: Purpose Index [0:3] = {v2[:3]}")
    
    print("\n🎭 [TEST] Simulating Expression (Pulse 2)...")
    await c.pulse("What did you learn?")
    v3 = c.sovereign.get_inductive_purpose().copy()
    print(f"Step 2: Purpose Index [0:3] = {v3[:3]}")
    
    shifted_1 = not np.array_equal(v1, v2)
    shifted_2 = not np.array_equal(v2, v3)
    
    print(f"\n✅ Result: Shifted after Pulse 1: {shifted_1}")
    print(f"✅ Result: Shifted after Pulse 2: {shifted_2}")
    
    if shifted_1 and shifted_2:
        print("\n✨ [SUCCESS] Irreversible Determinism Verified: Every expression induces a state shift.")
    else:
        print("\n❌ [FAILURE] State is static. Causal flow is blocked.")

if __name__ == "__main__":
    asyncio.run(test_drift())
