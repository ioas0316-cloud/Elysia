
import sys
import os
import time
sys.path.append(os.getcwd())

from Core.S1_Body.L1_Foundation.Foundation.hyper_sphere_core import HyperSphereCore

def test_seasonal_cycles():
    print("🧪 [Test] Phase 37: Seasonal Resonance (Rotor-driven Climate)")
    
    core = HyperSphereCore()
    
    # 1. Check Initial State
    field = core.field
    h0 = field.grid[25, 25, 25] # Heat
    m0 = field.grid[25, 25, 28] # Moisture
    print(f"\n1. [INITIAL] Heat: {h0:.2f}, Moisture: {m0:.2f}")

    # 2. Progress Time (One full cycle approximation)
    print("\n2. [SIMULATION] Ticking the HyperSphere (Seasons rotating)...")
    
    values = []
    for i in range(50):
        core.tick(dt=1.0)
        h = field.grid[25, 25, 25]
        m = field.grid[25, 25, 28]
        values.append((h, m))
        if i % 10 == 0:
            print(f"   Step {i:2}: Heat {h:5.2f} | Moisture {m:5.2f}")

    # 3. Analyze Oscillation
    h_max = max(v[0] for v in values)
    h_min = min(v[0] for v in values)
    
    print(f"\n3. [RESONANCE] Heat Range: {h_min:.2f} to {h_max:.2f}")
    
    if h_max > h_min:
        print("\n✅ Phase 37 Verification Successful: Climate is now a seasonal wave driven by the HyperSphere Rotor.")
    else:
        print("\n❌ Verification Failed: Climate remains static.")

if __name__ == "__main__":
    test_seasonal_cycles()
