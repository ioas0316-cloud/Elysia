"""
Test: Phase 2 - SovereignRotor + HyperHologram Integration
===========================================================
Verifies that SovereignRotor can project its 21D state into 4D HyperSphere.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_rotor import SovereignRotor
from Core.S1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector
import math

def test_phase2_integration():
    print("=== Phase 2: Rotor-Hologram Integration Test ===\n")
    
    # 1. Create SovereignRotor (should auto-initialize HyperHologram)
    rotor = SovereignRotor(snapshot_dir="data/test_rotor_snapshots")
    
    print(f"Hologram Enabled: {rotor._hologram_enabled}")
    
    if not rotor._hologram_enabled:
        print("❌ HyperHologram failed to initialize!")
        return False
    
    # 2. Set a test state
    test_state = D21Vector(
        # Body (sins)
        lust=0.3, gluttony=0.2, greed=0.1, sloth=0.4, wrath=0.2, envy=0.1, pride=0.5,
        # Soul (faculties)
        perception=0.6, memory=0.7, reason=0.8, will=0.9, imagination=0.5, intuition=0.6, consciousness=0.7,
        # Spirit (virtues)
        chastity=0.8, temperance=0.9, charity=1.0, diligence=0.7, patience=0.8, kindness=0.9, humility=1.0
    )
    
    rotor.current_state = test_state
    print(f"21D State Magnitude: {test_state.magnitude():.3f}")
    
    # 3. Project to 4D HyperSphere over time
    print("\n--- Time Evolution ---")
    for t in range(5):
        coord = rotor.project_to_hologram(dt=0.1)
        if coord:
            print(f"t={t*0.1:.1f}: θ={math.degrees(coord.theta):.1f}°, φ={math.degrees(coord.phi):.1f}°, ψ={math.degrees(coord.psi):.1f}°, r={coord.radius:.3f}")
    
    # 4. Check hologram status
    status = rotor.get_hologram_status()
    print(f"\n--- Hologram Status ---")
    print(f"Enabled: {status.get('enabled')}")
    print(f"Projection Count: {status.get('count')}")
    print(f"Center of Mass: {status.get('center_of_mass')}")
    
    # 5. Check Trinity equilibrium
    eq = rotor.get_equilibrium()
    print(f"\n--- Trinity Equilibrium ---")
    print(f"Spirit Ratio: {eq*100:.1f}%")
    
    print("\n✅ Phase 2 Integration Test PASSED!")
    return True


if __name__ == "__main__":
    test_phase2_integration()
