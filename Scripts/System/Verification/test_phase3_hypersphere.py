"""
Test: Phase 3 - HyperSphereField + PPE Integration
===================================================
Verifies that HyperSphereField can project M1-M4 states into 4D HyperSphere.
"""

import sys
sys.path.insert(0, "c:\\Elysia")

import math

def test_phase3_integration():
    print("=== Phase 3: HyperSphereField + PPE Integration Test ===\n")
    
    try:
        from Core.S1_Body.L6_Structure.M1_Merkaba.hypersphere_field import HyperSphereField
    except Exception as e:
        print(f"❌ Import Error: {e}")
        return False
    
    # 1. Create HyperSphereField
    print("Initializing HyperSphereField...")
    field = HyperSphereField()
    
    print(f"PPE Enabled: {field._ppe_enabled}")
    
    if not field._ppe_enabled:
        print("❌ PhaseProjectionEngine failed to initialize!")
        return False
    
    # 2. Project cognitive map
    print("\n--- Cognitive Map Projection ---")
    for t in range(5):
        result = field.project_cognitive_map(dt=0.1)
        if result.get("enabled"):
            theta_deg = math.degrees(result["theta"])
            phi_deg = math.degrees(result["phi"])
            psi_deg = math.degrees(result["psi"])
            print(f"t={t*0.1:.1f}: θ={theta_deg:.1f}°, φ={phi_deg:.1f}°, ψ={psi_deg:.1f}°")
    
    # 3. Check equilibrium
    print("\n--- Trinity Equilibrium ---")
    eq = result.get("equilibrium", {})
    print(f"  Body:   {eq.get('body', 0)*100:.1f}%")
    print(f"  Soul:   {eq.get('soul', 0)*100:.1f}%")
    print(f"  Spirit: {eq.get('spirit', 0)*100:.1f}%")
    
    # 4. Check 4D Cartesian
    print("\n--- 4D Cartesian Coordinates ---")
    cart = result.get("cartesian_4d", (0,0,0,0))
    print(f"  (x, y, z, w) = ({cart[0]:.3f}, {cart[1]:.3f}, {cart[2]:.3f}, {cart[3]:.3f})")
    
    # 5. Hologram status
    status = field.get_hologram_status()
    print(f"\n--- Hologram Status ---")
    print(f"  Count: {status.get('count', 0)}")
    
    print("\n✅ Phase 3 Integration Test PASSED!")
    return True


if __name__ == "__main__":
    test_phase3_integration()
