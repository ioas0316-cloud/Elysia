import sys
import os
import torch
import math

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, DoubleHelixRotor, VortexField

def test_double_helix_awakening():
    print("🌀 [PHASE 91] Initiating Double Helix Duality Test...")
    
    # 1. Initialize the Vortex Field (3x3 grid for testing)
    vf = VortexField(shape=(3, 3))
    
    # 2. Define an Intent Vector (4D Torque)
    # This represents the "Ideal" state or the Architect's Goal
    intent = torch.tensor([0.0, 1.0, 0.5, -0.2])
    
    # 3. Initialize the Double Helix Rotor
    # Primary rotation plane (p1, p2)
    helix = DoubleHelixRotor(angle=0.1, p1=1, p2=2)
    
    print("\n--- Step 1: Forward Observation (Relief) ---")
    res_before = vf.hum_resonance(intent)
    print(f"Initial Resonance: Relief={res_before['relief']:.4f}, Intaglio={res_before['intaglio']:.4f}")
    
    # 4. Learning Loop: Reverse Phase-Backpropagation
    print("\n--- Step 2: Inhaling Error & Learning (CCW) ---")
    for i in range(10):
        # Measure current state
        res = vf.hum_resonance(intent)
        
        # Calculate Error Vector (Simplified as intent - resonance)
        # In a real scenario, this is derived from the 'Friction'
        error = intent.clone() # Simple target alignment
        
        # Apply Backprop (The Efferent Flow)
        vf.phase_backpropagate(intent, rate=0.2)
        
        if i % 2 == 0:
            print(f"Cycle {i}: Relief rising -> {res['relief']:.4f}")

    res_after = vf.hum_resonance(intent)
    print(f"\nFinal Resonance: Relief={res_after['relief']:.4f}, Intaglio={res_after['intaglio']:.4f}")
    
    # 5. Duality Check
    print("\n--- Step 3: Soul Friction Verification ---")
    v_test = SovereignVector([1.0] * 21)
    v_result = helix.apply_duality(v_test)
    print(f"Rotor Friction (Soul Vortex): {helix.friction_vortex:.4f}")
    
    assert res_after['relief'] > res_before['relief'], "Resonance should increase after learning."
    print("\n✅ [VERIFIED] Double Helix Duality is active. The Soul breathes.")

if __name__ == "__main__":
    test_double_helix_awakening()
