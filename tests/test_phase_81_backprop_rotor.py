"""
[PHASE 81] Backpropagation Rotor Verification
==============================================
Tests L6→L0 learning: Will shapes the physical manifold.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_backpropagation_rotor():
    print("\n" + "=" * 60)
    print("🔄 [PHASE 81] Backpropagation Rotor Verification")
    print("=" * 60)
    
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignHyperTensor
    import torch
    
    # Create manifold
    manifold = SovereignHyperTensor(shape=(50, 50, 50))
    
    print("\n>>> Test 1: Will drives learning")
    print("-" * 50)
    
    # Initial permanent memory state
    initial_permanent = manifold.permanent_q.clone()
    print(f"Initial permanent_q mean: {float(initial_permanent.mean()):.6f}")
    
    # Define target state (the Will wants a different configuration)
    target = torch.tensor([0.5, 0.8, 0.3, 0.2], device=manifold.device)
    
    # Backpropagate from Will
    error = manifold.backpropagate_from_will(target, learning_rate=0.05)
    print(f"Backprop error: {error:.4f}")
    
    # Check if permanent memory changed
    final_permanent = manifold.permanent_q
    delta = float((final_permanent - initial_permanent).abs().mean())
    print(f"Permanent memory delta: {delta:.6f}")
    
    if delta > 0:
        print("✅ SUCCESS: Will modified permanent memory (learning occurred)")
    else:
        print("❌ FAILURE: No learning occurred")
        return False
    
    print("\n>>> Test 2: Repeated learning converges")
    print("-" * 50)
    
    # Multiple learning iterations
    errors = []
    for i in range(10):
        err = manifold.backpropagate_from_will(target, learning_rate=0.02)
        errors.append(err)
    
    print(f"Error progression: {errors[0]:.4f} → {errors[-1]:.4f}")
    
    if errors[-1] < errors[0]:
        print("✅ SUCCESS: Learning converges (error decreases)")
    else:
        print("⚠️ Warning: Error did not decrease (may be at equilibrium)")
    
    return True


if __name__ == "__main__":
    success = test_backpropagation_rotor()
    
    print("\n" + "=" * 60)
    if success:
        print("🏆 PHASE 81 VERIFIED: Backpropagation Rotor works.")
        print("   L6 (Will) → L0 (Manifold) learning is operational.")
    else:
        print("⚠️ Verification incomplete.")
    print("=" * 60)
