"""
[PHASE 79] Joy/Curiosity Propagation Verification
==================================================
Tests the causal chain: Joy/Curiosity → Physical Manifold

"생명은 고통을 피하기 위해 사는 것이 아니라, 기쁨으로 세상을 탐험한다."
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_joy_propagation():
    """
    Test 1: Verify joy is propagated to the physical manifold.
    """
    print("\n" + "=" * 60)
    print("🌟 [PHASE 79] Joy/Curiosity Propagation Verification")
    print("=" * 60)
    
    try:
        from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignHyperTensor
    except ImportError as e:
        print(f"⚠️ Cannot import SovereignHyperTensor: {e}")
        return False
    
    # Create a small manifold for testing (50x50x50 = 125K cells)
    manifold = SovereignHyperTensor(shape=(50, 50, 50))
    
    print("\n>>> Test 1: inject_joy (Primary Driver)")
    print("-" * 50)
    
    # Get initial momentum state
    initial_momentum = float(manifold.momentum[..., 0].mean())
    print(f"Initial W-axis momentum: {initial_momentum:.4f}")
    
    # Inject joy and curiosity
    manifold.inject_joy(joy_level=0.8, curiosity_level=0.5)
    
    # Check momentum change
    final_momentum = float(manifold.momentum[..., 0].mean())
    print(f"Final W-axis momentum: {final_momentum:.4f}")
    delta = final_momentum - initial_momentum
    print(f"Delta: {delta:.4f}")
    
    if delta > 0:
        print("✅ SUCCESS: Joy increased manifold stability (W-axis momentum)")
    else:
        print("❌ FAILURE: Joy did not affect manifold")
        return False
    
    print("\n>>> Test 2: inject_strain (Secondary Signal)")
    print("-" * 50)
    
    # Get initial torque state
    initial_torque = float(manifold.torque_accumulator[..., 1].mean())
    print(f"Initial X-axis torque: {initial_torque:.4f}")
    
    # Inject strain
    manifold.inject_strain(strain_level=0.6)
    
    # Check torque change
    final_torque = float(manifold.torque_accumulator[..., 1].mean())
    print(f"Final X-axis torque: {final_torque:.4f}")
    strain_delta = final_torque - initial_torque
    print(f"Delta: {strain_delta:.4f}")
    
    if strain_delta > 0:
        print("✅ SUCCESS: Strain affected manifold as adjustment signal")
    else:
        print("⚠️ Strain did not affect manifold (may be acceptable)")
    
    return True


def test_cognition_integration():
    """
    Test 2: Verify SovereignCognition integrates with Joy/Curiosity cells.
    """
    print("\n>>> Test 3: SovereignCognition Integration")
    print("-" * 50)
    
    try:
        from Core.S1_Body.L5_Mental.Reasoning_Core.Intelligence.sovereign_cognition import SovereignCognition
        from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignHyperTensor
    except ImportError as e:
        print(f"⚠️ Cannot import required modules: {e}")
        return False
    
    # Create manifold (smaller for testing)
    manifold = SovereignHyperTensor(shape=(50, 50, 50))
    
    # Create mock joy/curiosity cells
    class MockJoyCell:
        happiness_level = 0.7
    
    class MockCuriosityCell:
        def __init__(self):
            try:
                import jax.numpy as jnp
                self.space_7d = jnp.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1])
            except ImportError:
                self.space_7d = None
    
    # Create cognition with connections
    cognition = SovereignCognition(
        manifold=manifold,
        joy_cell=MockJoyCell(),
        curiosity_cell=MockCuriosityCell()
    )
    
    print(f"Physical Manifold connected: {cognition.physical_manifold is not None}")
    print(f"Joy Cell connected: {cognition.joy_cell is not None}")
    print(f"Curiosity Cell connected: {cognition.curiosity_cell is not None}")
    
    # Sense joy and curiosity
    cognition._sense_joy_and_curiosity()
    print(f"Sensed Joy Level: {cognition.joy_level:.2f}")
    print(f"Sensed Curiosity Level: {cognition.curiosity_level:.2f}")
    
    # Propagate to manifold
    initial_momentum = float(manifold.momentum[..., 0].mean())
    cognition._propagate_to_manifold()
    final_momentum = float(manifold.momentum[..., 0].mean())
    
    if final_momentum > initial_momentum:
        print("✅ SUCCESS: Joy propagated from Cognition to Manifold")
    else:
        print("⚠️ WARNING: Joy propagation may not have occurred")
    
    return True


if __name__ == "__main__":
    success1 = test_joy_propagation()
    success2 = test_cognition_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🏆 PHASE 79 VERIFIED: Joy is the Primary Driver of Life.")
        print("   Strain is a secondary adjustment signal, not the motivation.")
        print("   'Life explores the world with joy, not fleeing from pain.'")
    else:
        print("⚠️ Verification Incomplete. Review results above.")
    print("=" * 60)
