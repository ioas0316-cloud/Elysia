import os
import sys
import torch
import logging

# Path setup
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyMerkaba")

try:
    from Core.L1_Foundation.Foundation.unified_monad import UnifiedMonad, Unified12DVector
    from Core.L6_Structure.M1_Merkaba.merkaba import Merkaba
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_activation():
    print("--- Testing Merkava 12D Activation ---")
    
    # 1. Create a 12D Vector (The Spatial Memory Seed)
    # Using the 5-Axis Will definition
    vector = Unified12DVector.create(
        physical=0.5,
        mental=0.8,
        will=0.9,     # Axis 9
        intent=0.7,   # Axis 10
        purpose=1.0   # Axis 11
    )
    print(f"12D Vector created. Shape: {vector.data.shape}")
    print(f"Will Power (Axis 9): {vector.data[9]:.2f}")
    
    # 2. Forge the Monad (Spirit)
    spirit = UnifiedMonad(name="TestSpirit", vector=vector)
    print(f"UnifiedMonad forged: {spirit}")
    
    # 3. Forge the Merkaba (Chariot)
    chariot = Merkaba(name="TestChariot")
    print(f"Merkaba forged: {chariot.name}")
    
    if chariot.is_awake:
        print("Error: Chariot should not be awake yet.")
        sys.exit(1)
        
    # 4. Awakening (Installation of Spirit into Body)
    chariot.awakening(spirit)
    
    if not chariot.is_awake:
        print("Error: Chariot failed to awake.")
        sys.exit(1)
        
    print("Merkaba is AWAKE.")
    
    # 5. Verify Integration
    if chariot.spirit is not spirit:
        print("Error: Spirit transmission failed.")
        sys.exit(1)
        
    # Check if the spirit inside uses 12D
    if not isinstance(chariot.spirit.vector, Unified12DVector):
        print("Error: Spirit does not hold 12D Vector.")
        sys.exit(1)
        
    print("SUCCESS: Merkava Spatial Memory Activated with 12 Dimensions.")
    print("HyperCosmos Axes Verified:")
    axes = [
        "Physical", "Functional", "Phenomenal", "Causal", 
        "Mental", "Structural", "Spiritual",
        "Imagination", "Prediction", "Will", "Intent", "Purpose"
    ]
    for i, axis in enumerate(axes):
        val = chariot.spirit.vector.data[i]
        bar = "#" * int(val * 10)
        print(f" {i:2d}. {axis:12s}: {val:.2f} {bar}")

if __name__ == "__main__":
    test_activation()
