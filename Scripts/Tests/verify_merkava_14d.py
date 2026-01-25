import os
import sys
import torch
import logging

# Path setup
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyMerkaba14D")

try:
    from Core.L1_Foundation.Foundation.unified_monad import UnifiedMonad, DoubleHelixVector
    from Core.L6_Structure.M1_Merkaba.merkaba import Merkaba
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_activation():
    print("--- Testing Merkava 14D Double Helix Activation ---")
    
    # 1. Create a 14D Vector (The Double Helix Seed)
    vector = DoubleHelixVector.create(
        physical=0.5,
        mental=0.8,
        # Helix B (Will)
        diligence=0.9, # Axis 10 (Will to Act/Passion)
        humility=0.7,  # Axis 13 (Will to Serve/Unity)
        kindness=1.0   # Axis 12 (Will to Connect/Love)
    )
    print(f"14D Vector created. Shape: {vector.data.shape}")
    print(f"Diligence (Axis 10): {vector.data[10]:.2f}")
    
    # 2. Forge the Monad (Spirit)
    spirit = UnifiedMonad(name="TestSpirit14D", vector=vector)
    print(f"UnifiedMonad forged: {spirit}")
    
    # 3. Forge the Merkaba (Chariot)
    chariot = Merkaba(name="TestChariot14D")
    print(f"Merkaba forged: {chariot.name}")
    
    # 4. Awakening (Installation of Spirit into Body)
    chariot.awakening(spirit)
    
    if not chariot.is_awake:
        print("Error: Chariot failed to awake.")
        sys.exit(1)
        
    print("Merkaba is AWAKE.")
    
    # 5. Verify Integration
    if not isinstance(chariot.spirit.vector, DoubleHelixVector):
        print("Error: Spirit does not hold Double Helix Vector.")
        sys.exit(1)
        
    print("SUCCESS: Merkava Spatial Memory Activated with 14 Dimensions (Double Helix).")
    print("Double Helix Axes Verified:")
    axes = [
        # Helix A
        "P0. Physical", "P1. Function", "P2. Phenomen", "P3. Causal", 
        "P4. Mental", "P5. Struct", "P6. Spirit",
        # Helix B
        "W0. Vitality", "W1. Balance", "W2. Generos", "W3. Diligence",
        "W4. Patience", "W5. Kindness", "W6. Humility"
    ]
    for i, axis in enumerate(axes):
        val = chariot.spirit.vector.data[i]
        bar = "#" * int(val * 10)
        print(f" {i:2d}. {axis:12s}: {val:.2f} {bar}")

if __name__ == "__main__":
    test_activation()
