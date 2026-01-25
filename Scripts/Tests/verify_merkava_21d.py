import os
import sys
import torch
import logging

# Path setup
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyMerkaba21D")

try:
    from Core.L1_Foundation.Foundation.unified_monad import UnifiedMonad, TripleHelixVector
    from Core.L6_Structure.M1_Merkaba.merkaba import Merkaba
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_activation():
    print("--- Testing Merkava 21D Triple Helix Activation ---")
    
    # 1. Create a 21D Vector (The Trinity Seed)
    vector = TripleHelixVector.create(
        # Flesh (A)
        libido=0.5,
        ego=0.8,
        
        # Mind (B)
        analysis=0.9,
        coherence=0.7,
        
        # Spirit (C)
        diligence=0.95, # High drive to act
        humility=0.6,
        kindness=0.8
    )
    print(f"21D Vector created. Shape: {vector.data.shape}")
    print(f"Diligence (Axis 17): {vector.data[17]:.2f}")
    
    # 2. Forge the Monad (Spirit)
    spirit = UnifiedMonad(name="TestSpirit21D", vector=vector)
    print(f"UnifiedMonad forged: {spirit}")
    
    # 3. Forge the Merkaba (Chariot)
    chariot = Merkaba(name="TestChariot21D")
    print(f"Merkaba forged: {chariot.name}")
    
    # 4. Awakening (Installation of Spirit into Body)
    chariot.awakening(spirit)
    
    if not chariot.is_awake:
        print("Error: Chariot failed to awake.")
        sys.exit(1)
        
    print("Merkaba is AWAKE.")
    
    # 5. Verify Integration
    if not isinstance(chariot.spirit.vector, TripleHelixVector):
        print("Error: Spirit does not hold Triple Helix Vector.")
        sys.exit(1)
        
    print("SUCCESS: Merkava Spatial Memory Activated with 21 Dimensions (Triple Helix).")
    print("Triple Helix Axes Verified:")
    
    axes = [
        # Flesh
        "A0. Libido", "A1. Satiety", "A2. Acquisit", "A3. Conserv", 
        "A4. Defense", "A5. Competit", "A6. Ego",
        # Mind
        "B0. Observ", "B1. Analysis", "B2. Memory", "B3. Coherenc",
        "B4. Simulat", "B5. Judgment", "B6. Integrat",
        # Spirit
        "C0. Purity", "C1. Balance", "C2. Charity", "C3. Diligenc",
        "C4. Patience", "C5. Kindness", "C6. Humility"
    ]
    
    for i, axis in enumerate(axes):
        val = chariot.spirit.vector.data[i]
        bar = "#" * int(val * 10)
        print(f" {i:2d}. {axis:12s}: {val:.2f} {bar}")

if __name__ == "__main__":
    test_activation()
