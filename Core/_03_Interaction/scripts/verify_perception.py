
"""
Verify Perceptual Expansion
===========================
Demonstrates Elysia's ability to transcend text and perceive underlying principles.
User Challenge: "Ice is not just text, but Stasis." "Love is Warmth."
"""
import sys
import os

# Ensure we are in root for comfortable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Core._01_Foundation._05_Governance.Foundation.fractal_concept import ConceptDecomposer, ConceptNode

def verify_perception():
    decomposer = ConceptDecomposer()
    
    # Test Cases: Concepts that are metaphors or physical states
    inputs = [
        "The Frozen Wasteland of data",  # Should be ICE (Stasis)
        "The burning desire for freedom", # Should be FIRE (Combustion)
        "The gentle flow of conversation", # Should be WATER (Flow)
        "A logical proof of existence",    # Should be MATH (Logos)
    ]
    
    print("\n👁️ Perceptual Expansion Test")
    print("===========================")
    
    for text in inputs:
        # Extract Essence
        essence = decomposer.infer_principle(text)
        
        print(f"\n📝 Input Text: '{text}'")
        print(f"   > Detected:  {essence.get('keywords', ['Unknown'])[0].upper()}") 
        print(f"   > Principle: {essence.get('principle')}")
        print(f"   > Law:       {essence.get('law')}")
        print(f"   > Frequency: {essence.get('frequency')} Hz")
        
        # Verify correctness
        if "Ice" in text or "Frozen" in text:
            if essence['principle'] != "Stasis":
                print("   ❌ FAIL: Ice should be Stasis.")
            else:
                print("   ✅ PASS: Ice identified as Fixed Static Energy.")

if __name__ == "__main__":
    verify_perception()
