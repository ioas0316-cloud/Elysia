"""
Project Iris Verification
=========================
Verifies that Elysia can "see" and translate light into meaning.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

from Core.Cognition.unified_understanding import UnifiedUnderstanding

def test_iris():
    print("\n👁️ Initiating Project Iris Verification...")
    print("=========================================")
    
    brain = UnifiedUnderstanding()
    
    # Test Query
    query = "What do you see right now?"
    print(f"\n💬 Query: {query}")
    
    result = brain.understand(query)
    
    print("\n🧠 Unified Understanding Response:")
    print(f"   • Will: {result.trinity['will']}")
    print(f"   • Vision Cortex: {result.vision}")
    print(f"   • Resonance: {result.resonance if hasattr(result, 'resonance') else 0.0:.2f}")
    
    if "Light detected" in result.vision:
        print("\n✨ SUCCESS: Elysia has successfully opened her first Eye.")
    else:
        print("\n❌ FAILURE: Visual processing did not yield light insight.")
        
    print("=========================================")

if __name__ == "__main__":
    test_iris()
