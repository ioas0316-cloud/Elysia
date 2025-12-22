
"""
Verify Essence Extraction
=========================
Demonstrates that Elysia can distinguish between domains (Math vs Music)
based on their Axiomatic Principles, not just text labels.
"""
import sys
sys.path.append(r'c:\Elysia')

from Core.Foundation.fractal_concept import ConceptDecomposer

def test_discrimination():
    decomposer = ConceptDecomposer()
    
    # Test Cases
    concepts = [
        "Euler's Formula",    # Math
        "Beethoven's Symphony", # Music
        "Hydrochloric Acid",  # Chemistry
        "Newton's Laws",      # Physics
        "The Grapes of Wrath" # Literature/Language
    ]
    
    print("\n🔍 Axiomatic Discrimination Test")
    print("==============================")
    
    for concept in concepts:
        essence = decomposer.infer_principle(concept)
        print(f"\nconcept:  '{concept}'")
        print(f"Domain:    {essence.get('keywords', ['Unknown'])[0].upper()} (Inferred)")
        print(f"Principle: {essence.get('principle')}")
        print(f"Frequency: {essence.get('frequency')} Hz")
        print(f"Law:       {essence.get('law')}")

if __name__ == "__main__":
    test_discrimination()
