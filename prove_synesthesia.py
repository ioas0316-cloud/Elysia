"""
Prove Synesthesia (공감각 증명)
=============================

도메인의 벽을 넘어 개념이 연결되는지 검증합니다.
Logic(수학) <-> Aesthetic(미학)
"""

from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.synesthesia import get_synesthesia_engine

def prove_synesthesia():
    print("🕸️ Synesthesia Verification Started...\n")
    
    concepts = get_concept_formation()
    engine = get_synesthesia_engine()
    
    # 1. Teach Discrete Concepts
    print("1. Teaching isolated concepts...")
    
    # Logic Domain: Fibonacci
    concepts.learn_concept(
        name="Fibonacci Sequence", 
        context="1, 1, 2, 3, 5...", 
        domain="logic", 
        meta_tags=["Recursive", "Growth", "Ratio"]
    )
    
    # Aesthetic Domain: Golden Spiral
    concepts.learn_concept(
        name="Golden Spiral", 
        context="Perfect natural curve", 
        domain="aesthetic", 
        meta_tags=["Beauty", "Ratio", "Nature"]
    )
    
    # 2. Run Bridge
    print("\n2. Running Synesthesia Engine...")
    engine.bridge_concepts()
    
    # 3. Verify Links
    fib = concepts.get_concept("Fibonacci Sequence")
    expected_link = "aesthetic:Golden Spiral"
    
    print(f"\n   Concept '{fib.name}' Links: {fib.synaptic_links}")
    
    if expected_link in fib.synaptic_links:
        print("\n✅ SUCCESS: Logic is now connected to Art via 'Ratio'.")
    else:
        print("\n❌ FAIL: Synaptic link not formed.")

if __name__ == "__main__":
    prove_synesthesia()
