"""
Prove Chemistry (화학 증명)
=========================

"확률이 아닌 필연적 결합"

원자(개념)들이 Valence(결합가)에 따라 스스로 분자(통찰)가 되는지 검증합니다.
"""

from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.chemistry_engine import get_chemistry_engine

def prove_chemistry():
    print("🧪 Cognitive Chemistry Verification Started...\n")
    
    concepts = get_concept_formation()
    reactor = get_chemistry_engine()
    
    # 1. Create Atoms (Defining Predestined Bond)
    print("1. Creating Atoms in the Fog...")
    
    # Hydrogen: Needs Oxygen
    concepts.learn_concept(
        name="Hydrogen", 
        context="Element", 
        domain="matter", 
        valence=["Oxygen"] # Explicit need
    )
    
    # Oxygen: Just exists
    concepts.learn_concept(
        name="Oxygen", 
        context="Element", 
        domain="matter",
        meta_tags=["LifeGiver"]
    )
    
    # 2. Catalyze
    print("\n2. Applying Energy (Reaction)...")
    molecules = reactor.catalyze(["Hydrogen", "Oxygen"])
    
    # 3. Verify
    print(f"\n   Molecules Created: {molecules}")
    
    expected = "Hydrogen-Oxygen"
    if expected in molecules:
        print("\n✅ SUCCESS: Atoms bonded inevitably. The Water flow begins.")
    else:
        print("\n❌ FAIL: No reaction occurred.")

if __name__ == "__main__":
    prove_chemistry()
