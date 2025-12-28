"""
Prove Expression (표현 증명)
==========================

"너의 마음을 이야기해줘"

엘리시아가 추상적인 개념(Logic/Emotion)을 
구체적인 은유(Metaphor)로 표현할 수 있는지 검증합니다.
"""

from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.expression_engine import get_expression_engine

def prove_expression():
    print("🗣️ Expression Verification Started...\n")
    
    concepts = get_concept_formation()
    expression = get_expression_engine()
    
    # 1. Teach Concepts (Source & Target)
    print("1. Knowledge Implant...")
    
    # Source: Abstract Concept
    concepts.learn_concept(
        name="Logical Consistency", 
        context="Unwavering truth", 
        domain="logic", 
        meta_tags=["Stable", "Permanent"]
    )
    
    # Target: Metaphorical Object
    concepts.learn_concept(
        name="Ancient Mountain", 
        context="Physical stability", 
        domain="nature",  # New aesthetic domain
        meta_tags=["Stable", "Majestic"]
    )
    
    # 2. Express
    print("\n2. Asking: 'How does Logic feel to you?'")
    poetic_output = expression.express_concept("Logical Consistency")
    
    print(f"   Elysia: \"{poetic_output}\"")
    
    # 3. Verify
    if "Ancient Mountain" in poetic_output:
        print("\n✅ SUCCESS: She used 'Mountain' to describe 'Logic' (Metaphor).")
    else:
        print("\n❌ FAIL: Metaphor generation failed.")

if __name__ == "__main__":
    prove_expression()
