
import sys
sys.path.append(r'c:\Elysia')

from Core.Intelligence.logos_engine import LogosEngine
from Core.Foundation.fractal_knowledge import KnowledgeTriple

def verify_causality():
    logos = LogosEngine()
    
    print("\n🔮 Causal Reinterpretation Verification")
    print("=======================================")
    
    concept = "엘리시아"
    
    # 1. Simulate Fragmented/Probabilistic Data (Triples)
    print("\n[Step 1] Gathering Fragmented Triples...")
    raw_triples = [
        KnowledgeTriple("엘리시아", "IsA", "인공지능"),
        KnowledgeTriple("인공지능", "Requires", "데이터"),
        KnowledgeTriple("데이터", "Is", "파동"),
        KnowledgeTriple("파동", "Creates", "우주")
    ]
    for t in raw_triples:
        print(f"   - {t}")
        
    # 2. Reinterpret (The "Bridge" Action)
    print("\n[Step 2] Reinterpreting into Unified Causality...")
    narrative = logos.reinterpret_causality(concept, raw_triples, tone="logical")
    
    print(f"\n🗣️ Result:\n{narrative}")
    
    if "통합된 인과율" in narrative and "우주" in narrative:
        print("\n✅ Verification SUCCESS: Fragments unified into a causal flow.")
    else:
        print("\n❌ Verification FAILED.")

if __name__ == "__main__":
    verify_causality()
