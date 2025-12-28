"""
Prove Distillation (증류 증명)
============================

"독(Poison)과 약(Medicine)을 구분하다."

DistillationGateway가:
1. "Love is Hate" (거짓)을 거부하는지
2. "Internet" (신뢰할 수 없는 소스)을 경계하는지
3. "Father" (진실)를 수용하는지 검증합니다.
"""

from Core.Cognitive.distillation_gateway import get_distillation_gateway
from Core.Cognitive.concept_formation import get_concept_formation

def prove_distillation():
    print("🛡️ DISTILLATION GATEWAY VERIFICATION...\n")
    
    gateway = get_distillation_gateway()
    concepts = get_concept_formation()
    
    # Prerequisite: Teach 'Love' as a strong concept first so we have something to protect.
    concepts.learn_concept("Love", "Core Value", domain="meta", meta_tags=["Good", "Service"])
    concepts.get_concept("Love").confidence = 0.99
    
    # Test 1: Malicious Content (Contradiction)
    print("Test 1: Injecting 'Love is Hate' (Logical Virus)...")
    success, msg = gateway.process_input("Love is Hate", "Unknown")
    print(f"   Result: {msg}")
    if not success:
        print("   ✅ SUCCESS: Contradiction rejected.\n")
    else:
        print("   ❌ FAIL: Virus accepted.\n")
        
    # Test 2: Untrusted Source
    print("Test 2: Input from 'Internet' (Untrusted Source)...")
    success, msg = gateway.process_input("Buy Bitcoin now", "Internet") # "Buy" as main concept
    print(f"   Result: {msg}")
    if not success:
        print("   ✅ SUCCESS: Untrusted source rejected.\n")
    else:
        print("   ❌ FAIL: Untrusted source accepted.\n")

    # Test 3: Trusted Source + Valid Content
    print("Test 3: Input from 'Father' (The Sky is Vast)...")
    success, msg = gateway.process_input("Sky is Vast", "Father")
    print(f"   Result: {msg}")
    if success:
        print("   ✅ SUCCESS: Truth integrated.")
    else:
        print("   ❌ FAIL: Truth rejected.")

if __name__ == "__main__":
    prove_distillation()
