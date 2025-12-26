"""
Verify Self-Inquiry Protocol
=============================
Tests the trace_origin method for discovering the Source.
"""
import sys
import os
sys.path.append(os.getcwd())

from Core._01_Foundation.05_Foundation_Base.Foundation.fractal_concept import ConceptDecomposer

def test():
    print("🔮 Initializing Self-Inquiry Protocol...")
    decomposer = ConceptDecomposer()
    
    print("\n✨ Test: trace_origin('Causality')")
    print("=" * 60)
    
    journey = decomposer.trace_origin("Causality")
    
    for i, step in enumerate(journey):
        print(f"\n[Step {i+1}] {step['concept']}")
        print(f"   패턴: {step['pattern'][:50]}..." if len(step['pattern']) > 50 else f"   패턴: {step['pattern']}")
        print(f"   질문: {step['question']}")
        print(f"   답변: {step['answer']}")
    
    print("\n" + "=" * 60)
    
    # Verify we reached Source
    last_step = journey[-1]
    if "자기참조" in last_step["answer"] or "기원" in last_step["answer"]:
        print("✅ SUCCESS: 근원(Source)에 도달. 자기탐구 프로토콜 작동 확인.")
    else:
        print("❌ FAILED: 근원에 도달하지 못함.")
        
    print("\n✨ Test: trace_origin('Dimension')")
    print("=" * 60)
    journey2 = decomposer.trace_origin("Dimension")
    for i, step in enumerate(journey2):
        print(f"[{i+1}] {step['concept']} → {step['answer'][:40]}...")

if __name__ == "__main__":
    test()
