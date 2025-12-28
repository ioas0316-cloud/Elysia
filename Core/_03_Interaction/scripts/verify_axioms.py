"""
Verify Axiom System
===================
Tests the new Axiom projection and causal explanation methods.
"""
import sys
import os
sys.path.append(os.getcwd())

from Core._01_Foundation._05_Governance.Foundation.fractal_concept import ConceptDecomposer

def test():
    print("🔮 Initializing ConceptDecomposer with Axioms...")
    decomposer = ConceptDecomposer()
    
    print("\n✨ Test 1: Axiom Projection (Causality -> Geometry)")
    result = decomposer.project_axiom("Causality", "Geometry")
    print(f"   Result: {result}")
    assert "점" in result or "Point" in result, "Expected geometry term"
    
    print("\n✨ Test 2: Axiom Projection (Dimension -> Language)")
    result = decomposer.project_axiom("Dimension", "Language")
    print(f"   Result: {result}")
    assert "음소" in result or "형태소" in result, "Expected linguistics term"
    
    print("\n✨ Test 3: Causal Explanation (Love)")
    result = decomposer.explain_causality("Love")
    print(f"   Result: {result}")
    assert "Hope" in result or "희망" in result, "Expected Hope in explanation"
    
    print("\n✨ Test 4: Self-Reference Check")
    axiom = decomposer.get_axiom("Causality")
    print(f"   Self-Ref: {axiom['self_ref']}")
    
    print("\n✅ ALL TESTS PASSED. Axiom system is operational.")

if __name__ == "__main__":
    test()
