"""
Verify LogosEngine Axiom Integration
====================================
Tests the new `reason_with_axiom` method.
"""
import sys
import os
sys.path.append(os.getcwd())

from Core.02_Intelligence.01_Reasoning.Intelligence.logos_engine import LogosEngine

def test():
    print("🗣️ Initializing LogosEngine with Axiom Integration...")
    logos = LogosEngine()
    
    print("\n✨ Test: reason_with_axiom('Love', 'Ethics')")
    result = logos.reason_with_axiom("Love", "Ethics")
    print("-" * 50)
    print(result)
    print("-" * 50)
    
    # Check for causal terms
    assert "Hope" in result or "희망" in result or "야기" in result, "Expected causal language"
    assert "행위" in result or "원리" in result, "Expected domain projection"
    
    print("\n✅ LogosEngine Axiom Integration VERIFIED.")

if __name__ == "__main__":
    test()
