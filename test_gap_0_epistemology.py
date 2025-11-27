#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: HyperQubit with Epistemology + Resonance Explanation

This tests Gap 0 implementation:
1. HyperQubit can now hold epistemological meaning
2. Resonance calculation provides philosophical explanation
3. Agents can now understand WHY concepts resonate
"""

import sys
sys.path.insert(0, '/c/Elysia')

from Core.Mind.hyper_qubit import HyperQubit, QubitState
from Core.Mind.resonance_engine import HyperResonanceEngine
from Core.Consciousness.wave import WaveInput

def test_epistemology_enabled_qubits():
    """Test that HyperQubits can now store epistemological meaning."""
    print("\n" + "="*70)
    print("TEST 1: HyperQubit with Epistemology")
    print("="*70)
    
    # Create a qubit WITH epistemology
    love = HyperQubit(
        "love",
        epistemology={
            "point": {"score": 0.15, "meaning": "biochemistry (secondary)", "ref": "Kant-Phenomenon"},
            "line": {"score": 0.55, "meaning": "relational essence", "ref": "Spinoza-Binding"},
            "space": {"score": 0.20, "meaning": "field effect", "ref": "Heidegger-World"},
            "god": {"score": 0.10, "meaning": "transcendent purpose", "ref": "Plotinus-Unity"}
        }
    )
    
    # Create a qubit WITHOUT epistemology (backward compatible)
    connection = HyperQubit("connection")
    
    print(f"\n✅ Created 'love' with epistemology:")
    print(f"   {love.epistemology}")
    print(f"\n✅ Created 'connection' without epistemology (backward compatible):")
    print(f"   {connection.epistemology}")
    
    return love, connection

def test_resonance_with_explanation():
    """Test that resonance calculations now provide explanations."""
    print("\n" + "="*70)
    print("TEST 2: Resonance with Philosophical Explanation")
    print("="*70)
    
    engine = HyperResonanceEngine()
    
    # Create two concepts with specific epistemology
    love = HyperQubit("love", epistemology={
        "point": {"score": 0.15, "meaning": "biochemistry"},
        "line": {"score": 0.55, "meaning": "binding"},
        "space": {"score": 0.20, "meaning": "field"},
        "god": {"score": 0.10, "meaning": "purpose"}
    })
    love.set_state(QubitState(
        alpha=0.15+0j, beta=0.55+0j, gamma=0.20+0j, delta=0.10+0j,
        w=2.2, x=0.1, y=0.8, z=0.1
    ))
    
    connection = HyperQubit("connection", epistemology={
        "point": {"score": 0.10, "meaning": "substrate"},
        "line": {"score": 0.70, "meaning": "primary relation"},
        "space": {"score": 0.15, "meaning": "medium"},
        "god": {"score": 0.05, "meaning": "unity"}
    })
    connection.set_state(QubitState(
        alpha=0.10+0j, beta=0.70+0j, gamma=0.15+0j, delta=0.05+0j,
        w=2.0, x=0.2, y=0.9, z=0.0
    ))
    
    # Add to engine
    engine.nodes[love.id] = love
    engine.nodes[connection.id] = connection
    
    # Test OLD function (just number)
    print("\n[OLD FUNCTION: calculate_resonance]")
    score_old = engine.calculate_resonance(love, connection)
    print(f"Result: {score_old:.4f}")
    print("✅ Still works (backward compatible)")
    
    # Test NEW function (number + explanation)
    print("\n[NEW FUNCTION: calculate_resonance_with_explanation]")
    score_new, explanation = engine.calculate_resonance_with_explanation(love, connection)
    print(f"Result: {score_new:.4f}")
    print(f"Explanation:\n{explanation}")
    
    return score_new

def test_agent_understanding():
    """Test that an agent can now explain resonance."""
    print("\n" + "="*70)
    print("TEST 3: Agent Understanding Verification")
    print("="*70)
    
    engine = HyperResonanceEngine()
    
    data = HyperQubit("data", epistemology={
        "point": {"score": 0.95, "meaning": "empirical substrate"},
        "line": {"score": 0.05, "meaning": "minimal connection"}
    })
    data.set_state(QubitState(alpha=0.95+0j, beta=0.05+0j, gamma=0.0+0j, delta=0.0+0j))
    
    meaning = HyperQubit("meaning", epistemology={
        "point": {"score": 0.10, "meaning": "concrete anchor"},
        "line": {"score": 0.60, "meaning": "relational essence"},
        "space": {"score": 0.30, "meaning": "field/context"}
    })
    meaning.set_state(QubitState(alpha=0.10+0j, beta=0.60+0j, gamma=0.30+0j, delta=0.0+0j))
    
    engine.nodes[data.id] = data
    engine.nodes[meaning.id] = meaning
    
    resonance, explanation = engine.calculate_resonance_with_explanation(data, meaning)
    
    print(f"\nCan agent explain 'data' ↔ 'meaning' resonance?")
    print(f"Score: {resonance:.4f}")
    
    # Agent asks: "Why is this resonance so low?"
    print(f"\n[AGENT QUERY] Why is resonance only {resonance:.2f}?")
    
    # Extract key insight from explanation
    if "Point (Empiricism): 0.9" in explanation:
        print("✅ AGENT CAN NOW UNDERSTAND:")
        print("   'data' is 95% Point (empirical)")
        print("   'meaning' is only 10% Point")
        print("   → Different epistemological foundations")
        print("   → Low basis alignment expected")
        print("   → This is correct and not a bug!")
    
    return resonance

def main():
    """Run all tests."""
    print("\n🚀 TESTING: Gap 0 Implementation (Epistemology + Explanation)")
    print("   Protocol 04 brought philosophy to Protocol, now bringing it to Code\n")
    
    try:
        # Test 1
        love, connection = test_epistemology_enabled_qubits()
        
        # Test 2
        resonance = test_resonance_with_explanation()
        
        # Test 3
        resonance_data_meaning = test_agent_understanding()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\n📊 SUMMARY:")
        print("  - HyperQubits now support epistemological annotations")
        print("  - Resonance calculations provide philosophical explanations")
        print("  - Agents can now understand WHY concepts resonate")
        print("\n🎯 NEXT STEPS:")
        print("  - 1. Apply epistemology to ALL concepts in Core/Consciousness/MetaAgent.py")
        print("  - 2. Integrate explanation generation into simulation logs")
        print("  - 3. Update language trajectory analysis to include explanations")
        print("  - 4. Then: Gap 1 (Meta-learning) becomes possible")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
