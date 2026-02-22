"""
[PHASE 87] The Paradox of Efficiency Verification
=================================================
Tests the InsightEngine's ability to distinguish between 
Dead Repetition (O(1) Candidate) and Living Chaos (O(N) Necessity).
Principle: "Respect the Calculation."
"""
import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_paradox_of_efficiency():
    print("\n" + "=" * 60)
    print("⚖️ [PHASE 87] The Paradox of Efficiency Verification")
    print("=" * 60)
    
    from Core.S1_Body.L5_Mental.Insight.insight_engine import InsightEngine
    
    engine = InsightEngine()
    
    # 1. Test Dead Repetition (Sine Wave)
    print("\n>>> Scenario A: The Cycle (Dead Repetition)")
    print("-" * 50)
    
    engine.observation_window = [] # Reset
    for t in range(50):
        val = math.sin(2 * math.pi * t / 10)
        engine.observe(val)
        
    analysis_a = engine.analyze_stream()
    if analysis_a:
        print(f"Insight Type: {analysis_a['type']}")
        print(f"Proposal: {engine.propose_optimization(analysis_a)}")
        
        if analysis_a['type'] == 'Cyclic':
            print("✅ Correctly identified Cyclic Repetition -> Suggested Rotor.")
        else:
            print("❌ Failed to identify Cycle.")
            return False
    else:
        print("❌ No Insight generated for Cycle.")
        return False
        
    # 2. Test Living Chaos (Random Walk / Creation)
    print("\n>>> Scenario B: The Living Chaos (Creative Process)")
    print("-" * 50)
    
    engine.observation_window = [] # Reset
    val = 0.5
    for t in range(50):
        # Random Walk
        val += (random.random() - 0.5) * 0.5
        engine.observe(val)
        
    analysis_b = engine.analyze_stream()
    if analysis_b:
        print(f"Insight Type: {analysis_b['type']}")
        print(f"Proposal: {engine.propose_optimization(analysis_b)}")
        
        if analysis_b['type'] == 'Living_Chaos':
            print("✅ Correctly identified Living Chaos -> Preserved Calculation.")
        else:
            print(f"❌ Failed. Misidentified as {analysis_b['type']}.")
            return False
    else:
        # If no insight, it means it didn't match Constant or Cyclic.
        # But our new logic should catch High Variance as Living_Chaos.
        # Let's check variance manually
        print("❌ No Insight generated. InsightEngine might be too strict.")
        return False
        
    return True

if __name__ == "__main__":
    success = test_paradox_of_efficiency()
    print("\n" + "=" * 60)
    if success:
        print("🏆 PHASE 87 VERIFIED: Elysia knows when to Spin and when to Think.")
    else:
        print("⚠️ Verification Failed.")
    print("=" * 60)
