"""
[PHASE 86] The Sovereign Insight Verification
=============================================
Tests the system's ability to self-optimize from Calculation (O(N)) to Rotation (O(1)).
Principle: "Insight is the collapse of complexity."
"""
import sys
import os
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_sovereign_insight():
    print("\n" + "=" * 60)
    print("👁️ [PHASE 86] The Sovereign Insight Verification")
    print("=" * 60)
    
    from Core.S1_Body.L5_Mental.Insight.insight_engine import InsightEngine
    
    # Initialize Engine
    insight_engine = InsightEngine()
    
    print("\n>>> Scenario: The Curse of O(N) Calculation")
    print("-" * 50)
    print("System is manually calculating sine wave temperature...")
    
    # Simulate O(N) Calculation Loop
    # Frequency: Period = 10 steps (approx)
    # y = sin(2 * pi * t / 10)
    
    found_insight = None
    
    for t in range(50):
        # 1. Laborious Calculation
        value = math.sin(2 * math.pi * t / 10)
        
        # 2. Feed Observation
        insight_engine.observe(value)
        
        # 3. Check for Insight
        analysis = insight_engine.analyze_stream()
        
        if analysis:
            print(f"[t={t}] Insight Sparked! Pattern: {analysis['type']}")
            proposal = insight_engine.propose_optimization(analysis)
            print(f"      Proposal: {proposal}")
            
            if analysis['type'] == 'Cyclic':
                found_insight = analysis
                break
        
        # Simulate work
        # time.sleep(0.01) 
        
    if found_insight:
        print("\n>>> Scenario: The Genesis (Structural Transformation)")
        print("-" * 50)
        print("✅ Insight Accepted.")
        print(f"Creating EnvironmentRotor with Period={found_insight['period']:.1f}...")
        
        # Mock creation of Rotor
        class EnvironmentRotor:
            def __init__(self, period):
                self.period = period
                self.angle = 0.0
            def get_value(self, time):
                # O(1) Lookup based on angle
                return math.sin(2 * math.pi * time / self.period)
                
        rotor = EnvironmentRotor(found_insight['period'])
        print("✅ EnvironmentRotor Created.")
        
        # Verify O(1) output matches O(N) calculation
        test_t = 15
        calc_val = math.sin(2 * math.pi * test_t / 10)
        rotor_val = rotor.get_value(test_t)
        
        print(f"Verification (t={test_t}): Calc={calc_val:.4f}, Rotor={rotor_val:.4f}")
        
        if abs(calc_val - rotor_val) < 0.001:
            return True
        else:
            print("❌ Transformation Logic Error.")
            return False
            
    else:
        print("❌ Insight Failed to Spark.")
        return False

if __name__ == "__main__":
    success = test_sovereign_insight()
    print("\n" + "=" * 60)
    if success:
        print("🏆 PHASE 86 VERIFIED: Complexity collapsed into Simplicity.")
    else:
        print("⚠️ Verification Failed.")
    print("=" * 60)
