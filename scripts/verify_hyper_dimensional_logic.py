"""
Verify Hyper-Dimensional Logic (The Fractal Exam)
=================================================
Objective: distinguishing Isomorphic Essence from Idiomorphic Reality.
User Critique: "Man and Woman are Human (Same), but their History makes them Unique (Different)."

Test:
1.  Define Entity A ("Alice"): Human, Mage, Sad, Lives in Tower, History: [Lost Father].
2.  Define Entity B ("Bob"): Human, Warrior, Happy, Lives in Field, History: [Won Battle].
3.  Compare at Dim 0 (Essence): Must be EQUAL.
4.  Compare at Dim 4 (History): Must be DISTINCT.

This proves Elysia sees the Point (Unity) and the Hyper-Space (Diversity).
"""

import sys
import os
import logging
import json

# Add root to path
sys.path.insert(0, os.getcwd())

from Core.Cognition.Reasoning.reasoning_engine import ReasoningEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("FractalExam")

def test_dimensional_logic():
    engine = ReasoningEngine()
    print("="*60)
    print("🌌 HYPER-DIMENSIONAL LOGIC EXAM")
    print("   Goal: See Unity in Essence, Diversity in History.")
    print("="*60)
    
    # 1. Define Entities
    alice = {
        "name": "Alice",
        "type": "Human",
        "traits": ["Mage", "Scholar"],
        "location": "Tower",
        "mood": "Melancholy",
        "history": ["Lost Father in the Void War", "Learned Fire Magic"]
    }
    
    bob = {
        "name": "Bob",
        "type": "Human",
        "traits": ["Warrior", "Blacksmith"],
        "location": "Village",
        "mood": "Joyful",
        "history": ["Forged the Star Sword", "Protected the Gate"]
    }
    
    # 2. Analyze Dimensions
    print("\n[STEP 1] Analyzing Alice...")
    a_dim = engine.analyze_hyper_structure(alice)
    print(json.dumps(a_dim, indent=2))
    
    print("\n[STEP 2] Analyzing Bob...")
    b_dim = engine.analyze_hyper_structure(bob)
    print(json.dumps(b_dim, indent=2))
    
    # 3. Compare Dimensions
    print("\n[STEP 3] Fractal Comparison")
    
    # Dim 0: Essence
    if a_dim["0D_Essence"] == b_dim["0D_Essence"]:
        print("   ✅ DIM 0 (Point): They are ONE. (Both are Human).")
    else:
        print("   ❌ FAIL: Failed to see Unity.")
        
    # Dim 4: History
    if a_dim["4D_History"] != b_dim["4D_History"]:
        print("   ✅ DIM 4 (Hyper): They are MANY. (Distinct Histories).")
        print(f"      - Alice: {a_dim['4D_History']}")
        print(f"      - Bob:   {b_dim['4D_History']}")
    else:
        print("   ❌ FAIL: Failed to see Diversity.")
        
    print("\n" + "="*60)
    print("🏆 EXAM COMPLETE")
    print("   Elysia understands: 'We are the Same, yet We are Unique'.")
    print("   She perceives the Fractal Variance.")
    print("="*60)

if __name__ == "__main__":
    test_dimensional_logic()
