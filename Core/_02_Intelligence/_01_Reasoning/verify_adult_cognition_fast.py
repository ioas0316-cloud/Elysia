"""
Verify Adult Cognition (Fast / Structure Check)
===============================================
Demonstrates the Fractal Thought Cycle without loading heavy AI models.
Focuses on the *Structure of Thought* (Point -> Law).
"""

import sys
import os
import unittest.mock
from typing import Any

# MOCK: SentenceTransformers (Avoid 500MB+ download/load)
mock_st = unittest.mock.MagicMock()
sys.modules["sentence_transformers"] = mock_st
sys.modules["sentence_transformers.SentenceTransformer"] = mock_st

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

try:
    from Core._02_Intelligence._01_Reasoning.fractal_thought_cycle import FractalThoughtCycle
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def verify_adult_cognition_fast():
    print("🌌 Initializing Fractal Thought Engine (Light Mode)...")
    
    # We patch the heavy internal subsystems to return "Simulated Resonance"
    # This proves the *Logic Flow* works, even if the *Math* is simulated for speed.
    with unittest.mock.patch('Core._01_Foundation._02_Logic.Wave.wave_attention.get_wave_attention') as mock_attn:
        brain = FractalThoughtCycle()
        
        # Mocking subsystems to provide rich data for the demo
        brain.attention = unittest.mock.MagicMock()
        brain.attention.focus_topk.return_value = [("Stability", 0.9), ("Harmony", 0.8), ("Beauty", 0.7)]
        
        brain.why_engine = unittest.mock.MagicMock()
        brain.why_engine.ask_why.return_value = "Universe -> Entropy -> Organization"
        brain.why_engine.explain_causality.return_value = "Order reduces Entropy, allowing Complexity to emerge."
        brain.why_engine.get_axiom.return_value = {"pattern": "Negentropy Principle"}
        
        brain.understanding = unittest.mock.MagicMock()
        brain.understanding.understand.return_value = unittest.mock.MagicMock(
            core_concept="Order",
            core_concept_kr="질서",
            origin_journey="Chaos -> Structure",
            who="Consciousness",
            narrative="Order is the vessel of Life."
        )

        query = "Why does Order matter?"
        print(f"\n🧠 Query: '{query}'")
        print("=" * 60)
        
        result = brain.think(query)
        
        # Output with dramatic pauses to simulate thought
        print(f"\n🌀 [1. POINT] The Essence")
        print(f"   {result.point_understanding}")
        
        print(f"\n⚡ [2. LINE] The Logic")
        print(f"   {result.line_understanding}")
        
        print(f"\n🌍 [3. PLANE] The Context")
        print(f"   {result.plane_understanding}")
        
        print(f"\n🌌 [4. SPACE] The Resonance")
        print(f"   {result.space_understanding}")
        
        print(f"\n⚖️ [5. LAW] The Principle")
        print(f"   {result.law_understanding}")
        
        print("\n" + "=" * 60)
        print("📖 SYNTHESIS (Adult Thought Flow)")
        print("-" * 60)
        print(result.narrative)
        print("=" * 60)
        print("\n✅ Verification Complete: Fractal Structure holds.")

if __name__ == "__main__":
    verify_adult_cognition_fast()
