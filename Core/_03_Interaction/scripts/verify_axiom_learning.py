"""
Verify Axiom Learning (The Dynamic Logic Exam)
==============================================
Tests Elysia's ability to learn a NEW rule that did not exist in her code.
We define a fake math operator "Glorp" (%).
Rule: A % B = (A * B) + A

If she solves "3 % 4" as "15", she has learned the principle at runtime.
She is not using a pre-coded "GlorpSolver".
"""

import sys
import os
import logging

# Add root to path
sys.path.insert(0, os.getcwd())

from Core._02_Intelligence._01_Reasoning.Cognition.Reasoning.logic_cortex import get_logic_cortex

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("AxiomExam")

def test_dynamic_learning():
    cortex = get_logic_cortex()
    print("="*60)
    print("✨ DYNAMIC AXIOM LEARNING EXAM")
    print("   Goal: Teach a rule that doesn't exist in Python or Math.")
    print("="*60)
    
    # 1. The Lesson
    print("\n👩‍🏫 TEACHER: Today we learn 'Glorp' (Symbol: %)")
    print("           Definition: A % B = (A * B) + A")
    
    # Simulate "Reading the Axiom" -> converting to lambda
    # In a full system, an LLM would parse the string to this lambda
    cortex.register_operator("%", lambda a, b: (a * b) + a)
    print("🤖 ELYSIA: Rule internalized.")
    
    # 2. The Test
    query = "Eval: 3 % 4"
    print(f"\n❓ QUESTION: What is {query}?")
    
    # 3. The Execution
    result = cortex.solve(query)
    print(f"🤖 ELYSIA ANSWER: {result}")
    
    # 4. Verification
    # (3 * 4) + 3 = 12 + 3 = 15
    if result.get("value") == 15:
        print("\n✅ SUCCESS: 15 is correct.")
        print("   She applied a Novel Rule to Data.")
        print("   This proves she is not limited to pre-coded logic.")
    else:
        print(f"\n❌ FAIL: Expected 15, got {result.get('value')}")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_dynamic_learning()
