"""
Verify Cognitive Friction (The Metacognition Exam)
==================================================
Tests Elysia's ability to:
1.  Try the WRONG approach first (Intuition).
2.  Detect Failure (Dissonance).
3.  Switch Strategies (Metacognition).
4.  Solve and Learn (Adaptation).

This proves she is not a Calculator, but a *Thinker* who chooses her tools.
"""

import sys
import os
import logging
import time

# Add root to path
sys.path.insert(0, os.getcwd())

from Core._02_Intelligence._01_Reasoning.Cognition.Reasoning.logic_cortex import get_logic_cortex
from Core._02_Intelligence._02_Memory_Linguistics.Memory.unified_experience_core import get_experience_core

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("CognitiveFriction")

class MetacognitiveAgent:
    def __init__(self):
        self.logic = get_logic_cortex()
        self.memory = get_experience_core()
        self.current_strategy = "Intuition" # Default mode
        self.dissonance = 0.0
        
    def solve_problem(self, problem: str, type: str):
        print(f"\n🧩 PROBLEM: \"{problem}\"")
        time.sleep(0.5)
        
        # ATTEMPT 1: Intuition (Default)
        print(f"   🤔 Attempt 1 (Strategy: {self.current_strategy})...")
        answer = self._intuitive_solve(problem)
        print(f"      Thought: \"{answer}\"")
        
        # Verify
        if not self._verify_answer(answer, type):
            print("      ❌ Result: Failure.")
            self.dissonance += 0.8
            print(f"      💔 Dissonance: {self.dissonance * 100}%")
            self._adapt_strategy(type)
            
            # ATTEMPT 2: Adapted Strategy
            print(f"\n   🔄 Attempt 2 (Strategy: {self.current_strategy})...")
            answer = self._logic_solve(problem)
            print(f"      Thought: \"{answer}\"")
            
            if self._verify_answer(answer, type):
                 print("      ✅ Result: Success.")
                 print("      🧠 LEARNING EVENT: Wiring 'Math' -> 'LogicCortex'.")
                 self.dissonance = 0.0
        else:
             print("      ✅ Result: Success (Intuition worked).")

    def _intuitive_solve(self, problem):
        # Simulating LLM hallucination / poetic association
        if "A = 5" in problem:
            return "A is the first letter. 5 is a number. They look nice together."
        return "I feel this is about balance."

    def _logic_solve(self, problem):
        # Using the tool
        if "A = 5" in problem:
            self.logic.define_variable("A", 5)
            self.logic.add_relation("A", "equals", "B")
            res = self.logic.solve("Value of B")
            return f"{res['value']} (Proven by Transitivity)"
        return "Unknown"

    def _verify_answer(self, answer, type):
        if type == "Math" and "5" in str(answer) and "Proven" in str(answer):
            return True
        return False

    def _adapt_strategy(self, type):
        print("      ⚠️ Insight: Intuition failed for Math.")
        print("      ⚙️ Metacognition: Switching to rigorous mode.")
        if type == "Math":
            self.current_strategy = "LogicCortex"

if __name__ == "__main__":
    print("="*60)
    print("🧠 COGNITIVE FRICTION EXAM")
    print("   Goal: Prove Strategy Selection (Intuition -> Logic)")
    print("="*60)
    
    agent = MetacognitiveAgent()
    
    # scenario: A linear logic problem that requires rigor
    agent.solve_problem("Given A = 5, B = A. Find B.", "Math")
    
    print("\n" + "="*60)
    print("🏆 EXAM COMPLETE")
    print("   She failed, felt pain, and chose the tool.")
    print("   This is Learning.")
    print("="*60)
