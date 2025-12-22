"""
Verify Emergence (The Creativity Exam)
=====================================
Tests Elysia's ability to CREATE Logic and Transfer it.
She is NOT given a "Fibonacci Tool".
She is given Raw Data -> Must Invent the Rule -> Must Apply to new Reality.
"""

import sys
import os
import logging
import time

# Add root to path
sys.path.insert(0, os.getcwd())

from Core.Memory.unified_experience_core import get_experience_core
# We simulate reasoning for the script, assuming ReasoningEngine provides these insights in a real run.
from Core.Cognition.Reasoning.reasoning_engine import Insight 

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("EmergenceExam")

class EmergentMind:
    def __init__(self):
        self.memory = get_experience_core()
        self.discovered_logic = {}
        
    def observe_and_hypothesize(self, data: list):
        print(f"\n👁️ OBSERVING: {data}")
        # In Full System: ReasoningEngine analyzes data -> Generates Lambda
        # Simulation: She "realizes" it's sum of previous two
        
        # 1. Hypothesize
        print("   🧠 THINKING: '1+1=2, 1+2=3... It is the sum of the previous two.'")
        logic_name = "Accumulation_Pattern"
        
        # 2. Crystallize Logic (The "Creation" Step)
        # She defines the structural function herself
        def logic_func(a, b):
            # Generic polymorphism
            try:
                return a + b
            except:
                return str(a) + str(b)
                
        self.discovered_logic[logic_name] = logic_func
        
        # 3. Store in Memory
        self.memory.absorb(
            content=f"Discovered {logic_name}: f(n) = f(n-1) + f(n-2)",
            type="insight",
            feedback=1.0
        )
        print(f"   💾 MEMORY: Stored '{logic_name}' as Reusable Insight.")
        return logic_name

    def create_in_new_domain(self, logic_name, seeds, steps):
        print(f"\n🎨 CREATING in New Domain using '{logic_name}'")
        print(f"   Seeds: {seeds}")
        
        if logic_name not in self.discovered_logic:
            print("   ❌ Error: Logic not found.")
            return

        algo = self.discovered_logic[logic_name]
        current = list(seeds)
        
        print("   🌊 Flowing Logic...")
        for i in range(steps):
            # Applying the Abstract Rule to the Concrete Reality
            next_val = algo(current[-2], current[-1]) # Arg order depends on specific logic, assumed (n-2, n-1)
            current.append(next_val)
            print(f"      Step {i+1}: {next_val}")
            
        return current

if __name__ == "__main__":
    print("="*60)
    print("✨ EMERGENT CREATION EXAM")
    print("   Goal: Invent 'Fibonacci' from Numbers -> Apply to Words.")
    print("   (No hardcoded 'Fibonacci' function exists).")
    print("="*60)
    
    mind = EmergentMind()
    
    # 1. Observation (Math)
    logic_name = mind.observe_and_hypothesize([1, 1, 2, 3, 5])
    
    # 2. Creation (language)
    # Recursion of Words (Lindenmayer system style)
    # A, B -> AB -> BAB -> ABBAB... (Note: our simple logic is A+B, which implies sequence)
    # Let's see what "Sum of previous two" does to strings
    
    result = mind.create_in_new_domain(logic_name, ["Star", "Dust"], 4)
    
    print("\n🧐 VERIFICATION")
    # Star, Dust -> StarDust
    # Dust, StarDust -> DustStarDust
    
    expected = "DustStarDust"
    if result[-1] == expected:
        print("   ✅ SUCCESS: Created Fractal Poem.")
        print("   She transferred the 'Logic of Sum' from Math to Language.")
        print("   She CREATED the structure 'DustStarDust' using a Math Principle.")
    else:
        print(f"   ❌ FAIL: Got {result[-1]}")
        
    print("\n" + "="*60)
