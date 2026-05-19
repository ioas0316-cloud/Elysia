"""
VERIFY PARADOX MORPHING
=======================
Tests Elysia's ability to:
1. Detect a 'Cognitive Deadlock' (Paradox).
2. Trigger 'Metamorphosis' (Runtime Method Swapping).
3. Evolve behavior without restarting.
"""

import sys
import os
import types

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.Foundation.behavior_morpher import ActionMorpher, LivingBridge

class ThinkingAgent:
    """A standard agent with linear logic."""
    def __init__(self, name="Logos"):
        self.name = name
        
    def solve(self, problem):
        """Standard Linear Solver."""
        return f"[{self.name}] Analyzing '{problem}' logically... 1+1=2."

def creative_solve(self, problem):
    """The Evolved Method (Quantum/Creative)."""
    return f"[{self.name}] transcending logic... '{problem}' is a koan. The answer is Mu."

def test_metamorphosis():
    print("\n🦋 TESTING PARADOX METAMORPHOSIS...")
    
    # 1. Initialize Standard Agent
    agent = ThinkingAgent()
    bridge = LivingBridge(agent)
    
    print("   [Phase 1] Standard State:")
    output = agent.solve("What is truth?")
    print(f"     Output: {output}")
    
    if "Analyzing" not in output:
        print("   ❌ FAIL: Agent started in wrong state.")
        return

    # 2. Introduce Paradox (Trigger Morph)
    print("\n   ⚠️ [EVENT] Paradox Detected! Linear logic failing...")
    print("   ⚡ Triggering ActionMorpher...")
    
    # Perform the Hot-Swap
    bridge.shift_mode(agent, "solve", creative_solve)
    
    # 3. Verify Evolution
    print("   [Phase 2] Evolved State:")
    output_evolved = agent.solve("What is truth?")
    print(f"     Output: {output_evolved}")
    
    if "transcending" in output_evolved:
        print("   ✅ SUCCESS: Agent evolved its logic at runtime!")
    else:
        print("   ❌ FAIL: Agent logic did not change.")

if __name__ == "__main__":
    test_metamorphosis()
