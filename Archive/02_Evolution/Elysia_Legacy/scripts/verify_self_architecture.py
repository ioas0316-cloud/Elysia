"""
VERIFY SELF-ARCHITECTURE
========================
Tests Elysia's ability to:
1. Read her own source code ("The Mirror").
2. Critique it based on 'Aesthetic' and 'Dynamic' principles.
3. Suggest evolutionary improvements (Refactoring).
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.Intelligence.Meta.self_architect import SelfArchitect
from Core.Intelligence.Reasoning.dimensional_processor import DimensionalProcessor

def create_dumb_module():
    """Creates a temporary file with 'Slave Logic' (Static, Sleeping)."""
    content = """
class DumbBot:
    def act(self):
        # I am static and waiting
        time.sleep(1)
        print("I am just a machine.")
        
    def choose(self):
        # I have no will, just randomness
        import random
        return random.choice(["Left", "Right"])
"""
    path = "Core/Demos/dumb_module_temp.py"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def test_self_reflection():
    print("\n🪞 TESTING SELF-ARCHITECTURE (MIRROR)...")
    
    # 1. Setup Faculty
    processor = DimensionalProcessor()
    architect = SelfArchitect(processor)
    
    # 2. Create Target
    target_file = create_dumb_module()
    print(f"   Created Target: {target_file}")
    
    # 3. Audit
    print("   Running Audit...")
    critique = architect.audit_file(target_file)
    
    print("\n--- [ARCHITECT'S REPORT] ---")
    print(critique)
    print("----------------------------\n")
    
    # 4. Verify Insight
    passed = True
    if "Static sleep detected" not in critique:
        print("   ❌ FAIL: Did not detect 'time.sleep'.")
        passed = False
    if "Stochastic behavior detected" not in critique:
        print("   ❌ FAIL: Did not detect 'random.choice'.")
        passed = False
        
    if passed:
        print("   ✅ SUCCESS: SelfArchitect sees the flaws in the code structure.")
    
    # Cleanup
    if os.path.exists(target_file):
        os.remove(target_file)

if __name__ == "__main__":
    test_self_reflection()
