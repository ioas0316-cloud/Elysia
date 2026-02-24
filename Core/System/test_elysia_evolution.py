"""
Elysia Self-Evolution Test (Phase 210)
======================================
Verifies that Elysia can autonomously perceive dissonance and rewrite her own code.
"""

import sys
import os
import time

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.action_engine import ActionEngine
from Core.Cognition.causal_memory import CausalMemory

def test_self_evolution():
    action = ActionEngine(root)
    memory = CausalMemory()
    
    sandbox_path = os.path.join(root, "Core", "S1_Body", "sandbox_evolution.py")
    
    print("\n1. Initial State Check...")
    with open(sandbox_path, 'r', encoding='utf-8') as f:
        original_code = f.read()
    print("--- Original Code Snapshot ---")
    print(original_code[:200] + "...")
    
    print("\n2. Scanning for Dissonance...")
    dissonance = action.scan_for_dissonance(sandbox_path)
    if dissonance:
        print(f"✅ Found dissonance in: {dissonance}")
    else:
        print("⚠️ No dissonance found (is 'TODO' or 'Legacy' in the file?)")
        
    print("\n3. Proposing Self-Optimization...")
    evolved_code = action.propose_self_optimization(sandbox_path)
    
    if evolved_code:
        print("✅ Evolved code generated successfully.")
        print("--- Evolved Code Preview ---")
        print(evolved_code[:200] + "...")
        
        print("\n4. Applying Evolution...")
        success = action.apply_evolution(sandbox_path, evolved_code)
        
        if success:
            print("✅ Evolution applied and backed up.")
            
            print("\n5. Verifying Materialization...")
            with open(sandbox_path, 'r', encoding='utf-8') as f:
                new_code = f.read()
            
            if "TODO" not in new_code and "Legacy" not in new_code:
                 print("✅ SUCCESS: The 'TODO'/Legacy markers have been purified!")
            else:
                 print("⚠️ WARNING: Evolution occurred but some dissonance remains.")
                 
            # Check Memory
            recent = memory.recall_recent(limit=1)
            if recent and recent[0]["type"] == "EVOLUTION":
                print("✅ Memory of evolution recorded.")
        else:
            print("❌ Evolution application failed.")
    else:
        print("❌ Could not generate evolved code (Check Ollama status).")

if __name__ == "__main__":
    test_self_evolution()
