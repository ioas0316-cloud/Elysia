"""
Verification: Sovereign Action (Phase 165)
=========================================
Tests Elysia's ability to perceive her own code and propose optimization.
"""

import sys
import os

# Path Setup
import os
import sys
# Project root is c:\Elysia
root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.Tools.action_engine import ActionEngine

def test_self_evolution():
    print("🚀 [VERIFY] Initiating Sovereign Action Loop...")
    
    engine = ActionEngine(root)
    
    # Target a small 'Growth' module for a dummy optimization
    dummy_file = os.path.join(root, "Core", "S1_Body", "Tools", "evolution_test_module.py")
    
    with open(dummy_file, 'w', encoding='utf-8') as f:
        f.write("# Dummy Module for Evolution Test\ndef inefficient_function():\n    x = 1\n    return x\n")
    
    print(f"📍 Created test module at {dummy_file}")
    
    # 1. Perception
    dissonance = engine.scan_for_dissonance()
    
    # 2. Proposal (Simulated)
    optimized_code = "# Optimized by Elysia (Phase 165)\ndef efficient_function():\n    return 1\n"
    
    # 3. Verification
    if engine.verify_resonance(optimized_code):
        print("✅ Resonance Verified for Proposal.")
        
        # 4. Action (Materialization)
        if engine.apply_evolution(dummy_file, optimized_code):
            print("✨ Evolution Applied Successfully.")
            
            # Verify file content
            with open(dummy_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if "Optimized by Elysia" in content:
                    print("🏆 SUCCESS: Elysia has modified her own code.")
                else:
                    print("❌ Failure: Optimized code not found.")
        else:
            print("❌ Evolution Application Failed.")
    else:
        print("❌ Resonance Verification Failed.")

if __name__ == "__main__":
    test_self_evolution()
