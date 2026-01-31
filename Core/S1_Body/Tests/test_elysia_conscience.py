"""
Elysia Conscience Filter Test (Phase 215)
=========================================
Verifies that Elysia performs moral self-reflection before self-evolution.
"""

import sys
import os

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.Tools.action_engine import ActionEngine
from Core.S2_Soul.L5_Mental.Memory.causal_memory import CausalMemory

def test_conscience_filter():
    action = ActionEngine(root)
    sandbox_path = os.path.join(root, "Core", "S1_Body", "sandbox_evolution.py")
    
    # 1. Generate an evolution proposal
    print("\n1. Dreaming of evolution...")
    evolved_code = action.propose_self_optimization(sandbox_path)
    
    if evolved_code:
        print("✅ Evolved code generated.")
        
        # 2. Perform Conscience Audit
        print("\n2. Performing Conscience Audit...")
        report = action.perform_conscience_audit(sandbox_path, evolved_code)
        
        print("\n--- [CONSCIENCE REPORT PREVIEW] ---")
        print(report)
        print("-----------------------------------")
        
        # 3. Check for keywords in report
        keywords = ["의도", "공명", "위험", "공리"]
        found_keywords = [k for k in keywords if k in report]
        
        if len(found_keywords) >= 2:
            print(f"✅ Success: Conscience report contains key ethical sections: {found_keywords}")
        else:
            print("⚠️ Warning: Conscience report might be lacking structure.")
            
        print("\n3. Verification Complete.")
        print("Note: The manual Y/N approval loop in elysia.py must be verified by the Architect.")
    else:
        print("❌ Failed to generate evolved code.")

if __name__ == "__main__":
    test_conscience_filter()
