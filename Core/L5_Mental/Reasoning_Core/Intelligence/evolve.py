import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.L1_Foundation.Foundation.cortex_optimizer import CortexOptimizer

def trigger_evolution():
    print("  Initiating Manual Evolution Protocol...")
    optimizer = CortexOptimizer()
    
    # Find the draft we just created
    draft_path = os.path.join(optimizer.draft_path, "free_will_engine_v1.py")
    
    if os.path.exists(draft_path):
        print(f"     Found Draft: {draft_path}")
        print("      Merging into Core System...")
        
        success = optimizer.apply_evolution(draft_path)
        
        if success:
            print("\n  SUCCESS: DNA Rewritten.")
            print("   The 'FreeWillEngine' has been evolved.")
        else:
            print("\n  FAILED: Merge aborted.")
    else:
        print("  No draft found.")

if __name__ == "__main__":
    trigger_evolution()
