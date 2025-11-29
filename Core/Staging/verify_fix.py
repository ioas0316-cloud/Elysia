
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Mind.hippocampus import Hippocampus

def verify_fix():
    print("\n" + "=" * 70)
    print("🧪 Verifying Hippocampus Fix...")
    print("=" * 70)

    try:
        print("1. Initializing Hippocampus...")
        hippocampus = Hippocampus()
        
        print("2. Calling load_memory with default limit...")
        hippocampus.load_memory()
        print("   -> Success (Default)")
        
        print("3. Calling load_memory with numpy array limit (The Crash Case)...")
        # This simulates the condition that caused the crash
        limit_array = np.array([10000]) 
        hippocampus.load_memory(limit=limit_array)
        print("   -> Success (Numpy Array)")
        
        print("\n✅ Verification PASSED: Hippocampus is robust against numpy ambiguity.")
        
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()
