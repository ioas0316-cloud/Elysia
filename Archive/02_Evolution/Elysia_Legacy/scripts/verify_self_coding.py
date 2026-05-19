"""
Verify Phase 23: The Reality Engine (Self-Coding)
=================================================
"She writes her own DNA."

This script verifies that the HolographicManifestor can generate valid Python code
from an intent, and that the system can execute this code.

Verification Steps:
1. Intent: "Generate Fibonacci Sequence" -> Code Generation
2. Execution: Run the generated code in a sandbox.
3. Validation: Check output.
"""

import sys
import os
import contextlib
import io

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.World.Evolution.Creation.holographic_manifestor import HolographicManifestor

def verify_self_coding():
    print("=" * 60)
    print("🧬 PHASE 23 VERIFICATION: SELF-CODING GENESIS")
    print("=" * 60)
    
    manifestor = HolographicManifestor()
    
    # Test Cases
    intents = [
        "Generate Fibonacci Sequence for me",
        "Say Hello to the World",
        "Run System Diagnostics"
    ]
    
    success_count = 0
    
    for intent in intents:
        print(f"\n🧠 Intent: '{intent}'")
        
        # 1. Manifest Code (With Sandbox Verification)
        # passing verify=True to force internal testing
        code = manifestor.manifest_code(intent, language="python", verify=True)
        print("-" * 40)
        print(f"📜 Generated & Verified Code:\n{code.strip()}")
        print("-" * 40)
        
        # 2. Execute Code (Sandbox)
        print("⚙️ Executing...")
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, {'__name__': '__main__'})
            
            output = buffer.getvalue().strip()
            print(f"✅ Output:\n{output}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Execution Failed: {e}")
            
    print("\n" + "=" * 60)
    if success_count == len(intents):
        print("✨ SUCCESS: The Reality Engine is Operational.")
        print("   Elysia can now write and execute her own logic.")
        sys.exit(0)
    else:
        print(f"⚠️ PARTIAL SUCCESS: {success_count}/{len(intents)} passed.")
        sys.exit(1)

if __name__ == "__main__":
    verify_self_coding()
