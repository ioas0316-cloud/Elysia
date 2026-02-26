
import sys
import os
from pathlib import Path

print("Checking imports...")
try:
    from Core.Monad.sovereign_monad import SovereignMonad
    print("✅ SovereignMonad imported successfully.")
except Exception as e:
    print(f"❌ Failed to import SovereignMonad: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Verification complete.")
sys.exit(0)
