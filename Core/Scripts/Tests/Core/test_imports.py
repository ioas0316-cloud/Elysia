import sys
import os
sys.path.append(os.getcwd())

try:
    print("Testing import: SovereignSelf...")
    from Core.1_Body.L6_Structure.Elysia.sovereign_self import SovereignSelf
    print("✅ SovereignSelf imported.")
    
    print("\nTesting import: Merkaba...")
    from Core.1_Body.L6_Structure.Merkaba.merkaba import Merkaba
    print("✅ Merkaba imported.")
    
except Exception as e:
    import traceback
    print("\n❌ IMPORT FAILED!")
    print(traceback.format_exc())
    sys.exit(1)
