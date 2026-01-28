import sys
import os
sys.path.append(os.getcwd())

try:
    print("Testing import: SovereignSelf...")
    from Core.L1_Foundation.M1_Keystone.emergent_self import EmergentSelf as SovereignSelf
    print("✅ SovereignSelf imported.")
    
    print("\nTesting import: Merkaba...")
    from Core.L6_Structure.Merkaba.merkaba import Merkaba
    print("✅ Merkaba imported.")
    
except Exception as e:
    import traceback
    print("\n❌ IMPORT FAILED!")
    print(traceback.format_exc())
    sys.exit(1)
