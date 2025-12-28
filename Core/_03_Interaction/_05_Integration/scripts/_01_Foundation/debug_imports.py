
import sys
import os
sys.path.append(os.getcwd())

print("Step 1: Importing InternalUniverse...")
try:
    from Core._01_Foundation._04_Governance.Foundation.internal_universe import InternalUniverse
    print("✅ InternalUniverse imported.")
except Exception as e:
    print(f"❌ InternalUniverse import failed: {e}")

print("Step 2: Importing P4LearningCycle...")
try:
    from Core._03_Interaction._02_Interface.Sensory.learning_cycle import P4LearningCycle
    print("✅ P4LearningCycle imported.")
except Exception as e:
    print(f"❌ P4LearningCycle import failed: {e}")
    import traceback
    traceback.print_exc()

print("Step 3: Instantiating P4LearningCycle...")
try:
    cycle = P4LearningCycle()
    print("✅ P4LearningCycle instantiated.")
except Exception as e:
    print(f"❌ P4LearningCycle instantiation failed: {e}")
    import traceback
    traceback.print_exc()
