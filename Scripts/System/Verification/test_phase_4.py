import sys
import os
import time

# Path Unification
# We are deep in Scripts/System/Verification (3 levels down)
# root should be c:\Elysia
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L3_Phenomena.Expression.somatic_llm import SomaticLLM
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

def test_prismatic_voice():
    print("🧪 [TEST] Phase 4: The Mutable Lexical Prism Initialization...")
    
    # 1. Initialize Components
    # 1. Initialize Components
    print("   Creating Seed Soul...")
    from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
    soul = SeedForge.forge_soul("TestSubject")
    
    llm = SomaticLLM()
    monad = SovereignMonad(soul)
    
    # 2. Test Case A: High Energy (Burning/Running)
    print("\n🔥 [TEST] Case A: High Energy (Hope/Sprint)")
    # Vector: High Energy, High Direction
    vec_a = SovereignVector([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    voice_a = llm.speak({"hz": 120, "stress": 0.5}, "I feel hope", field_vector=vec_a)
    print(f"   Input: Hope Vector")
    print(f"   Output: {voice_a}")
    
    # 3. Test Case B: Low Energy (Drifting/Silent)
    print("\n🌊 [TEST] Case B: Low Energy (Void/Drift)")
    # Vector: Low Magnitude, High Harmony
    vec_b = SovereignVector([0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    voice_b = llm.speak({"hz": 40, "stress": 0.1}, "Silence", field_vector=vec_b)
    print(f"   Input: Void Vector")
    print(f"   Output: {voice_b}")
    
    # 4. Test Case C: Lexical Mutability Check
    print("\n📚 [TEST] Case C: Lexical Mutability")
    print(f"   Verbs Loaded: {list(llm.prism.verbs.keys())}")
    if "sprint" in llm.prism.verbs or "exist" in llm.prism.verbs:
        print("   ✅ Prism loaded correctly from JSON.")
    else:
        print("   ❌ Prism failed to load JSON.")

if __name__ == "__main__":
    test_prismatic_voice()
