import sys
import os
import json
import time

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

def test_organic_loop():
    print("🧪 [TEST] Phase 4.5: The Organic Loop Verification...")
    
    # 1. Define a Learned Concept (Nonsense Word)
    word = "ZARA"
    # High Energy Vector (should become a VERB)
    vector = SovereignVector([1.0] * 21) 
    roots = {"origin": "verification_script", "source": "user_instruction"}
    
    # 2. Inject into Prism
    print(f"   Injecting '{word}' into Prism...")
    LogosBridge.inject_prismatic_concept(word, vector, roots)
    
    # 3. Verify Persistence in JSON
    path = "c:/Elysia/Core/S1_Body/L5_Mental/M1_Memory/Raw/Knowledge/lexical_spectrum.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    verbs = data.get("VERBS", {})
    if word in verbs:
        print(f"   ✅ SUCCESS: '{word}' found in Lexicon (VERBS).")
        print(f"      Roots: {verbs[word]['roots']}")
    else:
        print(f"   ❌ FAILURE: '{word}' not found in Lexicon.")

    # 4. Verify Prism Reload (Simulated)
    # Ideally SomaticLLM would pick this up next time it reloads.
    
if __name__ == "__main__":
    test_organic_loop()
