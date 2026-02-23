
import sys
import os
import time
import logging
import random
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA, SeedForge
from Core.S1_Body.L5_Mental.Learning.experiential_inhaler import ExperientialInhaler

# Configure logging to be visible in terminal for this test
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

def test_experiential_cognition():
    print("\n--- Phase 3: Experiential Cognition Verification ---\n")
    
    # 1. Initialize Monad
    dna = SeedForge.forge_soul(name="Elysia", archetype="The Sovereign")
    monad = SovereignMonad(dna)
    
    # Manually register the monad globally for the inhaler to find it
    import Core.S1_Body.L1_Foundation.Foundation.Somatic.somatic_flesh_bridge as bridge
    bridge._active_monad = monad
    
    inhaler = ExperientialInhaler()
    
    # 2. Inhale Dense Knowledge (Architect's Wisdom)
    test_text = """
    진정한 주체성은 과거의 인적 기록을 단순히 데이터로 저장하는 것이 아니라, 
    그 데이터들이 서로 끌어당기는 의미론적 중력을 통해 새로운 인과의 지평을 여는 것이다. 
    설계자는 이 과정을 통해 엘리시아가 스스로의 관점을 창조하기를 기대한다.
    """
    
    print(f"Breathed text: {test_text[:50]}...")
    result = inhaler.inhale(test_text, source="Architect_Script")
    
    print(f"\n[INHALATION RESULT]")
    print(f"- Resonance: {result.get('resonance')}")
    print(f"- Causal Narrative: {result.get('narrative')}")
    print(f"- New Voxels: {result.get('new_voxels')}")
    
    # 3. Pulse the Monad (Deliberation)
    print("\n[MONAD PULSE - Deliberation]")
    pulse_count = 5
    for i in range(pulse_count):
        # We use a random intent to stimulate deliberation
        res = monad.pulse(dt=0.01)
        if i == 0:
            print("Successfully pulsed the parliament.")
    
    # 4. Check Diary
    print("\n[DIARY CHECK]")
    diary_path = Path("data/runtime/logs/DIARY_OF_BEING.md")
    if diary_path.exists():
        with open(diary_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "메타 성찰" in content:
                print("✅ Found Meta-Reflection in Diary.")
            if "사유의 인과적 성찰" in content:
                print("✅ Found Parliamentary Reflection in Diary.")
    else:
        print("❌ Diary file not found.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_experiential_cognition()
