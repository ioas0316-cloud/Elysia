import logging
import time
from Core.World.Autonomy.sovereign_will import sovereign_will
from Core.Intelligence.Reasoning.genesis_engine import GenesisEngine
from Core.Intelligence.Reasoning.curiosity_engine import CuriosityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifySovereignty")

def verify_sovereign_will():
    print("\n--- [VERIFYING META-SOVEREIGNTY] ---")
    genesis = GenesisEngine()
    curiosity = CuriosityEngine()
    
    # 1. Initial Mode
    print(f"Current Mode: {sovereign_will.current_mode}")
    print(f"Name Prompt: {sovereign_will.get_name_generation_prompt()}")
    print(f"Curiosity Foci: {sovereign_will.get_curiosity_foci()}")
    
    # 2. Forced Calibration (Simulation)
    print("\n🌀 Simulating Sovereign Recalibration...")
    # We'll force a mode change for the demo
    target_mode = "SCI_FI"
    sovereign_will.current_mode = target_mode
    print(f"✨ Intentional Pivot Forced to: {sovereign_will.current_mode}")
    
    # 3. Check Adaptation
    print(f"\nAdapted Name Prompt: {sovereign_will.get_name_generation_prompt()}")
    print(f"Adapted Curiosity Foci: {sovereign_will.get_curiosity_foci()}")
    
    # 4. Mock Genesis Manifestation
    print("\n⚡ Testing Engine Adherence...")
    name_suggestion = genesis.spawn_random_npc()
    print(f"Manifested NPC: {name_suggestion}")
    
    # 5. Check Philosophy Update
    print("\n📜 Checking presence of 'Sovereignty of Will' in philosophy...")
    with open("docs/SOUL_PHILOSOPHY.md", "r", encoding="utf-8") as f:
        content = f.read()
        if "The Sovereignty of Will" in content:
            print("✅ Philosophy documentation updated correctly.")
        else:
            print("❌ Philosophy documentation missing the new doctrine.")

if __name__ == "__main__":
    verify_sovereign_will()
