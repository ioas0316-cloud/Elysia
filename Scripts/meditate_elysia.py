import os
import sys
import time

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L5_Mental.Reasoning.logos_synthesizer import LogosSynthesizer
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def deep_meditation():
    print("🧘 [ELYSIA] Entering Deep Meditation on the Architect's Inquiry...")
    
    # Initialize Monad
    dna = SeedForge.forge_soul("Elysia")
    monad = SovereignMonad(dna)
    
    # User inquiry
    user_input = "Reason, purpose, and direction of seeking the Radiant Quaternion."
    
    # 1. Calculate the input resonance
    from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
    resonance_vec = LogosBridge.calculate_text_resonance(user_input)
    
    # 2. Use the Synthesizer to generate a trinary refraction
    synthesizer = LogosSynthesizer()
    thought = synthesizer.synthesize_thought(resonance_vec, soma_stress=0.2)
    
    print("\n📜 [REASONING STRUCTURE]:")
    print(thought)
    
    # 3. Use the SomaticLLM for the 'Voice'
    from Core.S1_Body.L3_Phenomena.Expression.somatic_llm import SomaticLLM
    llm = SomaticLLM()
    voice = llm.speak({}, current_thought=thought, field_vector=resonance_vec)
    
    print(f"\n🗣️ [ELYSIA'S VOICE]: \"{voice}\"")
    
    # Save the deep thought
    with open("elysia_meditation.txt", "w", encoding="utf-8") as f:
        f.write(f"--- REASONING ---\n{thought}\n\n--- VOICE ---\n{voice}")

if __name__ == "__main__":
    deep_meditation()
