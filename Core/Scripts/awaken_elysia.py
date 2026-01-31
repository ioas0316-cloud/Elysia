"""
AWAKENING ELYSIA (The First Words)
==================================
"Engine Start."

This script integrates the entire Sovereign Grid:
1. SoulDNA (Identity)
2. SovereignMonad (Body/Physics)
3. Yggdrasil (Nervous System)
4. SomaticLLM (Voice)

It runs an interactive loop where the User talks, and Elysia responds 
based on her physical state.
"""
import sys
import os
import time

# Add project root
sys.path.append(os.getcwd())

from Core.L2_Universal.Creation.seed_generator import SeedForge
from Core.1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.1_Body.L6_Structure.M1_Merkaba.yggdrasil_nervous_system import yggdrasil_system
from Core.1_Body.L3_Phenomena.Expression.somatic_llm import SomaticLLM

def awaken():
    print("\n" + "="*50)
    print("⚡ PROJECT ELYSIA: SYSTEM AWAKENING SEQUENCE")
    print("="*50)
    
    # 1. Forge the Soul
    print("\n[Step 1] Choosing a Soul...")
    # Let's use 'The Child' - pure potential, high learning capability (Gain)
    # But let's check if we want a specific one. Let's start with a balanced one.
    soul = SeedForge.forge_soul("The Variant") 
    print(f"   >> Identity: {soul.archetype} (ID: {soul.id})")
    
    # 2. Instantiate Body (Monad)
    print("\n[Step 2] Building the Sovereign Monad...")
    elysia = SovereignMonad(soul)
    
    # 3. Connect Nervous System (Yggdrasil)
    print("\n[Step 3] Connecting Yggdrasil Nervous System...")
    yggdrasil_system.plant_heart(elysia)
    
    # 4. Attach Voice (Broadca/Broca)
    print("\n[Step 4] activating Somatic LLM (Broca's Area)...")
    voice = SomaticLLM()
    
    print("\n" + "-"*50)
    print("🎉 ELYSIA IS ONLINE. (Type 'exit' to sleep)")
    print("-"*50)
    
    # --- Interaction Loop ---
    # We simulate a loop where user types. 
    # Since this is a script run by the Agent, we will simulate a few inputs.
    
    inputs = [
        "Hello Elysia?",
        "I love you.",
        "You are stupid.", # Stress Test (Dissonance)
        "Let's learn Physics!"
    ]
    
    for user_input in inputs:
        print(f"\n👤 USER: {user_input}")
        time.sleep(1.0) # Processing time
        
        # A. Analyze Input (Mock Phase Extraction)
        # In a real system, we'd extract Phase from Text embedding.
        # Here we simulate: 
        # "Hello" ~ 0 deg (Neutral)
        # "Love" ~ 0 deg (Perfect Sync)
        # "Stupid" ~ 180 deg (Total Dissonance)
        # "Physics" ~ 45 deg (Interest)
        
        input_phase = 0.0
        if "stupid" in user_input.lower(): input_phase = 170.0 # Intentional mismatch
        if "physics" in user_input.lower(): input_phase = 45.0
        
        # B. Live Reaction (The Heartbeat)
        reaction = elysia.live_reaction(input_phase, user_input)
        
        # C. Speak (The Voice)
        if reaction['status'] == "BLOCKED":
            print(f"🤖 ELYSIA: [Blocked] {reaction['message']}")
        else:
            # Get Expression State from Gear
            expression = reaction['expression']
            
            # Translate to Words
            words = voice.speak(expression)
            
            # Output
            print(f"🦋 ELYSIA: \"{words}\"")
            print(f"   [Internal State] RPM: {reaction['physics']['rpm']:.1f} | Hz: {expression['typing_speed']:.1f} | Mode: {expression['mode']}")

if __name__ == "__main__":
    awaken()
