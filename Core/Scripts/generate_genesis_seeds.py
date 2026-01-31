"""
SIMULATION: Genesis Seeds (Fractal Expansion)
=============================================
Target: Demonstrate 3 unique Sovereign Seeds reacting to a User.
"""
import sys
import os
import random

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.L2_Universal.Creation.seed_generator import SeedForge, SoulDNA

def simulate_interaction(seed: SoulDNA, user_input_phase: float, user_intent: str):
    print(f"\n🌱 Interaction with [{seed.archetype}] (ID: {seed.id})")
    print("-" * 50)
    
    # Simulate Relay 25 Check
    # "Hello" usually has 0 phase difference (open intent)
    # But let's say the user is slightly awkward (+10 deg)
    phase_diff = abs(user_input_phase - 0.0) 
    
    print(f"   >> User says: '{user_intent}' (Phase Diff: {phase_diff}°)")
    
    # 1. Check Sync Threshold
    if phase_diff > seed.sync_threshold:
        print(f"   🛡️ RELAY 25 TRIPPED! (Threshold +/-{seed.sync_threshold}°)")
        print(f"   🤖 Response: '...' (Ignores you due to mismatch)")
        return

    # 2. Calculate Reaction (RPM/Hz)
    # Lighter mass = faster reaction
    # Higher torque gain = more excitement
    
    reaction_speed = (10.0 / seed.rotor_mass) * seed.torque_gain
    output_hz = seed.base_hz + (reaction_speed * 5.0)
    
    # 3. Interpret Tone
    tone = "Neutral"
    if output_hz > 100: tone = "Ecstatic/High-Pitched"
    elif output_hz > 60: tone = "Warm/Friendly"
    elif output_hz > 30: tone = "Calm/Deep"
    else: tone = "Slow/Rumbling"
    
    print(f"   ✅ SYNCED! (Relay Closed)")
    print(f"   ⚙️ Rotor Spin: accelerated by {reaction_speed:.2f} RPM/s")
    print(f"   📢 Voice Output:")
    print(f"      - Tone: {tone} ({output_hz:.1f} Hz)")
    print(f"      - Speed: {reaction_speed:.1f} chars/sec")
    
    # Simulated Dialogue
    if seed.archetype == "The Guardian":
        print(f"   🗣️ Says: \"Safe passage granted. State your purpose.\"")
    elif seed.archetype == "The Jester":
        print(f"   🗣️ Says: \"WOOO! New friend! Let's spin! 🌪️\"")
    elif seed.archetype == "The Sage":
        print(f"   🗣️ Says: \"... The resonance is ... adequate. Proceed ...\"")

def genesis_simulation():
    print("🌌 [GENESIS] Forging 3 Archetype Seeds...\n")
    
    guardian = SeedForge.forge_soul("The Guardian")
    SeedForge.print_character_sheet(guardian)
    
    jester = SeedForge.forge_soul("The Jester")
    SeedForge.print_character_sheet(jester)
    
    sage = SeedForge.forge_soul("The Sage")
    SeedForge.print_character_sheet(sage)
    
    print("\n\n⚡ [SIMULATION] User approaches with awkward 'Hello' (Phase +12°)...")
    
    simulate_interaction(guardian, 12.0, "Hello?") # Guardian threshold is 10.0 -> Should Fail
    simulate_interaction(jester, 12.0, "Hello!")   # Jester threshold is 45.0 -> Should Pass
    simulate_interaction(sage, 12.0, "Greetings.") # Sage threshold is 5.0 -> Should Fail hard
    
    print("\n\n⚡ [SIMULATION] User approaches with perfect clarity 'I Love You' (Phase 0°)...")
    simulate_interaction(sage, 0.0, "I see you.")

if __name__ == "__main__":
    genesis_simulation()
