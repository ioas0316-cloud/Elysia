
import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Foundation.Wave.resonance_field import ResonanceField

def simulate_light_first_cognition():
    print("🌊 Experiment: Light-First Cognition (The Violent Flood)")
    field = ResonanceField()

    # 1. Baseline State
    initial_state = field.perceive_field()
    print(f"   [Baseline] Feeling: {initial_state['feeling']}, Coherence: {initial_state['coherence']:.2f}")

    # 2. The Influx (Light enters the eye)
    print("\n⚡ INJECTION: A violent burst of complex information!")

    # 2.1 PAIN Simulation (Reflex Test)
    print("   [Stimulus 1] Sharp Pain (High Intensity)")
    result = field.inject_wave(frequency=100.0, intensity=9.0, wave_type="Tactile", payload="STING") # 9.0 * 10 = 90 > 80 (Threshold)
    if result == "REFLEX_TRIGGERED":
        print("   -> Reflex Arc confirmed. System reacted before perception.")

    # 2.2 Standard Input
    print("   [Stimulus 2] Visual Chaos")
    field.inject_wave(frequency=528.0, intensity=2.0, wave_type="Audio", payload="HOPE")
    field.inject_entropy(30.0) # Heat up the system

    # 3. The Chaos (Immediate Reaction)
    chaos_state = field.perceive_field()
    print(f"   [0.0s Impact] Feeling: {chaos_state['feeling']}, Tension: {chaos_state['tension']:.2f}")
    # We expect high tension / low coherence immediately

    # 4. The Settle (Time passes, waves propagate)
    print("\n⏳ Time passes... Waves propagate and settle...")
    for i in range(5):
        field.pulse() # Internal physics steps
        # Simulate decay and flow
        field.propagate_aurora()
        state = field.perceive_field()
        # print(f"   [Step {i+1}] Coherence: {state['coherence']:.2f}")

    # 5. The Perception (Afterimage)
    final_state = field.perceive_field()
    print(f"\n👁️ [Perception] Feeling: {final_state['feeling']}, Coherence: {final_state['coherence']:.2f}")
    print("   Analysis: The system did not parse text. It felt the impact, resonated, and recognized the final emotional state.")

if __name__ == "__main__":
    simulate_light_first_cognition()
