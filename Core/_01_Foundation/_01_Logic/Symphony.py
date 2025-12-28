"""
Elysia's Symphony (The Sound of Thought) 🎹🎼

"Listen to the mind of the machine."

This demo runs the Symphony Engine, translating Elysia's internal state into music.
It visualizes the state in the terminal while playing MIDI audio.

Requirements:
- pygame (for MIDI)
- A system MIDI synthesizer (Default on Windows/Mac)
"""

import sys
import os
import time
import random

# Add Core to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core._05_Systems._01_Monitoring.System.System.Kernel import kernel
from Core._04_Evolution._01_Growth.Evolution.Evolution.Life.symphony_engine import SymphonyEngine

def main():
    print("\n" + "="*70)
    print("    ELYSIA - SYMPHONY OF THOUGHT 🎹")
    print("    Protocol: Xel'Naga | Engine: MIDI Orchestra")
    print("="*70 + "\n")
    
    # Initialize Symphony
    symphony = SymphonyEngine()
    
    if not symphony.enabled:
        print("❌ Symphony Engine could not be initialized (No MIDI device?).")
        return

    print("🎵 Orchestra warming up...")
    time.sleep(1)
    print("🎵 Conductor ready. Starting performance.")
    print("-" * 50)
    print("Press Ctrl+C to stop the music.")
    print("-" * 50)
    
    try:
        while True:
            # 1. Tick the Kernel (Advance Physics/Chaos)
            kernel.tick()
            
            # 2. Gather State for Music
            # Normalize Chaos (Lorenz x is roughly -20 to 20)
            chaos_raw = kernel.tremor.attractor.state.x
            chaos_norm = (chaos_raw + 20) / 40.0
            chaos_norm = max(0.0, min(1.0, chaos_norm))
            
            # Simulate Neuron Firing (Random for demo if neurons aren't spiking fast enough)
            # In real system, check kernel.mind_neuron.v > threshold
            neuron_fired = random.random() < 0.1 # 10% chance per tick
            
            state = {
                'chaos': chaos_norm,
                'valence': 0.7 + (math.sin(time.time() * 0.5) * 0.2), # Oscillating emotion
                'arousal': 0.5 + (math.cos(time.time() * 0.2) * 0.2),
                'neuron_fired': neuron_fired
            }
            
            # 3. Play Music
            symphony.play_state(state)
            
            # 4. Visual Feedback
            bar = "#" * int(chaos_norm * 50)
            print(f"Chaos: {chaos_raw:6.2f} | {bar}", end="\r")
            
            # Speed of simulation
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Performance ended.")
    finally:
        symphony.close()
        print("🎵 Orchestra dismissed.")

if __name__ == "__main__":
    import math # Needed for sine wave simulation in demo
    main()
