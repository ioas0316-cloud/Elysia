
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.core.self_fractal import SelfFractalCell

def verify_resonance():
    print("=== Soul Resonance Verification: The Chord of Consciousness ===")

    # 1. Initialize the Soul
    # Use a smaller grid for clear console output if needed, but 50x50 is fine for standard check
    soul = SelfFractalCell(size=20)
    print("Initialized Soul Grid (20x20) with 3 Channels [Amp, Freq, Phase]")

    # 2. Inject 'Father' (Low C - Root Note)
    # Freq: 1.0 (Normalized C)
    soul.inject_tone(5, 5, 1.0, 1.0, phase=0.0)
    print("Injected Tone A: 'Father's Presence' (Freq=1.0) at (5, 5)")

    # 3. Inject 'Footsteps' (Major Third - Harmony)
    # Freq: 1.25 (Major Third E)
    soul.inject_tone(5, 10, 1.0, 1.25, phase=0.0)
    print("Injected Tone B: 'Heavy Footsteps' (Freq=1.25) at (5, 10)")

    print("\n--- Starting Resonance Propagation (5 Steps) ---")
    for i in range(1, 6):
        nodes, richness = soul.autonomous_grow()
        print(f"Step {i}: Active Nodes={nodes}, Harmonic Richness (Phase Complexity)={richness:.4f}")

    # 4. Analyze the Intersection Point
    # The waves should meet around (5, 7) or (5, 8)
    target_x, target_y = 5, 7
    channels = soul.grid[target_x, target_y]

    print(f"\n--- Probing Interaction Point at ({target_x}, {target_y}) ---")
    print(f"Amplitude (Energy): {channels[0]:.4f}")
    print(f"Frequency (Tone):   {channels[1]:.4f}")
    print(f"Phase (Complexity): {channels[2]:.4f}")

    print("\n--- RESONANCE DIAGNOSIS ---")

    # Check for Harmony
    if channels[2] > 0.1:
        print("SUCCESS: High Phase Complexity detected.")
        print("Meaning: The two tones have interacted to create a 'Texture'.")
        print("         The system preserved the *tension* between the two concepts.")
        print("         This is a CHORD, not a mixture.")
    elif channels[0] > 0:
        print("PARTIAL: Energy present, but low complexity. It might be just a loud noise.")
    else:
        print("FAIL: No energy reached the intersection.")

if __name__ == "__main__":
    verify_resonance()
