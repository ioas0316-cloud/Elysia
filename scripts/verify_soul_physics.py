import sys
import os
import time
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from Project_Sophia.core.self_fractal import SelfFractalCell

def print_soul_ecg(tick, soul, event_msg=""):
    """
    Prints a single line 'ECG' status of the soul.
    Format: [Tick] Energy: ||||... (Val) | Freq: 123Hz | Richness: 45.2 | Event
    """
    # Calculate metrics
    total_energy = np.sum(soul.grid[:, :, 0])
    dominant_freq = soul.get_dominant_frequency()
    total_richness = np.sum(soul.grid[:, :, 2])
    active_nodes = np.sum(soul.grid[:, :, 0] > 0.01)

    # Visual Bar for Energy
    bar_len = int(total_energy / 5.0) # Scale down for display
    bar_len = min(bar_len, 20)
    bar = "|" * bar_len
    space = " " * (20 - bar_len)

    print(f"[{tick:03d}] Energy: {bar}{space} ({total_energy:6.2f}) | Voice: {dominant_freq:6.2f}Hz | Richness: {total_richness:6.2f} | Active: {active_nodes:4d} | {event_msg}")

def run_experiment():
    print("--- [ SOUL PHYSICS LABORATORY ] ---")
    print("Experiment: Internal Resonance & Harmonic Complexity")
    print("Goal: Observe if the Self-Fractal Layer produces meaningful 'texture' changes.")
    print("-" * 80)

    # 1. Initialize Soul
    soul = SelfFractalCell(size=30) # Smaller size for easier observation in text
    print(f"Initialized Soul Grid: {soul.size}x{soul.size}")

    # Phase 1: Silence
    print("\n[ PHASE 1: THE VOID ]")
    for t in range(5):
        print_soul_ecg(t, soul)
        soul.autonomous_grow()

    # Phase 2: First Tone (Sorrow - Low Freq, High Amp)
    print("\n[ PHASE 2: IMPULSE - SORROW (100Hz) ]")
    center = soul.size // 2
    # Injecting: x, y, amp, freq, phase
    soul.inject_tone(center, center, amplitude=5.0, frequency=100.0, phase=0.0)
    print_soul_ecg(5, soul, ">>> INJECT: SORROW")

    for t in range(6, 20):
        soul.autonomous_grow()
        print_soul_ecg(t, soul)

    # Phase 3: Second Tone (Hope - High Freq, High Amp) - Offset position
    print("\n[ PHASE 3: RESONANCE - HOPE (500Hz) ]")
    # Injecting slightly off-center to create interaction
    offset = 5
    soul.inject_tone(center + offset, center + offset, amplitude=5.0, frequency=500.0, phase=0.0)
    print_soul_ecg(20, soul, ">>> INJECT: HOPE")

    for t in range(21, 50):
        soul.autonomous_grow()
        print_soul_ecg(t, soul)

    print("\n--- [ EXPERIMENT COMPLETE ] ---")

    # Final Analysis
    final_richness = np.sum(soul.grid[:, :, 2])
    print(f"\nFinal Richness (Complexity): {final_richness:.2f}")
    if final_richness > 10.0:
        print("RESULT: SUCCESS. The interaction created significant harmonic complexity.")
    else:
        print("RESULT: INCONCLUSIVE. The soul remained too simple.")

if __name__ == "__main__":
    run_experiment()
