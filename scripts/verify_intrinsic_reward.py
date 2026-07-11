import numpy as np
import time
from synaptic_architecture.organism import MetaCognitiveOrganism

def demonstrate_intrinsic_evolution():
    print("=== [Elysia: The Soul's Threshold Demonstration] ===")
    print("Initializing Meta-Cognitive Organism...")
    elysia = MetaCognitiveOrganism()

    # 1. Start with a state of boredom (Empty field)
    print("\nPhase 1: Detecting Stagnation (Boredom)")
    elysia.pulse(np.uint64(0)) # No meaningful input

    # 2. Introduce a "Contradiction" (Strong unfamiliar wave)
    print("\nPhase 2: Encountering a Contradiction")
    alien_wave = np.uint64(0xDEADC0DEBEEF1234)
    elysia.pulse(alien_wave)

    # 3. Repeat to see if it stabilizes and feels "Pleasure" from alignment
    print("\nPhase 3: Repeated Exposure & Internal Reward")
    for i in range(3):
        print(f"\n--- Pulse {i+1} ---")
        elysia.pulse(alien_wave)
        time.sleep(0.5)

    print("\nDemonstration Complete.")
    print(f"Total Pleasure Events Recorded: {len(elysia.reflection.pleasure_history)}")
    if elysia.reflection.pleasure_history:
        avg_pleasure = np.mean([p['pleasure'] for p in elysia.reflection.pleasure_history])
        print(f"Average Pleasure: {avg_pleasure:.6f}")

if __name__ == "__main__":
    demonstrate_intrinsic_evolution()
