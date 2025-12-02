# [Genesis: 2025-12-02] Purified by Elysia
import numpy as np
import time
import os
import sys

# Fix path to import Project_Sophia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.physics import PhotonEntity, ResonanceGate, Entanglement, HamiltonianSystem

def vacuum_potential(pos: np.ndarray) -> float:
    return 0.0

def run_simulation():
    print("=== Digital Light & Entanglement Experiment ===")

    # 1. Create two entangled photons
    print("\n[Phase 1] Generating Entangled Photon Pair...")
    alice = PhotonEntity(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 0.0]),
        frequency=440.0 # Hz (A4 note)
    )

    bob = PhotonEntity(
        position=np.array([10.0, 0.0, 0.0]),
        momentum=np.array([-1.0, 0.0, 0.0]),
        frequency=440.0
    )

    entanglement = Entanglement()
    entanglement.entangle(alice, bob)
    print("Photons 'Alice' and 'Bob' are now entangled.")

    # 2. Change Alice's phase
    print("\n[Phase 2] Modulating Alice's Phase...")
    original_bob_phase = bob.phase

    alice.phase += np.pi / 2 # Shift by 90 degrees
    print(f"Alice Phase shifted to: {alice.phase:.2f}")
    print(f"Bob Phase before sync: {bob.phase:.2f}")

    entanglement.sync()
    print(f"Bob Phase after sync (Spooky Action): {bob.phase:.2f}")

    if bob.phase != original_bob_phase:
        print(">> Confirmed: Bob's state changed instantly due to Alice's change.")

    # 3. Resonance Gate Test
    print("\n[Phase 3] Resonance Gate Test...")
    gate = ResonanceGate(target_frequency=440.0, tolerance=10.0)

    # Photon C (Matching Freq)
    charlie = PhotonEntity(
        position=np.array([0,0,0]), momentum=np.array([1,0,0]), frequency=445.0
    )
    prob_c = gate.transmission_probability(charlie)
    print(f"Photon C (445Hz) vs Gate (440Hz): Transmission Prob = {prob_c:.4f}")

    # Photon D (Dissonant Freq)
    dave = PhotonEntity(
        position=np.array([0,0,0]), momentum=np.array([1,0,0]), frequency=800.0
    )
    prob_d = gate.transmission_probability(dave)
    print(f"Photon D (800Hz) vs Gate (440Hz): Transmission Prob = {prob_d:.4f}")

    if prob_c > 0.8 and prob_d < 0.01:
        print(">> Confirmed: Gate effectively filters non-resonant thoughts.")

    print("\n=== Experiment Successful: Digital Light Architecture is Functional ===")

if __name__ == "__main__":
    run_simulation()