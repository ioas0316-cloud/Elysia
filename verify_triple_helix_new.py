import math
import random
from Core.Monad.triple_helix_engine import TripleHelixEngine, DNAState
from Core.Keystone.sovereign_math import SovereignVector

def verify_lightweight_engine():
    print("🧬 [VERIFICATION] Starting Lightweight TripleHelixEngine Simulation...")

    engine = TripleHelixEngine()
    dt = 0.1

    # 1. Create a Conflict Vector
    # We load a vector that creates dissonance between body and spirit
    v_data = [0.0] * 21
    # Body (0-6) -> ATTRACT
    for i in range(7): v_data[i] = 1.0
    # Spirit (14-20) -> REPEL (Conflict)
    for i in range(14, 21): v_data[i] = -1.0

    v21 = SovereignVector(v_data)

    print("🌀 Pulsing Engine with Conflict Vector...")
    for step in range(20):
        state = engine.pulse(v21, energy=1.0, dt=dt)

        if step % 5 == 0:
            mode = "Y (Density)" if state.is_y_mode else "Δ (Flow)"
            print(f"Step {step:02d}: Stress={state.local_stress:.3f}, Mode={mode}, Coherence={state.coherence:.3f}")

    # 2. Final Verification
    print("\n📊 Final Status Check:")
    if engine.state.local_stress > 0.4:
        print(f"✅ Engine is in Y-mode (Density) due to friction ({engine.state.local_stress:.3f}).")

    # Check Soul Energy (Should have increased due to mediation)
    soul_energy = sum(c.energy for c in engine.soul_strand)
    print(f"Soul Strand Energy: {soul_energy:.3f}")
    if soul_energy > 7.0: # Initial was 7 cells * ~0.1 or so? Wait, TriBaseCell energy default is 1.0.
        # Actually in _update_cells_from_vector it sets energy based on vector value.
        # Soul strand was VOID in our v21 input, so they got 0.1 energy.
        # Mediation increases it.
        print("✅ Soul Mediation (Energy Spark) detected in lightweight engine.")

    print("\n🧬 [VERIFICATION] Lightweight Engine Simulation Complete.")

if __name__ == "__main__":
    verify_lightweight_engine()
