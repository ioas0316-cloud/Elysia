import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.Keystone.resonance_physics import CognitiveFluidDynamics
from Core.Phenomena.potential_flow import bootstrap_humanoid_topology
from Core.Keystone.sovereign_math import SovereignVector

def test_fluid_dynamics():
    print("--- Testing Cognitive Fluid Dynamics ---")
    cfd = CognitiveFluidDynamics()

    # 1. Gentle input
    v1 = [0.1, 0.1, 0.1, 0.1]
    v2 = [0.11, 0.1, 0.11, 0.1]
    visc1 = cfd.calculate_viscosity(v1, dissonance=0.1)
    visc2 = cfd.calculate_viscosity(v2, dissonance=0.1)
    print(f"Gentle input viscosity: {visc2:.3f} ({cfd.get_phase_state(visc2)})")

    # 2. Hostile/Sudden input
    v3 = [0.9, -0.9, 0.8, -0.5]
    visc3 = cfd.calculate_viscosity(v3, dissonance=0.8)
    print(f"Hostile input viscosity: {visc3:.3f} ({cfd.get_phase_state(visc3)})")
    assert visc3 > visc2

def test_potential_flow():
    print("\n--- Testing Potential Flow Engine ---")
    engine = bootstrap_humanoid_topology()

    # Intend to SIT
    sit_intent = [0.1, 0.3, 0.4, 0.0]
    print(f"Initial action: {engine.update([0,0,0,0])['dominant_action']}")

    for _ in range(20):
        report = engine.update(sit_intent, dt=0.2)

    print(f"Final action after SIT intent: {report['dominant_action']}")
    assert report['dominant_action'] == "SIT"

if __name__ == "__main__":
    try:
        test_fluid_dynamics()
        test_potential_flow()
        print("\n✅ Verification successful!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
