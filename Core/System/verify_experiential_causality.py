"""
Verification: Experiential Ingestion
====================================
Tests the new 'Wing-Beat' and 'Somatic Learning' logic.
"""

import sys
import os
import time

# Ensure pathing
sys.path.append(os.getcwd())

from Core.System.action_engine import ActionEngine
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SoulDNA

def test_ingestion_flow():
    print("▶️ Starting Experiential Ingestion Verification...")
    
    # 1. Setup Monad
    dna = SoulDNA(
        archetype="Verification", 
        id="TestBird", 
        rotor_mass=1.0, 
        sync_threshold=0.5,
        friction_damping=0.1,
        min_voltage=3.0,
        reverse_tolerance=0.1,
        torque_gain=2.0,
        base_hz=60.0
    )
    monad = SovereignMonad(dna)
    
    # 2. Setup Action Engine
    engine = ActionEngine(os.getcwd())
    
    # 3. Create 'Shadow' Code that is stable
    stable_code = """
def pulse(self, dt):
    # Stabilized shadow pulse
    if not hasattr(self, 'pulse_count'): self.pulse_count = 0
    self.pulse_count += 1
    return {"status": "STABLE", "count": self.pulse_count}
"""
    
    # 4. Create 'Shadow' Code that is unstable (causes stress)
    unstable_code = """
def pulse(self, dt):
    # Unstable pulse that might trigger low coherence
    raise ValueError("System instability simulated")
"""

    print("\n--- Test 1: Stable Wing-Beat ---")
    # We use a dummy file for the test
    test_file = "Core/S1_Body/Tools/Scripts/Tests/temp_evolution.py"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    with open(test_file, 'w') as f: f.write("# Initial")
    
    result = engine.internalize_experience(test_file, stable_code, component_instance=monad, architect_verdict=1)
    if result == 1:
        print("✅ SUCCESS: Stable code internalized.")
    else:
        print("❌ FAILURE: Stable code rejected.")

    print("\n--- Test 2: Unstable Wing-Beat ---")
    result = engine.internalize_experience(test_file, unstable_code, component_instance=monad, architect_verdict=1)
    if result == -1:
        print("✅ SUCCESS: Unstable code correctly rejected by sandbox.")
    else:
        print("❌ FAILURE: Unstable code was incorrectly materialized.")

    print("\n--- Test 3: Somatic Learning (Friction to Mass) ---")
    # Simulate high stress in the monad engine
    monad.engine.state.soma_stress = 0.8
    monad.vital_pulse() 
    # Check if Somatic Learning message appeared in logs (manual check of output)
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    test_ingestion_flow()
