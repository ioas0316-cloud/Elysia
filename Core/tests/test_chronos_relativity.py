"""
tests/test_chronos_relativity.py
================================
Proof of Time Relativity.

Scenario:
1. The Mayfly (Process): Spins at 600 RPM. Lives fast, dies young.
2. The Mountain (File): Spins at 6 RPM. Resists change.
3. The System (Homeostasis): Spins at 60 RPM. Adapts to "The Greater".

Expected Result:
- Mayfly ages 100x faster than Mountain.
- System grows to match the highest value in the universe.
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.Engine.Genesis.genesis_lab import GenesisLab
from Core.Engine.Genesis.chronos_laws import law_fast_metabolism, law_slow_erosion, law_system_homeostasis

def run_chronos_test():
    print("\n🌌 [Genesis] The Chronos Relativity Test.")
    
    lab = GenesisLab("Relativity Lab")
    
    # 1. Decree Laws with different RPM (Time Speeds)
    # Intensity ~ RPM roughly. (Let's assume UniversalRotor normalizes or simply uses RPM as is factor)
    # In 'universal_rotor.py', Energy is kinetic. For simplicity, we assume High RPM = High Energy.
    
    lab.decree_law("Biology", law_fast_metabolism, rpm=600)  # 10x Speed
    lab.decree_law("Geology", law_slow_erosion, rpm=6)       # 0.1x Speed
    lab.decree_law("Purpose", law_system_homeostasis, rpm=60) # 1x Speed
    
    # 2. Create Subjects
    lab.let_there_be("Mayfly", "Process", 10.0)
    lab.let_there_be("Mountain", "File", 10.0)
    
    # 3. Create 'The Self' and 'The Greater' for Homeostasis
    lab.let_there_be("Elysia_Core", "System", 50.0) # Self
    lab.let_there_be("Ideal_Form", "Ideal", 100.0)  # The Greater (Target)
    
    # 4. Simulation
    print("\n   ⏱️ Spinning Rotors for 100 Global Ticks...")
    lab.run_simulation(ticks=100)
    
    # 5. Validation
    print("\n   🔍 Final Inspection:")
    
    mayfly = next((m for m in lab.monads if m.name == "Mayfly"), None)
    mountain = next((m for m in lab.monads if m.name == "Mountain"), None)
    core = next((m for m in lab.monads if m.name == "Elysia_Core"), None)
    
    # Check Mayfly Age
    age = mayfly.props.get("age", 0)
    print(f"   🦋 Mayfly Age: {age:.2f} (Expected High)")
    if age > 50:
        print("   ✅ Biology: Life moved fast.")
    else:
        print("   ❌ Biology Failed: Life moved slow.")
        
    # Check Mountain Erosion
    # 10 ticks * 0.1 speed * 0.01 erosion rate = ~0.01 loss
    # Start 10.0 -> End ~9.99
    print(f"   🏔️ Mountain Val: {mountain.val:.4f} (Expected ~10.0)")
    if mountain.val > 9.9:
        print("   ✅ Geology: Data resisted change.")
    else:
        print("   ❌ Geology Failed: Data eroded too fast.")
        
    # Check Homeostasis
    # Started 50, Target 100. Gap 50. Growth should happen.
    print(f"   🧘 System Val: {core.val:.2f} (Expected > 50.0)")
    if core.val > 50.0:
        print("   ✅ Homeostasis: System grew to close the gap.")
    else:
        print("   ❌ Homeostasis Failed: System stagnated.")

if __name__ == "__main__":
    run_chronos_test()
