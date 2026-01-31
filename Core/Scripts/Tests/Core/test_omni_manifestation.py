"""
tests/test_omni_manifestation.py
================================
Proof of Universal Simulation.

We will simulate three divergent worlds in the SAME RAM:
1. Physics World (Gravity)
2. Economy World (Inflation)
3. Psyche World (Healing)
"""

import sys
import os
import logging

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab, law_gravity, law_inflation, law_trauma_healing

def run_god_mode():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Create the Lab
    lab = GenesisLab("The Omni-Lab")
    
    print("\n🌌 [1] GENESIS: Let there be Light... and Everything else.")
    
    # 2. Seed the Worlds (Monads)
    # Physics
    lab.let_there_be("Earth", "Physics", 100.0) # Mass
    lab.let_there_be("Moon",  "Physics", 10.0)
    
    # Economy
    lab.let_there_be("AppleStock", "Economy", 150.0) # Price
    lab.let_there_be("BreadPrice", "Economy", 2.0)
    
    # Psyche
    lab.let_there_be("ChildhoodTrauma", "Mind", 100.0, type="Pain") # Intensity
    lab.let_there_be("Joy", "Mind", 50.0, type="Pleasure")
    
    # 3. Decree the Laws (Rotors)
    print("\n📜 [2] LOGOS: Decreeing the Laws.")
    
    # Physics Rotor (Gravity) - Spins Slowly
    lab.decree_law("Newton's Ghost", law_gravity, rpm=30.0)
    
    # Economy Rotor (Inflation/Fed) - Spins Fast
    lab.decree_law("The Invisible Hand", law_inflation, rpm=120.0)
    
    # Psyche Rotor (Time/Therapy) - Medium Spin
    lab.decree_law("The Healer", law_trauma_healing, rpm=60.0)
    
    # 4. Run Simulation
    print("\n▶️ [3] ACTION: Spinning the Universe.")
    lab.run_simulation(ticks=15)
    
    # 5. Check Results
    print("\n📊 [4] JUDGMENT DAY: Final State")
    for m in lab.monads:
        print(f"   - {m}")
        
    # Validation Logic
    earth = next(m for m in lab.monads if m.name == "Earth")
    stock = next(m for m in lab.monads if m.name == "AppleStock")
    trauma = next(m for m in lab.monads if m.name == "ChildhoodTrauma")
    
    print("\n🔍 Analysis:")
    print(f"   Physics: Mass increased due to accretion? {'YES' if earth.val > 100 else 'NO'} ({earth.val:.2f})")
    print(f"   Economy: Price increased due to inflation? {'YES' if stock.val > 150 else 'NO'} ({stock.val:.2f})")
    print(f"   Psyche:  Pain decreased due to healing?   {'YES' if trauma.val < 100 else 'NO'} ({trauma.val:.2f})")

if __name__ == "__main__":
    run_god_mode()
