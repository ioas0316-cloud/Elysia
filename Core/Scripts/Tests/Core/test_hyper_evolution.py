"""
tests/test_hyper_evolution.py
=============================
The Grand Unification of Principles.

This script demonstrates the "Time Axis Rotor" in action.
It spins the universe through:
1. Physics (NAND Gates)
2. Architecture (ALU Addition)
3. Operating System (Round Robin & Paging)

Showing that Elysia now contains the seed of the entire system.
"""

import sys
import os

# Ensure Core is visible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Core.1_Body.L6_Structure.Engine.Genesis.genesis_lab import GenesisLab
from Core.1_Body.L6_Structure.Engine.Genesis.hyper_time_rotor import HyperTimeRotor

def run_grand_evolution():
    print("\n🌌 [Genesis] The Hyper-Time Evolution Begins.")
    
    # Initialize the Laboratory
    lab = GenesisLab("The Omni-Computer")
    
    # Initialize the Time Engine
    time_rotor = HyperTimeRotor(lab)
    
    # Execute the History of Computing
    time_rotor.run_evolution_sequence()
    
    print("\n✅ The Seed has Germinated. History is Internalized.")

if __name__ == "__main__":
    run_grand_evolution()
