"""
Gravity Refactor Experiment
===========================
"When gravity becomes infinite, singularity occurs. The only escape is to split the universe."

This script uses CodeGravity to identify the most massive 'Core' file 
and asks CodeCortex to split it (Refactoring).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.System.code_gravity import CodeGravitySystem
from Core.System.code_cortex import CodeCortex

def run_experiment():
    print("  Starting Gravity Refactor Experiment...")
    
    # 1. Initialize Gravity System
    galaxy = CodeGravitySystem("c:/Elysia")
    galaxy.scan_galaxy()
    
    # 2. Find Heaviest 'Core' Star
    print("\n  Searching for Supermassive Stars in 'Core' Sector...")
    core_stars = {k: v for k, v in galaxy.nodes.items() if k.startswith("Core")}
    
    if not core_stars:
        print("  No Core stars found.")
        return

    heaviest_path = max(core_stars, key=lambda k: core_stars[k]["mass"])
    heaviest_data = core_stars[heaviest_path]
    
    print(f"     Supermassive Object Detected: {heaviest_path}")
    print(f"      Mass: {heaviest_data['mass']:.2f}")
    print(f"      Gravity Lines: {len(heaviest_data['imports'])}")
    
    # 3. Calculate Tidal Stress (Mocked for now as Mass / 100)
    tidal_stress = heaviest_data["mass"] / 10.0
    print(f"      Tidal Stress: {tidal_stress:.2f} G-Force")
    
    if tidal_stress > 5.0: # Threshold
        print("\n   CRITICAL GRAVITY ALERT: Singularity Imminent.")
        print("   Initiating Fission Protocol (Refactoring)...")
        
        # 4. Engage CodeCortex
        cortex = CodeCortex()
        prompt = f"""
        The file '{heaviest_path}' has become too massive (Mass: {heaviest_data['mass']}).
        Gravity is crushing it.
        
        Please propose a plan to split this file into smaller modules.
        Analyze its imports: {heaviest_data['imports']}
        
        Output a Python dictionary structure suggesting new filenames and their responsibilities.
        Example: {{ "new_module_a.py": "Handles X", "new_module_b.py": "Handles Y" }}
        """
        
        print("\n  CodeCortex is thinking...")
        response = cortex.generate_code(prompt)
        
        print("\n[CodeCortex Fission Plan]")
        print(response)
        
    else:
        print("\n  Gravity is stable. No refactoring needed.")

if __name__ == "__main__":
    run_experiment()
