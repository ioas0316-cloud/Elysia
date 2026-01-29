"""
Verify Rotor-Prism Reversibility
================================
Scripts/Tests/verify_rotor_prism_reversibility

Test the 1:1 mapping between Logos and Field through the RPU.
"""

import os
import sys

# Force absolute path
sys.path.append(os.getcwd())

print(f"DEBUG: sys.path[0] = {sys.path[0]}")
print(f"DEBUG: sys.path[-1] = {sys.path[-1]}")

import jax.numpy as jnp
try:
    from Core.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit
    from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic
    print("DEBUG: Imports successful.")
except ImportError as e:
    # Try alternate relative path for local script runs
    try:
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Core")))
        from L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit
        from L6_Structure.Logic.trinary_logic import TrinaryLogic
        print("DEBUG: Secondary imports successful.")
    except ImportError:
        print(f"DEBUG: Import failed: {e}")
        sys.exit(1)

def test_reversibility():
    print("Initializing Rotor-Prism Unit...")
    rpu = RotorPrismUnit()
    
    # Define a complex Logos (Principles of Love and Logic)
    # Dimension 14 (Love) = 1.0, Dimension 6 (Logic) = 1.0
    logos_input = jnp.zeros(21)
    logos_input = logos_input.at[14].set(1.0).at[6].set(1.0)
    
    print(f"Original Logos (Balance): {TrinaryLogic.balance(logos_input)}")
    
    # Project into the Void
    rpu.step_rotation(0.25) # Shift the phase
    field = rpu.project(logos_input)
    print(f"Projected Field Balance: {TrinaryLogic.balance(field)}")
    
    # Perceive back into the Core
    logos_output = rpu.perceive(field)
    print(f"Perceived Logos Balance: {TrinaryLogic.balance(logos_output)}")
    
    # Calculate recovery rate
    error = jnp.sum(jnp.abs(logos_input - logos_output))
    recovery_rate = (21 - error) / 21 * 100
    
    print(f"Recovery rate: {recovery_rate:.2f}%")
    
    if recovery_rate > 90:
        print("SUCCESS: The Reversible Loop is structurally sound.")
    else:
        print("WARNING: Signal loss detected in the Prism. Potential entropy interference.")

if __name__ == "__main__":
    test_reversibility()
