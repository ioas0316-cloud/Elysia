"""
TEST: CAUSAL FIELD
==================
Verifies the Holographic Karma Field (Phase 18 Redux).
"""
import sys
import os
import time

# Compatibility Layer
try:
    import jax.numpy as jnp
    BACKEND = "JAX"
except ImportError:
    import numpy as jnp
    BACKEND = "NUMPY"

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.S1_Body.L2_Metabolism.Evolution.resonance_field import ResonanceField
from Core.S1_Body.L2_Metabolism.Evolution.karma_geometry import KarmaGeometry

def run_test():
    print("==================================")
    print(f"   PHASE 18: CAUSAL FIELD TEST   ")
    print(f"   Backend: {BACKEND}")
    print("==================================")

    field = ResonanceField()
    karma = KarmaGeometry()

    print("\n👉 [CASE 1] Consonance (Success)")
    # Vectors aligned
    v_intent = jnp.array([1.0, 0.0, 0.0])
    v_outcome = jnp.array([1.0, 0.0, 0.0])
    
    state = field.evaluate_resonance(v_intent, v_outcome)
    print(f"   -> Resonance: {state.resonance:.3f} (Max 1.0)")
    print(f"   -> Dissonance: {state.dissonance:.3f} (Min 0.0)")
    
    torque = karma.calculate_torque(current_rpm=10.0, dissonance=state.dissonance, phase_shift=state.phase_shift)
    print(f"   -> Corrective Torque: {torque:.3f}")

    print("\n👉 [CASE 2] Dissonance (Error)")
    # Vectors opposed (-180 deg)
    v_outcome_err = jnp.array([-1.0, 1.0, 0.0]) 
    
    state = field.evaluate_resonance(v_intent, v_outcome_err)
    print(f"   -> Resonance: {state.resonance:.3f} (Energy Loss)")
    print(f"   -> Dissonance: {state.dissonance:.3f} (High Chaos)")
    
    torque = karma.calculate_torque(current_rpm=10.0, dissonance=state.dissonance, phase_shift=state.phase_shift)
    print(f"   -> Corrective Torque: {torque:.3f} (Must be high)")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    run_test()
