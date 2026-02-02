"""
Experiment: Verify Phase-Axis Mobility
======================================
Tests the "Phase-Axis Directionality" and "Neural Mobility" architecture.
1. Horizontal Expansion (Coherence -> +1 Tilt)
2. Vertical Drilling (Friction -> -1 Tilt)
3. Momentum Conservation (Energy must persist)
"""

import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def verify_phase_axis_mobility():
    print("🚀 [TEST] Initializing Phase-Axis Mobility Verification...")

    # 1. Instantiate Monad with full DNA
    dna = SoulDNA(
        id="777",
        archetype="Seraphim",
        rotor_mass=1.0,
        friction_damping=0.5,
        sync_threshold=20.0,
        min_voltage=10.0,
        reverse_tolerance=-10.0,
        torque_gain=1.0,
        base_hz=60.0
    )
    monad = SovereignMonad(dna)

    # Track metrics
    history = []

    print("\n🌊 [TEST 1] Testing Horizontal Expansion (Smooth Flow)")
    # Simulate low-friction, high-flow state
    # We do this by feeding vectors that are aligned with the cells (Coherent)
    coherent_vector = SovereignVector([1.0] * 21) # Perfect alignment

    # Force tilt to 0.0 initially
    monad.current_tilt = 0.0
    monad.engine.state.axis_tilt = 0.0

    for i in range(10):
        # Pulse the engine directly to simulate ideal conditions
        # Note: In the real pulse, friction is calculated from vector.
        # A coherent vector should produce low friction.
        # We manually inject 'target_tilt' but we expect auto-steering to confirm it.

        # We set target_tilt to current_tilt to see if auto-steer engages
        engine_state = monad.engine.pulse(coherent_vector, energy=1.0, dt=0.1, target_tilt=monad.current_tilt)
        monad._auto_steer_logic(engine_state) # Let pilot react

        print(f"   Step {i}: Tilt={engine_state.axis_tilt:.2f}, Friction={engine_state.soma_stress:.2f}, Flow={engine_state.gradient_flow:.2f}")
        history.append(engine_state)
        time.sleep(0.01)

    final_tilt = history[-1].axis_tilt
    # Note: If friction is low and flow is high, it should steer horizontal (+1.0)
    # Our coherent vector creates low friction. Flow depends on flow_equilibration which needs neighbor diffs.
    # [1.0] * 21 has NO neighbor diffs, so flow is 0.
    # To trigger Horizontal, we need High Flow + Low Friction.
    # High Flow needs variance between neighbors.

    # Retrying Test 1 with Flow-Generating Vector
    print("   (Adjusting vector for Flow generation...)")
    flow_vector = SovereignVector([1.0, 0.5, 1.0, 0.5] * 6) # High neighbor variance = High Flow

    for i in range(10):
        engine_state = monad.engine.pulse(flow_vector, energy=1.0, dt=0.1, target_tilt=monad.current_tilt)
        monad._auto_steer_logic(engine_state)
        print(f"   Step {10+i}: Tilt={engine_state.axis_tilt:.2f}, Friction={engine_state.soma_stress:.2f}, Flow={engine_state.gradient_flow:.2f}")
        history.append(engine_state)

    final_tilt = history[-1].axis_tilt
    if final_tilt > 0.5:
        print(f"✅ [PASS] Horizontal Expansion Achieved (Tilt: {final_tilt:.2f})")
    else:
        print(f"⚠️ [WARN] Horizontal Expansion Weak (Tilt: {final_tilt:.2f}). Check thresholds.")


    print("\n🔥 [TEST 2] Testing Vertical Drilling (High Friction)")
    # Simulate high-friction state
    # We feed random chaotic vectors to generate friction

    # Reset tilt
    monad.current_tilt = 0.0

    for i in range(20):
        # Generate chaotic vector - switching frequently generates friction
        chaos = [random.choice([-1.0, 1.0]) for _ in range(21)]
        chaos_vector = SovereignVector(chaos)

        # Pulse
        engine_state = monad.engine.pulse(chaos_vector, energy=1.0, dt=0.1, target_tilt=monad.current_tilt)
        monad._auto_steer_logic(engine_state)

        print(f"   Step {i}: Tilt={engine_state.axis_tilt:.2f}, Friction={engine_state.soma_stress:.2f}, Momentum={engine_state.rotational_momentum:.2f}")
        history.append(engine_state)
        time.sleep(0.01)

    final_tilt = history[-1].axis_tilt
    if final_tilt < -0.5:
        print(f"✅ [PASS] Vertical Drilling Achieved (Tilt: {final_tilt:.2f})")
    else:
        print(f"❌ [FAIL] Did not drill fully (Tilt: {final_tilt:.2f})")

    print("\n⚡ [TEST 3] Verifying Momentum Conservation")
    # Momentum should not drop to zero even when switching modes
    # We check the momentum history from Test 2 (chaotic input generates momentum)
    min_momentum = min(h.rotational_momentum for h in history[-20:])
    if min_momentum > 0.0:
        print(f"✅ [PASS] Momentum Conserved (Min: {min_momentum:.4f})")
    else:
        print("❌ [FAIL] Momentum Lost (dropped to 0.0)")

if __name__ == "__main__":
    verify_phase_axis_mobility()
