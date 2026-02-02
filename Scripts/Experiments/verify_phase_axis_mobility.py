"""
Experiment: Verify Phase-Axis Mobility with HUD
===============================================
Tests the "Phase-Axis Directionality" and "Neural Mobility" architecture.
Visualizes the transition using the PhaseHUD Tesseract.

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
from Core.S1_Body.Tools.Debug.phase_hud import PhaseHUD

def verify_phase_axis_mobility():
    print("🚀 [TEST] Initializing Phase-Axis Mobility Verification (with HUD)...")

    # 1. Instantiate Monad
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
    hud = PhaseHUD()

    # Track metrics
    history = []

    print("\n🌊 [TEST 1] Testing Horizontal Expansion (Smooth Flow)")
    time.sleep(1)

    # Flow Vector (Alternating High/Low to create neighbor gradients)
    flow_vector = SovereignVector([1.0, 0.5, 1.0, 0.5] * 6)

    monad.current_tilt_vector = [0.0]

    for i in range(15):
        engine_state = monad.engine.pulse(flow_vector, energy=1.0, dt=0.1, target_tilt=monad.current_tilt_vector)
        monad._auto_steer_logic(engine_state)

        # Render HUD instead of print
        # Clear screen (Partial) for animation effect
        print("\033[H\033[J", end="")
        hud.render(engine_state)

        history.append(engine_state)
        time.sleep(0.1) # Slow down for visualization

    final_tilt = history[-1].axis_tilt[0]
    if final_tilt > 0.5:
        print(f"\n✅ [PASS] Horizontal Expansion Achieved (Tilt: {final_tilt:.2f})")
    else:
        print(f"\n⚠️ [WARN] Horizontal Expansion Weak (Tilt: {final_tilt:.2f})")
    time.sleep(1)


    print("\n🔥 [TEST 2] Testing Vertical Drilling (High Friction)")
    time.sleep(1)

    # Reset tilt
    monad.current_tilt_vector = [0.0]

    for i in range(20):
        # Generate chaotic vector
        chaos = [random.choice([-1.0, 1.0]) for _ in range(21)]
        chaos_vector = SovereignVector(chaos)

        engine_state = monad.engine.pulse(chaos_vector, energy=1.0, dt=0.1, target_tilt=monad.current_tilt_vector)
        monad._auto_steer_logic(engine_state)

        # Render HUD
        print("\033[H\033[J", end="")
        hud.render(engine_state)

        history.append(engine_state)
        time.sleep(0.1)

    final_tilt = history[-1].axis_tilt[0]
    if final_tilt < -0.5:
        print(f"\n✅ [PASS] Vertical Drilling Achieved (Tilt: {final_tilt:.2f})")
    else:
        print(f"\n❌ [FAIL] Did not drill fully (Tilt: {final_tilt:.2f})")
    time.sleep(1)

    print("\n⚡ [TEST 3] Verifying Momentum Conservation")
    min_momentum = min(h.rotational_momentum for h in history[-20:])
    if min_momentum > 0.0:
        print(f"✅ [PASS] Momentum Conserved (Min: {min_momentum:.4f})")
    else:
        print("❌ [FAIL] Momentum Lost (dropped to 0.0)")

if __name__ == "__main__":
    verify_phase_axis_mobility()
